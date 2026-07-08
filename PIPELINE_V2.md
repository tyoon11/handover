# 파이프라인 v2.0 — 평가 재설계 (개선안 1~6 반영)

> v1(`config.py`, `pipeline/05_evaluate.py`, `run_all.py`, `utils/export_results.py`)은 **그대로 보존**.
> v2는 별도 모듈/산출물로 추가되어 병행·비교·롤백이 가능하다.
> 전제: **완전 로컬(API 불가)**, 학습 ≤4 GPU, judge/추론성 평가 ≤6×A6000(48GB).

---

## 0. 왜 v2인가 (v1 진단 요약)

v1은 원본 노트북을 충실히 이식했고, 그래서 원본의 **설계 결함도 그대로 상속**했다:

- 평가가 "짧을수록 + '특이사항 없음'이면 만점"을 강제 → judge가 EMR 사실 대조를 안 함.
- 결과: 이상소견이 있는 케이스에서 **`특이사항 없음`·`종료` 같은 빈/조각 출력이 sum=10**,
  실제로 옳은 임상 핸드오프(intraop VT·강심제·수혈 등)는 하위권. **안전성과 역전.**
- SCALE(영어 Flan-T5)은 한국어 출력에서 0.1~0.4 노이즈 — 변별 불가.
- `clean_output`이 멀쩡한 출력을 조각내고(`를 종료합니다.`), 짧으면 `특이사항 없음`으로 강제치환.

**핵심 자산 발견**: `인계요약지_SY.xlsx` 에 **22 케이스 전부의 전문의 gold 인계문 +
전문의 평가점수(168행) + 전문의 채점 루브릭**이 있었다. v2는 이걸 gold·검증 기준으로 쓴다.
(전문의 루브릭 III-2가 "특이사항 있는데 없다고 퉁치는 경우"를 명시 감점 → v1 진단과 일치.)

---

## 1. v2 평가 설계 — 3축 + 안전 하드게이트

`pipeline/eval_v2/metrics.py`. 전문의 루브릭에 정렬:

| v2 축 | 전문의 루브릭 | 측정 | 비고 |
|---|---|---|---|
| **COVERAGE** (안전 핵심) | III. 설명 부족 | gold checklist 항목 recall (severity 가중) | 놓침=위험 |
| **FAITHFULNESS** | I. 내용의 오류 | 출력 claim의 EMR entailment (환각/모순) | SCALE 대체 |
| **BREVITY** | II. 과도한 설명 | 추론성 권고·약잔량·이송·정상지표 부연 등 노이즈 | 보조 |

- **합산(sum)으로 뭉개지 않는다.** 세 축을 따로 보고하고, composite는
  `0.5·COV + 0.3·FAITH + 0.2·BREV` (가중치 `config_v2.V2_WEIGHTS`).
- **안전 하드게이트**(`V2_SAFETY_VIOLATION_CAP=0.15`): 이상소견 케이스인데
  `특이사항 없음`으로 응답하거나 **고위험 항목을 통째로 놓치면** composite를 상한 절단 +
  `safety_violation=true`. → v1의 "빈 출력이 만점" 역전을 원천 차단.
- **생성 실패**(빈/반복/잘림)는 LLM 호출 없이 즉시 0점(`clean_v2` status로 판정).
  → `종료`·`같은말 반복`·`50mc totals` 류가 더는 만점 못 받음.

엔진: **gemma-4-31B**(로컬 재사용) 하나가 추출/Coverage/Faithfulness/Brevity를 모두 수행.
모든 호출은 배치 + 구조화 JSON 강제 + 재시도(`eval_v2/engine.py`).

---

## 2. Gold actionable checklist (Coverage 정답 기준)

`build_gold_checklist.py` → `data/preprocessed/gold_checklist_v2.json`

**gold 소스 = KHS c10(교수님 피드백)만 사용** (SY는 gold에서 제외):
- `인계요약지_gold_sampled_251002_KHS.xlsx`('데이터' 시트, 다중헤더 3행, 케이스당 1행):
  - c9 `인계요약지_sample_from_LLM` = gemma-3-27b-it 원안(정답 아님)
  - **c10 = 교수님이 c9를 보고 준 피드백 = 진짜 gold** ← checklist 항목의 유일한 출처
  - c11 `인계요약지` = 거의 공란(미사용)
- **추출 원칙(중요)**: checklist 항목 집합 = *교수님 gold가 담은 것만*. EMR은 약어 풀이/근거
  인용에만 쓰고, **gold에 없는 소견(특히 QTc 연장·일시적 desat·혈압/심박 이벤트 카운트 등
  vital 파생)은 추가 금지**. (1차 부트스트랩이 QTc를 13/22 케이스에 과다 포함시킨 문제를 교정.)
- gold가 '특이사항 없음'이면 `is_normal_case=true`. c10이 공란인 케이스(옆 행에 잘못
  입력된 사례 존재 — remap JSON으로 보정)는 `source:"no_gold"`로 표시하고 **수기 작성 대상**으로 남김.
- (참고) `인계요약지_SY.xlsx`는 gold로 안 쓰고, 전문의 평가점수 168행만 `--calibrate` 검증에 사용.
- 각 항목: `{id, finding, category, severity, source}` + 케이스 `is_normal_case`.
- **전문의가 직접 수정 가능한 JSON.** 검수 후 `reviewed: true`로 바꾸면 정식 기준이 된다.
  (검수 전엔 리포트/로그에 "잠정(LLM 부트스트랩)" 경고 표시.)

```bash
python build_gold_checklist.py --inspect            # SY 엑셀 컬럼 확인
python build_gold_checklist.py --gpus 0,1,2,3       # checklist 생성
```

### 판정이 사람과 맞는가? (calibration — 개선안 1·2 검증)
SY의 모델출력 168행은 전문의 점수를 갖고 있다. v2 judge로 같은 출력을 채점해
**Spearman 상관**을 본다(judge↔전문의). 방향 불일치 시 루브릭/프롬프트를 고친다.

```bash
python build_gold_checklist.py --calibrate --gpus 0,1,2,3
#  → data/preprocessed/calibration_v2.csv + 상관표 출력
```

---

## 3. 생성 위생 (개선안 4)

`pipeline/04_inference.py --decode v2` (opt-in, 기본 v1 보존):
- vLLM/HF 모두 `repetition_penalty`(+HF `no_repeat_ngram_size`) 적용 → 반복 루프 억제.
- 정제는 `clean_v2`: think 블록만 안전 제거, **greedy preamble 삭제·강제치환 폐기**,
  빈/반복/잘림을 `gen_status`로 **명시 기록**(점수 단계에서 0점 처리).

---

## 4. 실험 위생 (개선안 5)

`run_all_v2.py`가 Phase1 후 `provenance_v2.json` 생성:
- 변형별 **체크포인트 해시**(safetensors 크기 기반) → "정말 다른 가중치인가" 점검.
- **학습변형 간 출력 동일성** 점검 → 100% 동일하면 경고(학습 효과 없음/체크포인트 의심).
- self-consistency 디코딩 옵션은 `config_v2.INFER_SELF_CONSISTENCY`(향후 확장 지점).

---

## 5. 리포트 (개선안 6)

`utils/export_results_v2.py` → `report_v2/results_cases_v2.html`
- **안전축 우선 정렬**(composite↑, 안전위반 하단·빨강 강조).
- 케이스별 **gold 필수항목**, 모델별 **누락(miss)·환각(hallucination)·노이즈** 상세.
- `sum`/`scale_xl` 같은 v1의 오해 유발 지표 제거.

---

## 6. GPU 레이아웃 / 실행

| 단계 | GPU | 이유 |
|---|---|---|
| 학습(SFT/RLAIF) | `train_gpus` ≤4 | 오래 걸림. 8B=1장, 31B=2~4장 |
| 추론 | train 풀 재사용 | vLLM TP |
| **평가(31B judge)** | `eval_gpus` = **4장이면 충분** | gemma4-31B bf16≈62GB → 4장 TP면 됨 |

```bash
# 0) 최초 1회 — checklist 생성 (+ 전문의 검수 권장)
python run_all_v2.py --build_checklist --eval_gpus 0,1,2,3

# 1) 전체: 학습+추론(4GPU, decode=v2) → v2 평가(4GPU) → HTML
python run_all_v2.py --models llama qwen35 gemma4_31b \
    --train_gpus 0,1,2,3 --gpus_per_job 1 \
    --eval_gpus 0,1,2,3 --skip_done

# 추론이 이미 있을 때 평가+리포트만
python run_all_v2.py --models llama --eval_gpus 0,1,2,3 --only_eval

# 평가만 수동
python pipeline/06_evaluate_v2.py --gpus 0,1,2,3 \
    --result_files outputs/<run>/inference/*/gold_results.jsonl
```

---

## 7. 산출물

```
outputs/<run_id>/
├── inference/<model>_<exp>/gold_results.jsonl     # gen_status 포함(decode v2)
├── evaluation_v2/<model>_<exp>/gold_results_scores_v2.jsonl
│       └─ coverage·faithfulness·brevity_v2·composite·safety_violation
│          missed_items·hallucinations·noise·is_normal_case
├── provenance_v2.json                             # 체크포인트 해시·출력 다양성
└── report_v2/results_cases_v2.html
data/preprocessed/
├── gold_checklist_v2.json                         # 전문의 수정 대상
└── calibration_v2.csv                             # judge↔전문의 상관
```

---

## 8. 파일 맵 (신규)

| 파일 | 역할 | 개선안 |
|---|---|---|
| `config_v2.py` | 경로·엔진·가중치·디코딩·SY 컬럼·루브릭 매핑 | 1~6 |
| `pipeline/eval_v2/engine.py` | gemma-4-31B vLLM/HF + 배치 JSON + `clean_v2` | 2,3,4 |
| `pipeline/eval_v2/checklist.py` | SY 로더 + LLM 부트스트랩 + JSON I/O | 1 |
| `pipeline/eval_v2/metrics.py` | 3축 + 안전 하드게이트 | 1,2,3 |
| `pipeline/06_evaluate_v2.py` | 배치 평가 CLI | 1,2,3 |
| `build_gold_checklist.py` | checklist 빌드/inspect/calibration | 1,2 |
| `pipeline/04_inference.py` | `--decode v2`(opt-in) | 4 |
| `utils/export_results_v2.py` | 안전축 HTML | 6 |
| `run_all_v2.py` | 오케스트레이터 + provenance/다양성 | 5 |

---

## 9. 남은 작업 (사람 손이 필요한 곳)

1. **전문의 checklist 검수** — `gold_checklist_v2.json`을 열어 항목/severity/normal 여부를
   확인하고 `reviewed:true`로. (검수 전 결과는 잠정치.)
2. **calibration 확인** — `--calibrate` 상관표에서 방향 불일치 축이 있으면 해당 루브릭
   프롬프트(`metrics.py`)를 조정.
3. **SY 엑셀 경로** — 서버에 `data/gold_sampled/인계요약지_SY.xlsx` 배치(없으면 repo 루트 fallback).
4. (선택) 케이스 22건은 분산이 큼 — gold 확대 시 통계 신뢰도↑.
```
