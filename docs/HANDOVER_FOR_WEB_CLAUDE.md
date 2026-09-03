# 프로젝트 인수인계 — 소아수술실 인계요약지 생성 파이프라인

> **이 문서의 용도**: 리포지토리에 접근할 수 없는 Claude(claude.ai 웹)에게 프로젝트 전체
> 맥락을 한 번에 넘기기 위한 자립형 브리핑. 코드를 읽지 않아도 설계·현황·함정·다음 할 일을
> 판단할 수 있게 썼다. 작성 기준: 2026-09-01, 커밋 `4abd52e`.

---

## 0. 웹 Claude에게 (먼저 읽을 것)

- **PHI 절대 반입 금지.** 이 프로젝트는 서울대병원 소아 수술 환자의 실제 EMR·바이탈을 다룬다.
  실제 수술ID(9자리), 환자 텍스트, `*.pkl / *.xlsx / *.html / *.jsonl` 산출물은 웹에 올리지 않는다.
  이 문서에도 PHI는 없다. 붙여넣을 때 리포트 HTML·엑셀을 같이 올리지 말 것.
- **연구 데이터는 폐쇄망 서버에만 있다.** 웹 Claude는 코드를 실행할 수 없다. 도와줄 수 있는 범위는
  ① 설계·통계·논문 서술 검토, ② 프롬프트/루브릭 문안 개선, ③ 결과 해석과 limitation 정리,
  ④ 코드 스니펫을 붙여넣었을 때의 리뷰 — 이 네 가지다.
- **v1 결과(`sum_score` 기반 순위)는 어떤 맥락에서도 인용하면 안 된다.** 평가셋 유출 + judge 순환 +
  절단 버그가 중첩돼 있다. 유효한 숫자는 v3 계열뿐이다.

---

## 1. 한 문단 요약

소아 수술 종료 후 수술방(OR)에서 회복실(PACU)/중환자실(ICU)로 넘길 때 쓰는 **초간결 인계요약문
(한국어 1~5문장)** 을 LLM이 EMR + 수술 중 바이탈 시계열로부터 자동 생성하게 만드는 연구다.
학술적 질문은 **"SFT와 RLAIF(DPO·SimPO)가 이 과업에서 각각 얼마나 기여하는가"** 이고,
공학적 질문은 **"LLM-as-judge 평가를 임상적으로 신뢰 가능하게 만들 수 있는가"** 다.
현재 코드는 3세대(`pipeline_v3`, v3.1)까지 왔고, 1·2세대는 연구 타당성 결함 때문에 봉인됐다.

- IRB: **E-2601-138-1712** (텍스트 + 바이탈 멀티모달) / DRB: **DRB-E(I)-2026-02-04**
  → ※ 두 문서 모두 **연구자 명단 수정이 아직 안 됐다** (미해결 행정 과제).

---

## 2. 프로젝트 계보 — 왜 v3인가

| 세대 | 코드 | 상태 | 요지 |
|---|---|---|---|
| 다린(선행연구) | `HANDOVER_인계용_다린/` | 참조 전용 | 엄다린 연구원의 원 실험(2026-02). 데이터 전처리·split·체크포인트의 출처. v3 리포트에 **비교군으로 병기**한다. |
| v1 | `pipeline/`, `config.py` | **봉인** | 돌아가긴 했으나 평가셋 유출·judge 순환·프롬프트 절단 버그. 수치 인용 금지. |
| v2 | `config_v2.py`, `pipeline/eval_v2/` | **봉인** | 3축 평가를 도입했으나 "실패를 0점으로 변환"하는 구조적 결함 잔존. |
| **v3** | `pipeline_v3/` | **유일한 유효 코드** | 재설계. 결함 목록·근거는 `CODE_REVIEW_V3_PROPOSAL.md`, 프로토콜은 `PIPELINE_V3.md`. |
| **v3.1** | 〃 (2026-08) | 현재 | 임계값 교과서 근거화 + 이상소견 정량화 + 필수 항목군 + 교수님 재검수 GT. |

### v3 재설계 3원칙 (모든 판단의 기준)

1. **실패는 크게** — 생성/파싱/매칭/추출의 실패를 0.0이나 1.0 점수로 바꾸지 않는다.
   `None` + 플래그 + '제외'로 집계하고, 유효 케이스 비율 < 80%면 평가 자체가 `exit 1`.
2. **한 벌만** — 프롬프트 빌드·절단·chat template·모델 키·태그 파싱은 `pipeline_v3` 안에 단 한 벌.
3. **순환 금지** — 데이터 생성 judge ≠ 평가 judge ≠ 평가 대상 계열.

---

## 3. 데이터 설계

| split | 크기 | 역할 | 원천 파일(서버) |
|---|---|---|---|
| **GOLD** | 22 | **최종 test — 봉인.** `--final`에서만 개봉. 모델/하이퍼 선택 금지 | `gold_sampled_251008.pkl` |
| SFT_SRC (jsft) | 2736 | SFT 타깃 생성용 | `jsft_251008.pkl` |
| SELFJUDGE_SRC | 2736 | **DPO 선호쌍 생성용** (SFT와 분리) | `selfjudge_251008.pkl` |
| RLHF | 150 | few-shot pool 40 + **dev set 110** (seed=42) | `rlhf_251008.pkl` |
| 바이탈 | — | 수술 중 시계열 → 텍스트 요약 맵 | `vitalsheet_260216.pkl` → `vital_summary_map.pkl` |

- 모든 split 로드는 `data_splits.load_splits()` 한 곳을 통과하고, **gold와 교집합이 생기면
  `LeakageError`로 즉시 중단**한다.
- few-shot 예시는 gold가 아니라 **held-out 40건에 gemma4_31b가 zero-shot으로 쓴 초안**
  (`fewshot_bank_v3.json`). 행마다 `random_state=row_idx`로 로테이션해 특정 케이스 스타일이
  학습셋 전체에 박히는 것을 막는다.
- run마다 `split_manifest.json`에 sid의 sha1 지문을 남긴다(PHI 없음).

### 정답(GT)의 출처 — 이게 이 연구의 핵심 자산

1. 김희수 교수님이 22건에 대해 직접 작성한 인계요약지(`인계요약지_..._KHS.xlsx`)의 **c10 피드백 열**만
   정답으로 쓴다. (c9 원안은 참고)
2. 2026-08-14 **재검수본**을 병합해 최종 GT를 확정 → `khs_gold_override.json` (PHI, gitignored).
   병합 방침(연구책임자 확정): 의견 없음=유지 / 보완=병합 / 재작성=교체 / 부정=교체.
   자동 판정이 애매한 케이스는 `merged_conflict`로 표시해 사람이 확정.
   결과: **replaced 6 · replaced_conflict 2 · merged 5 · kept 9 = 22건.**
3. GT 문장 → LLM이 **채점용 checklist 항목**으로 분해 (`gold_checklist_v3.json`).
   `reviewed: true`는 **사람만** 설정 가능. `--accept`는 `accepted_without_review` 플래그를 따로 세운다.
   → 리포트에는 항상 "전문의검수 n / 잠정채택 n / 잠정 n"이 그대로 찍힌다.

> **현재 `reviewed=0`이다.** 즉 지금 나오는 coverage 수치는 전부 "잠정 gold" 기준이다.
> 논문에 정식 보고하려면 전문의가 checklist JSON을 직접 검수해야 한다. **가장 중요한 미해결 과제.**

---

## 4. 실험 매트릭스

**모델 × 7변형**:
`raw` / `sft_1ep` / `sft_3ep` / `rlaif_dpo` / `rlaif_simpo` / `sft_1ep_dpo` / `sft_3ep_dpo`

학습 후보 모델(`TRAIN_KEYS`): `llama`(Llama-3.1-8B-Instruct), `qwen`(Qwen3-8B),
`qwen35`(Qwen3.5-9B), `gemma4`(gemma-4-E4B-it), `gemma4_31b`(gemma-4-31B-it), `hari`(snuh/hari-q3-8b).

- **LoRA** r=8, α=16, dropout=0.1, 표준 7모듈. (v1과 동일 — 비교 연속성 유지)
- SFT: lr 2e-5, bs 2×grad_accum 4, cosine. RLAIF: lr 5e-6, bs 1×8, constant_with_warmup.
- DPO β=0.1 / SimPO β=2.0, γ/β=0.25.
- 토큰 예산 **한 벌**: `MAX_PROMPT_TOKENS=2560`, `MAX_COMPLETION=512`, `MAX_SEQ=3072`,
  추론 `max_model_len=4096`, judge `8192`. SFT collator·DPO(`max_prompt_length`)·SimPO 전부 동일값.
  (v1은 TRL 기본 512로 조용히 잘리고 있었다.)

---

## 5. 파이프라인 5단계

### ① 후보·선호쌍 생성 (`gen_pairs.py`)
- 후보 = 모델당 **greedy 1 + temperature 0.8 샘플 3**, `clean_v3` 통과분만, 정규화 dedup.
  생성 예외/붕괴는 후보에서 **제외**한다(v1처럼 "특이사항 없음"으로 치환하지 않음).
- 채점 judge = **prometheus-8x7b, reference-free.** 남의 환자 기준답안과 비교하던 v1 방식 폐기.
  루브릭 2종: `coverage`(가중 2배; 이상소견 있는데 '특이사항 없음'이면 **1점** → "없음 최적해" 붕괴 차단),
  `fidelity+brevity`(EMR 근거 없는 창작/장황이면 1점).
- judge 입력은 8192 예산에서 **EMR만 좌측절단** — 응답·루브릭은 절대 안 잘린다.
- 점수 파싱은 `[RESULT] n`만 신뢰, 실패 시 후보 drop(준랜덤 점수 주입 금지).
- 쌍 선정: chosen=최고, rejected=최저, **chosen≠rejected AND margin ≥ 2.0** 강제.
  탈락 사유 통계는 `pairs_*_meta.json`, 전 후보 점수는 `*_audit.jsonl`에 남는다.

### ② SFT (`sft_train.py`)
`pairs_sft.pkl`의 chosen을 타깃으로 학습. judge-augmented(A/B) 샘플은 기본 포함,
`--no_judge_aug`로 ablation 가능. 라벨 마스킹은 프롬프트 길이 기반(템플릿 문자열 매칭 아님).

### ③ RLAIF (`dpo_train.py`)
**정책별 on-policy 쌍**을 쓴다 — `rlaif_*`는 base 정책 쌍, `sft_Xep_dpo`는 해당 SFT 정책 쌍.
SFT 데이터 재사용 금지(off-policy 오염 차단).

### ④ 추론 (`inference.py`)
- vLLM 우선(`max_model_len=4096`), HF fallback은 left-truncation/left-padding.
- 레코드에 `engine` 필드를 남기고 **전 변형이 동일 엔진**이어야 한다. 섞이면 경고.
  단, `qwen35`·`gemma4`는 **HF 강제**(`VLLM_INCOMPATIBLE_MODELS`) — 텍스트 LoRA를 멀티모달 base에
  merge하면 vLLM이 arch를 잘못 잡아 raw만 vLLM을 통과한다. 그러면 "학습 효과"가 아니라
  "디코딩 엔진 차이"를 재는 꼴이 되므로 전 변형을 HF로 통일한다.
- LoRA merge 캐시 키 = adapter **내용 해시** → 재학습하면 자동 무효화.
- gold split은 `--allow_gold` 없이는 실행 거부(봉인 장치).

### ⑤ 평가 (`eval_v3/`)
아래 §6 전체.

### 오케스트레이터 Phase 구조 (`run_all_v3.py`)
| Phase | 작업 | GPU 배분 |
|---|---|---|
| A | SFT 학습 | 2-GPU **병렬** |
| B | DPO 선호쌍 생성(prometheus judge, 47B MoE) | **전체 GPU 직렬** (≥3장 필요) |
| C | RLAIF 학습 | 2-GPU **병렬** |
| D | 추론 dev(+gold if `--final`) | 2-GPU **병렬** |
| E | 평가 + 리포트 | eval_gpus / CPU |

같은 정책을 쓰는 여러 loss(dpo/simpo)는 쌍을 1회만 만들어 공유한다.
각 잡은 `.done` 마커 + 체크포인트 유효성으로 skip을 판정한다.

---

## 6. 평가 프로토콜 (연구의 방어선)

### 6.1 3축 composite
`composite = 0.5·coverage + 0.3·faithfulness + 0.2·brevity`

| 축 | 정의 |
|---|---|
| coverage | 전문의 gold checklist의 recall (macro=케이스 평균, micro=항목 pooled 병기) |
| faithfulness | 각 claim이 EMR에 의해 지지되는가(entailment) |
| brevity | 과설명·행정 노이즈 감점 |

### 6.2 안전 게이트
- **이상소견 케이스인데 "특이사항 없음"** → coverage=0 **그리고** faithfulness=0
  (그 주장 자체가 EMR과 모순) → composite=0.0, `gate="missed_abnormal"`.
- 생성 붕괴(빈/반복/garbage/프롬프트 누출) → 0.0, `gate="degenerate"`.
- normal 케이스 + 실질 보고 → coverage 정의불가(None), faith+brev 재정규화.
- `is_no_issue_v3()`는 **문자열 전체 일치**만 no-issue로 본다.
  "특이사항 없음. intraop VT 발생" → no-issue 아님(v2는 접두 매칭이라 오판했다).

### 6.3 "실패는 점수가 아니다"
| 상황 | v2(버그) | v3 |
|---|---|---|
| judge JSON 파싱 실패 | coverage 0.0 | 축=None, 케이스 **제외** + `judge_failed` |
| coverage verdict에 항목 누락 | 누락=no 처리 | judge 실패로 간주 → 제외 (항목 id 대조검증) |
| sid 매칭 실패 | normal-case 간주 → 1.0 가능 | **하드 에러**(파이프라인 중단) |
| checklist 추출 실패 | items=[] → coverage 1.0 | `gold_llm_failed` → 제외 |
| gold 없음 | faith+brev로 채점 | 제외(별도 표에 명시) |

유효 케이스 비율 < `MIN_VALID_CASE_RATIO(0.8)` → 평가 `exit 1`. 성적이 아니라 인프라 문제로 취급.

### 6.4 교차 judge (순환 방지)
- 평가 judge = **gemma4_31b + qwen35 둘 다**로 전 출력을 채점.
- 공식 점수 = **평가 대상과 다른 family의 judge만** 평균
  (gemma 계열 대상 → qwen35만, qwen 계열 → gemma4_31b만, llama → 둘 평균).
- 두 judge 일치도(Spearman ρ / MAD)를 변형마다 **항상** 보고한다.
- 선호쌍 judge는 prometheus(mixtral 계열) — 학습 신호 judge ≠ 평가 judge이므로
  v1의 "평가 judge에 직접 최적화(reward hacking)" 해석 불능 문제가 제거된다.

### 6.5 프롬프트 주입 방어
- faithfulness의 premise는 **EMR 원문 + vital summary만** (생성 지시문 미포함).
- 모델 출력은 `<<<MODEL_HANDOFF>>>` 구분자로 감싸 "지시가 아니라 채점 대상 텍스트"임을 명시.

### 6.6 통계
- 축별 **bootstrap 95% CI** (케이스 단위, B=10,000).
- 변형 vs `{model}_raw`: **paired permutation test**(공통 유효 케이스, 20,000회) + **Holm 보정**.
  보조로 t·Wilcoxon(전부-0 가드 포함). seed=42.

### 6.7 Calibration (gold 개봉 전 필수 관문)
`eval_v3/calibrate.py` — SY 엑셀의 전문의 채점 168행을 **두 judge 모두로** 재채점해
judge별 × 생성모델 family별 Spearman 표(`calibration_v3.csv`)를 만든다.
**방향이 뒤집힌 축이 있으면 루브릭을 고치고 재실행**한다. 여기서 통과 못 하면 gold를 열지 않는다.

---

## 7. v3.1에서 바뀐 것 (2026-08-11 ~ 08-24)

### 7.1 바이탈 임계값을 교과서 전거로 전면 재설정
전거: **Smith's Anesthesia for Infants and Children 9e (2021)**, **Miller's Anesthesia 10e (2024)**.
표·페이지 단위 근거는 `docs/THRESHOLDS.md`, 코드 단일 출처는 `utils/vital_thresholds.py`.

판정은 **2-tier** — `[유의]`(소생·개입 기준 초과 = 임상적 유의) / 표시 없음(연령별 참조범위 이탈).

| 항목 | `[유의]` 기준 | 정상범위 기준 | 전거 |
|---|---|---|---|
| HR | 서맥 <60 · 빈맥 >220/190/180/150 | 연령별 mean±2SD(9구간) | Smith T.57.3 / T.18.1 |
| SBP | 신생아<60 · 영아<70 · 1–10세 <70+2×age · >10세 <90 | 고혈압 = 95th pct 초과 | Smith T.57.3 / T.18.2 |
| MBP | `min(1.5×age+40, 65)` 미만 | — | 관례식 + Miller Ch.4 |
| DBP | **판정 안 함** (소아 하한 문헌 없음) | 고혈압 = 95th pct | Smith T.18.3 |
| SpO2 | <90% | 목표미달 90–93%(목표 94–99%) | Smith Ch.57 |
| T1 | <35.5 · >38.0°C | 저체온<36.0 · >37.5 | Smith Ch.21 / Ch.7 |
| QTc | >480 ms | >470(신생아)/>440 | Miller / Smith Ch.5 |
| UO | 핍뇨 <0.5 mL/kg/hr | — | Miller Ch.24 |
| EBL | >10% EBV · >50% EBV | — | Smith T.21.6 / Ch.18 |
| Ppeak | **판정 안 함** (문헌 없음) | — | — |

**왜 중요한가**: v1 임계값은 출처 불명이라 정상 영아를 빈맥으로, 정상 학령전아를 서맥으로
오표기했다. QTc는 단일 기준이라 임상적으로 무의미한 "연장"을 대량 생성해 gold checklist를 오염시켰다.
DBP 하한·Ppeak처럼 **문헌 근거가 없는 값은 판정 자체를 삭제**한 것이 설계 판단이다.

### 7.2 이상소견 정량화 — "몇 회"에서 "얼마나 오래·얼마나 벗어남"으로
모든 이벤트에 **최장 episode·총 지속시간·기준 대비 편차**를 병기한다
(Miller Ch.4의 *time below threshold* 방식).

예: `[유의]저혈압(<70): 4회 20분(최장 15분), 최저 55 (기준-15)`

> 여기서 핵심은 **생성 프롬프트·brevity judge·gen_pairs 루브릭을 함께 고쳤다**는 점이다.
> 안 고치면 coverage와 brevity가 서로 반대로 당기고, DPO가 정량화 표현을 학습에서 오히려 빼버린다.

### 7.3 인계문 필수 항목군 6개
`기저질환·약물` / `기도관리` / `수술 중 이벤트 및 처치` / `수혈·수액` / `수술 전 검사이상` / `감기 유무`

**조건부 필수** — EMR에 소견이 있는 군은 반드시 전달하고, **없는 군은 "없음"조차 쓰지 않는다**
(brevity 축과 '특이사항 없음' 규칙을 보호하기 위해).
생성 프롬프트·checklist 추출·coverage 채점 **3곳에 동시 반영**돼 있고,
coverage는 항목군별 recall(`category_coverage` / `missed_categories`)을 함께 낸다.
`missed_categories`는 **진단 지표이지 감점 요소가 아니다** — 안전게이트로 승격할지는 전문의 검수 후 결정.
정의는 `docs/REQUIRED_CATEGORIES.md`, 코드는 `pipeline_v3/required_categories.py`.

### 7.4 프로덕션 버그 3건 (전부 "코드는 맞는데 실제로는 빠지던" 종류)
1. **UO 속도 5배 과대평가** — v1은 `rate = total / (weight × 기록건수 / 60)`, 즉 *기록 1건=1분*을
   가정했다. 실제 기록 간격이 5분 안팎이라 속도가 ~5배 부풀어 **핍뇨가 단 한 건도 검출되지 않았다.**
   → 실제 타임스탬프 경과시간으로 수정.
2. **`build_vital_map`이 `weight_kg`를 안 넘김** — EBL의 EBV 대비 %와 핍뇨 판정이 프로덕션에서
   통째로 빠져 있었다. → 나이/체중 매핑 배선 + 매핑 실패 시 명시적 경고.
3. **EBL 절대 mL의 무의미성** — 5 kg 영아의 100 mL ≈ EBV의 24%. EBV 대비 %를 병기하도록 수정.

### 7.5 학습 전멸 원인 — CUDA 스트림 race
260824 run에서 llama·qwen SFT 4변형이 전부 죽어 21변형 중 13종만 나왔다. OOM이 아니라
**비동기 CUDA 스트림 race**(loss 계산 중 unspecified launch failure)였고, 대응 env
`CUDA_LAUNCH_BLOCKING=1`이 `GEMMA4_BASES`에만 걸려 있어 gemma4만 7변형 전부 생존했다.
→ **전 모델로 확대 적용.** 끄려면 `HANDOVER_NO_LAUNCH_BLOCKING=1` (속도↔안정성 트레이드오프).

### 7.6 측정오류 배제는 "좁게"만
생성 인계문에 `T1 최저 20.3°C`, `SpO2 42%`가 실려 나왔다. 하지만 **이 코호트는 소아 심장수술을
포함**하므로 DHCA 저체온 18~20°C와 청색성 심질환 SpO2 40~70%는 **실제 값**이다.
임상적으로 드문 값을 지우면 정작 인계해야 할 소견이 사라진다.
→ 물리적으로 불가능하거나 센서 이탈로만 설명되는 값만 배제(`HR>300`, `SpO2>100`, `T1<14 or >45`,
`BP<5`). 애매한 값은 남긴다. 정의는 `PLAUSIBLE_RANGE`, 배제 건수는 `DROPPED_ARTIFACTS`에 기록.

### 7.7 프롬프트 토큰 예산
필수 항목군·정량화 지시로 지시문이 커지자 few-shot이 붙는 `gen_pairs`에서 EMR을 전부 비워도
2560 토큰을 넘겨 `PromptTruncationError`가 났다.
→ 지시문 2641→1968자 압축(6개 항목군 상세는 SYSTEM에만, USER는 한 줄 참조),
few-shot 예시 EMR 1500→700자(`PAIRGEN["fewshot_emr_chars"]`).
`scripts/check_prompt_budget.py`로 실제 tokenizer 기준 고정비용·EMR 여유를 **사전 측정**한다
(몇 시간 돌린 뒤 죽는 것을 방지).

---

## 8. 현재 상태 (2026-09-01 기준)

### 8.1 마지막 run 결과 — **전부 v3.0 기준이며 v3.1 재실행 전이다**

**dev (n=110), run 260824** — SFT 변형이 CUDA race로 죽어 13종만 산출:

| tag | n(유효/제외) | composite [95% CI] | faith | brev | judge ρ | vs raw (Holm) |
|---|---|---|---|---|---|---|
| llama_rlaif_dpo | 110/0 | **0.906** [0.878,0.932] | 0.930 | 0.871 | 0.648 | +0.300 (p<0.001) ✓ |
| llama_rlaif_simpo | 110/0 | 0.856 [0.824,0.886] | 0.858 | 0.852 | 0.612 | +0.249 (p<0.001) ✓ |
| llama_raw | 110/0 | 0.607 [0.578,0.634] | 0.609 | 0.602 | 0.489 | — |
| qwen_raw | 110/0 | 0.820 [0.797,0.842] | 0.782 | 0.877 | 0.571 | — |
| qwen_rlaif_dpo | 110/0 | 0.816 [0.793,0.839] | 0.792 | 0.852 | 0.545 | -0.004 (p=0.730) |
| qwen_rlaif_simpo | 110/0 | 0.789 [0.762,0.815] | 0.781 | 0.800 | 0.549 | -0.031 (p=0.046) ✓ |
| gemma4_sft_3ep_dpo | 109/1 | 0.762 [0.729,0.793] | 0.827 | 0.665 | 0.737 | +0.083 (p=0.001) ✓ |
| gemma4_raw | 110/0 | 0.682 [0.652,0.710] | 0.730 | 0.609 | 0.454 | — |
| gemma4_sft_3ep | 109/1 | 0.483 [0.436,0.530] | 0.707 | 0.147 | 0.551 | -0.198 (p<0.001) ✓ |
| gemma4_sft_1ep | 103/7 | 0.480 [0.430,0.527] | 0.763 | 0.056 | 0.489 | -0.204 (p<0.001) ✓ |

**gold (n=22), run 260709** — 참고용(잠정 gold 기준):

| tag | composite [95% CI] | coverage(macro/micro) | faith | brev | vs raw (Holm) |
|---|---|---|---|---|---|
| qwen_rlaif_dpo | 0.568 [0.500,0.641] | 0.240/0.218 | 0.763 | 0.909 | +0.024 (p=0.561) |
| qwen_rlaif_simpo | 0.565 [0.499,0.634] | 0.236/0.268 | 0.787 | 0.886 | +0.021 (p=0.561) |
| llama_rlaif_dpo | 0.547 [0.482,0.626] | 0.023/0.035 | 0.971 | 0.903 | +0.162 (p=0.004) ✓ |
| llama_rlaif_simpo | 0.527 [0.454,0.611] | 0.081/0.063 | 0.854 | 0.881 | +0.148 (p=0.007) ✓ |
| qwen_raw | 0.544 [0.480,0.615] | 0.206/0.190 | 0.746 | 0.886 | — |
| gemma4_raw | 0.413 [0.358,0.475] | 0.080/0.070 | 0.756 | 0.557 | — |
| llama_raw | 0.389 [0.317,0.468] | 0.202/0.232 | 0.550 | 0.506 | — |

### 8.2 이 숫자들을 읽을 때의 경고

1. **v3.0 산출물이다.** v3.1에서 임계값·프롬프트·gold가 전부 바뀌었으므로 v3.1 결과와
   직접 비교하면 안 된다. composite 변화에 "모델 개선"과 "측정 기준 변화"가 섞인다.
2. **dev의 coverage가 `—`인 것은 정상이다.** dev(110)에는 전문의 gold checklist가 없어
   coverage가 정의불가이고, faith+brev를 재정규화한 값이 composite다.
   **즉 dev composite과 gold composite은 다른 척도다.** 절대 나란히 놓고 비교하지 말 것.
3. **gold의 coverage 절대값이 매우 낮다**(0.02~0.24). 잠정 gold checklist가 LLM 부트스트랩
   상태라 항목이 과다 생성됐을 가능성이 크다. 전문의 검수 전에는 이 값을 성능으로 해석하지 않는다.
4. **judge 일치도 ρ가 0.45~0.85로 넓다.** 낮은 쪽(raw 변형, ρ≈0.45~0.49)은 두 judge가 사실상
   다른 것을 재고 있다는 뜻이므로 해당 행의 결론은 약하게 써야 한다.
5. **llama_rlaif_dpo의 gold coverage 0.023 vs faith 0.971**은 전형적인 신호다 —
   "안전하지만 아무것도 말하지 않는" 출력으로 수렴했을 가능성. 케이스 원문 확인 필요.

### 8.3 무엇이 끝났고 무엇이 남았나

**끝남**: v3.1 코드 전부(임계값·정량화·항목군·GT override·버그수정·CUDA 안정화·토큰예산),
재실행 런북, 검토 자료 추출기, 테스트 17종.

**남음** (우선순위 순):
1. **v3.1 전체 재실행** — `docs/RERUN_RUNBOOK.md` 절차대로. 아직 안 돌렸다. 지금 가진 수치는 전부 구식.
2. **gold checklist 전문의 검수** (`reviewed: true`) — 이게 안 되면 coverage는 논문에 못 쓴다.
3. **calibration 통과 확인** — judge↔전문의 Spearman 방향 검증. gold 개봉 전 관문.
4. **IRB/DRB 연구자 명단 수정.**
5. **보안 마무리**: ① `.env`의 GitHub PAT revoke 후 파일 삭제,
   ② repo private 전환 또는 `git filter-repo`로 과거 이력의 수술ID 스크럽
   (이미 push된 이력은 새 커밋으로 지워지지 않는다).
6. `missed_categories`를 안전게이트로 승격할지 결정(전문의 검수 후).

---

## 9. 리포지토리 지도

```
01_HANDOVER/
├── README.md                     # 시작점 (설치·실행·임계값 표·PHI 규칙)
├── PIPELINE_V3.md                # v3 프로토콜 전체 (이 문서의 원본 중 하나)
├── PROMPTS_V3.md                 # 전 프롬프트 원문 레퍼런스 (생성/쌍생성/평가/checklist)
├── VITAL_SUMMARY_RULES.md        # 바이탈 → 텍스트 요약 생성규칙
├── CODE_REVIEW_V3_PROPOSAL.md    # v1/v2 결함 목록 + v3 재설계 근거 (P0~P2)
├── docs/
│   ├── THRESHOLDS.md             # 임계값 전거 (Smith/Miller 표·페이지 단위)
│   ├── REQUIRED_CATEGORIES.md    # 필수 항목군 6개 정의·근거·반영지점
│   ├── UPGRADES_v3.1.md          # v3.1 고도화 내역
│   ├── RERUN_RUNBOOK.md          # v3.1 재실행 절차서  ← 다음 작업의 출발점
│   └── references/               # 교과서 PDF (저작권 — gitignored)
├── pipeline_v3/                  # ★ 유일한 유효 코드
│   ├── config_v3.py              # 단일 설정 소스 (경로/모델/토큰예산/하이퍼/judge배정)
│   ├── data_splits.py            # split 로드 + LeakageError
│   ├── prompt_utils.py           # 프롬프트 빌드·EMR 좌측절단 (한 벌)
│   ├── required_categories.py    # 6개 항목군 단일 출처
│   ├── make_fewshot_bank.py      # held-out 40건 few-shot 뱅크
│   ├── gen_pairs.py              # 후보 생성 + prometheus 채점 + 쌍 선정
│   ├── sft_train.py / dpo_train.py / merging.py
│   ├── inference.py              # vLLM/HF, engine 기록, gold 봉인
│   ├── build_gold_checklist_v3.py
│   ├── eval_v3/{engine,checklist,metrics,cleaning,stats,evaluate,calibrate}.py
│   ├── report_v3.py              # 리치 HTML/MD/CSV 리포트
│   ├── provenance.py             # 내용해시 + 동일출력 경고 + git rev
│   └── run_all_v3.py             # Phase A~E 오케스트레이터
├── utils/
│   ├── vital_thresholds.py       # ★ 임계값 단일 출처
│   ├── vital_summarizer.py       # 바이탈 시계열 → 텍스트 요약
│   └── download_models.py, verify_output.py, check_seq_length.py
├── scripts/
│   ├── install_hooks.sh          # pre-commit PHI 가드 (필수)
│   ├── invalidate_v3.py          # 임계값 변경 시 캐시 무효화
│   ├── rebuild_vital_map.py      # vital_summary_map 재생성 + 자가진단
│   ├── import_khs_review.py      # 교수님 재검수본 → GT 병합/분류
│   ├── build_gold_override.py    # 확정본 → khs_gold_override.json
│   └── check_prompt_budget.py    # 토큰 예산 사전 점검
├── export_gold_cases_full.py     # gold 케이스별 완전 EMR 추출 (사람 검토용, 캡 없음)
├── reinfer_darin_on_v3sids.py    # 선행연구 체크포인트를 v3 sid에 재추론 (병기 비교용)
├── tests_v3/test_v3.py           # 17개 회귀 테스트
├── HANDOVER_인계용_다린/          # 선행연구 패키지 (PHI 포함 — gitignored)
├── pipeline/, config.py, config_v2.py   # ← 봉인. 수정·인용 금지
└── data/, outputs*/, old_outputs_v3/    # PHI — gitignored
```

### 테스트 17종이 지키는 것
config import 부작용 없음 / no-issue 전체일치 / clean_v3 오탐·진탐 / `missed_abnormal` 게이트 /
normal 케이스 만점 / degenerate는 0점이지 제외 아님 / no_gold·추출실패 제외 / judge 실패 제외 /
coverage verdict id 대조검증 / 교차 judge 배정 / 통계 가드 / micro coverage pooling /
쌍 선정 margin·동일성 / 프롬프트 EMR-only 절단 / 예산 불가 시 loud fail.

---

## 10. 실행 커맨드 (폐쇄망 서버)

```bash
git pull
bash scripts/install_hooks.sh                        # PHI pre-commit 가드 — 필수
pip install -r requirements.txt

export HANDOVER_BASE_DIR=/home/coder/workspace/data/handover
export HANDOVER_MODEL_DIR=/home/coder/workspace/data/local_models
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6   # ★ 세션당 1회, 셸에서 export

# 준비물 (1회)
python -m pipeline_v3.make_fewshot_bank        --gpus 0,1,2,3
python -m pipeline_v3.build_gold_checklist_v3  --gpus 0,1,2,3
python -m pipeline_v3.eval_v3.calibrate        --gpus 0,1,2,3

# SFT 타깃 생성 (1회 공유)
python -m pipeline_v3.gen_pairs --split sft --models llama qwen --gpus 0,1,2,3

# 학습 → 쌍 → RLAIF → 추론 → dev 평가  (모델 선택은 여기까지로만)
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 --skip_done

# 최종 1회 — gold 22 개봉 (dev로 선택 끝난 뒤에만)
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 --skip_done --final

# 선행연구(다린) 병기 최종 리포트
python reinfer_darin_on_v3sids.py --gpus 0,1 --split gold --skip_done
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 \
    --skip_done --final --include_source --include_darin
```

**v3.1 재실행이면 위를 그냥 돌리면 안 된다** — `docs/RERUN_RUNBOOK.md`:
① 재검수 GT 반영 → ② `invalidate_v3.py --apply --archive` → ③ `rebuild_vital_map.py --apply`
(자가진단 4줄 확인) → ④ 준비물 재생성 → ⑤ 학습~dev → ⑥ gold → ⑦ 검토본 재수출.

### 산출물 레이아웃
```
{WORK_DIR}/outputs_v3/<run_id>/
├── split_manifest.json                # split 지문 (sid 해시, PHI 없음)
├── pairs/pairs_sft.pkl, pairs_dpo_*.pkl, *_audit.jsonl, *_meta.json
├── sft/<model>_<ep>ep/{final/, .done}
├── rlaif/<model>_<...>/{final/, .done}
├── inference/<model>_<exp>/{dev,gold}_results.jsonl   # engine 필드 포함
├── evaluation/<model>_<exp>/*_scores_v3.jsonl + _summary_v3.json
├── report/results_{dev,gold}_v3.{csv,md,html}
└── provenance_v3.json                 # 내용해시 + 동일출력 경고 + git rev
```

---

## 11. 알려진 함정 (사람이 실제로 밟았던 것들)

| 함정 | 증상 | 대응 |
|---|---|---|
| `LD_PRELOAD` 미설정 | vLLM이 **조용히** HF로 폴백해 극단적으로 느려짐 | 셸에서 export(프로세스 시작 시점에만 유효). `strings $CONDA_PREFIX/lib/libstdc++.so.6 \| grep -c GLIBCXX_3.4.29` ≥1 확인 |
| `--skip_done`을 v3.1에서 그대로 사용 | 옛 `vital_summary_map.pkl` 재사용 → **새 임계값이 전혀 반영 안 됨**, 조용히 틀린 결과 | 먼저 `invalidate_v3.py --apply --archive` |
| vLLM/HF 엔진 혼용 | raw만 vLLM 통과 → "학습 효과"가 아니라 **디코딩 엔진 차이**를 잼 | `VLLM_INCOMPATIBLE_MODELS`로 전 변형 HF 통일, `engine` 필드 확인 |
| CUDA 스트림 race | SFT가 loss 계산 중 unspecified launch failure로 전멸 | `CUDA_LAUNCH_BLOCKING=1` 전 모델 적용(기본값) |
| 토큰 예산 초과 | 몇 시간 돌린 뒤 `PromptTruncationError` | `scripts/check_prompt_budget.py`로 사전 측정 |
| `rebuild_vital_map` 자가진단 무시 | 체중 미매핑 시 EBL %·핍뇨가 **통째로 생략**됨 | 출력의 `체중 컬럼 사용` / `나이 매핑 n/N` / `[유의] 포함 n/N` / `지속시간 표기 n/N` 4줄 확인. 0이면 중단 |
| 완전 EMR과 모델 입력 혼동 | `export_gold_cases_full.py` 출력은 캡 없는 **사람 검토용 원본**, 모델 실제 입력은 좌측절단본 | 리포트 근거로 인용할 때 구분 |

---

## 12. 논문 작성 체크리스트

보고 전에 반드시:
- [ ] v1 `sum_score` 순위는 어디에도 인용하지 않는다.
- [ ] gold checklist `reviewed` 상태를 표에 명시한다 (잠정이면 "잠정"이라고 쓴다).
- [ ] calibration 표(judge↔전문의 Spearman, family별)를 첨부한다.
- [ ] 두 judge 일치도(ρ)와 공식 점수의 judge 배정 규칙을 명시한다.
- [ ] 제외 케이스 수와 사유(judge 실패 / no_gold)를 명시한다 — 조용히 빼지 않는다.
- [ ] CI와 Holm 보정 p를 함께 쓴다. "n=22 평균 한 줄"로 결론 내지 않는다.
- [ ] 전 변형이 동일 추론 엔진임을 `engine` 필드로 확인한다.
- [ ] v3 점수를 RLAIF 보상으로 직접 쓰지 않았음을 명시한다(쌍 생성은 prometheus 경로).

### Limitations에 쓸 것
- gold **n=22** — CI가 넓다. dev(110)로 선택하고 gold는 확증에만 썼다.
- gold checklist가 전문의 검수 전이면 LLM 부트스트랩에 의한 계통 편향 가능.
- gemma 계열 평가 대상은 qwen35 **단독** judge — judge 1개 분산.
- few-shot bank가 모델 초안(스타일 앵커) — 전문의 검수로 업그레이드 권장.
- vital summarizer의 나이 파싱·UO 가정은 v1 자산 재사용분이 남아 있다.
- MAP 소아 절대 기준은 관례식(de Graaff 2016 원표 미확보), <1세 고혈압 기준은 관례값,
  미숙아 EBV·혈압 참조범위와 baseline 대비 상대 저혈압은 미도입.

---

## 13. 용어집

| 용어 | 뜻 |
|---|---|
| **인계요약지/인계문** | OR→PACU/ICU 구두·문서 인계 요약. 목표 출력물. 한국어 1~5문장 |
| **sid** | 수술ID(9자리). **PHI** — 코드·문서에 절대 쓰지 않는다 |
| **다린** | 엄다린 연구원의 선행연구(2026-02). 전처리·split·체크포인트의 출처이자 비교군 |
| **KHS c10** | 김희수 교수님 gold 엑셀의 피드백 열. **정답의 유일한 소스** |
| **jsft** | judge-augmented SFT. 합성데이터에 A/B judge 내용이 포함된 지도학습 |
| **selfjudge** | 모델이 자기 출력 2개 중 나은 것을 골라 학습하는 RLAIF 방식(선행연구 명칭) |
| **RLAIF** | AI 피드백 기반 강화학습. 여기서는 DPO·SimPO 두 loss |
| **prometheus** | 선호쌍 생성 전용 judge(8x7B MoE, mixtral 계열). 평가 judge와 분리 |
| **composite** | 0.5·coverage + 0.3·faithfulness + 0.2·brevity |
| **게이트** | `missed_abnormal`(이상소견을 '없음'으로 보고) / `degenerate`(생성 붕괴). composite=0 |
| **제외** | judge 실패·gold 부재 등 **인프라 문제**. 점수가 아니라 별도 집계 |
| **`[유의]`** | 소생·개입 기준 초과 = 임상적으로 유의한 바이탈 이벤트 (구 표기 ⚑) |
| **EBV** | estimated blood volume. EBL은 절대 mL이 아니라 EBV 대비 %로 판정 |
| **DHCA** | deep hypothermic circulatory arrest. 저체온 18~20°C가 **정상 소견**인 케이스 |
| **봉인(seal)** | gold 22건은 `--final` + `--allow_gold` 없이는 추론조차 거부된다 |

---

## 14. 웹 Claude에게 시킬 만한 일

**적합**
- 논문 Methods/Results/Limitations 초안 작성 및 논리 검토 (이 문서의 §6·§8·§12가 재료).
- 평가 프로토콜의 통계적 타당성 리뷰 — paired permutation + Holm, bootstrap CI, judge 순환 차단.
- 프롬프트·루브릭 문안 개선 (원문은 `PROMPTS_V3.md`를 붙여넣어 주면 된다).
- §8.2의 이상 신호 해석 — 특히 "높은 faithfulness + 낮은 coverage" 패턴 진단.
- 미해결 임계값 항목(MAP 소아 기준, 영아 고혈압, 미숙아 EBV)에 대한 문헌 탐색 방향 제안.

**부적합**
- 실제 데이터가 필요한 모든 것(재실행·수치 재계산·케이스 검토) — 폐쇄망 서버에서만 가능.
- PHI가 포함된 리포트 HTML·엑셀 분석.
- v1/v2 코드 수정 — 봉인 상태다.
