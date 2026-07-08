# 코드 리뷰 종합 + v3 재설계 제안

> 2026-07-06. 세 영역(학습 파이프라인 / 평가 v2 / 오케스트레이션·위생)을 병렬 정밀 리뷰한 결과.
> 핵심 주장(few-shot 유출, 게이트 부재, DPO 프롬프트 절단, PHI 노출 등)은 코드/데이터를 직접 열어 재검증함.

---

## TL;DR — 결론 3줄

1. **연구 타당성**: 현재 실험 매트릭스(README의 11개 변형 비교)는 그대로 보고하면 안 된다.
   평가셋 22케이스가 학습데이터 생성에 유출되었고(few-shot + judge reference), 선호쌍을 만든 judge와
   최종 평가 judge가 같으며(순환), DPO는 프롬프트가 512토큰으로 잘려 사실상 EMR 없이 학습됐다.
2. **안전성**: v2 평가가 v1의 "빈 출력이 만점" 역전을 절반만 고쳤다. 문서(PIPELINE_V2.md)가 약속한
   안전 하드게이트는 코드에서 제거되어 존재하지 않고, 이상소견 케이스에 "특이사항 없음"이라 답해도
   composite 0.5를 받는다.
3. **PHI/보안**: 환자 실명·등록번호가 든 `results_cases_v2.html`과 실제 수술ID가 든
   `gold_checklist_v2.json`이 repo 루트에 gitignore 없이 놓여 있었다(→ 이번에 .gitignore 보강 완료).
   그러나 **이미 public GitHub에 push된 파일들에 실제 수술ID·희귀진단명이 들어 있다**(아래 P0-3).
   `.env`의 GitHub PAT는 즉시 폐기 권장.

---

## 0. 이번 리뷰에서 이미 조치한 것

- `.gitignore` 패턴 기반으로 보강: `*.html`, `*.jsonl`, `*.log`, `gold_checklist*.json`,
  `calibration*.csv`, `khs_dump*`, `data/`, `report_v2/`, `logs_*/` 추가.
  (`_v2`처럼 버전이 붙으면 정확명 규칙을 빠져나가는 사고의 재발 방지)
- `git check-ignore`로 `results_cases_v2.html`, `gold_checklist_v2.json`이 이제 무시됨을 확인.
- 이 문서 작성.

---

## P0 — 즉시 조치 필요 (PHI / 보안)

### P0-1. repo 루트의 PHI 파일 (gitignore는 했지만 파일 자체 처리 필요)
- `results_cases_v2.html`(670KB): **환자 실명, 병원등록번호, 성별/나이, 신장/체중, 집도의 실명** +
  22케이스 전체 EMR 원문. (실제 값 검증됨 — 이 문서에는 미기재)
- `gold_checklist_v2.json`: 실제 수술ID 키 + 희귀 진단명, 수술일자, 수술중 이벤트.
- **조치**: 두 파일을 repo 밖(또는 `outputs/` 밑 원래 자리 `data/preprocessed/`)으로 이동.
  원래 export 경로는 정상(`utils/export_results_v2.py:251` → `REPORT_V2_OUT`) — 루트 복사 습관이 원인.

### P0-2. `.env` = 평문 GitHub PAT
- 40바이트짜리 `ghp_…` 토큰 하나. gitignore돼 있고 커밋된 적은 없으나(git 이력 전수 확인),
  코드 어디서도 이 파일을 읽지 않는다 — 순수 자격증명 방치.
- **조치**: GitHub에서 해당 토큰 **즉시 revoke** 후 파일 삭제. `gh auth login` 또는
  fine-grained PAT + credential helper로 대체.

### P0-3. **이미 public에 push된** 실제 수술ID·진단명 (gitignore로 해결 불가)
- `config_v2.py:80-82`: `KHS_GOLD_REMAP`에 실제 수술ID 2건 + 희귀진단명 언급 주석 (→ 조치됨: 데이터 폴더 JSON 로드로 치환)
- `pipeline/eval_v2/checklist.py:11`: docstring에 실제 수술ID + 실제 수술명/소견 (→ 조치됨: 플레이스홀더로 치환)
- `PIPELINE_V2.md:61`: 희귀진단 케이스 명시 (→ 조치됨: 일반 표현으로 치환)
- 소아 전문센터에서 희귀 선천질환 + 내부 수술ID 조합은 재식별 벡터이며 DRB/IRB 조건 위반 소지.
- **조치(택1, 병행 권장)**: ① repo를 private로 전환(가장 빠름), ② sid를 가짜 플레이스홀더로 치환하고
  `git filter-repo`로 이력 재작성 후 force-push. gitignore 수정으로는 해결되지 않는다.
- **재발 방지**: pre-commit hook으로 staged 파일에서 8자리 sid 패턴/등록번호 grep → 차단.

### P0-4. 기타 보안
- `utils/download_models.py:159-180`: `_patch_httpx_ssl()`이 **무조건** TLS 검증을 끔 —
  HF 토큰이 미검증 TLS로 전송. env 플래그(`HANDOVER_INSECURE_SSL=1`)로 게이트하거나
  `REQUESTS_CA_BUNDLE`로 병원 프록시 CA 지정.
- `khs_to_txt.py`: 환자ID 컬럼 포함 gold 엑셀을 CWD의 `khs_dump.txt`로 덤프(모듈 최상위에서 실행됨 —
  `__pycache__`에 임포트 흔적 존재). 출력 경로를 ignored 디렉토리로 옮길 것.

---

## P1 — 연구 타당성을 깨는 문제 (Critical)

### 학습 데이터 생성 (`pipeline/01_synthetic_gen.py`)

**T1. 평가셋 유출** (`01_synthetic_gen.py:165-184, 195`)
- few-shot 예시를 `GOLD_REF_PKL`(= 평가 전용 22케이스)에서 뽑는다. `random_state=RANDOM_SEED` 고정이라
  **같은 gold 케이스 2~4건의 EMR+인계문이 2736행 모든 생성 프롬프트에 들어감**.
- 결과: 모든 SFT/DPO 학습 타깃이 평가셋의 내용·스타일 영향 아래 생성됨. 22케이스뿐인 평가에서
  "SFT가 raw보다 좋다"는 결과의 일부는 평가셋 스타일 암기를 측정한 것.
- **수정**: few-shot은 미사용 `RLHF_PKL`(150케이스) 등 held-out에서, 행마다 로테이션(`random_state=idx`).
  `00_preprocess.py`에 `assert gold_ids ∩ sft_ids == ∅` 추가.

**T2. Judge 기준답안이 남의 환자 것 하나로 고정** (`01_synthetic_gen.py:97-114, 204`)
- `_get_handoff()`가 gold_df에서 **첫 번째** 비어있지 않은 인계문 하나를 뽑아 2736케이스 전부의
  "5점 기준답안"으로 사용. fallback 컬럼은 마취기록 원문(모델 입력과 동일 필드), 최종 fallback은
  리터럴 `"특이사항 없음"`.
- 선호쌍(SFT/DPO의 전체 감독신호)이 임상적으로 틀린 기준으로 채점된 것.
- **수정**: 케이스별 KHS c10 gold를 쓰거나, eval_v2처럼 해당 케이스 EMR 대비 reference-free 채점.

**T3. "특이사항 없음" 편향 + 에러가 학습 정답으로 둔갑** (`01_synthetic_gen.py:73-83, 243-245, 157-162`)
- 두 루브릭 모두 "짧고 None이면 만점" 구조, 사실 커버리지 축 없음 → "특이사항 없음" 상시 출력이 최적해.
- 생성 예외 시 `response = "특이사항 없음"` 대입 → 크래시 플레이스홀더가 chosen으로 뽑혀 학습 타깃이 됨.
- 점수 파싱 실패 시 "피드백 텍스트의 마지막 1~5 숫자 아무거나" fallback → 준랜덤 점수 주입.
- **수정**: 커버리지/사실성 루브릭 추가, 예외는 후보 제외(치환 금지), 파싱 실패는 drop+로깅.

**T4. 쌍 생성 judge 입력이 4096토큰 우측 절단** (`01_synthetic_gen.py:146-148`)
- 긴 EMR 케이스에서 **채점 대상 response·기준답안·루브릭이 통째로 잘림** → 파싱 실패 → 둘 다 3.0 →
  chosen/rejected가 `random.choice`. 임상적으로 가장 복잡한 케이스일수록 선호 라벨이 랜덤.
- v1 평가(`05_evaluate.py`)는 같은 버그를 이미 고쳤는데(8192 + EMR만 절단) 쌍 생성 쪽은 방치됨.

### 학습 (`pipeline/02_sft_train.py`, `03_rlaif_train.py`, `04_inference.py`)

**T5. DPO 프롬프트가 512토큰으로 절단** (`03_rlaif_train.py:250-259`)
- SimPO는 `max_prompt_length=1792` 명시(`:231`), DPO는 미설정 → TRL 기본값 512.
- EMR+vital 프롬프트는 보통 1~2.5k토큰 → **DPO는 프롬프트 꼬리 512토큰(≈ 출력 지시문)만 보고 학습**.
  EMR 없이 선호를 학습하므로 "EMR 무관하게 특이사항 없음" 정책을 조장. DPO vs SimPO 비교도 3.5배
  컨텍스트 차이로 confound.
- **수정**: `DPOConfig(max_prompt_length=1792, ...)` — SimPO와 동일하게.

**T6. HF 추론 경로 우측 절단 → 프롬프트 끝(### OUTPUT + assistant 헤더)이 잘림** (`04_inference.py:421-426`)
- 학습은 `truncation_side="left"`(`02_sft_train.py:249-250`), HF 추론은 기본 right + 2048 →
  긴 케이스에서 생성 지시가 사라지고 모델이 EMR을 "이어쓰기". v1의 `종료`, `50mc totals` 류
  쓰레기 출력의 유력 원인. vLLM 경로(4096, 무절단)와도 달라 엔진 간 결과 비교 불가.
- **수정**: `load_model()`에서 `truncation_side="left"` + 두 엔진 컨텍스트 예산 통일.

**T7. judge 순환 평가** (`01_synthetic_gen.py` ↔ `05_evaluate.py`, 둘 다 `EVAL_JUDGE_MODEL`)
- 선호쌍을 만든 Prometheus + 같은 루브릭이 최종 `sum_score`도 매김 → 학습된 모델은 평가 지표의
  judge에 직접 최적화된 상태. v1 수치는 reward hacking과 구분 불가.
- v2도 유사 문제: `EVAL_V2_LLM = gemma4_31b`가 **평가 대상 모델이자, gold checklist 부트스트랩 모델이자,
  gold의 원안(c9=gemma-3-27b) 계열** — 4중 순환(`config_v2.py:36`, `checklist.py:82-117`).
- **수정**: 교차 채점(Qwen이 gemma 출력을, gemma가 Qwen 출력을) 또는 2-judge 앙상블 + 일치도 보고.
  168행 전문의 점수로 **모델 계열별** calibration Spearman 필수.

**T8. 퇴화 선호쌍** (`01_synthetic_gen.py:233, 258-274`)
- 후보 = 2모델 × 2 shot, 전부 greedy → 동일 문자열 빈발. `chosen == rejected` 검사 없음, 점수 마진 없음,
  동점은 랜덤. DPO 입장에서 상당수가 무의미 쌍 or 라벨 노이즈.
- **수정**: temperature 샘플링으로 후보 확대, dedup, `score(chosen) − score(rejected) ≥ margin` 요구.

**T9. SFT와 DPO가 같은 데이터 재사용** (`03_rlaif_train.py:136` = `02_sft_train.py:240`, 둘 다 `SYNTH_PKL`)
- SFT→DPO 구성에서 DPO의 chosen은 방금 지도학습으로 loss≈0까지 외운 문자열 → 그래디언트가 rejected
  억제로만 쏠려 퇴행 위험. 원래 이 단계용으로 보이는 `RLHF_PKL`(150행)은 전혀 사용 안 됨.

### 평가 v2 (`pipeline/eval_v2/`, `06_evaluate_v2.py`)

**E1. 문서가 약속한 안전 하드게이트가 코드에 없음** (`metrics.py:198-220` "(게이트 없음)" ↔ `PIPELINE_V2.md:37-39`)
- 커밋 `7f70665`에서 게이트 제거, 문서는 미갱신 → PIPELINE_V2.md가 "원천 차단"이라고 **거짓 기술**.
  `V2_SAFETY_VIOLATION_CAP`, `_SEV_W`, `safety_violation` 읽기(`build_gold_checklist.py:153`)는 죽은 코드.
- 문서·코드 불일치 상태로 교수님/임상진에 결과를 보여주면 안 됨.

**E2. 이상소견 케이스 + "특이사항 없음" = composite 0.5** (`06_evaluate_v2.py:77-85`)
- fast path가 `faithfulness=1.0, brevity=1.0` 부여. 그러나 VT/수혈이 있는 EMR에 "특이사항 없음"은
  **모순된 주장**(이 파이프라인 자신의 faithfulness 정의상 0이어야 함).
- 0.5면 성실하지만 불완전한 인계(cov 0.3/faith 0.7/brev 0.8 ≈ 0.52)와 통계적 동률 — v1 역전이 절반 생존.
- **수정(권장)**: 이 분기에서 `faithfulness=0.0` → composite ≈ 0.1. 마법 상수 cap 없이 문서의 게이트가
  자연 구현됨. 문서도 함께 갱신.

**E3. 실패가 점수로 둔갑하는 경로들**
- judge JSON 실패 → `parse_coverage(None)` → 전 항목 "no" → **coverage 0.0** (인프라 장애 = 임상 전부 누락으로 기록,
  `metrics.py:64-88`). faithfulness/brevity는 None→0.5로 처리되는 것과 비대칭.
- sid 매칭 실패 → `is_normal_case=True` fallback (`06:64-68`) → "특이사항 없음" 출력이면 **composite 1.0**.
  v1의 "빈 출력 만점"이 데이터 배관 오류로 재유입되는 뒷문.
- checklist 추출 실패(`source="gold_llm_failed"`, `checklist.py:188`) → items=[] → **coverage 1.0** (`06:135-136`).
- **수정 원칙**: "조용히 낮게/높게"가 아니라 **크게 실패** — `None` + `judge_failed` 플래그, 평균에서 제외.

**E4. `clean_v2` 오탐으로 정상 인계문이 0점** (`engine.py:114-140`, 실행 검증됨)
- 조사(josa) 휴리스틱: `"…저혈당 주의"`처럼 의/로/과/가로 끝나는 정상 문장 → `truncated` → 0점.
- 문자벽: 마크다운 구분선 `--------` → `repetition` → 0점. (`_RE_CODE_FENCE`는 정의만 되고 미사용)
- 외국문자 비율: 곱슬따옴표·전각문장부호·호환자모가 허용셋에 없어 garbage 오탐 가능.
- n=22에서 오탐 2~3건이면 모델 순위가 뒤집힌다.
- **수정**: truncated는 0점 대신 정상 채점+플래그, `-`/`=`/`#` 연속은 제외, 허용 문자셋 확장.

**E5. `is_no_issue` 양방향 오탐** (`config_v2.py:190-203`, 실행 검증됨)
- `"특이사항 없음. intraop VT 발생"`(40자 미만) → True → **VT를 언급했는데도** coverage 0 fast path.
- `"None significant except bleeding"` → True(startswith "none") → 정상 케이스면 자동 composite 1.0.
- **수정**: 접두 매칭 금지, 전체 내용(구두점 제거 후)이 no-issue 문구와 일치할 때만 True.

**E6. `--accept`가 "전문의 검수"를 조작** (`build_gold_checklist.py:87-91`)
- LLM 부트스트랩 gold 21/22건에 `reviewed=true` 자동 세팅 → 리포트가 "전문의 검수 21건"으로 표시.
  의학 AI 논문에서 과학적 진실성 문제.
- **수정**: `accepted_without_review=true` 별도 플래그, 잠정 경고 유지. `reviewed`는 사람만 설정.

**E7. faithfulness premise가 EMR이 아니라 생성 프롬프트 전체** (`06_evaluate_v2.py:49-53`)
- "특이사항 없음만 출력하라 … ### OUTPUT" 지시문이 judge의 '원본 EMR'로 주입됨 — 지시문 오염 + 주입 위험.
  모델 출력도 구분자 없이 judge 프롬프트에 삽입(`metrics.py:41-53` 등) — v2 점수를 RLAIF 보상으로 쓰는 순간
  reward hacking 표준 경로가 열림.
- **수정**: premise = `build_emr_text + vital`만. 출력은 명시적 구분자로 감싸고 verdict id를 checklist와 대조 검증.

**E8. 통계 부재** (`06:187-190`)
- n≈19(유효)인데 평균만 출력. v1엔 있던 paired test(`05:775-843`)가 v2엔 없음. 케이스별 항목수
  1~9개 편차를 macro 평균으로 뭉갬. calibration은 n≥5에서 Spearman 출력(무의미).
- **수정**: 케이스 bootstrap CI + paired permutation test, 항목 pooled micro 평균 병기.

---

## P2 — 주요 버그 (High/Medium, 요약)

| # | 위치 | 문제 | 수정 |
|---|---|---|---|
| B1 | `run_all_v2.py:43-54` | provenance 해시가 파일명+크기만 사용 → **모든 LoRA 변형이 같은 해시**(존재 이유 무효) | 내용 sha1 |
| B2 | `run_all.py:109-112, 175-213` | `_ckpt_exists`가 config 파일만 확인 → 크래시 잔해를 "완료"로 스킵; 부분 작성 jsonl도 영구 동결 | safetensors 존재+행수 검증, tmp→atomic rename, DONE 마커 |
| B3 | `04_inference.py:228-231` | LoRA merge 캐시가 adapter 갱신을 감지 못함 → **옛 가중치로 평가** 가능 | adapter 해시/mtime 키 |
| B4 | `run_all.py:549-550` | t임계값 1.984 하드코딩(n=22면 2.080), wilcoxon 전부-0 diff에서 크래시 | `t.ppf(0.975, n-1)`, guard |
| B5 | `engine.py:240-259` | HF fallback: right padding+우측 절단 — v1이 고친 버그(`05:177-192`)를 v2가 재도입 | left pad, EMR만 절단 |
| B6 | `engine.py:261-286` | 재시도가 greedy에 동일 프롬프트 재전송 → 보장된 no-op | vLLM guided_json으로 대체 |
| B7 | `vital_summarizer.py:308-321` | 나이 파싱 1건 실패 시 **전 케이스 5세 기준** 적용(서맥/빈맥 오판) | 원소별 coerce, 미상은 명시 |
| B8 | `vital_summarizer.py:239-245` | UO rate가 "차팅 행수=분" 가정(시간당 차팅이면 60배 과대) — 현재는 dead code지만 지뢰 | timestamp 기반, cumulative 여부 확인 |
| B9 | `vital_summarizer.py:181-192` | 치명 desat 있으면 90–95% 구간 카운트 소실; 이벤트에 아티팩트 필터 없음 | 양 구간 병기, k연속/중앙값 필터 |
| B10 | `metrics.py:83-84` | coverage "partial"이 +0.5 받으면서 missed에도 기록 → 리포트 누락 과대 | partial 별도 목록 |
| B11 | `config.py:11-12` | import 시 서버 경로 mkdir → **로컬에서 아무것도 import 불가**(테스트 전멸) | env override + lazy mkdir |
| B12 | `smoke_test.py:295,344` | 존재하지 않는 모듈 import + `importlib.util` 몽키패치(죽은 깨진 코드); 실패해도 exit 0 | 삭제 + exit code |
| B13 | `tests/test_sft.py` | 파이프라인과 무관한 평행 재구현(lr도 2e-4로 다름) — 통과해도 증명하는 것 없음 | 삭제 |
| B14 | `download_models.py:60` | 모델 키 `gemma4-31b`(하이픈) vs 전역 `gemma4_31b` → CLI 에러 | config에서 파생 |

중복/드리프트: 모델 키 4벌(`config.py:97`/`export_results_v2.py:48`/`smoke_test.py:562`/`download_models.py:60`),
`_split_tag` 3벌, EMR 빌더 3벌, chat template 헬퍼 3벌, no-issue 판정 2벌(`diagnose_v2.py:40-47`는 또 다른 정의),
리포트 렌더러 3벌. requirements.txt 없음(vllm, openpyxl 누락), README 파일맵은 `pipeline/` 이동 전 기준으로 낡음.

---

## v3 재설계 제안 (우선순위 로드맵)

### 원칙
- **실패는 크게**: 어떤 실패(생성/파싱/매칭/추출)도 0.0이나 1.0 같은 점수로 변환하지 않는다. None + 플래그.
- **한 벌만**: 프롬프트 빌드·절단·chat template·모델 키·태그 파싱은 단일 모듈. v1은 legacy로 봉인.
- **순환 금지**: 데이터 생성 judge ≠ 평가 judge ≠ 평가 대상 계열. gold는 사람 검수 전엔 항상 "잠정" 표기.

### Phase 0 — 오늘 (PHI/보안, 코드 변경 거의 없음)
1. PAT revoke + `.env` 삭제. repo private 전환(또는 P0-3 sid 스크럽 + 이력 재작성).
2. 루트의 `results_cases*.html`, `gold_checklist_v2.json` 이동. (gitignore는 보강 완료)
3. pre-commit hook: staged 파일에서 `\b1005\d{5}\b`·등록번호 패턴 검출 시 커밋 차단.

### Phase 1 — 평가를 먼저 고친다 (모델 재학습 전에 자로 쓸 것이므로)
1. **E2 수정**이 최소 핵심: 이상소견+no-issue fast path에서 faithfulness=0 → 문서의 게이트가 구현됨.
   PIPELINE_V2.md를 코드에 맞게 갱신(severity 가중 제거 사실 포함), 죽은 게이트 상수 삭제.
2. 실패 경로 전부 `None`+플래그로 (E3), `clean_v2` 오탐 수정 (E4), `is_no_issue` 전체 일치로 (E5).
3. 교차 judge (T7/C3): gemma 출력은 Qwen3.5가, 나머지는 gemma가 채점 — 계열별 편향을 상관표로 보고.
4. **calibration을 실제로 실행**: 168행 전문의 점수 ↔ v2 judge Spearman(배치화, `build_gold_checklist.py:140-153`
   순차 호출 수정). 가중치 0.5/0.3/0.2는 이 회귀로 근거 부여.
5. bootstrap CI + paired permutation test. 전문의 checklist 실검수(`--accept` 조작 금지, E6).

### Phase 2 — 학습 데이터 재생성 (가장 레버리지 큰 작업)
1. few-shot을 held-out(RLHF_PKL)에서, 행별 로테이션 (T1). preprocess에 gold 분리 assert (M1).
2. 케이스별 gold(KHS c10) 또는 EMR-기반 reference-free 채점으로 선호쌍 재구축 (T2/T3).
   후보는 SFT 정책에서 temperature 샘플링(vLLM 배치) + dedup + 점수 마진 (T8) + 중간 체크포인트 저장.
3. 절단 통일: 단일 `prompt_utils.py` — `truncation_side="left"`, 컨텍스트 예산 하나,
   `### OUTPUT`+generation header 생존 assert. DPO `max_prompt_length=1792` (T5, T6).
4. SFT/DPO 데이터 분리 (T9): SYNTH를 스테이지별로 나누거나 진짜 RLAIF(정책 자체 출력에서 쌍 생성).

### Phase 3 — 재실험
- 22 gold는 **최종 1회 비교 전까지 봉인**. 모델 선택은 별도 dev set(RLHF_PKL 일부)으로.
- provenance: 내용 해시(B1) + merge 캐시 무효화(B3) + 부분 산출물 검증(B2)을 로드 시점에 강제.
- 결과 보고: 3축 분리 + CI + 계열별 judge 일치도 + "gold 잠정/검수" 상태 명시.

### Phase 4 — 정리 (병행 가능)
- `orchestration.py`(GpuPool/run_cmd/phase1/_split_tag/_ckpt) + `report_common.py`로 v1/v2 공통화.
- `tests/test_sft.py`, `monitor.py`, `scripts/run_train.sh·launch.sh`(깨진 3번째 오케스트레이터) 삭제 또는 재작성.
- smoke_test exit code, GPU-free 부분 pytest화(내용 해시 테스트가 B1을 잡았을 것).
- `requirements.txt` 생성(vllm, openpyxl 포함), README를 `pipeline/` 구조·run_all 진입점 기준으로 갱신.
- `config.py` env override(`HANDOVER_WORK_DIR`) — 이것만으로 로컬에서 테스트 가능해짐.

### 하지 말 것
- v1 sum_score 기반 순위를 어디에도 보고하지 말 것 (T1/T2/T7이 겹쳐 해석 불능).
- v2 점수를 그대로 RLAIF 보상으로 쓰지 말 것 (E7 주입 방어 + E2 게이트 수정 전까지).
- gold checklist를 LLM으로 재부트스트랩하며 `reviewed`를 자동 세팅하지 말 것.

---

## 부록: 발견 전체 목록 위치

이 문서는 종합본이며, 영역별 상세(파일:줄 + 코드 발췌 + 개별 수정안)는 리뷰 세션 로그에 있음.
심각도 분류: P0(보안/PHI) 7건, Critical(타당성) 17건, High 11건, Medium 12건, Low/위생 20여 건.
