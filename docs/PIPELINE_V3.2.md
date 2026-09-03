# v3.2 파이프라인 재정립

작성 2026-09-02 · 기준 커밋 `4abd52e` (v3.1) · 상태: 1·2·3·6 확정 / 7 결정 대기

v3.1은 "임계값·평가 기준을 임상적으로 근거 있게" 만든 릴리스였다. v3.2는 **학습 신호(합성데이터)의
품질과 평가의 공정성**을 손본다. 바꾸는 축은 7개다.

| # | 결정 | 성격 |
|---|---|---|
| 1 | 모든 프롬프트를 영어로 (출력은 한국어 유지) | **적용 완료** |
| 2 | 합성데이터 생성을 대형 teacher 모델로 | **확정: Qwen 72B AWQ4** |
| 3 | 생성 시 **GT를 few-shot 예시**로 제공 | **확정: gold 22 → few-shot 6 / test 16** |
| 4 | raw 출력으로 **프롬프트를 먼저 확정**하고 동결 후 실험 | 설계 확정, 구현 예정 |
| 5 | 합성데이터를 **사람이 검토**한 뒤 학습 진행 | 설계 확정, 구현 예정 |
| 6 | judge를 **전 모델 동일 패널**로 고정하고 평균 | **확정: prometheus + medgemma27b + llama70b** |
| 7 | 합성데이터 chosen/rejected 판정 방식 재설계 | 비용·규칙 구체화 (§7), **결정 대기** |

---

## 1. 프롬프트 영어화 — 적용 완료

**규칙**: 지시·루브릭·판정기준은 전부 영어. 모델 **출력은 한국어**. 프롬프트에 한국어가 남는 곳은
세 군데뿐이고 전부 '데이터 리터럴'이다.

1. 필수 출력 문구 `"특이사항 없음"` — 이 문자열이 평가 로직(`is_no_issue_v3`)의 판정 기준이다.
2. vital summary의 유의 이벤트 마커 `[유의]` — 입력 데이터 안의 토큰이다.
3. 한국어 출력 형식 예시 `"20분간 저혈압(최저 55mmHg)"` — 출력 형식 규정이므로 한국어여야 한다.

**바뀐 파일**

| 파일 | 내용 |
|---|---|
| `pipeline_v3/prompt_utils.py` | `SYSTEM_PROMPT`·`build_user_prompt` 전면 영어화 + `PROMPT_SPEC_VERSION="v3.2-en"` |
| `pipeline_v3/required_categories.py` | 6개 항목군의 **영어 라벨·설명** 추가(`CATEGORY_EN`), `prompt_block(lang="en")` 기본 |
| `pipeline_v3/eval_v3/metrics.py` | coverage·faithfulness·brevity 템플릿 영어화, 항목 태그 영어 라벨 |
| `pipeline_v3/eval_v3/checklist.py` | gold checklist 추출 템플릿 영어화 (단 `finding`/`source`는 **한국어로 산출** — 전문의 검수 대상이므로) |
| `pipeline_v3/prompts_pairgen.py` (신규) | gen_pairs judge 템플릿·루브릭을 별도 모듈로 분리 (프롬프트 '한 벌' 원칙) |
| `pipeline_v3/prompt_registry.py` (신규) | 프롬프트 전체 목록 + 내용 지문(sha1), 동결 게이트용 |

**왜 영어인가** — ① 지시문 토큰이 줄어 EMR 좌측절단이 덜 일어난다(한국어는 BPE에서 글자당 토큰
비용이 크다), ② llama/gemma 계열은 영어 지시 준수율이 한국어보다 높다, ③ judge 계열
(prometheus·gemma·llama)이 영어 루브릭에 맞춰 학습돼 있어 루브릭 해석 편차가 줄어든다.

```bash
python -m pipeline_v3.prompt_registry              # 지문 + 프롬프트별 길이/한글자수
python -m pipeline_v3.prompt_registry --dump /tmp/prompts   # 사람 검토용 전문 덤프
python scripts/check_prompt_budget.py --models llama qwen35 gemma4   # ★ 서버에서 필수 재확인
```

> **주의**: 프롬프트가 바뀌었으므로 `gold_checklist_v3.json`, `fewshot_bank_v3.json`,
> `calibration_v3.csv` 는 **전부 재생성**해야 한다. 옛 산출물과 섞으면 조용히 틀린 결과가 난다.
> `prompt_registry.assert_same()` 이 산출물 메타의 지문과 코드 지문을 대조해 이걸 막는다.

---

## 2. 합성데이터 생성 = 대형 teacher 모델 — **Qwen 72B AWQ4 확정**

### 2.1 역할 분리 (핵심)

| 용도 | 모델 | 근거 |
|---|---|---|
| **SFT 타깃 생성** | **teacher = `qwen72b`** (Qwen 72B Instruct AWQ4) | 타깃 품질이 SFT 상한을 정한다. 8B 자기생성 타깃은 상한이 낮다 |
| **DPO/SimPO 쌍 생성** | **학생 정책 자신** (on-policy 유지) | teacher 출력을 chosen으로 쓰면 RLAIF가 아니라 증류다 — 연구 질문이 무너진다 |
| 선호쌍 judge | `qwen72b` (§6.5 참조) | 평가 패널(mixtral/gemma/llama)과 계열이 겹치지 않는 유일한 선택 |
| gold checklist 추출기 | `qwen72b` (`CHECKLIST_EXTRACTOR`) | 채점자가 아니라 GT→항목 변환기. 엄격 JSON이 필요해 가장 강한 모델 |
| 평가 judge | `prometheus` + `medgemma27b` + `llama70b` | 전 모델 동일 패널 (§6) |

**teacher는 SFT 타깃에만** 들어간다. 논문 서술도 "SFT = teacher 증류 + rejection sampling,
RLAIF = on-policy 선호학습"으로 정확히 쓴다. 이 경계가 흐려지면 `sft_3ep_dpo` 의 이득이
증류 효과인지 선호학습 효과인지 구분할 수 없다.

### 2.2 배치 (A6000 48G × 6 = 288GB, Ampere)

- A6000은 **FP8 하드웨어 가속이 없다** → 양자화는 **AWQ/GPTQ INT4**(Marlin 커널) 또는 bf16.
- Qwen 72B AWQ4 ≈ **40GB** → **TP=2 × 3인스턴스** 동시 가동이 가능하다. 후보 생성은 배치
  처리량 문제이므로, 큰 모델 1인스턴스보다 이 구성이 3배 가깝게 빠르다.
- TP 크기는 attention head 수의 약수여야 한다 → 72B는 TP=2/4/8 가능, **TP=6은 불가**.
- `gen_pairs` 는 현재 `tensor_parallel_size=torch.cuda.device_count()` 로 고정이므로,
  **3인스턴스 병렬을 쓰려면 `--gpus 0,1` / `2,3` / `4,5` 로 프로세스를 3개 띄우고 행 범위를
  나눠주는 옵션(`--row_shard i/n`)이 필요하다** (§9 구현대기 4번).
- AWQ 모델은 vLLM에 `quantization="awq_marlin"` (또는 자동감지)로 로드된다. `model_quant()` 가
  레지스트리에서 포맷을 준다.

> **확인 필요**: 폐쇄망 미러의 실제 경로/버전. `config_v3.MODELS["qwen72b"]["dir"]` 기본값은
> `Qwen2.5-72B-Instruct-AWQ` 다. 미러에 더 최신(예: Qwen3 계열 72B급 AWQ)이 있으면
> `dir`/`repo` 만 바꾸면 되고 나머지 코드는 그대로다.

### 2.3 순환 방지

teacher 계열(qwen)은 **평가 패널에 없다**(mixtral/gemma/llama) — 우연이 아니라 그렇게 고른
결과다. teacher 문체를 배운 학생이 같은 계열 judge에게 후한 점수를 받는 self-preference 경로가
차단된다. 대신 llama70b judge ↔ llama 학생, medgemma judge ↔ gemma 학생은 계열이 겹치므로
**LOO 민감도 분석**(§6.4)이 필수다.

### 2.4 예상 비용 (2736행 기준, 후보 4개/행)

생성 ~11k건, 입력 ~28M 토큰. Qwen72B-AWQ TP=2×3인스턴스면 **1~2시간**.
judge 비용이 더 크므로 §7에서 구조를 바꿔 줄인다.

## 3. GT를 few-shot 예시로 — **gold 22 → few-shot 6 / test 16 확정**

### 3.1 결정과 그 대가

전문의 GT는 gold 22건에만 있다(교수님 재검수 override 22/22 확보, 중위 100자).
GT를 few-shot으로 쓰면서 유출을 막는 방법으로 **gold를 GT few-shot 6 / 최종 test 16으로
분할**하기로 했다.

- 구현: `data_splits.split_gold()` — sid 정렬 → GT의 '특이사항 없음' 여부로 **층화** →
  seed=42 결정적 분할. `load_splits()["gold"]` 는 이제 **test 16**을 돌려주므로 추론·평가·리포트가
  자동으로 16건만 본다. 22 전체는 `["gold_full"]`, few-shot 몫은 `["gold_gt_fewshot"]`.
- 교집합은 `assert_disjoint` 로 하드 차단하고, 두 집합의 sid 해시는 `split_manifest.json` 에 남는다.
- 층화를 넣은 이유: n=22에서 무작위로 6을 뽑으면 정상('특이사항 없음') 케이스가 한쪽에 몰려
  test 16의 구성이 크게 흔들린다.

**대가를 명시적으로 적어둔다.**

1. 최종 test **n=22 → 16**. paired permutation 검정력이 떨어진다 —
   n=16에서는 중간 효과크기(d≈0.5)를 잡을 확률이 50%를 밑돈다. 따라서 **gold는 "확정용"이 아니라
   "dev 결론의 확인용"** 이고, 주 결론은 dev(110)에서 나온다는 점을 리포트에 못박는다.
2. v3.1 이전 gold(n=22) 숫자와 **직접 비교 금지** — 케이스 집합이 다르다.
3. coverage는 gold 16건에서만 정의된다(dev에는 GT checklist가 없음). 기존 경고 그대로 유지.

### 3.2 사용 규칙

- GT few-shot 6건은 **SFT 타깃 생성(teacher) 프롬프트에만** 들어간다. 학생 정책의 추론
  프롬프트에는 few-shot을 넣지 않는다(학습–추론 입력 형태 일치, 추론 비용).
- 기존 `fewshot_bank_v3.json`(모델 초안 40건)은 **GT 6건으로 교체**한다. 남기려면
  `source` 필드로 구분해 섞이지 않게 한다 (`gt` vs `model_draft`).
- GT few-shot은 **coverage를 구조적으로 밀어올린다**(같은 GT 문체 → GT 기반 checklist를 더 맞춤).
  그래서 모델 1종은 **GT-few-shot 없이 만든 SFT 타깃**으로도 학습해 ablation을 남긴다.
- 예시 EMR은 앞 `PAIRGEN["fewshot_emr_chars"]`(700)자만 붙인다. GT 본문은 짧아 예산 영향이 작다.

## 4. raw 출력으로 프롬프트 먼저 확정 → 동결

### 4.1 왜 먼저인가

프롬프트가 바뀌면 SFT 타깃·선호쌍·gold checklist·calibration이 전부 무효다. v3.1까지는
"학습 다 돌린 뒤 프롬프트 결함 발견 → 재실행" 이 반복됐다. 순서를 뒤집는다.

### 4.2 `pipeline_v3/prompt_lab.py` (구현 예정)

```bash
python -m pipeline_v3.prompt_lab --models llama qwen gemma4 --variants A B C \
       --n 24 --gpus 0,1,2,3        # raw 모델만, dev에서 결정적 서브샘플
```

- 대상: **raw 모델만** (학습 전). split은 **dev만** — gold는 열지 않는다.
- 변형: `prompt_variants.py` 에 A/B/C를 등록(현행 = A). 변형 간 diff를 리포트에 같이 싣는다.
- 산출: `outputs_v3/<run>/promptlab/<stamp>/` 에 `outputs.jsonl` + **검토용 md/xlsx**
  (케이스별로 EMR 요약·vital summary·변형별 출력 나란히).
- **자동 지표는 judge가 아니라 규칙 기반**으로 낸다 (judge 점수로 프롬프트를 고르면 프롬프트를
  judge에 과적합시킨다):

| 지표 | 계산 |
|---|---|
| `must_hit` | vital summary의 `[유의]` 이벤트가 출력에 언급된 비율 |
| `quantified` | 언급된 이상 바이탈 중 지속시간·최저/최고가 붙은 비율 |
| `num_grounded` | 출력의 수치가 EMR/vital에 실제로 존재하는 비율 (환각 수치 검출) |
| `n_sent` | 문장 수 분포 (1~5 준수율) |
| `no_issue_rate` | "특이사항 없음" 출력 비율 (이상 케이스에서 나오면 실패) |
| `clean_status` | `clean_v3` 붕괴율 |

- 최종 선택은 **사람이** 위 지표 + 출력 실물을 보고 한다. 선택 후 동결:

```bash
python -m pipeline_v3.prompt_registry > outputs_v3/<run>/prompt_frozen.json
```

이후 모든 스테이지가 산출물 메타에 지문을 기록하고 `assert_same()` 으로 대조한다.

---

## 5. 합성데이터 사람 검토 게이트

`scripts/export_pairs_review.py` (구현 예정) — `pairs_*.pkl` + `*_audit.jsonl` → 검토 xlsx.

- **Sheet1 통계**: 쌍 수, 탈락 사유(too_few/no_margin/identical), margin 분포, chosen 길이·문장수
  분포, chosen의 `[유의]` 커버율·정량화율·수치환각율, 항목군별 커버 분포, chosen 중
  "특이사항 없음" 비율, teacher greedy가 chosen이 된 비율.
- **Sheet2 케이스별**: EMR 요약 · vital summary · chosen · rejected · 점수/마진 · 규칙 플래그 ·
  **사람 판정 열**(동의 / 비동의 / 코멘트).

**통과 기준 (이걸 못 넘으면 학습을 시작하지 않는다)**

1. 무작위 50쌍 검토에서 사람–judge chosen 일치 **≥ 80%**
2. chosen의 `[유의]` 이벤트 커버율 **≥ 90%**, 수치환각 **0건**
3. 이상 케이스인데 chosen이 "특이사항 없음" **0건**
4. chosen 길이 분포가 GT 분포와 크게 어긋나지 않음(중위 문장수 차이 ≤ 1)

---

## 6. 평가 judge = 전 모델 동일 패널 + 평균 — **패널 확정**

### 6.1 무엇을 바꾸는가

v3.1은 "평가 대상과 **같은 계열 judge를 제외**"했다. 순환은 막지만 **모델마다 채점자가
달라진다**(qwen 대상 → gemma judge만, llama 대상 → 둘 평균). 서로 다른 자로 잰 값을 한 표에
놓고 순위를 매기고 있었다. v3.2는 뒤집는다.

- 평가 패널 = **고정 3 judge**, 전 변형·전 모델에 **동일 적용** (`judges_for()` 는 이제 대상과
  무관하게 패널 전체를 돌려준다 — family 제외 로직 폐기)
- 공식 점수 = **축별로 판정자 평균** (composite를 평균하는 게 아니라 coverage/faith/brevity 각각을
  평균한 뒤 가중합 — 실패 축이 섞여도 정의가 깨지지 않는다)
- 항상 병기: judge별 점수, 판정자 간 일치도(Spearman ρ / Krippendorff α), **LOO 순위 안정성**

### 6.2 패널 (확정)

```python
EVAL_JUDGES = ["prometheus", "medgemma27b", "llama70b"]     # mixtral / gemma / llama
EVAL_JUDGES_RESERVE = ["gemma4_31b", "qwen35"]              # 탈락 시 승계
MIN_JUDGES_PER_CASE = 2
JUDGE_ADMISSION = dict(min_spearman=0.4, max_parse_fail_ratio=0.10)
```

세 계열이 서로 다르고, teacher(qwen)·pair-judge(qwen)와도 겹치지 않는다.

### 6.3 ⚠ 이 패널의 실제 위험 두 가지 (착수 전 확인 필요)

**(1) prometheus의 엄격 JSON 취약성** — prometheus-8x7b-v2.0은
`(feedback) [RESULT] (score)` 절대채점 포맷으로 학습된 모델이다. 그런데 v3 평가는
coverage(항목별 verdict 배열)·faithfulness(claim 배열) 모두 **구조화 JSON**을 요구한다.
파싱 실패는 v3 규칙상 '점수 0'이 아니라 '케이스 제외'이므로, 실패율이 높으면 조용히 틀리는 대신
**유효 케이스 비율이 무너져 평가가 exit 1** 한다.
→ 게이트 **G4a 판정포맷 파일럿**을 먼저 돌린다: dev 출력 30건 × 3축 × 3 judge에서
`parse_fail_ratio` 측정. prometheus가 10%를 넘으면 `EVAL_JUDGES_RESERVE` 로 승계한다.
승계도 **모델 점수를 보기 전에** 결정하고, 전 모델에 동일 적용한다.

**(2) 계열 중복** — `llama70b` judge ↔ llama 학생, `medgemma27b`(gemma) ↔ gemma 학생이 겹친다.
qwen 학생만 같은 계열 judge가 없다(비대칭). 이 비대칭은 **LOO 민감도**(§6.4)로 정량화해 보고한다.

또한 judge의 양자화 포맷(llama70b=AWQ4)은 연구 전체에서 **고정**한다. 중간에 bf16으로 바꾸면
같은 judge가 다른 자가 된다. 포맷은 provenance에 기록한다.

### 6.4 자기선호 민감도 분석 (family 제외를 대신하는 장치)

각 모델 계열에 대해 **같은 계열 judge를 뺀 재계산**을 리포트에 병기한다
(`same_family_judges()` 가 목록을 준다). 순위가 뒤집히면 그 결론은 "judge 편향에 민감"으로
명시한다. 데이터를 버리지 않고 민감도로 다루므로 ① 채점 균일성과 ② 순환 우려를 동시에 만족한다.

### 6.5 pair-judge 재배치 (prometheus가 패널로 올라간 결과)

prometheus는 v3.1의 **선호쌍 생성 judge**였다. 평가 패널에 넣는 순간 "학습 신호 judge = 평가
judge"가 되어 v3가 없애려던 reward-hacking 해석불능이 되살아난다. 따라서 쌍 생성 judge를
비워줘야 한다. 남은 계열은 **qwen 하나**뿐이므로 `PAIRGEN_JUDGE = "qwen72b"` 로 옮겼다.

- DPO 쌍의 후보는 **학생 정책의 출력**이므로 자기채점이 아니다.
- 다만 **SFT 타깃 선별**은 teacher 후보를 teacher가 고르는 자기채점이 된다.
  → §7.4에서 이 단계는 **LLM judge를 아예 쓰지 않고 규칙 점수로 선별**하도록 바꾼다.

### 6.6 실패 처리 (규칙 완화 + 명시)

- 케이스별 유효 judge **≥ 2**면 유효 judge 평균으로 채점하고 `n_judges` 를 레코드에 남긴다.
  (현행: 한 judge라도 실패하면 케이스 제외 → 3 judge에선 제외율이 과도해진다)
- 유효 judge < 2 → 케이스 제외. 전체 유효비율 < `MIN_VALID_CASE_RATIO(0.8)` → exit 1 유지.
- paired permutation은 지금처럼 **변형 간 공통 유효 케이스**에서만 수행한다.

## 7. 합성데이터 chosen/rejected 판정 — 비용·규칙 구체화 (결정 대기)

### 7.1 현행(v3.1)과 그 한계

prometheus 단일 judge, reference-free 절대점수 루브릭 2종(coverage×2 + fidelity/brevity, 15점),
`chosen=최고 / rejected=최저`, `margin ≥ 2.0`.

| 한계 | 왜 문제인가 |
|---|---|
| 절대점수 1~5의 압축 | 후보 4개가 전부 4점 → margin 미달로 대량 탈락 (쌍 수 급감) |
| 단일 judge | 판정 오류가 그대로 학습 신호가 된다. 검증 장치가 없다 |
| `margin ≥ 2.0` 의 임의성 | 15점 스케일에서 2점의 임상적 의미가 정의돼 있지 않다 |
| rejected가 '그냥 낮은 점수' | 고치고 싶은 실패(유의 이벤트 누락·정량화 누락·수치 환각)와 무관할 수 있다 |
| **EMR을 후보×루브릭마다 재전송** | 비용의 대부분이 EMR 프리필이다 (아래 7.2) |
| prometheus가 이제 평가 패널 | 순환 — 쌍 judge를 옮겨야 한다 (§6.5, 이미 `qwen72b` 로 이동) |

### 7.2 비용 계산 (여기가 설계를 정한다)

가정: 행당 EMR+vital ≈ 2,000 토큰, 후보 1개 ≈ 150 토큰, 루브릭 ≈ 250 토큰,
대상 2,736행. 쌍 생성이 필요한 정책 수 = 모델당 3개(base / sft_1ep / sft_3ep) × 모델 2~3종
→ **6~9회**. 여기에 SFT 타깃 생성 1회.

| 방식 | 행당 judge 호출 | 행당 입력 토큰 | 1회 실행 | 6~9회 총합 |
|---|---|---|---|---|
| **현행** 절대점수(후보 4 × 루브릭 2) | 8 | ~8 × 2,400 ≈ **19k** | prometheus로 **4~5시간** | **28~41시간** |
| pairwise 개별쌍(top-3 = 3쌍 × 2순서) | 6 | ~6 × 2,550 ≈ **15k** | ~3시간 | 20~28시간 |
| **listwise 순위 ×2순서 (권고)** | **2** | ~2 × 2,900 ≈ **5.8k** | **40~60분** | **4~9시간** |

핵심은 판정 방식이 아니라 **EMR을 몇 번 보내는가**다. 후보 전체를 한 프롬프트에 넣고 순위를
매기게 하면 EMR 프리필이 8회 → 2회로 줄어 **비용이 3배 이상 싸지고**, 동시에 순서 스왑으로
자기검증까지 얻는다. (2순서 = 후보 나열 순서를 뒤집은 두 번째 호출)

### 7.3 권고안: L1 규칙 게이트 → L2 listwise 순위(×2순서) → L3 수락규칙

```
후보 N (greedy 1 + 샘플 3) → clean_v3 통과 + 정규화 dedup
  │
  ├─ L1 규칙 게이트 (결정적·무료, vital summary와 EMR에서 계산)
  │     hard(위반 시 chosen 자격 박탈 / rejected 우선 채택):
  │        h1 수치 환각      출력의 수치가 EMR·vital에 없음 (허용오차 내 매칭 실패)
  │        h2 no-issue 오용   이상 케이스인데 "특이사항 없음"
  │        h3 길이 위반      6문장 이상
  │        h4 붕괴           clean_v3 status != ok
  │     soft(정렬용 rule_score, 벌점 아님):
  │        s1 [유의] 커버율   vital의 [유의] 이벤트 중 언급된 비율
  │        s2 정량화율        언급된 이상 바이탈 중 지속시간+최저/최고가 붙은 비율
  │        s3 항목군 커버     EMR에 소견이 있는 필수 6군 중 다뤄진 비율
  │
  ├─ L2 listwise 순위 (규칙 통과 후보 2~4개를 한 프롬프트에)
  │     pair-judge(qwen72b)가 EMR 대비 순위를 매긴다 → 후보 나열 순서를 뒤집어 1회 더
  │     두 순서에서 **1위와 최하위가 동일**한 행만 채택 (position bias 자기검증)
  │
  └─ L3 쌍 수락 규칙
        chosen  : hard 위반 0 AND 양쪽 순서 1위 AND rule_score(chosen) ≥ rule_score(rejected)
        rejected: hard 위반 1+ 또는 양쪽 순서 최하위
        + 구성적 negative를 전체의 최대 30%까지 섞는다 (7.5)
```

### 7.4 SFT 타깃 선별은 LLM judge를 쓰지 않는다

teacher 후보를 teacher가 고르면 자기채점이다(§6.5). 대신:

```
SFT 타깃 = L1 hard 위반 0 인 후보 중 rule_score 최고 (동점이면 더 짧은 것)
           → 하나도 없으면 그 행은 버린다 (개수를 리포트)
```

teacher + GT few-shot 조건에서 greedy 출력은 대체로 충분하고, 규칙 점수는 우리가 실제로 요구하는
것([유의] 커버·정량화·수치 근거)을 **결정적으로** 잰다. judge 호출 1회분(≈40분)이 통째로 사라지고
재현성도 올라간다. `--select_by judge` 옵션으로 judge 선별 ablation을 남긴다.

### 7.5 구성적 negative — 규칙과 예시

chosen에서 **최소 편집**으로 만든다(한 곳만 손댄다). 목적은 DPO 기울기를 우리가 벌하려는 실패에
정확히 정렬시키는 것이다.

| 유형 | 조작 | 예시 (chosen → rejected) | 공격 대상 축 |
|---|---|---|---|
| C1 유의 이벤트 삭제 | `[유의]` 대응 절 하나 제거 | "20분간 저혈압(최저 55mmHg)으로 승압제 투여, 발관 후 croup으로 산소 필요합니다." → "발관 후 croup으로 산소 필요합니다." | coverage |
| C2 정량화 제거 | 지속시간·수치만 제거 | "20분간 저혈압(최저 55mmHg)" → "저혈압이 있었습니다" | coverage(정량화) |
| C3 행정 노이즈 삽입 | 이송·반납·부연 문구 추가 | "…산소 필요합니다." → "…산소 필요합니다. 잔여 마약성 진통제는 반납하였으며 회복실로 이송 예정입니다." | brevity |
| C4 r/o → 확진 | 의심 소견을 단정으로 | "심근염 의심(r/o) 소견" → "심근염으로 진단되어" | faithfulness |
| C5 수치 환각 | EMR에 없는 값으로 치환 | "최저 55mmHg" → "최저 38mmHg" | faithfulness |
| C6 no-issue 오용 | 이상 케이스에 no-issue | (이상소견 있는 케이스) → "특이사항 없음" | 안전게이트 |

**위험과 완화**

- *너무 쉬운 negative* → DPO가 표면 단서만 학습. → 비율 상한 **30%**, 최소 편집 원칙,
  `rejected_source` 태깅 후 **ablation**(0% / 30%)으로 효과 확인.
- *길이가 단서가 되는 문제* → C1·C2는 rejected를 **짧게**, C3는 **길게** 만든다.
  유형 비율을 맞춰 **길이와 chosen 여부의 상관을 0 근처로** 유지한다 (검토 시트에서 확인).
- *분포 이탈* → 구성적 negative는 **정책 자신의 chosen을 편집**해 만들므로 정책 분포 안에 있다.
- C5는 "EMR에 없는 수치"를 넣어야 하므로, 숫자 치환 후 h1 규칙으로 **실제로 환각인지 검증**한다
  (검증 실패 시 그 negative는 버린다).

### 7.6 남은 선택지 (채택 안 함, 기록용)

- **(b) pair-judge 패널화(2~3 judge)**: 판정 안정성↑이지만 평가 패널과 겹치지 않는 계열이 없다
  (mixtral/gemma/llama 전부 패널). qwen 하나로 2 judge를 만들면 계열 다양성이 없어 의미가 작다.
- **(f) teacher greedy = chosen / 정책 출력 = rejected**: 마진 크고 저렴하지만 증류다.
  RLAIF 해석이 불가능해지므로 **ablation arm으로만**.

### 7.7 결정해야 할 것

1. L2를 **listwise ×2순서**로 갈지, 현행 절대점수 + L1만 얹을지 (비용 4~9h vs 28~41h)
2. 구성적 negative 채택 여부와 비율 상한(권고 30%)
3. SFT 타깃 선별을 규칙 점수로 할지(권고), judge 선별을 유지할지

## 8. 실행 순서 (게이트 포함)

```
G0  프롬프트 예산 확인            scripts/check_prompt_budget.py           ← 영어화 후 필수
G1  프롬프트 확정·동결            prompt_lab (raw only, dev)  → prompt_frozen.json
G2  GT few-shot 뱅크 구축         make_fewshot_bank --source gt  (gold_gt_fewshot 6건)
G3  gold checklist 재추출          build_gold_checklist_v3 (영어 프롬프트, extractor=qwen72b)
G4a judge 판정포맷 파일럿          dev 30건 × 3축 × 3judge → parse_fail_ratio ≤ 10%  (§6.3)
G4b judge 패널 calibration         eval_v3.calibrate — 3 judge 전원, 축별 부호·ρ ≥ 0.4
G5  SFT 타깃 생성 (teacher)        gen_pairs --split sft --teacher qwen72b (규칙 선별)
G6  ★ 합성데이터 사람 검토         export_pairs_review → 통과기준 §5
G7  SFT 학습 → on-policy 쌍 → RLAIF → dev 추론/평가       run_all_v3
G8  dev로 모델·설정 확정 (gold 16 미개봉)
G9  gold 16 1회 개봉 + 최종 리포트  run_all_v3 --final
```

G1~G4 중 하나라도 실패하면 다음 단계로 가지 않는다. 각 산출물에는 프롬프트 지문·split 해시·
teacher/judge 키·양자화 포맷이 메타로 박힌다.

## 8b. 폐쇄망 모델 보유 현황과 '로컬 구성' (2026-09-03)

프록시가 `us.aws.cdn.hf.co` 를 차단해 teacher 3후보·`mprometheus`·`llama70b`(245GB)를 받지
못하고 있다(§8c). 보유분만으로 파이프라인 전체가 돌아가므로 **역할을 env 로 교체해 지금
진행**하고, 프록시가 열리면 목표 구성으로 되돌린다.

| 역할 | 목표 구성 | **로컬 구성(현재 가능)** |
|---|---|---|
| teacher | `qwen35_122b` (78.9G, 미보유) | **`gemma4_31b`** (62.5G ✓) |
| 쌍 judge | `prometheus` (93.4G ✓) | 동일 |
| 평가 패널 | `medgemma27b` + `llama70b` + `mprometheus` | **`medgemma27b` + `qwen35` + `hari`** |
| checklist 추출기 | `qwen35_122b` | `gemma4_31b` |
| 학습 base | llama / qwen (+gemma4·hari·qwen35) | 동일 (base 6종 전부 ✓) |

```bash
export HANDOVER_TEACHER=gemma4_31b
export HANDOVER_EVAL_JUDGES=medgemma27b,qwen35,hari
export HANDOVER_CHECKLIST_EXTRACTOR=gemma4_31b
```

**순환 금지를 코드가 강제한다** — `config_v3.validate_roles()`:

- 하드 실패: teacher ∈ 평가패널 / teacher == 쌍judge / 쌍judge ∈ 평가패널
- 경고(LOO 보고 필수): 평가 judge 가 학습 대상이거나 같은 family, 패널 family 중복,
  checklist 추출기 == 평가 judge

`gemma4_31b` 를 teacher 로 쓰면서 패널에도 두는 실수를 이 검증이 잡는다(실제로 설계 중
한 번 그렇게 쓸 뻔했다). `assert_models_available()` 는 스테이지 시작 시 미보유 모델을
즉시 실패로 알린다 — 몇 시간 뒤 죽는 것을 막는다.

**로컬 구성의 약점**: 패널이 gemma 1 + qwen 2 라 **qwen 계열 학습 대상이 유리해질 수 있다.**
그래서 **dev 는 로컬 구성으로 진행(모델·설정 선택), gold 최종 평가는 목표 구성 확보 후**로
미룬다. dev/gold 분리 원칙과 그대로 맞는다 — 재평가는 추론 산출물 재채점(수 시간)이고
학습을 다시 하지 않는다.

## 8c. HF 다운로드 차단 (프록시) — 진단 기록

| 호스트 | 결과 |
|---|---|
| `huggingface.co` | 200 OK (API·소용량 정상) |
| `cdn-lfs-us-1.hf.co` · `cas-bridge.xethub.hf.co` · `transfer.xethub.hf.co` | CONNECT 200 → CloudFront 403 = **도달 가능** |
| **`us.aws.cdn.hf.co`** | **20초 무응답 · 4연결 모두 0바이트 (curl exit 28)** |

대용량 파일은 전부 `us.aws.cdn.hf.co/xet-bridge-us/...` 로 302 되고, `HF_HUB_DISABLE_XET=1`
으로도 목적지가 바뀌지 않는다(서버가 정한다). **클라이언트 우회는 없다** → 프록시 allowlist 에
`*.hf.co` 추가가 필요하다. TLS 가로채기도 확인됐다
(`issuer=forward-proxy (Seoul National University Hospital)`) — `--fix-certifi` 로 해결.

진단: `python utils/download_models.py --probe --probe-cdn`

## 9. 적용 상태

**코드 반영 완료 (2026-09-02)**

| 항목 | 파일 |
|---|---|
| 프롬프트 전면 영어화 + 지문/동결 게이트 | `prompt_utils.py`, `required_categories.py`, `eval_v3/metrics.py`, `eval_v3/checklist.py`, `prompts_pairgen.py`(신규), `prompt_registry.py`(신규) |
| teacher/judge 레지스트리 (실측 repo·용량, `qwen35_122b`·`mprometheus`·`llama70b`) + 양자화 필드 | `config_v3.py` |
| 평가 패널 3 고정 + `same_family_judges()` + `MIN_JUDGES_PER_CASE` + 입장기준 | `config_v3.py` |
| pair-judge → `prometheus`, teacher 키, checklist 추출기 분리 | `config_v3.py`, `build_gold_checklist_v3.py` |
| 역할 env 오버라이드 + `validate_roles()` 순환 검증 + 보유 확인 | `config_v3.py` |
| gold 22 → GT few-shot 6 / test 16 층화 분할 | `data_splits.py` |
| 회귀 테스트 3종 (동일 패널·gold 분할·프롬프트 영어화) | `tests_v3/test_v3.py` |

**구현 대기 (아래 순서대로)**

1. `prompt_lab.py` + `prompt_variants.py` — 규칙 지표 6종, 검토 md/xlsx (§4)
2. `rule_metrics.py` — L1/soft 규칙의 **단일 소스**. prompt_lab·gen_pairs·검토시트가 공유 (§7.3)
3. `make_fewshot_bank.py` 개편 — `--source gt` (gold_gt_fewshot 6건의 전문의 GT 사용) (§3.2)
4. `gen_pairs.py` 개편 — teacher 생성, `--row_shard i/n` 3인스턴스 병렬, 규칙 선별,
   listwise 판정(결정 후), 구성적 negative(결정 후) (§2.2/§7)
5. `eval_v3/evaluate.py` 개편 — 축별 판정자 평균, `n_judges ≥ 2`, LOO 민감도, per-judge 기록 (§6)
6. `scripts/export_pairs_review.py` — 검토 xlsx + 통과기준 자동 판정 (§5)
7. `eval_v3/calibrate.py` — 3 judge + 입장기준 + G4a 파일럿 모드 (§6.3)
8. `report_v3.py` — LOO 표, n=16 검정력 경고, teacher/judge/양자화 메타 표기

**서버에서 먼저 확인할 것**

- `python -m tests_v3.test_v3` (로컬에는 numpy/pandas가 없어 gold 분할 테스트를 못 돌렸다)
- `python scripts/check_prompt_budget.py --models llama qwen35 gemma4` — 영어화로 지시문
  토큰이 줄었는지 실측 (예상: 감소, EMR 여유 증가)
- ~~미러의 Qwen 72B AWQ 경로 확정~~ → HF 실측으로 확정 완료. 남은 문제는 프록시 차단(§8c)
- 프롬프트가 바뀌었으므로 `gold_checklist_v3.json`·`fewshot_bank_v3.json`·`calibration_v3.csv`
  전부 재생성 (`scripts/invalidate_v3.py`)
