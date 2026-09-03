# v3.2 전체 설계 — 데이터부터 리포트까지 · 선행연구 대비 · 논의 목록

작성 2026-09-02 · 기준 커밋 `4abd52e` + v3.2 작업분 · PHI 없음(수술ID·환자텍스트 미포함)

이 문서 하나로 "무엇을 어떻게 왜"가 다 읽히게 쓴다. 변경 이력은
[PIPELINE_V3.2.md](PIPELINE_V3.2.md), v3.1 이전 배경은 [UPGRADES_v3.1.md](UPGRADES_v3.1.md),
프롬프트 원문은 `python -m pipeline_v3.prompt_registry --dump DIR`.

---

## 0. 연구 질문과 산출물

**과제** — 소아 수술 종료 후 OR → PACU/ICU 로 넘기는 **초간결 인계문(한국어 1~5문장)** 을
마취기록 EMR + 수술 중 바이탈 시계열에서 자동 생성한다.

**학술 질문 두 개**

1. 이 과업에서 **SFT(증류)와 RLAIF(DPO·SimPO)가 각각 얼마나 기여하는가**
2. **LLM-as-judge 평가를 임상적으로 신뢰 가능하게 만들 수 있는가** (우리 쪽 진짜 기여)

**산출물** — 모델 3계열 × 7변형의 3축 점수 + CI + 검정, 전문의 GT 기반 coverage,
judge↔전문의 calibration, 케이스별 검토 리포트(EMR·GT·모델출력 병기).

IRB E-2601-138-1712 / DRB DRB-E(I)-2026-02-04 (연구자 명단 수정 미결 — §14-L).

---

## 1. 전체 흐름

```
     EMR pkl (마취기록 MultiIndex)          Vital pkl (시계열)
             │                                    │
             │                       utils/vital_thresholds.py (교과서 임계값)
             │                       utils/vital_summarizer.py
             │                                    │
             └───────────────┬────────────────────┘
                             ▼
                    build_emr_text + vital_summary          ← 모델 입력 한 벌
                             │
        ┌────────────────────┼─────────────────────────────┐
        ▼                    ▼                             ▼
  [G1] prompt_lab      [G5] gen_pairs(teacher)      [G3] gold checklist
  raw 출력로 프롬프트     SFT 타깃 = 규칙 선별          교수님 GT → 항목화
  확정·동결              DPO 쌍 = on-policy + 판정      (전문의 검수)
        │                    │                             │
        │              [G6] 사람 검토 게이트                │
        │                    ▼                             │
        │            SFT → merge → DPO/SimPO               │
        │                    ▼                             │
        └──────────────► 추론 (dev 110 / gold 16) ◄─────────┘
                             ▼
                  평가 3축 (판정 패널 3, 축별 평균)
                             ▼
              통계(bootstrap CI · paired permutation · Holm)
                             ▼
                  리포트 (md/html/csv) + provenance
```

게이트 `G*` 는 §15.

---

## 2. 데이터와 split

원천은 선행연구(다린)가 전처리한 pkl을 그대로 쓴다 — 전처리를 다시 하지 않는 게 비교
연속성에 유리하고, 애초에 IRB 데이터 추출 경로가 그쪽이다.

| split | 크기 | 원천 | 역할 |
|---|---|---|---|
| **GOLD** | 22 → **GT few-shot 6 / test 16** | `gold_sampled_251008.pkl` | 전문의 GT 보유. v3.2에서 둘로 쪼갠다 |
| **SFT_SRC** (jsft) | 2,736 | `jsft_251008.pkl` | SFT 타깃 생성 원천 |
| **SELFJUDGE_SRC** | 2,736 | `selfjudge_251008.pkl` | DPO 선호쌍 생성 원천 (SFT와 분리) |
| **RLHF** | 150 | `rlhf_251008.pkl` | few-shot pool 40 + **dev set 110** |
| vital map | — | `vital_summary_map.pkl` | `{수술ID: 바이탈요약}` |

- 로드 경로는 `data_splits.load_splits()` **하나**. 로드 시점에 `assert_disjoint`가
  gold∩(sft/selfjudge/rlhf), fewshot∩dev, **GT few-shot∩gold test** 를 검사하고
  하나라도 겹치면 `LeakageError`로 즉시 중단한다.
- `SPLIT_SEED=42`. RLHF 150은 sid 정렬 후 결정적 셔플 → 앞 40 few-shot / 나머지 110 dev.
- gold 22 분할은 GT의 '특이사항 없음' 여부로 **층화**(n=22에서 무작위로 6을 뽑으면 정상
  케이스가 한쪽에 몰린다). `load_splits()["gold"]` 는 **test 16** 을 반환하므로 추론·평가·
  리포트가 자동으로 16건만 본다. 22 전체는 `["gold_full"]`.
- run마다 `split_manifest.json` 에 sid 정렬 SHA1 16자리만 기록(PHI 없음).

**역할 고정**: 한 split = 한 역할. dev(110)로 모델·설정을 고르고, gold test(16)는
**`--final` 에서 1회만** 개봉한다. `inference.py` 는 `--allow_gold` 없이는 gold 실행을 거부한다.

---

## 3. 모델 입력

**(a) EMR 텍스트** — 마취기록 MultiIndex에서 4개 섹션만 뽑아 조립한다
(`prompt_utils.build_emr_text`).

```
- Pre-anesthetic Patient Conditions Summary   (마취전 환자상태 요약)
- Preoperative Preparations and Premedication (전처치)
- Anesthetic TOTALS                           (약물·수액·소변량 집계)
- Anesthetic Record                           (시간순 마취기록 전문)
```

**(b) 바이탈 요약** — v3의 고유 입력. 시계열을 그대로 넣지 않고 **임계값 기반 이벤트 요약**
으로 압축한다. 판정은 2-tier이고, 모든 이벤트에 지속시간과 편차가 붙는다.

```
[유의]저혈압(<70): 4회 20분(최장 15분), 최저 55 (기준-15)
```

- `[유의]` = 소생·개입 기준 초과(임상적 유의) / 표시 없음 = 연령별 참조범위 이탈.
- 임계값 전거는 **Smith's Anesthesia for Infants and Children 9e (2021)** ·
  **Miller's Anesthesia 10e (2024)** 표·페이지 단위([THRESHOLDS.md](THRESHOLDS.md)).
  문헌 근거 없는 항목(DBP 하한, Ppeak)은 **판정 자체를 삭제**했다.
- 측정 아티팩트는 **좁게만** 배제한다 — 이 코호트는 소아 심장수술을 포함해 DHCA 저체온
  18~20°C와 청색성 심질환 SpO2 40~70%가 실제 값이다(`PLAUSIBLE_RANGE`).

**(c) 필수 항목군 6개** — 인계문이 다뤄야 하는 군. **조건부 필수**(EMR에 소견이 있는 군만;
없는 군은 "없음"조차 쓰지 않는다 — brevity 축 보호).

기저질환·약물 / 기도관리 / 수술 중 이벤트 및 처치 / 수혈·수액 / 수술 전 검사이상 / 감기 유무

단일 출처 `pipeline_v3/required_categories.py` 를 **생성 프롬프트 · checklist 추출 ·
coverage 채점** 세 곳이 공유한다(한국어 라벨 = 리포트용, 영어 라벨·설명 = 프롬프트용).

**(d) 토큰 예산 한 벌** — 학습·추론·쌍생성·평가 전부 같은 값.

```
MAX_PROMPT_TOKENS 2560 · MAX_COMPLETION 512 · MAX_SEQ 3072
INFER_MAX_MODEL_LEN 4096 · JUDGE_MAX_MODEL_LEN 8192
```

초과 시 **EMR 본문만 좌측(오래된 쪽)** 을 이진탐색으로 잘라 맞춘다. 지시문·`### OUTPUT`
헤더는 절대 자르지 않고, 사라지면 `PromptTruncationError`(조용한 절단 금지).

---

## 4. 프롬프트 (v3.2에서 전면 영어화)

**규칙**: 지시·루브릭·판정기준은 **영어**, 모델 출력은 **한국어**. 프롬프트에 한국어가
남는 곳은 세 군데뿐이고 전부 데이터 리터럴이다 — 필수 출력 문구 `"특이사항 없음"`,
바이탈 마커 `[유의]`, 한국어 출력 형식 예시 `"20분간 저혈압(최저 55mmHg)"`.

관리 대상 프롬프트는 **14개**이고 전부 `prompt_registry.all_prompts()` 한 곳에 등록된다.

| 그룹 | 프롬프트 |
|---|---|
| 생성 | `gen.system`, `gen.user` (학습·추론·후보생성 공통) |
| 선호쌍 judge | `pairgen.judge_tmpl`, `rubric_coverage`, `rubric_fidelity_brevity` |
| gold checklist | `checklist.system`, `checklist.tmpl` |
| 평가 3축 | `eval.{coverage,faith,brevity}_{system,tmpl}` |
| 항목군 블록 | `categories.en` |

`fingerprint()` 가 전체 sha1과 프롬프트별 해시를 낸다. 산출물 메타에 지문을 박고
`assert_same()` 으로 대조하므로, **프롬프트를 고친 뒤 옛 캐시를 재사용하는 사고**가 막힌다
(v3.1에서 실제로 반복된 사고다).

생성 프롬프트의 핵심 지시 3줄:
① 이상소견 없으면 정확히 `특이사항 없음` 만, ② 있으면 1~5문장 한국어,
③ **필수 6군 중 소견 있는 군은 빠뜨리면 최악의 오류**, 바이탈 이상은 지속시간+최저/최고 병기,
`[유의]` 표시는 반드시 포함.

---

## 5. 합성데이터 — SFT 타깃과 DPO 쌍

### 5.1 역할 분리 (v3.2의 핵심 결정)

| 용도 | 생성 주체 | 왜 |
|---|---|---|
| **SFT 타깃** | **teacher = Qwen 72B AWQ4** + **전문의 GT few-shot** | 타깃 품질이 SFT 상한을 정한다 |
| **DPO/SimPO 쌍** | **학생 정책 자신** (on-policy) | teacher 출력을 chosen으로 쓰면 RLAIF가 아니라 증류다 |

이 경계가 흐려지면 `sft_3ep_dpo` 의 이득이 증류 효과인지 선호학습 효과인지 분해할 수 없다.

### 5.2 GT few-shot

- 전문의 GT는 gold 22건에만 있다(교수님 재검수본 22/22 확보, 중위 100자).
- 그래서 gold를 **GT few-shot 6 / test 16** 으로 쪼갠다. 6건의 GT는 **teacher 프롬프트의
  few-shot 예시로만** 쓰이고, 학생 정책의 추론 프롬프트에는 few-shot을 넣지 않는다
  (학습-추론 입력 일치, 추론 비용).
- 예시 EMR은 앞 700자만 붙인다(`PAIRGEN["fewshot_emr_chars"]`, 스타일 앵커 목적).
- 행마다 `RandomState(42+row_idx)` 로 예시 조합을 로테이션 → 특정 케이스 스타일이 전
  학습셋에 박히는 것을 막는다.
- **대가**: test n=22→16, GT 문체 모방이 coverage를 구조적으로 올린다 → 모델 1종은
  **GT-few-shot 없는 SFT 타깃**으로도 학습해 ablation을 남긴다.

### 5.3 후보 생성

모델당 **greedy 1 + temperature 0.8/top_p 0.95 샘플 3 = 4후보**.
`clean_v3` 통과분만 남기고(붕괴 후보는 **drop**, v1처럼 "특이사항 없음"으로 치환하지 않는다)
정규화 dedup. teacher AWQ4는 40GB라 **TP=2 × 3인스턴스** 병렬이 가능하다(행 샤딩 필요).

### 5.4 규칙 게이트 (L1) — v3.2 신설, judge 없이 결정적으로 계산

| 종류 | 항목 |
|---|---|
| **hard** (위반 시 chosen 자격 박탈 / rejected 우선 채택) | 수치 환각(출력 수치가 EMR·vital에 없음) · 이상 케이스에 "특이사항 없음" · 6문장 이상 · `clean_v3` 실패 |
| **soft** (정렬용 `rule_score`, 벌점 아님) | `[유의]` 이벤트 커버율 · 정량화율(지속시간+최저/최고) · 필수 항목군 커버율 |

이 지표는 `rule_metrics.py` 한 곳에서 계산하고 **프롬프트 랩(§4/G1) · 쌍 생성 · 검토
시트**가 공유한다.

### 5.5 SFT 타깃 선별 — LLM judge 미사용

```
SFT 타깃 = hard 위반 0 인 후보 중 rule_score 최고 (동점이면 짧은 것)
           → 하나도 없으면 그 행은 버리고 개수를 리포트
```

teacher 후보를 teacher가 고르면 자기채점이 되고, pair-judge가 teacher와 같은 계열이라
그 경로가 막혀 있다(§8.4). 규칙 점수는 우리가 실제로 요구하는 것을 결정적으로 재므로
재현성도 올라가고 judge 호출 1회분(≈40분)이 사라진다. `--select_by judge` 로 ablation.

### 5.6 DPO 쌍 판정 (L2/L3) — **결정 대기, §14-A**

권고안: 규칙 통과 후보 2~4개를 **한 프롬프트에 넣어 listwise 순위**를 받고, 후보 나열
순서를 뒤집어 **1회 더** 받아 두 순서에서 1위·최하위가 같은 행만 채택(position bias
자기검증). chosen은 hard 위반 0 + 양쪽 1위, rejected는 hard 위반 또는 양쪽 최하위.
여기에 **구성적 negative**(chosen 최소 편집: 유의 이벤트 삭제 / 정량화 제거 / 행정 노이즈
삽입 / r/o→확진 / 수치 환각 / no-issue 오용)를 최대 30% 섞는다.

비용 근거(2736행, 쌍 생성 6~9회 필요):

| 방식 | 행당 judge 호출 | 총합 |
|---|---|---|
| 현행 절대점수(후보4 × 루브릭2) | 8 | 28~41시간 |
| pairwise 개별쌍 | 6 | 20~28시간 |
| **listwise ×2순서** | **2** | **4~9시간** |

지배 요인은 판정 방식이 아니라 **EMR 프리필 횟수**다.

### 5.7 산출물

`pairs/pairs_sft.pkl`(chosen만 타깃) · `pairs/pairs_dpo_{tag}.pkl` ·
감사로그 `*_audit.jsonl`(전 후보 점수) · 통계 `*_meta.json`(탈락 사유별 카운트).

---

## 6. 학습

### 6.1 실험 매트릭스 (모델당 7변형)

| 변형 | 내용 | 쌍/타깃 |
|---|---|---|
| `raw` | 학습 없음 | — |
| `sft_1ep` / `sft_3ep` | SFT만 | `pairs_sft.pkl` chosen |
| `rlaif_dpo` / `rlaif_simpo` | base 위에 바로 선호학습 | base 정책 on-policy 쌍 |
| `sft_1ep_dpo` / `sft_3ep_dpo` | SFT → merge → DPO | 각 SFT 정책 on-policy 쌍 |

같은 정책을 쓰는 dpo/simpo는 쌍을 1회만 만들어 공유한다.

### 6.2 설정

- **LoRA** r=8, α=16, dropout=0.1, 타깃 `q/k/v/o_proj, gate/up/down_proj`
  (gemma4는 `model.language_model.layers.…` 정규식 분기). v1과 동일 — 비교 연속성.
- **SFT** lr 2e-5, bs 2 × grad_accum 4, warmup 50, cosine, bf16, 3ep(또는 1ep),
  `max_length=3072`. collator는 좌측 패딩·좌측 절단, 프롬프트 토큰 `-100` 마스킹,
  학습 토큰 0이면 `RuntimeError`(loss=0 방지).
- **judge-augmentation** (기본 ON, `--no_judge_aug` 로 ablation): 행마다 생성 샘플 +
  (rejected 있으면) A/B 판정 샘플 2개를 추가해 self-judge 능력을 보존한다.
- **DPO** lr 5e-6, bs 1 × 8, constant_with_warmup, `beta=0.1`, `loss_type=sigmoid`,
  `ref_model=None`(LoRA). **SimPO** 로컬 커스텀 트레이너, `beta=2.0, γ/β=0.25`.
- **선행 SFT 정책 로딩**: adapter를 base에 `merge_and_unload()` 한 뒤 그 위에 새 LoRA.
- merge 캐시 키 = **adapter 내용 해시** → 재학습하면 자동 무효화.
- 전 모델 `CUDA_LAUNCH_BLOCKING=1`(비동기 스트림 race로 SFT가 전멸한 사고 대응,
  끄려면 `HANDOVER_NO_LAUNCH_BLOCKING=1`).

---

## 7. 추론

- vLLM 우선, HF fallback(좌측 절단·좌측 패딩). 레코드에 `engine` 필드를 남기고
  **전 변형이 동일 엔진**이어야 한다 — 섞이면 "학습 효과"가 아니라 "디코딩 엔진 차이"를 잰다.
- `VLLM_INCOMPATIBLE_MODELS`(qwen35·gemma4)는 전 변형 HF 강제: 텍스트 LoRA를 멀티모달
  base에 merge하면 vLLM이 arch를 잘못 잡아 raw만 통과하기 때문이다.
- 디코딩 `do_sample=False`, `repetition_penalty=1.15`, `no_repeat_ngram_size=4`(HF 전용).
- gold split은 `--allow_gold` 없이 실행 거부.

---

## 8. 평가

### 8.1 3축 composite

`composite = 0.5·coverage + 0.3·faithfulness + 0.2·brevity`

| 축 | 정의 | 계산 |
|---|---|---|
| **coverage** | 전문의 GT checklist recall | yes 1.0 / partial 0.5 / no 0 → 항목 평균(macro) + 항목 pooled(micro), 항목군별 recall 병기 |
| **faithfulness** | claim의 EMR entailment | claim을 supported/contradicted/unsupported로 3분류 → supported 비율 |
| **brevity** | 간결성 1~5 | `(s-1)/4` 정규화. **이상 바이탈에 붙은 지속시간·수치는 감점 대상이 아니다** |

coverage 정의불가(normal 케이스/항목 0개)면 faith·brev를 0.6/0.4로 재정규화한다.
→ **dev(110)에는 GT가 없어 coverage가 전부 `None`** 이고, dev composite과 gold composite은
**서로 다른 척도**다. 나란히 놓고 비교하면 안 된다.

### 8.2 안전 게이트 (상수 캡 없이 자연 도출)

- `missed_abnormal` — 이상소견 케이스인데 "특이사항 없음" → coverage=0 **그리고**
  faithfulness=0(그 주장 자체가 EMR과 모순) → composite 0.0.
- `degenerate` — 생성 붕괴(empty/repetition/garbage/leak) → 0.0.
- `is_no_issue_v3()` 는 **전체 문자열 일치**만 no-issue로 본다
  ("특이사항 없음. intraop VT 발생" → no-issue 아님).

### 8.3 "실패는 점수가 아니다"

| 상황 | 처리 |
|---|---|
| judge JSON 파싱 실패 | 축=None, 케이스 **제외** + `judge_failed` |
| verdict에 항목 id 누락 | judge 실패로 간주 → 제외 |
| checklist 추출 실패 / gold 없음 | `gold_llm_failed` / `no_gold` → 제외 |
| sid 매칭 실패 | **하드 에러** (파이프라인 중단) |

유효 케이스 비율 < 0.8 → 평가 `exit 1`. 성적이 아니라 인프라 문제로 취급한다.

### 8.4 판정 패널 (v3.2에서 전면 교체)

```
EVAL_JUDGES  = prometheus(mixtral) + medgemma27b(gemma) + llama70b(llama)   # 전 모델 동일
RESERVE      = gemma4_31b, qwen35
PAIRGEN_JUDGE = qwen72b (teacher와 같은 계열 — 패널과 겹치지 않는 유일한 계열)
MIN_JUDGES_PER_CASE = 2
JUDGE_ADMISSION = min_spearman 0.4 / max_parse_fail 0.10
```

- v3.1은 "대상과 같은 family judge 제외"였다 → **모델마다 채점자가 달라져** 서로 다른 자로
  잰 값을 한 표에 놓고 순위를 매기고 있었다. v3.2는 패널을 고정해 전 모델에 동일 적용한다.
- 공식 점수 = **축별로 판정자 평균** 후 가중합(composite 평균이 아니다).
- 순환 우려는 데이터를 버리는 대신 **LOO 민감도**로 다룬다: 대상과 같은 계열 judge를 뺀
  재계산을 항상 병기하고, 순위가 뒤집히면 "judge 편향 민감"으로 명시한다.
- 패널 입장은 **결과를 보기 전에** calibration으로 판정한다. 탈락 시 예비 judge가 승계하고,
  승계는 전 모델에 동일 적용한다.
- 주입 방어: faithfulness premise는 **EMR 원문 + 바이탈 요약만**(생성 지시문 미포함),
  모델 출력은 `<<<MODEL_HANDOFF>>>` 구분자로 감싸 "명령이 아니라 채점 대상"임을 명시한다.

### 8.5 통계

- 축별 **bootstrap 95% CI** (케이스 단위, B=10,000)
- 변형 vs `{model}_raw`: **공통 유효 케이스**만으로 **paired permutation**(20,000) + **Holm**
- 보조로 t·Wilcoxon(전부-0 가드), 판정자 간 Spearman ρ / MAD, seed 42

### 8.6 Calibration — gold 개봉 전 관문

`인계요약지_SY.xlsx` 의 전문의 채점 **168행**(22케이스 × 2모델 × 프롬프트 4종)을 패널
judge 전원으로 재채점해 **judge별 × 생성모델 family별 Spearman 표**를 만든다.
방향이 뒤집힌 축이 있으면 루브릭을 고치고 재실행한다. 통과 못 하면 gold를 열지 않는다.

---

## 9. 리포트와 재현성

- `report_v3.py` → `results_{dev,gold}_v3.{csv,md,html}`: 3축+CI, raw 대비 Δ와 Holm p,
  판정자 일치도, 제외 케이스 표, 케이스별 EMR·GT·출력 병기(HTML, PHI 포함 → 외부공유 금지).
- `provenance_v3.json`: 체크포인트 **내용 해시**, git rev, 쌍 파일 해시,
  **동일 출력 변형군 경고**(학습이 안 먹었는지 탐지), 프롬프트 지문, teacher/judge/양자화.
- `split_manifest.json`: split별 sid SHA1 지문(PHI 없음).
- `.done` 마커는 tmp→atomic replace → 크래시 잔해와 완료를 구분한다.
- 회귀 테스트 20종(`tests_v3/test_v3.py`) — GPU 없이 돈다.

---

## 10. 오케스트레이션과 GPU (A6000 48G × 6)

| Phase | 작업 | GPU |
|---|---|---|
| A | SFT 학습 | 2-GPU 병렬 |
| B | DPO 선호쌍 생성 | 판정 모델 크기에 따라 전체 직렬 또는 TP=2×3 병렬 |
| C | RLAIF 학습 | 2-GPU 병렬 |
| D | 추론 (dev, `--final`이면 +gold) | 2-GPU 병렬 |
| E | 평가 + 통계 + 리포트 | eval_gpus / CPU |

- A6000은 **Ampere** → FP8 가속 없음. 양자화는 **AWQ/GPTQ INT4**(Marlin) 또는 bf16.
- TP 크기는 head 수의 약수 → **TP=2 또는 4**. 6장은 `TP=2 × 3인스턴스`가 처리량 최적.
- `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` 를 셸에서 export(세션당 1회).
  안 하면 vLLM이 **조용히** HF로 폴백해 극단적으로 느려진다.

---

## 11. 선행연구(다린)는 어떻게 했나

엄다린 연구원, 2026-02. `HANDOVER_인계용_다린/` (PHI 포함 — gitignored).
전처리·split·체크포인트의 출처이자 v3 리포트의 **비교군**이다.

### 11.1 데이터·입력

- 전처리 산출물이 지금 우리가 쓰는 pkl들이다(gold 22 / jsft 2736 / selfjudge 2736 /
  rlhf 150 / evaluation 100).
- 입력은 **EMR only**. 바이탈 시계열은 나중에 받아서(`vitalsheet_260216`) EDA까지만 했고
  **학습·추론 입력에는 넣지 않았다**. → 바이탈을 모델 입력으로 쓰는 것은 v3의 추가분이다.
- EMR 조립 포맷(4섹션)과 생성 시스템/유저 프롬프트는 **v3가 그대로 물려받았다**
  (v3.1에서 필수 항목군·정량화 지시 추가, v3.2에서 영어화).

### 11.2 합성데이터 생성

| 항목 | 내용 |
|---|---|
| 생성 모델 | `gemma-3-27b-it`, `medgemma-27b-it`, `Qwen3-32B`, `Llama-3.3-70B-Instruct` (Mixtral-8x7B는 생성 품질 문제로 제외) |
| 후보 수 | 4모델 × {2-shot, 4-shot} = **행당 8후보** |
| 디코딩 | vLLM, greedy(`temperature=0.0`), `max_tokens=768`, `repetition_penalty=1.03`, seed 42, `max_model_len=8192` |
| few-shot 예시 | **gold 22의 전문의 GT(`Human_handoff`)**. BERT(`bert-base-multilingual-cased`) 임베딩 → K-means(k=2 → 2-shot, k=4 → 4-shot, seed 42)로 대표 케이스 선택 |

> **여기가 중요하다.** 지금 우리가 하려는 "GT를 few-shot으로" 는 **다린이 이미 하던 방식**이다.
> v3가 그걸 없앤 이유는 방법이 나빠서가 아니라 **예시로 쓴 gold 22가 곧 평가셋이어서**
> (T1 유출)였다. v3.2는 gold를 6/16으로 쪼개 유출 없이 그 장점을 되찾는다.

### 11.3 합성데이터 채점과 쌍 선정

- judge: `prometheus-8x7b-v2.0`, `prometheus_eval` 의 `ABSOLUTE_PROMPT_WO_REF`
  (reference-free 절대채점), TP=4, `temperature=1.0, top_p=0.9, logprobs=20`.
- 루브릭 **2종**(최종본): `brevity_relevance`, `critical_focus`.
  둘 다 "**정상 소견·안정 언급·재보증은 노이즈이므로 감점**", "안정 환자면 `None`/`No issues`
  가 **유일한 5점 답**", "길이는 보상 요인이 아니다" 를 명시한다.
  (초기 버전은 4루브릭 합산 0~20점 체계였고 평균 ≈16/20)
- 쌍: `chosen` = 최고점 후보, **`rejected` = 나머지 중 랜덤 1개**.

**구조적 결과** — 루브릭에 **누락(coverage) 벌점이 없다**. "특이사항 없음"이 안전하게
고득점이므로 학습 신호가 *말을 덜 하는 쪽*으로 끌린다. v3가 coverage 루브릭(놓치면 1점)을
신설하고 `margin ≥ 2.0` 을 강제한 이유가 이것이다.

### 11.4 학습

- **jsft (SFT)**: 행당 3샘플 — ① EMR→chosen 생성, ② EMR+A/B→`Winner: A`,
  ③ 순서 뒤집어 `Winner: B`. `JudgeAugmentedSFTCollator`(v3가 이 아이디어를 계승).
  LoRA, bs 2 × grad_accum 16, 3ep, **lr 1e-6**, cosine, warmup 0.1, flash-attn2, seed 42.
  Qwen3는 `enable_thinking=False` 필수(안 하면 빈 `<think>` 블록이 학습 데이터에 섞인다).
- **self-judge (RLAIF)**: DPO(sigmoid) / SimPO(커스텀 트레이너, `git_new/SIMPO`).
  변형 = `{llama,qwen} × {dpo, simpo, dpo_jsft_1epoch, dpo_jsft_3epoch}`.
- 체크포인트는 `experiments/{jsft,self_judge}/{모델}/` 에 남아 있다 → v3 병기 비교의 재료.

### 11.5 추론·평가

- 추론: HF(transformers), greedy, `max_new_tokens=768`. (0_Readme에 "transformers와 vLLM
  결과가 다르다"는 경고가 있다 — v3가 엔진 통일을 규칙으로 못박은 배경.)
- 자동 평가: **합성 채점과 동일한 prometheus 루브릭**으로 모델 출력을 채점.
  → **학습 신호 judge = 평가 judge**. 이 상태로는 점수 상승이 "실력"인지 "그 judge에 맞춘
  최적화(reward hacking)"인지 구분할 수 없다. v3의 '순환 금지' 원칙이 여기서 나왔다.
- 사람 평가: 교수님들 human evaluation 실시(`human_evaluation_modeltype포함_251223.xlsx`,
  `llm_sum` 지표, 케이스 층화 분석). **v3에는 아직 없는 자산이다**(§14-K).

### 11.6 v3가 다린 대비 바꾼 것 (요약)

| # | 다린 | v3 / v3.2 |
|---|---|---|
| 1 | few-shot을 gold 22(=평가셋)에서 | held-out pool(v3) → **gold 6/16 분할 + GT few-shot**(v3.2) |
| 2 | 평가 judge = 학습 신호 judge (prometheus 동일 루브릭) | 3중 분리: 생성 teacher ≠ 쌍 judge ≠ 평가 패널 |
| 3 | 루브릭에 coverage 없음 → "없음"이 최적해 | coverage 루브릭 + **전문의 GT checklist recall** + `missed_abnormal` 안전게이트 |
| 4 | `rejected` = 랜덤 | 마진·순서스왑·규칙 게이트로 검증된 쌍 |
| 5 | chosen이 다른 모델(70B) 출력 → off-policy | DPO 쌍은 **정책별 on-policy** 재생성 |
| 6 | 바이탈 미사용 | 교과서 임계값 기반 **2-tier + 지속시간 + 편차** 요약을 입력으로 |
| 7 | 절단 미관리(TRL 기본값) | 예산 한 벌 + EMR-only 좌측절단 + loud fail |
| 8 | 실패를 점수로 흡수 | 실패는 `None`+제외, 유효비율 <80%면 `exit 1` |
| 9 | 단일 점수 평균 비교 | bootstrap CI + paired permutation + Holm, 제외 케이스 명시 |
| 10 | 사람 평가 있음 | 아직 없음 (되찾아야 할 자산) |

---

## 12. 다린 비교군을 리포트에 병기하는 방법

`reinfer_darin_on_v3sids.py` — 다린 기존 추론 출력은 `evaluation_251008`(100건) 전용이라
v3 gold/dev와 sid가 **0 겹침**이다. 그래서 다린 **체크포인트를 v3의 바로 그 sid에 재추론**한다.

- 다린 `make_self_judge_format` 을 verbatim 재현 → **바이탈을 넣지 않는다**
  (다린 모델은 학습 때 바이탈을 본 적이 없어 넣으면 왜곡된다).
- 따라서 병기의 의미는 "**EMR-only 옛 모델** vs **EMR+바이탈 v3**" 의 **총효과**다.
  바이탈 입력 추가분과 파이프라인 개편분이 섞여 있다 → 분해하려면 §14-G의 ablation이 필요하다.
- 출력은 `{darin_root}/{raw|jsft|self_judge}/{model}.pkl` → `report_v3 --include_darin` 이 읽는다.

---

## 13. 실행 순서와 게이트

```
G0  프롬프트 예산 확인        scripts/check_prompt_budget.py
G1  프롬프트 확정·동결        prompt_lab (raw only, dev 24건) → prompt_frozen.json
G2  GT few-shot 뱅크          make_fewshot_bank --source gt (gold_gt_fewshot 6건)
G3  gold checklist 재추출      build_gold_checklist_v3 (extractor=qwen72b) + 전문의 검수
G4a 판정포맷 파일럿           dev 30건 × 3축 × 3judge → parse_fail ≤ 10%
G4b 패널 calibration          SY 168행, 축별 부호·ρ ≥ 0.4
G5  SFT 타깃 생성             gen_pairs --split sft (teacher + GT few-shot, 규칙 선별)
G6  ★ 합성데이터 사람 검토     export_pairs_review → 50쌍 일치 ≥80%, [유의] 커버 ≥90%, 환각 0
G7  학습 → on-policy 쌍 → RLAIF → dev 추론/평가
G8  dev(110)로 모델·설정 확정  (gold 미개봉)
G9  gold test(16) 1회 개봉 + 최종 리포트 (+ 다린 병기)
```

하나라도 실패하면 다음으로 가지 않는다. G3의 전문의 검수(`reviewed: true`)가 안 되면
coverage는 논문에 쓸 수 없다(잠정 표기 필수).

---

## 14. 논의할 부분

### A. DPO 쌍 판정 방식 (§5.6) — 가장 시급
listwise ×2순서(4~9h) vs 현행 절대점수+규칙게이트(28~41h). 구성적 negative 채택 여부와
상한(권고 30%). SFT 타깃을 규칙 선별로 할지(권고) judge 선별을 유지할지.

### B. GT 수량이 진짜 병목이다
현재 전문의 GT는 22건뿐이고, 그중 6건을 few-shot으로 쓰면 **coverage를 잴 수 있는 케이스가
16건**이다. dev 110에는 GT가 없어 coverage가 아예 정의되지 않는다.
→ 즉 **연구의 주지표를 n=16에서만 측정**하는 구조다. 선택지:
 1. 교수님께 **few-shot pool(RLHF 40) 중 8~10건 GT 신규 작성** 요청(teacher 초안 → 수정) →
    gold 22 전체를 test로 되돌린다. 전문의 1~2시간.
 2. **dev 일부(예: 30건)에 GT 확보** → dev에서도 coverage 측정 가능(검정력 확보).
    전문의 3~5시간. 효과가 가장 크다.
 3. 현행 유지(16건) + 규칙 기반 `[유의]` 커버율을 dev 주지표로 병기.

### C. teacher 증류를 논문에서 어떻게 위치시킬지
SFT 타깃이 72B teacher 출력이면 `sft_*` 변형은 **증류**다. "SFT vs RLAIF" 프레이밍을
"**증류 + 선호최적화**"로 바꿔야 정확하다. 동시에 teacher 없는 self-generated SFT arm을
1종 남길지(비교 기준) 결정해야 한다.

### D. prometheus를 평가 패널에 두는 것의 실무 위험
`(feedback) [RESULT] (score)` 포맷 모델에 **항목별/claim별 JSON**을 요구한다. 파싱 실패는
케이스 제외로 이어져 유효비율을 깎는다. G4a 파일럿 결과에 따라 예비 judge 승계.
대안: prometheus에는 brevity 축(단일 점수)만 맡기고 coverage/faith는 다른 2 judge로 →
**축별로 판정자 구성이 달라지는** 비용을 감수할지.

### E. 판정자 계열 비대칭
llama70b↔llama 학생, medgemma↔gemma 학생은 계열이 겹치고 qwen 학생만 겹치지 않는다.
LOO 민감도로 보고하는 것으로 충분한지, 아니면 qwen 계열 judge를 하나 넣어 대칭을 맞출지
(그러면 teacher와 겹친다).

### F. 축 가중치 0.5/0.3/0.2의 근거
현재는 설계자 판단이다. SY 168행 전문의 채점에 대한 **회귀로 재추정**해 근거를 붙일지,
가중치 민감도(예: 0.6/0.25/0.15)를 부록으로 낼지.

### G. 바이탈 입력의 순효과 분해
다린 병기는 "EMR-only vs EMR+바이탈" 총효과다. v3.2에서 **바이탈 없는 arm**을 1종 돌리면
바이탈 입력의 순효과를 뽑을 수 있다(추론만 추가하면 되므로 비용은 작다).

### H. `missed_categories` 를 안전게이트로 승격할지
지금은 진단 지표다. 필수 항목군 누락을 감점(또는 0점)으로 올리면 임상적으로는 맞지만
brevity와 충돌하고 점수 분산이 커진다. 전문의 검수 결과를 보고 결정.

### I. 학습 대상 모델 몇 개로 갈지
`TRAIN_KEYS` 후보는 llama / qwen / qwen35 / gemma4 / gemma4_31b / hari(SNUH 한국어 의료 8B).
7변형 × 모델수 × (쌍 생성 3회/모델)이라 **모델 1개 추가 = 상당한 시간**이다.
`hari` 는 언어·도메인이 맞아 매력적이지만 계열이 qwen(=teacher·pair-judge)과 같다.

### J. 학생 추론에 few-shot을 넣을지
현재는 안 넣는다(학습-추론 일치). GT few-shot을 추론에도 주면 raw 성능이 올라가지만
"학습 효과" 측정이 흐려진다. raw 기준선을 few-shot 유/무 둘로 낼지.

### K. human evaluation 재실시
다린은 교수님 blind 평가를 받았다(2025-12). v3.2 최종 결과에 대해 다시 받으면
**judge 신뢰성 주장의 최상위 근거**가 된다. 케이스 수·평가자 수·설계(쌍비교 vs 절대점수)를
정해야 한다. 권고: gold 16 × 상위 3~4변형, 쌍비교 + 누락/환각 체크리스트.

### L. 행정·보안 미결
IRB·DRB 연구자 명단 수정 / `.env` GitHub PAT revoke 후 파일 삭제 /
repo private 전환 또는 `git filter-repo` 로 과거 이력의 수술ID 스크럽(이미 push된 이력은
새 커밋으로 지워지지 않는다).

---

## 15. 문서 지도

| 문서 | 내용 |
|---|---|
| **이 문서** | v3.2 전체 설계 + 다린 대비 + 논의 목록 |
| [PIPELINE_V3.2.md](PIPELINE_V3.2.md) | v3.2 변경 7건의 결정·근거·구현 상태 |
| [UPGRADES_v3.1.md](UPGRADES_v3.1.md) | v3.1 고도화(임계값·정량화·항목군·GT override·버그) |
| [THRESHOLDS.md](THRESHOLDS.md) | 바이탈 임계값 전거 (Smith/Miller 표·페이지) |
| [REQUIRED_CATEGORIES.md](REQUIRED_CATEGORIES.md) | 필수 항목군 6개 정의·근거 |
| [RERUN_RUNBOOK.md](RERUN_RUNBOOK.md) | 재실행 절차(캐시 무효화 순서) |
| [../PIPELINE_V3.md](../PIPELINE_V3.md) · [../PROMPTS_V3.md](../PROMPTS_V3.md) | v3 프로토콜·프롬프트 원문 |
| [../CODE_REVIEW_V3_PROPOSAL.md](../CODE_REVIEW_V3_PROPOSAL.md) | v1/v2 결함 목록과 재설계 근거 |
| `HANDOVER_인계용_다린/0_Readme.md` · `CLAUDE_CONTEXT.md` | 선행연구 원문 (PHI — gitignored) |
