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

## 2. 합성데이터 생성 = 대형 teacher — **Qwen3.5 122B-A10B GPTQ-Int4 확정**

### 2.1 역할 배치 (확정 2026-09-04)

| 용도 | 모델 | 근거 |
|---|---|---|
| **SFT 타깃 생성** | **teacher = `qwen35_122b`** (Qwen3.5 122B-A10B, 활성 10B, GPTQ-Int4 · 78.9GB) | 타깃 품질이 SFT 상한을 정한다. MoE라 활성 10B → 27B dense급 속도로 122B급 품질 |
| **DPO/SimPO 쌍 생성** | **학생 정책 자신** (on-policy) | teacher 출력을 chosen으로 쓰면 RLAIF가 아니라 증류다 |
| 선호쌍 judge | **`prometheus`** (8x7B, 절대채점·pairwise 둘 다 학습된 판정 전용 모델) | 원래 용도에 맞는 배치. 평가 패널에서 빠졌으므로 학습신호 ≠ 평가 |
| gold checklist 추출기 | **`qwen35_122b`** (`CHECKLIST_EXTRACTOR`) | 채점자가 아니라 GT→항목 변환기. 엄격 JSON이 필요해 가장 강한 모델 |
| 평가 패널 | **`medgemma27b` + `llama70b` + `mprometheus`** | 3계열(gemma/llama/qwen), 전 모델 동일 적용 (§6) |

**teacher는 SFT 타깃에만** 들어간다. 논문 서술도 "SFT = teacher 증류 + rejection sampling,
RLAIF = on-policy 선호학습"으로 정확히 쓴다. 이 경계가 흐려지면 `sft_3ep_dpo` 의 이득이
증류 효과인지 선호학습 효과인지 구분할 수 없다.

### 2.2 teacher 적재 (A6000 48G · 상세 배치는 §8d)

- A6000은 **Ampere → FP8 하드웨어 가속이 없다.** 양자화는 **AWQ/GPTQ INT4**(Marlin) 또는 bf16.
  teacher를 bf16으로 쓰면 244GB라 6장으로도 KV가 안 남는다 → **INT4 필수**(공식 GPTQ-Int4 사용).
- 78.9GB → **TP=2 (2장, 96GB)** 에 적재. 여유 ~17GB(util 0.95 기준 ~12GB)를 KV로 쓰므로
  `max_model_len 3200` · `max_num_seqs 8~16` 로 제한한다.
- TP=4면 KV 여유가 90GB+로 넉넉하지만, **A6000은 NVLink가 없거나 2-way라 4-way all-reduce가
  PCIe를 탄다.** 프리필이 지배하는 우리 작업에서는 통신 손실이 더 크다
  → **TP=2 × 3인스턴스(6장)** 가 최적.
- `gen_pairs` 는 `tensor_parallel_size=torch.cuda.device_count()` 고정이므로 3인스턴스 병렬에는
  `--gpus 0,1 / 2,3 / 4,5` + **`--row_shard i/n`** 이 필요하다 (§9 구현대기).

### 2.3 순환 방지와 남은 계열 겹침

- **pair judge(prometheus·mixtral) ∉ 평가 패널** → 학습신호 judge ≠ 평가 judge.
  다린 파이프라인의 최대 결함(합성 채점 = 평가에 동일 루브릭·동일 모델)이 제거된다.
- teacher(`qwen35_122b`) 자체는 패널에 없다. **단 `mprometheus` 는 Qwen2.5-14B 기반이라
  teacher와 같은 qwen 계열이다** — 한국어 판정 능력을 얻는 대가로 계열 겹침 하나를 받아들인 것이다.
  → LOO(mprometheus 제외) 민감도와 판정자 오류상관·n_eff를 **필수 보고**(§6.4).
- 겹침 전체 목록: `llama70b`↔llama 학생 / `medgemma27b`↔gemma 학생 /
  `mprometheus`↔qwen 학생·teacher. qwen 학생은 teacher와도 계열이 같아 이중으로 유리할 수 있다.

### 2.4 예상 비용 (2736행 · 후보 4개/행)

프리필 ~14M 토큰(greedy 패스 별도) + 디코딩 ~2.2M 토큰.
TP=2 × 3인스턴스면 **1~2시간**. judge 비용이 더 크므로 §7에서 구조로 줄인다.

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
EVAL_JUDGES = ["medgemma27b", "llama70b", "mprometheus"]    # gemma / llama / qwen
EVAL_JUDGES_RESERVE = ["gemma4_31b", "qwen35"]              # 탈락 시 승계
MIN_JUDGES_PER_CASE = 2
JUDGE_ADMISSION = dict(min_spearman=0.4, max_parse_fail_ratio=0.10)
```

세 계열이 서로 다르고, **pair-judge(prometheus·mixtral)와 겹치지 않는다** —
학습 신호 judge와 평가 judge의 분리가 이 패널의 존재 이유다.
`mprometheus`(M-Prometheus-14B)는 20개 이상 언어로 학습된 다국어 판정 전용 모델로,
**우리 출력이 전부 한국어**라는 점 때문에 넣었다(영어 전용 judge의 비영어 평가 품질 격차는
M-Prometheus 논문의 출발점이다). 대가는 teacher와 같은 qwen 계열이라는 것(§2.3).

### 6.3 ⚠ 이 패널의 실제 위험 세 가지 (착수 전 확인 필요)

**(1) 엄격 JSON 취약성 — 이제 `mprometheus` 쪽 위험이다.** M-Prometheus는
`(feedback) [RESULT] (score)` 계열 포맷으로 학습된 판정 모델인데, v3 평가는 coverage(항목별
verdict 배열)·faithfulness(claim 배열) 모두 **구조화 JSON**을 요구한다. 파싱 실패는 v3 규칙상
'점수 0'이 아니라 '케이스 제외'라, 실패율이 높으면 **유효 케이스 비율이 무너져 평가가 exit 1** 한다.
(`medgemma27b`·`llama70b` 는 범용 instruct라 JSON은 상대적으로 안전하지만, 반대로 한국어
임상 판정력이 미검증이다.)
→ 게이트 **G4a 판정포맷 파일럿**: dev 출력 30건 × 3축 × 3 judge에서 `parse_fail_ratio` 측정.
10%를 넘는 judge는 `EVAL_JUDGES_RESERVE` 로 승계한다. 승계는 **모델 점수를 보기 전에** 결정하고
전 모델에 동일 적용한다.

**(2) 계열 중복** — `llama70b`↔llama 학생, `medgemma27b`↔gemma 학생, `mprometheus`↔qwen
학생**이자 teacher**. 세 계열 모두 겹치는 학생이 하나씩 있고, qwen 학생만 teacher와도 계열이
같아 이중으로 유리할 수 있다. **LOO 민감도**(§6.4)와 판정자 간 오류상관·n_eff로 정량화해 보고한다.

**(3) 평균의 이득을 과대 주장하지 않는다** — 판정자 패널의 유효 표본수는 판정자 수보다
훨씬 작다(*Nine Judges, Two Effective Votes*, 2026: 9 judge 패널의 n_eff ≈ 2.18, 평균 오류상관
φ≈0.39, 계열당 1개로 제한하면 오히려 n_eff 감소). 그래서 리포트에 **Kish n_eff와 판정자 쌍별
오류상관**을 넣고, `n_eff/k < 0.5` 면 "판정자 다양성 부족"으로 명시한다.
반대 방향 근거도 있다 — 헬스케어 LLM-judge 스코핑 리뷰(2026, 134편)에서 3~4 judge 앙상블이
신뢰도 상위(agreement ~0.90)에 몰려 있고, 전문의 정렬 현실 기준선은 agreement 중위 0.83,
κ 중위 0.78, 상관 중위 0.69다. 우리 입장 기준 ρ≥0.4는 그 분포의 하위 경계다.

또한 judge의 양자화 포맷(llama70b=AWQ4)은 연구 전체에서 **고정**한다. 중간에 bf16으로 바꾸면
같은 judge가 다른 자가 된다. 포맷은 provenance에 기록한다.

### 6.4 자기선호 민감도 분석 (family 제외를 대신하는 장치)

각 모델 계열에 대해 **같은 계열 judge를 뺀 재계산**을 리포트에 병기한다
(`same_family_judges()` 가 목록을 준다). 순위가 뒤집히면 그 결론은 "judge 편향에 민감"으로
명시한다. 데이터를 버리지 않고 민감도로 다루므로 ① 채점 균일성과 ② 순환 우려를 동시에 만족한다.

### 6.5 pair-judge = prometheus (원래 용도로 복귀)

prometheus-8x7b-v2.0은 **판정 전용으로 학습된 모델**이고 절대채점과 pairwise 비교를 둘 다
지원한다. 그 능력이 필요한 곳은 평가 3축(엄격 JSON)이 아니라 **선호쌍 판정**이다.
그래서 `PAIRGEN_JUDGE = "prometheus"`, 평가 패널에서는 제외한다.

- 순환 차단: 학습 신호를 만든 judge가 평가에 참여하지 않는다 → DPO가 평가지표를 직접
  최적화하는 경로가 없다. (다린 파이프라인은 이 둘이 같아서 점수 상승의 해석이 불가능했다.)
- pairwise 모드를 쓴다 — 다린은 절대채점만 썼는데, 순서 스왑 판정(§7.3)이 이 모델의 원래
  능력에 더 맞고 margin 임의 임계값도 사라진다.
- **SFT 타깃 선별에는 judge를 쓰지 않는다**(§7.4) — teacher 후보를 어떤 LLM이 고르든
  자기채점·계열편향이 끼어들기 때문에, 규칙 점수로 결정적으로 고른다.
- 한국어 판정력 확인 과제: prometheus-8x7b는 영어 중심이다. SY 168행 calibration에
  **prometheus vs mprometheus 를 같이 태워** 쌍 judge를 최종 확정한다(인프라 이미 있음).

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

## 7b. 바이탈 요약 재설계 — 편향의 원인과 처방 (v3.2 신설)

### 7b.1 진단: 편향은 심증이 아니라 구조다

`utils/vital_summarizer.py` 를 실제 포맷으로 파싱해 보면 원인이 셋으로 갈린다.

1. **이벤트가 없는 항목도 항상 중앙값·범위를 출력한다.** `_events([])` 가 `", 이상 없음"` 을
   붙이는 구조라, 정상 케이스도 바이탈 블록이 9~10줄 숫자로 채워진다. 샘플 1건 기준
   **10줄 중 4줄이 순수 노이즈**(DBP·MBP·Ppeak·QTc)였다.
2. **1표본 이벤트도 등재되고 `[유의]` 까지 붙는다.** NIBP가 한 번 55mmHg면
   `[유의]저혈압: 1회 <1분` 이 되고, 프롬프트는 "`[유의]` 는 반드시 포함"이라 모델이 반드시 쓴다.
   같은 샘플에서 `[유의]` 6개 중 **2개가 1표본 <1분**이었다.
3. **깊이·지속·baseline·개입을 판정에 쓰지 않는다.** SBP 69가 1회와 SBP 45가 10분이 동급으로
   등재된다. Miller Ch.4의 술중저혈압 정의는 `MAP<65` **또는 baseline 대비 -20%` 인데
   절대 기준만 쓴다.

여기에 프롬프트의 "`[유의]` 필수 + 정량화 필수"가 곱해져 **바이탈 문장이 5문장 예산을 잠식**하고
기저질환·기도관리·검사이상·감기가 밀려난다. 즉 편향 = 과대등재 × 강제 포함.

### 7b.2 원칙 (한 줄)

> 인계문에 들어갈 바이탈은 **① 종료 시점까지 영향이 남은 것 · ② 개입이 필요했던 것 ·
> ③ 지속과 심도가 임계 이상인 것** 중 하나여야 한다. 나머지는 참고이며 프롬프트에 넣지 않는다.

### 7b.3 새 요약 구조 (3블록)

```
### VITAL — REPORTABLE
- 저혈압(<70): 최장 15분 / 총 20분, 최저 55 (기준-15, baseline 78 대비 -29%) — 개입: 승압제 13:20
- 실혈 120 mL (EBV의 24%) — pRBC 1U
- 핍뇨 0.3 mL/kg/hr
### VITAL — AT HANDOFF (마지막 15분)
- HR 150 / SBP 82 / SpO2 99%(room air) / T1 36.8 — 승압제 유지 중
### VITAL — MINOR  (감사 로그 전용, 프롬프트 미포함)
- 빈맥>190 최장 5분 · SpO2 85% 1회 · 저체온 1표본 · 고혈압 1표본
```

`AT HANDOFF` 블록은 **신설**이다. 인계에서 가장 중요한 정보인데 지금 요약에 아예 없다.

### 7b.4 REPORTABLE 등재 규칙 (OR — 하나만 만족하면 등재)

| 규칙 | 내용 | 근거 |
|---|---|---|
| **R1 개입 동반** | 마취기록 타임라인에 대응 개입(승압제·아트로핀·수혈·에피네프린·재삽관·제세동)이 이벤트 시각 ±10분에 있으면 무조건 등재 | 의사가 반응했다면 실제로 문제였다 — "실제 문제인가"의 최선 대리지표 |
| **R2 지속×심도** | 최장 episode ≥ 10분(또는 연속 2표본) **AND** 편차 ≥ 항목별 최소편차 | Miller Ch.4 *time below threshold* |
| **R3 종료 시점 지속** | 마지막 15분 창에서도 위반 | 인계 시점에 남아 있는 문제 |
| **R4 절대 위험선** | 항목별 하드라인은 1표본이라도 무조건 (SpO2<80 · SBP<0.7×PALS · MBP<0.7×기준 · HR<50 · T1<34 또는 >39 · QTc>500) | 짧아도 보고해야 하는 사건 |

MINOR로 강등: R1~R4 전부 미달 / QTc tier2(정상상한 초과) / tier2(정상범위 이탈)의 R2 경로.

**구현하면서 확정한 정밀화 두 가지** (selftest가 잡아낸 실제 오류다)

1. **baseline 게이트는 R2·R3·R4 공통이고 R1에는 적용하지 않는다.**
   처음엔 baseline 조건을 R2에만 걸었는데, 그러면 기저 SpO2 75%인 청색성 심질환 환자가
   **R4(위험선 <80)와 R3(종료 시점에도 75%)로 그대로 등재**된다. 그 환자에게 75%는 사건이
   아니라 평소 상태이고, 그 상태는 AT HANDOFF 블록이 이미 전달한다.
   → "환자의 기저값이 이미 그 선 아래면 그 선은 이 환자에게 정보가 없다."
   단 **처치가 있었다면(R1) 게이트와 무관하게 등재**한다.
2. **R3는 "마지막 창에 1표본 스쳐감"을 지속으로 보지 않는다.**
   episode가 2표본 이상이거나 케이스의 마지막 표본 자체가 위반일 때만 인정한다.
   (이 조건이 없으면 종료 15분 안의 아무 blip이나 R3로 승격된다 — 1표본 고혈압이
   REPORTABLE로 올라오는 것을 selftest가 잡았다.)

`tier2`(연령별 정상범위 이탈이지만 임상 개입 기준 미달)는 **R2로 승격하지 않는다** —
개입이 있었거나(R1) 인계 시점에 남아 있을 때(R3)만 인계 대상이다.

### 7b.5 baseline 상대 기준 — 이 코호트에서 특히 중요

유도 전(첫 10분) 중앙값을 baseline으로 잡고, **저산소·저혈압·저체온은
`절대 기준 위반 AND baseline 대비 -20% 이상`** 을 요구한다(R4 하드라인은 baseline과 무관하게 항상 등재).

이유: 이 코호트는 소아 심장수술을 포함한다. **청색성 심질환 환자의 기저 SpO2가 75%면 절대기준
90%는 수술 내내 위반**이고, CPB/DHCA의 계획된 저체온도 마찬가지다. v3.1은 이걸 "좁은 아티팩트
배제"로 다루려 했지만 근본 해법이 아니다. baseline AND 규칙을 넣으면 **기저 저산소·계획된
저체온이 자동으로 걸러지고 별도 예외 규칙이 대부분 불필요해진다.**

예외 하나는 남는다 — DHCA는 baseline 36.5 → 20°C라 -20%를 크게 넘는다. 마취기록에
`CPB / 체외순환 / DHCA` 키워드가 있으면 저체온은 이벤트 나열이 아니라
`계획된 저체온(최저 20.1, 종료 시 36.4)` 한 줄 **상태 표기**로 바꾼다. 임상적으로도 인계에서
중요한 것은 "회복됐는지"다.

### 7b.6 프롬프트 동반 수정 (필수)

- `[유의]` 마커 → 블록 구조로 대체. 지시문은
  "**REPORTABLE 의 모든 항목과 AT HANDOFF 의 이상은 반드시 포함. MINOR 는 언급 금지**".
- 정량화 요구(지속시간·최저/최고)는 **REPORTABLE 항목에만** 적용.
- 규칙 지표 `must_hit` 을 **새 REPORTABLE 집합**으로 재정의한다. 옛 `[유의]` 집합으로 두면
  지표가 편향을 오히려 강화한다.

### 7b.7 컷오프를 어떻게 정하는가 (감이 아니라 데이터)

1. **`scripts/vital_flag_audit.py`** — 전문의 GT가 언급한 카테고리를 정답으로 tier·지속·횟수
   버킷별 **PPV(언급률)** 를 낸다. "1표본 <1분 이벤트의 GT 언급률"이 숫자로 나오면 R2 컷오프가
   결정된다. 케이스별 `요약 줄수 / 이벤트0 항목줄 / 이벤트 수` 도 같이 나와 프롬프트 부담이 정량화된다.
   ⚠ **컷오프 결정은 fewshot 6건 열로만.** test 16의 GT로 고르면 입력 표현을 test에 맞추는 것이다.
2. **전문의 이진 라벨 20건**(pool/dev 케이스의 이벤트 목록에 "인계 대상 / 아님" 체크, 1~2시간).
   GT 인계문은 5문장 제약이 있어 "언급 안 됨 ≠ 안 중요함" — PPV가 과소평가된다. 이 라벨이
   그 편향을 없애고 test 유출 없이 컷오프를 정하게 해준다.
3. **바이탈 ablation** — 바이탈 블록 유/무로 raw 출력을 비교해 ① 출력 문장 중 바이탈 유래 비율,
   ② 비바이탈 항목군 커버율 변화를 본다. 편향 자체를 숫자로 만든다(추론만 추가하므로 저렴).

### 7b.8 구현 상태 (2026-09-04 적용 완료)

| # | 항목 | 상태 | 파일 |
|---|---|---|---|
| 1 | 이벤트0 항목줄 삭제 + `AT HANDOFF` 블록 신설 | ✅ | `utils/vital_summarizer.py` |
| 2 | R2/R4 컷오프 + REPORTABLE/MINOR 분리 | ✅ | `utils/vital_thresholds.py` (`VITAL_REPORT_RULES`·`MIN_DEVIATION`·`hard_lines`) |
| 3 | baseline 상대 기준 + CPB/DHCA 계획된 저체온 예외 | ✅ | 〃 (`BASELINE_DROP_REQUIRED`·`PLANNED_HYPOTHERMIA_KEYWORDS`) |
| 4 | R1 개입 연동 (마취기록 타임라인 파싱) | ✅ (키워드 확인 대기) | `utils/anesthetic_record.py`(신규) |
| 5 | QTc tier2 폐기 · tier2는 R1/R3만 | ✅ | `utils/vital_summarizer.py` |
| 6 | 프롬프트 블록 지시로 교체 (`[유의]` 마커 폐기) | ✅ | `pipeline_v3/prompt_utils.py` (`PROMPT_SPEC_VERSION=v3.2-en+vital3blk`) |
| 7 | 구조화 이벤트 감사본 저장 | ✅ | `VITAL_AUDIT_PKL`, `scripts/rebuild_vital_map.py` |
| 8 | 감사 스크립트가 구조화본을 읽고 층별·규칙별 PPV | ✅ | `scripts/vital_flag_audit.py` |
| 9 | 회귀 테스트 (합성 시계열 11조건 + 개입 매칭 5조건) | ✅ 20/20 | `utils/vital_summarizer.py --selftest`, `tests_v3/test_v3.py` |
| — | NIBP 단발 이상치 강등 | ⏸ 보류 | R2 지속 조건이 사실상 같은 역할을 한다 — PPV 결과 보고 판단 |

**NIBP 단발 강등을 보류한 이유**: "전후 표본이 정상인 단발 이탈"은 R2의 `2표본/10분` 조건에
이미 걸러진다. 별도 규칙을 더하면 R4(위험선)까지 막을 위험이 있어, 먼저 감사 PPV를 보고
필요하면 넣는다.

**서버 실행 순서 (G-1)**

```bash
python -m utils.vital_summarizer --selftest          # 30초, 규칙 검증 (PHI·GPU 불필요)
python scripts/vital_flag_audit.py --source text --out /tmp/audit_v31.md   # 개편 전 기준선
python scripts/invalidate_v3.py --apply --archive --scope vital,checklist,fewshot,calib
python scripts/rebuild_vital_map.py --apply --force  # selftest 자동 선행 + 자가진단 8줄
python scripts/vital_flag_audit.py --out /tmp/audit_v32.md                 # 개편 후
#   → 두 md 를 비교한다. 기대: MINOR 로 내려간 이벤트 다수 + REPORTABLE PPV 상승
python scripts/check_prompt_budget.py --models llama qwen35 gemma4
```

**파급**: `vital_summary_map.pkl` 재생성 → 프롬프트 지문 변경 → `gold_checklist_v3.json`
재추출 → calibration 재실행. 즉 **G1 프롬프트 동결보다 앞**이다(§8).

### 7b.9 결정해야 할 것

1. R1 개입 연동 도입 여부 + 약물/처치 키워드 목록 확정 (교수님 확인)
2. baseline 상대 기준 도입 여부 — **강력 권고**
3. R4 하드라인 값 (위 표의 초기값은 잠정)
4. 종료 시점 창 길이 15분 vs 5분
5. MINOR 블록을 프롬프트에서 완전히 제외할지(권고), 남기고 "언급 금지"로 지시할지
6. 전문의 이진 라벨 20건 요청 여부

## 8. 실행 순서 (게이트 포함)

```
G-1 바이탈 요약 재설계 + 재생성   §7b → rebuild_vital_map · invalidate_v3  ← 최우선
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

## 8d. GPU 배치와 소요 시간 (A6000 48GB)

용량은 `config_v3.MODELS[*]["size_gb"]` 실측치. 카드당 가용은 `gpu_memory_utilization 0.90`
기준 **≈43GB**(0.95면 45.6GB).

### 8d.1 모델별 적재

| 모델 | weights | TP | 카드 | 카드당 | KV 여유 | 비고 |
|---|---|---|---|---|---|---|
| `qwen35_122b` GPTQ-Int4 (teacher·추출기) | 78.9 | **2** | 2 | 39.5 | ~6GB(util .95) | `max_model_len 3200` · `max_num_seqs 8~16` |
| 〃 | 〃 | 4 | 4 | 19.7 | ~90GB | KV는 넉넉하나 PCIe 4-way all-reduce 손실 |
| `prometheus` 8x7B bf16 (쌍 judge) | 93.4 | **4** | 4 | 23.4 | ~78GB | TP=2는 46.7GB/장으로 KV 없음 → **4장 필수** |
| `medgemma27b` bf16 (판정) | 54.9 | **2** | 2 | 27.5 | ~31GB | |
| `llama70b` AWQ4 (판정) | 39.8 | **2** | 2 | 19.9 | ~46GB | TP=1도 적재는 되나 KV ~3GB로 위험 |
| `mprometheus` 14B bf16 (판정) | 29.5 | **1** | 1 | 29.5 | ~13GB | 1장이라 병렬화가 쉽다 |
| 학생 8~9B bf16 (학습·추론) | 16~19 | 1 | 1~2 | 19 | ~24GB | 학습은 2장 권장(현 `gpus_per_job=2`) |

**bf16 teacher는 불가**: 122B × 2byte = 244GB → 6장(258GB 가용)에 KV가 남지 않는다. INT4 필수.

### 8d.2 단계별 카드 수와 6장 배치

| 단계 | 필요 최소 | 6장 배치 | 예상 시간(6장) |
|---|---|---|---|
| G1 프롬프트 랩 (raw 학생) | 1 | 3잡 병렬 | 10~20분 |
| teacher 선발전 (후보 3종) | 2 | 122b(2) + 27b(2) + 72b(2) 동시 | 30~60분 |
| G3 checklist 추출 (22건) | 2 | 2장 | 5~10분 |
| G4a 판정포맷 파일럿 (30건×3축) | 1 | medgemma(2)+llama70b(2)+mprom(1) 동시 | 20~40분 |
| G4b calibration (168행 × 후보 5) | 1~2 | 3판정 동시 + 나머지 순차 | 1~2시간 |
| G5 SFT 타깃 생성 (2736행) | 2 | **TP=2 × 3인스턴스** (row_shard) | 1~2시간 |
| Phase A SFT 학습 | 1(2 권장) | 3잡 병렬 (2장×3) | 웨이브당 5~7시간 |
| **Phase B 쌍 생성** | **4** (prometheus) | prometheus 4장 직렬 + 학생 후보생성 2장 | 쌍세트당 1~2시간 |
| Phase C RLAIF 학습 | 1(2 권장) | 3잡 병렬 | 웨이브당 4~6시간 |
| Phase D 추론 (21변형) | 1 | 6잡 병렬 | 1~2시간 |
| **Phase E 평가 (패널 3)** | **5** | **3 judge 동시**(2+2+1) | 1.5~2.5시간 |

### 8d.3 결론 — 몇 장이 필요한가

- **하한 4장** — `prometheus` bf16 TP=4가 유일한 하드 제약이다(4장 미만이면 KV가 없어 못 띄운다).
  4장이면 전 단계가 순차로 돌아가고, 평가 패널도 순차(2→2→1)로 돌리면 된다.
- **현재 보유 6장 = 권장 구성.** teacher 3인스턴스 병렬(생성 1~2시간), 판정 3개 동시(평가
  1.5~2.5시간), 학습 3잡 병렬이 모두 가능하다. Phase B만 4장을 통째로 쓰므로 그 시간에는
  남은 2장으로 학생 후보 생성을 겹쳐 돌린다.
- **8장이면** Phase B(prometheus 4장)와 학습(2장×2잡)을 완전히 겹칠 수 있어 총 시간이 **30~40%**
  줄어든다. 10장 이상은 Phase B가 직렬이라 수익이 급감한다.

### 8d.4 전체 소요 (2모델 = llama + qwen, 7변형, 6장 기준)

```
준비 G1~G5          5 ~  7시간
Phase A  SFT 4잡    10 ~ 14시간   (3슬롯 → 2웨이브)
Phase B  쌍 6세트    8 ~ 14시간   (1500행 서브샘플이면 절반)
Phase C  RLAIF 8잡  12 ~ 18시간   (3슬롯 → 3웨이브)
Phase D+E           4 ~  6시간
────────────────────────────────
합계               40 ~ 55시간 ≈ 2 ~ 2.5일 연속 가동
```

3모델이면 Phase A/B/C가 1.5배 → **3.5~4일**. `--skip_done` 으로 나눠 돌릴 수 있다.

### 8d.5 디스크

모델 weight 합계 ≈ **445GB**(teacher 78.9 + prometheus 93.4 + 판정 124.2 + 학생 51.8 +
teacher 후보 97.2) + merged LoRA 캐시(변형당 16~19GB × 다수) + 산출물.
**여유 700GB 이상**을 확보한다. merged 캐시는 adapter 내용 해시로 이름이 붙으므로
재학습 때마다 새로 쌓인다 — 주기적으로 정리한다.

### 8d.6 "생성은 6장 · 이후는 4장" 시나리오 (결정 2026-09-04)

teacher 생성(G5)만 6장을 쓰고 **이후 전 단계를 4장 안에서** 돌리는 구성. 결론부터: 돌아간다.
그리고 학습은 오히려 **빨라진다**.

#### (1) 학습 — `--gpus_per_job 1` 로 바꾸면 4장이 6장보다 낫다

`sft_train.py` / `dpo_train.py` 는 `device_map="auto"` + `max_memory={i:"40GiB"}` 로 모델을
적재한다. 즉 **DDP(데이터 병렬)가 아니라 naive 모델 병렬**이다. 결과가 두 가지다.

- 유효 배치는 `per_device_batch × grad_accum` 으로 **GPU 수와 무관**하다
  (SFT 2×4=8, DPO 1×8=8). → 1장으로 줄여도 **하이퍼파라미터가 안 바뀐다.** 비교 연속성 유지.
- naive 모델 병렬은 한 번에 한 장만 계산한다 → **2장을 줘도 속도가 거의 같다.**
  현재 기본값 `gpus_per_job=2` 는 8~9B 모델에서 사실상 카드 절반을 낭비하고 있다.

| 구성 | 슬롯 | SFT 4잡 | RLAIF 8잡 |
|---|---|---|---|
| 6장 · 2장/잡 (현행) | 3 | 2웨이브 | 3웨이브 |
| **4장 · 1장/잡** | **4** | **1웨이브** | **2웨이브** |

8~9B(llama·qwen·qwen35·gemma4·hari)는 bf16 16~19GB + LoRA + checkpointing 활성값 → 40GiB 캡
안에서 여유롭다. **`gemma4_31b`(62.5GB)만 2장 필요.** 착수 전 1잡으로 메모리 스모크 테스트 권장
(4잡 동시면 dataloader 워커 16개 + 체크포인트 4개 동시 저장이라 CPU·디스크도 같이 본다).

#### (2) Phase B(쌍 생성) — 4장을 통째로 쓴다. 그래서 **2단계로 쪼개야 한다**

`prometheus` bf16 93.4GB → TP=4가 **하한**이다(TP=2면 46.7GB/장으로 KV가 없다).
4장 체제에서는 Phase B 동안 다른 잡을 겹칠 수 없다. 현재 코드는 쌍 세트마다
`후보 생성 → prometheus 적재 → 판정 → 해제` 를 반복하므로 **93GB 모델을 6~9번 다시 올린다.**

→ `gen_pairs` 를 `--stage {candidates,judge,both}` 로 분리한다.

```
Stage 1  후보 생성 : 정책별 merge + 생성 → cands_{tag}.jsonl   (8B, 1장/잡 × 4 병렬, ~1시간)
Stage 2  판정      : prometheus TP=4 를 **한 번만** 올려 전 tag 일괄 판정 → pairs_dpo_*.pkl
```

이득: 모델 적재 5~8회 절감(회당 수 분~십수 분) + 프리필 배치가 꽉 차서 처리량 상승 +
Stage 1을 4장 병렬로 돌릴 수 있다. (§9 구현대기)

판정 비용 자체는 줄지 않는다 — 6세트 × 2736행 × 2호출 × ~3k토큰 ≈ 98M 토큰,
TP=4에서 **약 9시간**. `PAIRGEN["max_rows"]=1500` 서브샘플이면 **약 5시간**.

> **선택지(고효율)**: `prometheus` INT4(AWQ/GPTQ) ≈ 26GB → **1장**. 그러면 판정 4인스턴스
> 병렬로 Phase B가 9시간 → 3시간 이하가 된다. 대가는 ① 미러에 양자화본이 없으면 자체 양자화
> (AutoAWQ/llm-compressor, MoE 지원 확인 필요, 1~2시간 + 디스크), ② **판정자가 바뀌므로
> 연구 시작 전에 확정**해야 한다(중간 변경 금지). Phase B가 4장 체제의 지배 비용이라
> 투자 대비 효과가 가장 크다.

#### (3) Phase E(평가 패널) — 5장이 필요하니 4장에선 2웨이브

| 방식 | 배치 | 시간 | 비고 |
|---|---|---|---|
| **A (권고)** | `medgemma27b(2) + llama70b(2)` → `mprometheus(1)` | ~2.5~3시간 | 추가 양자화 없음 |
| B | `medgemma27b` INT4(1) + `llama70b` AWQ(2) + `mprometheus`(1) = 4장 동시 | ~1.5시간 | medgemma 양자화본 필요. 판정자 포맷은 이미 혼재(llama70b=AWQ4)이므로 원칙 위반은 아니되 **사전 확정** 필수 |

`llama70b` AWQ4는 39.8GB로 1장 적재는 되지만 judge 예산 8192 토큰에서 KV가 ~3GB밖에 안 남는다
→ **TP=2 유지**.

#### (4) 그 외

- G3 checklist 추출(`qwen35_122b` TP=2, 22건) · G4 calibration · Phase D 추론(8B 1장 × 4병렬)
  전부 4장 안에서 문제없다.
- **6장을 쓸 수 있는 창에서 teacher 작업을 전부 끝내라**: SFT 타깃 생성뿐 아니라
  teacher 선발전 · GT few-shot 뱅크 초안 · gold checklist 추출까지. 나중에 4장으로 줄면
  `qwen35_122b`(TP=2)는 여전히 돌지만 다른 잡과 겹치기가 어려워진다.

#### (5) 4장 기준 총 소요 (2모델 · 7변형)

```
G5 생성 (6장 창)          1 ~  2시간
준비 G-1~G4 (4장)         3 ~  5시간
Phase A  SFT 4잡 1웨이브   5 ~  7시간
Phase B  후보 1h + 판정    6 ~ 10시간   (1500행이면 4~6 / prometheus INT4면 3~4)
Phase C  RLAIF 2웨이브     8 ~ 12시간
Phase D+E                 4 ~  5시간
─────────────────────────────────────
합계                     27 ~ 41시간
```

6장 추정(40~55시간)보다 오히려 짧다. 이유는 위 (1) — 지금 6장 구성이 카드를 절반 낭비하고 있어서다.
**6장을 쓸 수 있으면 `--gpus_per_job 1` + 6슬롯으로 더 줄어든다**(SFT·RLAIF가 각각 1·2웨이브).

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

**바이탈 요약 재설계 반영 완료 (2026-09-04, §7b)**

| 항목 | 파일 |
|---|---|
| REPORTABLE/AT HANDOFF 2블록 + MINOR 분리, R1~R4 판정, baseline 게이트, 계획된 저체온 | `utils/vital_summarizer.py` (전면 재작성) |
| 판정 규칙 상수 (`VITAL_REPORT_RULES`·`MIN_DEVIATION`·`BASELINE_DROP_REQUIRED`·`hard_lines`) | `utils/vital_thresholds.py` |
| 마취기록 타임라인 → 개입 추출 (R1), 가온·최초삽관 의도적 제외 | `utils/anesthetic_record.py`(신규) |
| 프롬프트 블록 지시 교체 + 지문 `v3.2-en+vital3blk` | `pipeline_v3/prompt_utils.py` |
| 구조화 이벤트 감사본 경로 + 무효화 scope | `config_v3.py`(`VITAL_AUDIT_PKL`), `scripts/invalidate_v3.py` |
| 재생성 시 selftest 자동 선행 + 자가진단 8줄 + 감사본 저장 | `scripts/rebuild_vital_map.py` |
| 층별·규칙별 PPV 감사 (구조화본 우선, 구 포맷 폴백) | `scripts/vital_flag_audit.py` |
| 회귀 테스트 2종 추가 → **20/20 통과** | `tests_v3/test_v3.py`, `utils/vital_summarizer.py --selftest` |

**구현 대기 (아래 순서대로)**

1. `gen_pairs.py` **2단계 분리** (`--stage candidates|judge|both`) — 4장 제약에서
   prometheus 93GB를 6~9회 재적재하는 낭비 제거 (§8d.6-2)
2. `prompt_lab.py` + `prompt_variants.py` — 규칙 지표 6종, 검토 md/xlsx (§4)
3. `rule_metrics.py` — L1/soft 규칙의 **단일 소스**. prompt_lab·gen_pairs·검토시트가 공유 (§7.3)
4. `make_fewshot_bank.py` 개편 — `--source gt` (gold_gt_fewshot 6건의 전문의 GT 사용) (§3.2)
5. `gen_pairs.py` 개편 — teacher 생성, `--row_shard i/n` 3인스턴스 병렬, 규칙 선별,
   listwise 판정(결정 후), 구성적 negative(결정 후) (§2.2/§7)
6. `eval_v3/evaluate.py` 개편 — 축별 판정자 평균, `n_judges ≥ 2`, LOO 민감도, per-judge 기록 (§6)
7. `scripts/export_pairs_review.py` — 검토 xlsx + 통과기준 자동 판정 (§5)
8. `eval_v3/calibrate.py` — 3 judge + 입장기준 + G4a 파일럿 모드 (§6.3)
9. `report_v3.py` — LOO 표, n=16 검정력 경고, teacher/judge/양자화 메타 표기

**서버에서 먼저 확인할 것**

- `python -m tests_v3.test_v3` — **로컬 venv(pandas 3.0.5 + scipy)에서 20/20 통과 확인.**
  서버 pandas 버전(2.x)에서 한 번 더 돌린다
- `python -m utils.vital_summarizer --selftest` — 바이탈 규칙 11조건 (재생성이 자동 선행)
- `python scripts/check_prompt_budget.py --models llama qwen35 gemma4` — 영어화로 지시문
  토큰이 줄었는지 실측 (예상: 감소, EMR 여유 증가)
- ~~미러의 Qwen 72B AWQ 경로 확정~~ → HF 실측으로 확정 완료. 남은 문제는 프록시 차단(§8c)
- 프롬프트가 바뀌었으므로 `gold_checklist_v3.json`·`fewshot_bank_v3.json`·`calibration_v3.csv`
  전부 재생성 (`scripts/invalidate_v3.py`)
