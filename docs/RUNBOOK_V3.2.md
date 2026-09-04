# v3.2 실행 런북 (폐쇄망 서버)

기준 2026-09-04 · 대상: 프롬프트 영어화 + 역할 재배치 + gold 6/16 분할 + **바이탈 요약 재설계**
이후의 첫 실행. v3.1 런북([RERUN_RUNBOOK.md](RERUN_RUNBOOK.md))은 교수님 GT 반영 절차(§1)만
아직 유효하고 나머지는 이 문서가 대체한다.

> **핵심 순서 하나만 기억할 것**: `vital_flag_audit --source text` 로 **개편 전 기준선을 먼저
> 캡처**한 다음에 무효화·재생성한다. 순서를 바꾸면 비교 대상이 사라진다.

---

## 0. 로컬(작업 PC)에서 먼저 — 커밋·푸시

서버는 `git pull` 로만 코드를 받는다. 푸시 전에는 아래 전부 무의미하다.

```bash
git log --oneline -1          # v3.2 바이탈 재설계 커밋이 있는지
git push
```

---

## 1. 서버: pull + 환경

```bash
cd /home/coder/workspace/.../01_HANDOVER      # repo 경로
git pull
bash scripts/install_hooks.sh                 # PHI pre-commit 훅 (최초 1회)

# 경로
export HANDOVER_BASE_DIR=/home/coder/workspace/data/handover
export HANDOVER_MODEL_DIR=/home/coder/workspace/data/local_models

# vLLM GLIBCXX — 세션당 1회, 반드시 셸에서 export
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# 새 run_id — 옛 산출물과 섞이지 않게
export HANDOVER_RUN_ID=v32_20260904
```

**모델을 물리 이관(외장SSD)해 목표 구성이 갖춰지면 env 오버라이드는 필요 없다** —
`config_v3.py` 기본값이 이미 목표 배치다.

| 역할 | 기본값(목표) | 용량 |
|---|---|---|
| teacher · checklist 추출기 | `qwen35_122b` (Qwen3.5 122B-A10B GPTQ-Int4) | 78.9G |
| 쌍 judge | `prometheus` (8x7B) | 93.4G |
| 평가 패널 | `medgemma27b` + `llama70b`(AWQ) + `mprometheus`(14B) | 54.9 + 39.8 + 29.5G |
| 학습 base | `llama` · `qwen` (+ `qwen35` · `gemma4` · `hari`) | 16~19G each |

일부만 도착했다면 **도착한 것만으로 진행**한다(폴백, §8b):

```bash
# 예: teacher·판정 일부가 아직 없을 때만
export HANDOVER_TEACHER=gemma4_31b
export HANDOVER_EVAL_JUDGES=medgemma27b,qwen35,hari
export HANDOVER_CHECKLIST_EXTRACTOR=gemma4_31b
```

> 폴백 구성으로는 **dev까지만** 간다(모델·설정 선택). gold 최종 평가는 목표 구성 확보 후.
> 재평가는 추론 산출물 재채점(수 시간)이고 학습을 다시 하지 않는다.

## 1b. ★ 이관한 모델 검증 (하기 전에 아무것도 시작하지 말 것)

물리 이동은 **조용히 깨진다** — 잘린 샤드, 빠진 파일, 복사 중단. 몇 시간 학습을 돌린 뒤
발견하는 게 최악이다.

```bash
# ① 무결성 대조 (받은 쪽에서 --checksums write 로 만든 checksums.sha256 필요)
python utils/download_models.py --checksums verify --group v32
python utils/download_models.py --checksums verify --group base

# ② 레지스트리 기준 보유 확인 (용량·필수 파일)
python utils/download_models.py --check --group all

# ③ 디스크 여유 — weight 합계 ≈ 445GB + merged LoRA 캐시(변형당 16~19GB)
df -h $HANDOVER_MODEL_DIR $HANDOVER_BASE_DIR
```

`verify` 는 **누락 / 크기 불일치 / 해시 불일치를 각각 구분해** 보고한다. 하나라도 나오면
그 모델만 다시 옮긴다(전체 재복사 불필요).

### ④ vLLM 적재 스모크 (모델당 3~5분 — 가장 중요)

`qwen35_122b` 는 **GPTQ-Int4 MoE**다. Ampere(A6000)에서 vLLM이 MoE 양자화 커널을 잡지 못하면
로드 자체가 실패한다. 여기서 걸러야 한다.

```bash
# teacher (TP=2, KV 여유가 빡빡하므로 max_model_len 을 낮춰 시험)
CUDA_VISIBLE_DEVICES=0,1 python -c "
from pipeline_v3.config_v3 import model_path
from pipeline_v3.eval_v3.engine import JudgeEngine
e = JudgeEngine(model_path('qwen35_122b'), backend='vllm',
                gen_cfg=dict(max_new_tokens=32, temperature=0.0), max_model_len=3200)
print(e.chat(['Reply with OK only.'], system=None)); e.unload()"

# 판정 3종
CUDA_VISIBLE_DEVICES=0,1 python -c "
from pipeline_v3.config_v3 import model_path
from pipeline_v3.eval_v3.engine import JudgeEngine
for k in ('medgemma27b','llama70b'):
    e = JudgeEngine(model_path(k), backend='vllm', gen_cfg=dict(max_new_tokens=32))
    print(k, e.chat(['Reply with OK only.'], system=None)); e.unload()"
CUDA_VISIBLE_DEVICES=0 python -c "
from pipeline_v3.config_v3 import model_path
from pipeline_v3.eval_v3.engine import JudgeEngine
e = JudgeEngine(model_path('mprometheus'), backend='vllm', gen_cfg=dict(max_new_tokens=32))
print(e.chat(['Reply with OK only.'], system=None)); e.unload()"

# 쌍 judge (TP=4 필수 — 93.4GB)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -c "
from pipeline_v3.config_v3 import model_path
from pipeline_v3.eval_v3.engine import JudgeEngine
e = JudgeEngine(model_path('prometheus'), backend='vllm', gen_cfg=dict(max_new_tokens=32))
print(e.chat(['Reply with OK only.'], system=None)); e.unload()"
```

`backend='vllm'` 은 실패 시 **HF로 조용히 폴백하지 않고 예외를 낸다** — 스모크 테스트가
의미를 가지는 이유다. 실패하면:

| 실패 | 대응 |
|---|---|
| GPTQ MoE 커널 없음 | vLLM 버전 확인 → 안 되면 teacher 를 `qwen35_27b`(bf16 55.6G, TP=2) 로 교체 |
| KV 부족(OOM) | `max_model_len` ↓, `gpu_memory_utilization 0.95`, `max_num_seqs 8` |
| arch 미지원 | `infer_engine_for()` 에 HF 강제 추가 (변형 간 엔진 통일 필수) |

## 2. 코드·역할 검증 (GPU 불필요, 2분)

```bash
python -m tests_v3.test_v3                       # 기대: 20/20 통과
python -m utils.vital_summarizer --selftest      # 기대: 11조건 전부 ✓
python -m pipeline_v3.prompt_registry            # spec_version = v3.2-en+vital3blk
python -c "from pipeline_v3.config_v3 import role_summary, validate_roles, \
assert_models_available, EVAL_JUDGES, TEACHER_KEY, PAIRGEN_JUDGE, CHECKLIST_EXTRACTOR; \
print(role_summary()); validate_roles(); \
assert_models_available([TEACHER_KEY, PAIRGEN_JUDGE, CHECKLIST_EXTRACTOR] + EVAL_JUDGES)"
```

- `validate_roles()` 가 순환(teacher ∈ 패널 / teacher == 쌍judge / 쌍judge ∈ 패널)을 **하드 실패**로
  잡는다. 경고는 로컬 구성의 알려진 약점이다(패널에 qwen 2개) — 그대로 진행하고 LOO로 보고한다.
- `assert_models_available()` 이 미보유 모델을 **지금** 알려준다. 몇 시간 뒤 죽는 것을 막는다.

**여기서 실패하면 아래로 내려가지 않는다.**

## 3. ★ 바이탈 개편 전 기준선 캡처 (무효화 전에!)

```bash
python scripts/vital_flag_audit.py --source text --out ~/audit_v31.md
```

구 포맷(`[유의]` 마커) `vital_summary_map.pkl` 을 파싱해 GT 대조 PPV를 낸다.
**이 파일이 개편 효과의 유일한 비교 대상이다.** 지우기 전에 반드시 뽑는다.

## 4. 캐시 무효화

```bash
python scripts/invalidate_v3.py --apply --archive --scope vital,checklist,fewshot,calib
```

`--archive` 로 옛 파일은 타임스탬프 폴더로 옮겨진다(삭제 아님).
`pairs/`·`sft/`·`rlaif/` 는 아직 만들지 않았으니 scope에서 뺐다.

## 5. 바이탈 맵 재생성 (CPU만 — GPU 안 씀)

```bash
python scripts/rebuild_vital_map.py --apply --force
```

selftest가 자동 선행되고, 끝나면 **자가진단 8줄**이 나온다. 판독법:

| 줄 | 정상 | 이상하면 |
|---|---|---|
| `REPORTABLE 있음 n/N` | 케이스의 일부만 (전부면 의심) | 전부면 규칙이 안 걸리는 것 |
| `AT HANDOFF 블록 n/N` | ≈ N | 적으면 타임스탬프 결측 |
| `개입 연동(R1) 표기 n/N` | **>0** | **0이면 마취기록 매핑 실패** → `_find_record_map` 경고 확인 |
| `계획된 저체온 n/N` | 심장수술 건수 수준 | 0이면 CPB 키워드 미검출 |
| `이벤트 X개 중 REPORTABLE Y개` | **30~60%** | **>70%면 v3.1 수준 과대등재** (경고 출력됨) |
| `측정오류 배제` | 소수 | 많으면 `PLAUSIBLE_RANGE` 확인 |
| `ebl_no_weight / uo_no_weight` | 0에 가깝게 | 많으면 체중 컬럼 매핑 실패 |

`--sample 3` 미리보기로 실제 3블록 출력을 눈으로 확인한다.
**GPU를 안 쓰므로 6·7단계와 동시에 돌릴 수 있다.**

## 6. 개편 후 감사 + 비교 → 컷오프 결정

```bash
python scripts/vital_flag_audit.py --out ~/audit_v32.md
```

두 md를 나란히 본다. 판독 기준:

| 지표 | 기대 | 아니면 |
|---|---|---|
| **REPORTABLE PPV vs MINOR PPV** | 격차가 커야 함 (예: 70% vs 20%) | 비슷하면 **규칙이 정보를 못 담고 있다** → 컷오프 재설계 |
| 규칙별 PPV | R1·R4 ≥ R2 ≥ R3 정도 | R1이 낮으면 개입 키워드가 틀린 것 |
| Recall (GT 언급 중 REPORTABLE 존재) | v3.1과 비슷하거나 약간 낮게 | 크게 떨어지면 R2 지속 컷오프(10분)가 과함 |
| `1회` / `<1분` 버킷 | v3.2에선 대부분 MINOR | REPORTABLE에 남아 있으면 R3/R4 조건 확인 |

- **컷오프 조정은 fewshot(6건) 열로만.** test(16건) 열로 고르면 입력 표현을 test에 맞추는 것이다.
- 조정 대상: `MIN_DEVIATION`, `VITAL_REPORT_RULES["min_longest_min"]`, `hard_lines()`.
  전부 `utils/vital_thresholds.py` 한 곳이다. 고치면 5단계부터 다시.
- GT는 1~5문장 제약이 있어 "언급 안 됨 ≠ 안 중요함" → **PPV는 하한**이다. 확정에는 전문의
  이진 라벨 20건이 필요하다(§7b.7-2).

## 7. gold checklist 재추출 (추출기 TP=2 · 22건 · 10~20분)

```bash
python -m pipeline_v3.build_gold_checklist_v3 --gpus 0,1
python -c "
import json; from pipeline_v3.config_v3 import GOLD_CHECKLIST_JSON as P
d = json.loads(P.read_text(encoding='utf-8'))
from collections import Counter
print(Counter(v.get('source') for v in d.values()))
print('reviewed:', sum(1 for v in d.values() if v.get('reviewed')), '/', len(d))
print('items 총합:', sum(len(v.get('items', [])) for v in d.values()))"
```

- `source` 분포에 `gold_llm_failed` 가 많으면 추출기(`qwen35_122b`)의 JSON 실패 →
  `--extractor` 로 교체(보유분 중 가장 강한 모델).
- **`reviewed: 0` 이면 coverage는 "잠정"이다.** 논문에 쓰려면 전문의 검수가 필요하다(§8 G3).
- 프롬프트가 영어화됐으므로 항목 문구가 이전과 다를 수 있다. `finding`/`source`는 **한국어**여야
  한다(추출 프롬프트가 그렇게 지시한다) — 영어로 나오면 프롬프트 회귀다.

## 8. few-shot bank + 프롬프트 예산

```bash
python -m pipeline_v3.make_fewshot_bank --gpus 0,1        # 현행: 모델 초안(스타일 앵커)
python scripts/check_prompt_budget.py --models llama qwen35 gemma4
```

- `make_fewshot_bank --source gt`(gold GT 6건 사용)는 **아직 구현 대기**다(§11). 지금은 현행
  방식으로 채워 프롬프트 예산만 먼저 확인한다.
- 예산 확인 기대: 영어화 + 바이탈 블록 축소로 **지시문 토큰 감소, EMR 여유 증가**.
  `PromptTruncationError` 가 나면 `PAIRGEN["fewshot_emr_chars"]` 부터 줄인다.

## 9. calibration — 판정 패널 자격 검증 (GPU · 1~2시간)

```bash
python -m pipeline_v3.eval_v3.calibrate --gpus 0,1
cat $HANDOVER_BASE_DIR/data/preprocessed/calibration_v3.csv    # 경로는 출력에 찍힌다
```

SY 엑셀 전문의 채점 168행을 **패널 3개 전원**으로 재채점해 judge별 × 생성모델 family별
Spearman을 낸다. `calibrate.py` 는 `EVAL_JUDGES` 를 그대로 순회하므로 3판정도 코드 수정 없이 돈다.

**쌍 judge도 여기서 함께 검증한다** — `prometheus`(영어 중심)와 `mprometheus`(20+개 언어)를
같은 168행에 태워 한국어 임상 판정력을 비교하고 선호쌍 judge를 최종 확정한다.
`HANDOVER_EVAL_JUDGES=medgemma27b,llama70b,mprometheus,prometheus` 로 한 번 더 돌리면
같은 표에 나란히 나온다(이 실행은 **비교 전용** — 실제 패널은 3개 그대로).

**입장 기준** (`JUDGE_ADMISSION`): 축별 Spearman 부호가 맞고 `ρ ≥ 0.4`, 파싱 실패 `< 10%`.
탈락하면 `EVAL_JUDGES_RESERVE`(`gemma4_31b`, `qwen35`)로 승계하고 **전 모델에 동일 적용**한다.
승계는 **모델 점수를 보기 전에** 결정한다 — 결과를 보고 judge를 갈아치우면 평가가 후처리 선택이 된다.

## 10. (선택 · GPU 0장) teacher 문체 사전조사 — 다린 합성데이터 재활용

다린이 4개 모델(`gemma-3-27b` · `medgemma-27b` · `Qwen3-32B` · `Llama-3.3-70B`)로 **같은 2736행에
이미 생성해 둔 출력**이 디스크에 있다. 어느 teacher가 한국어 인계 문체·지시 준수에서 앞서는지
GPU 없이 지금 볼 수 있다.

```bash
python - <<'PY'
import pandas as pd, ast, collections
p = "<다린>/data/synthetic/jsft_score_251213.pkl"
df = pd.read_pickle(p)
print(collections.Counter(
    ast.literal_eval(x)["chosen"] for x in df.chosen_rejected_info.dropna()))
for c in [c for c in df.columns if c.startswith("score_")]:
    print(f"{c:34s} mean={pd.to_numeric(df[c], errors='coerce').mean():.2f}")
PY
```

`chosen` 점유율이 곧 "다린의 prometheus 기준 승률"이다. teacher 는 `qwen35_122b` 로 이미
확정했으므로 **필수는 아니다** — 다만 ① 스모크 테스트에서 teacher 가 안 뜰 때의 대안 순위,
② 논문의 "teacher 선택 근거" 한 줄을 GPU 0장으로 확보할 수 있다.

---

## 11. 여기까지가 오늘 가능한 전부 — 다음은 코드 대기

아래가 없으면 G5(SFT 타깃 생성)를 **설계대로** 돌릴 수 없다. 작업 PC에서 구현 중이다.

| # | 항목 | 왜 필요한가 |
|---|---|---|
| 1 | `gen_pairs --stage candidates\|judge\|both` | 4장 제약에서 prometheus 93GB를 6~9회 재적재하는 낭비 제거 (§8d.6) |
| 2 | `gen_pairs` teacher 모드 + `--row_shard i/n` + 규칙 선별 | 3인스턴스 병렬 생성, judge 없는 SFT 타깃 선별 (§7.4) |
| 3 | `rule_metrics.py` | L1 규칙의 단일 소스 — prompt_lab·gen_pairs·검토시트 공용 |
| 4 | `make_fewshot_bank --source gt` | GT few-shot(gold 6건) (§3.2) |
| 5 | `prompt_lab.py` | raw 출력으로 프롬프트 확정·동결 (§4) |
| 6 | `eval_v3/evaluate.py` 3판정 개편 | 축별 평균 · `n_judges ≥ 2` · LOO · n_eff (§6) |
| 7 | `export_pairs_review.py` | 합성데이터 사람 검토 게이트 (§5) |

## 12. 그 다음 순서 (코드가 오면)

```
G1  프롬프트 확정·동결      prompt_lab (raw only, dev 24건) → prompt_frozen.json
G2  GT few-shot 뱅크        make_fewshot_bank --source gt
G5  SFT 타깃 생성           gen_pairs --split sft --stage both   (teacher 6장 창에서)
G6  ★ 합성데이터 사람 검토   export_pairs_review → 50쌍 일치 ≥80%
G7  학습 → on-policy 쌍 → RLAIF → dev 추론/평가   (4장, --gpus_per_job 1)
G8  dev(110)로 모델·설정 확정 — gold 미개봉
G9  gold test(16) 1회 개봉 — **목표 모델 확보 후에만** (로컬 패널로 최종 평가 금지)
```

- **학습 대상은 `--models llama qwen` 를 권한다.** 로컬 패널이 `medgemma27b·qwen35·hari` 라
  `qwen35`·`hari` 를 학습하면 **자기가 자기를 채점**한다. teacher 도 `gemma4_31b` 이므로
  `gemma4_31b` 학습은 자기증류가 된다.
- 학습은 `--gpus_per_job 1` — `device_map="auto"` 는 naive 모델 병렬이라 2장을 줘도 속도가
  같고, 유효 배치는 GPU 수와 무관하다(§8d.6). 4장이면 4잡 병렬.
- **gold(16)는 목표 구성에서만 개봉한다.** 이관이 끝났다면 그대로 G9까지 갈 수 있다.
  폴백 구성으로 dev를 돌렸다면, 목표 구성이 갖춰진 뒤 **추론 산출물만 재채점**한다(수 시간,
  학습 불필요).

---

## 부록: 실패 시 대응

| 증상 | 원인 | 대응 |
|---|---|---|
| `assert_models_available` 실패 | 미보유 모델 | env 로 로컬 구성 교체 (§1) |
| `validate_roles` 하드 실패 | 역할 순환 | env 재확인 — teacher/쌍judge/패널이 겹쳤다 |
| rebuild 자가진단 `개입 연동 0건` | 마취기록 컬럼 미매핑 | `_find_record_map` 경고 확인, EMR pkl 구조 점검 |
| rebuild `REPORTABLE >70%` 경고 | 컷오프가 느슨함 | `MIN_DEVIATION`·`min_longest_min` 상향 후 5단계 재실행 |
| audit Recall 급락 | R2 컷오프 과함 | `min_longest_min` 10분 → 5분 하향 검토 |
| checklist `gold_llm_failed` 다수 | 추출기 JSON 실패 | `--extractor` 교체 (보유분 중 가장 강한 모델) |
| calibration ρ < 0.4 | judge 부적격 | `EVAL_JUDGES_RESERVE` 승계, 점수 보기 전에 결정 |
| vLLM이 갑자기 느림 | `LD_PRELOAD` 미설정 | 셸에서 export (프로세스 시작 시점에만 유효) |
