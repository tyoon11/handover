# 마취 인계요약(PACU/ICU Handoff) 생성 파이프라인 — 전체 개요

> 이 문서는 프로젝트 전체(데이터 → 학습 → 추론 → 평가)와 현재까지 발견·수정된 이슈를 한 곳에 정리한 것입니다. Claude(웹)에게 맥락을 설명하기 위한 자기완결적 문서입니다.

---

## 1. 목표

소아 수술 환자의 **마취기록(EMR)을 입력**받아, PACU/ICU로 넘길 때 쓰는 **초간결(ultra-brief) 인계요약문**을 생성하는 LLM을 만든다. 핵심 철학은 **"exception-based handoff"** — 정상/안정 소견은 노이즈로 보고, **임상적으로 조치가 필요한 이상소견만** 간결히 전달.

여러 베이스 모델을 SFT/RLAIF로 학습하고, LLM-as-Judge + factual-consistency 지표로 비교한다.

---

## 2. 데이터

| 데이터 | 파일 | 내용 |
|---|---|---|
| 학습(합성) | `jsft_score_251213.pkl` (2736행) | `chosen`/`rejected` 쌍. 둘 다 **여러 LLM이 생성한 요약 초안**을 Prometheus judge로 점수매겨 고른 것 (chosen=최고점, rejected=무작위 하위) |
| 평가(gold) | `gold_sampled_251008.pkl` (22행 × 228컬럼) | gold 케이스의 전체 EMR. 추론 입력으로 사용 |
| Vital 요약 | `vital_summary_map.pkl` | `{수술ID: vital 요약문}` |

### 입력 EMR 구성 (`build_emr_text`)
```
- Pre-anesthetic Patient Conditions Summary  (마취 전 환자상태 요약)
- Preoperative Preparations and Premedication
- Anesthetic TOTALS                          (약물/수액/수혈 총량)
- Anesthetic Record                          (마취기록 원문 — 타임스탬프 이벤트 로그)
+ INTRAOPERATIVE VITAL SUMMARY               (vital_map에서)
```

### ⚠️ "정답(human reference)"에 관한 중요한 사실
- **clinician이 직접 쓴 인계요약 정답은 존재하지 않는다.** 어노테이션 엑셀(`인계요약지_gold_sampled_251002_KHS.xlsx`)에서 사람이 작성해야 했던 칸(`인계요약지`)은 **23건 전부 비어 있음(0/23)**.
- 채워져 있는 건 `참고용` 컬럼의 **gemma-3-27b-it가 생성한 초안(21/23)**뿐. (원본 노트북이 `Human_handoff`라 부른 것이 사실 이 gemma 초안)
- 따라서 **정답 기반(reference-based) 평가는 불가능**하고, 평가는 reference-free judge + factual consistency로만 한다 (아래 6장).
- export 결과물에서는 이 gemma-3-27b 초안을 **"참고 인계요약(정답 아님)"**으로만 표시한다.

---

## 3. 파이프라인 단계

```
01 전처리 → 02 SFT → 03 RLAIF(DPO/SimPO) → 04 추론 → 05 평가
            └────────── run_all.py 가 오케스트레이션 ──────────┘
```

| 스크립트 | 역할 |
|---|---|
| `pipeline/02_sft_train.py` | Synthetic SFT (LoRA). chosen 요약 + judge(A/B) augmentation |
| `pipeline/03_rlaif_train.py` | Self-Judge RLAIF — DPO 또는 SimPO (LoRA) |
| `pipeline/04_inference.py` | 배치 추론. **vLLM(tensor-parallel) 우선**, 실패 시 HF fallback |
| `pipeline/05_evaluate.py` | LLM-as-Judge(Prometheus) + SCALE(factual consistency) |
| `run_all.py` | GPU 풀 기반 병렬 오케스트레이션, 2단계(학습+추론 / judge 평가) |
| `tests/smoke_test.py` | 각 단계를 소량 샘플로 빠르게 검증 |
| `utils/export_results.py` | 결과를 Excel/Notion/HTML로 정리 |

### 실험 매트릭스 (모델 × 7 실험)
`run_all.py`의 `EXPERIMENTS`:
| 실험 key | SFT epoch | RLAIF | 선행 의존 |
|---|---|---|---|
| `raw` | – | – | – |
| `rlaif_dpo` | – | DPO (raw에서) | – |
| `rlaif_simpo` | – | SimPO (raw에서) | – |
| `sft_1ep` | 1 | – | – |
| `sft_3ep` | 3 | – | – |
| `sft_1ep_dpo` | 1 | DPO | sft_1ep |
| `sft_3ep_dpo` | 3 | DPO | sft_3ep |

---

## 4. 모델 (`SFT_MODELS`)

| key | 경로 | 비고 |
|---|---|---|
| `llama` | Llama-3.1-8B-Instruct | |
| `qwen` | Qwen3-8B | thinking 모델 (학습/추론 시 `enable_thinking=False`) |
| `gemma4` | gemma-4-E4B-it | gemma 특수처리(LoRA 타깃·attn) |
| `gemma4_31b` | gemma-4-31B-it | 31B, 1장에 안 들어감(≥2 GPU) |
| `qwen35` | Qwen3.5-9B | thinking 모델 |
| `hari` | hari-q3-8b | Qwen 기반 |

- `GEMMA4_BASES = {"gemma4","gemma4_31b"}` — 이 모델들은 LoRA 타깃이 언어모델 레이어 regex(`LORA_TARGET_MODULES_GEMMA4`), flash-attn 비활성.
- LoRA: `r=8, alpha=16, dropout=0.1`, target = `q/k/v/o_proj, gate/up/down_proj`.

---

## 5. 학습 세부

### 5-1. SFT (`02_sft_train.py`)
- **1행 → 3 샘플** 생성 (원본 JudgeAugmented 방식):
  1. **generation**: `chosen` 인계문 전체에 loss
  2. **judge A>B**: "EMR + A=chosen + B=rejected" → 정답 토큰 `"A"`
  3. **judge B>A**: 순서 뒤집고 → 정답 토큰 `"B"`
- **라벨 마스킹**: 프롬프트(system+user, `add_generation_prompt=True`) 길이만큼 `-100` 마스킹, 그 뒤 응답 토큰에만 loss. (이전엔 응답 템플릿 문자열 검색 방식이었으나 gemma에서 실패 → 7장 참고)
- 하이퍼파라미터 (`SFT_CONFIG`, 원본 `llama_jsft_3epoch.yaml` 기준):
  - `lr=2e-5`, `per_device_batch=2`, `grad_accum=4`, `epochs=3`, `cosine`, `warmup_steps=50`, `bf16`
- `Trainer` + `TrainingArguments`, `gradient_checkpointing=True (use_reentrant=False)`
- 저장: PEFT adapter (`adapter_config.json` + `adapter_model.safetensors`). 저장 후 파일 존재 검증.

### 5-2. RLAIF (`03_rlaif_train.py`)
- `chosen`/`rejected` 쌍으로 DPO 또는 SimPO.
- **DPO**: `loss_type=sigmoid`, `beta=0.1` (원본 `llama_dpo.yaml`)
- **SimPO**: `beta=2.0`, `gamma_beta_ratio=0.25` (커스텀 `trainers/simpo_trainer.py`, `simpo_config.py`)
- 하이퍼파라미터 (`RLAIF_CONFIG`):
  - `lr=5e-6`, `per_device_batch=1`, `grad_accum=8`(effective batch 8 유지), `epochs=3`, `constant_with_warmup`, `warmup_steps=50`
- SFT 체크포인트에서 시작 시: PEFT adapter를 `merge_and_unload` 후 새 LoRA 부착.

---

## 6. 평가 (`05_evaluate.py`) — **reference-free**

정답이 없으므로(2장 참고), 원본 방식 그대로 **참조 없이** 평가한다.

### 6-1. LLM-as-Judge (Prometheus-8x7b-v2.0)
- 2개 루브릭으로 각각 1~5점 채점 → 합(`sum_score`, 2~10):
  - **BREVITY**: 간결성/시간효율. 정상소견 나열 = 'CLINICAL NOISE'로 감점. 안정 환자엔 'None/특이사항 없음'이 만점.
  - **CRITICAL**: 임상적으로 조치가 필요한 이상소견을 정확히 짚는가.
- 시스템 프롬프트가 "길이/완전성/유창성에 점수 주지 말고 루브릭만 따르라"고 강제.
- **연속 점수화**: `[RESULT] N` 정수 파싱은 변별력이 낮아(만점 포화), 점수 토큰(1~5)의 **확률분포 기대값**으로 연속값(예: 4.7) 산출. (정수 파싱은 fallback)

### 6-2. SCALE (factual consistency)
- Flan-T5 large/xl 기반. **premise = EMR, hypothesis = 모델 출력** 의 사실 일치도(0~1, `scale_xl`).
- "출력이 EMR 사실과 얼마나 맞는가(환각 여부)"를 측정. judge(간결성)와 **다른 축**.

### 6-3. 두 지표를 함께 봐야 하는 이유
- judge sum이 높아도 scale_xl이 낮으면 = "간결하지만 사실과 안 맞는(환각 가능)" 출력.
- 실제로 judge sum과 scale_xl은 **약한 음의 상관**을 보임 (간결성에 치우치면 사실성↓).

---

## 7. 알려진 이슈 & 수정 이력

| 이슈 | 상태 | 설명 |
|---|---|---|
| 새 transformers/TRL 호환 | ✅ 수정 | `tokenizer→processing_class`, `compute_loss(num_items_in_batch)`, `log(start_time)`, `get_batch_samples` 이름충돌, `warmup_ratio` deprecation |
| SimPO Qwen 크래시 | ✅ 수정 | `bos_token_id=None`인 Qwen 계열에서 BOS prepend 시 collator 오류 → BOS 없으면 건너뜀 |
| judge 입력 truncation | ✅ 수정 | 긴 EMR(2~3k토큰)이 잘려 response/rubric이 사라짐 → judge가 최저점. 이제 EMR만 예산에 맞춰 자르고 한도 8192 |
| qwen35 thinking 누출 | ✅ 수정 | 답 뒤에 `<think>`(안 닫힘) trailing → clean_output 강화(닫힘/안닫힘/태그없는 CoT 모두 제거) |
| SCALE GPU/경로 | ✅ 수정 | HF Hub SSL 차단 → 로컬 Flan-T5 경로 + monkey-patch + 명시적 `.to(device)` |
| **SFT loss=0 (gemma)** | ✅ 수정 | 라벨 마스킹이 응답템플릿 문자열 검색 의존 → gemma에서 못 찾아 전체 마스킹 → 학습 안 됨. **gemma4(E4B) 기존 결과 무효, 재학습 필요.** 프롬프트 길이 마스킹으로 교체 + sanity check |
| run_all `--skip_done` 버그 | ✅ 수정 | 체크포인트 없어도 스킵 + PEFT는 adapter_config.json인데 config.json으로 확인 → `_ckpt_exists()`로 통일 |
| 평가 "정답" 오해 | ✅ 수정 | 마취기록 원문을 정답처럼 표시하던 것 → gemma-3-27b 참고 초안으로 교체, "정답 아님" 명시 |

### 아직 남은 한계
- **clean human reference 부재** — 정답 기반 평가 불가, judge/SCALE로만 비교.
- **judge 점수 포화** — 다수 케이스 brevity 5/5. 연속 점수화로 완화했으나 근본은 루브릭 특성.
- **학습은 아직 `device_map="auto"`(pipeline-parallel)** — 31B 학습은 느림. 추론만 vLLM 가속됨.

---

## 8. 인프라 / 실행

- **GPU**: 4 × RTX A6000 (48GB).
- **추론**: vLLM tensor-parallel (`tensor_parallel_size=GPU수`) — 31B도 4장에 분산, 배치 처리. `--engine auto`(기본)로 vLLM 우선, 미지원/실패 시 HF fallback. PEFT는 CPU에서 merge 후 vLLM 로드(캐시).
- **학습**: HF `Trainer` + `device_map="auto"` + `max_memory` (모델을 여러 GPU에 분할 — 메모리는 분산되나 연산은 순차).
- **judge(Prometheus 8x7b ~93GB)**: 4장에 분산, GPU별 실제 여유 메모리 기준 `max_memory` 자동 설정.
- vLLM `GLIBCXX` 문제: conda libstdc++를 `LD_PRELOAD`로 자동 추가.

### 2단계 오케스트레이션 (`run_all.py`)
- **Phase 1**: 모든 모델·실험의 학습+추론을 GPU 풀로 병렬.
- **Phase 2**: judge 모델을 **1회만 로드**해 모든 추론 결과를 batch 평가 (judge 로딩 시간 절약 + GPU 충돌 방지).

### 주요 명령
```bash
# smoke (단계별 빠른 검증)
python tests/smoke_test.py --all --gpus 4,5,6,7 --models all

# 전체 파이프라인 (예: gemma4_31b)
python run_all.py --models gemma4_31b --gpus 4,5,6,7 --gpus_per_job 4 --skip_done

# SCALE만 추가 (judge 끝난 뒤)
python run_all.py --scale_only --run_id <RUN_ID> --eval_gpus 4,5,6,7

# 결과 정리 (Excel + Notion MD + 임상검토용 HTML)
python run_all.py --export --run_id <RUN_ID>
```

### 산출물 (`outputs/<run_id>/`)
- `inference/<model>_<exp>/gold_results.jsonl` — 추론 결과 (`generated`, `generated_raw`)
- `evaluation/<model>_<exp>/gold_results_scores.jsonl` — judge 점수 (+`_scale.jsonl`)
- `results.xlsx` / `results_notion.md` / `results_cases.html` — 비교/검토용

---

## 9. 한 줄 요약
소아 마취기록(EMR)→초간결 PACU/ICU 인계문 생성. 6개 베이스 모델을 SFT/DPO/SimPO로 학습하고, **정답이 없으므로** Prometheus judge(간결성·중요성, 연속 점수) + SCALE(사실 일치도)로 비교. 추론은 vLLM tensor-parallel로 가속. 현재 gemma 계열 SFT 라벨 마스킹 버그를 막 수정해 재학습이 필요한 상태.
