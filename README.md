# 소아수술실 인계요약지 생성 파이프라인

## 파일 구조

```
handover/
├── config.py            # 전체 설정 (경로·하이퍼파라미터·프롬프트)
├── vital_summarizer.py  # 바이탈 측정값 → 한국어 요약 텍스트 (rule-based)
├── 00_preprocess.py     # EMR 전처리 + split + 바이탈 summary map 생성
├── 01_synthetic_gen.py  # 합성데이터 생성 + LLM-as-Judge 채점 + chosen/rejected 선정
├── 02_sft_train.py      # Synthetic SFT (LoRA)
├── 03_rlaif_train.py    # Self-Judge RLAIF (DPO / SimPO)
├── 04_inference.py      # 학습된 모델 배치 inference
└── 05_evaluate.py       # LLM-as-Judge (Prometheus) + SCALE 평가
```

## 실행 순서

```bash
# 1. 전처리 (1회)
python 00_preprocess.py

# 2. 합성데이터 생성 (시간 오래 걸림, GPU 필요)
python 01_synthetic_gen.py

# 3. 학습 (모델 × epoch 수 조합)
python 02_sft_train.py --base llama --epochs 1
python 02_sft_train.py --base llama --epochs 3
python 02_sft_train.py --base qwen  --epochs 1
python 02_sft_train.py --base qwen  --epochs 3

# 4. RLAIF (loss × SFT 유무 조합)
python 03_rlaif_train.py --base llama --loss dpo                   # Raw + DPO
python 03_rlaif_train.py --base llama --loss simpo                 # Raw + SimPO
python 03_rlaif_train.py --base llama --loss dpo  --sft_epochs 1  # SFT1ep + DPO
python 03_rlaif_train.py --base llama --loss dpo  --sft_epochs 3  # SFT3ep + DPO
python 03_rlaif_train.py --base qwen  --loss dpo  --sft_epochs 3

# 5. Inference (모델별)
python 04_inference.py --model_path models/sft_llama_3ep/final
python 04_inference.py --model_path models/rlaif_sft3ep_dpo_llama/final
# ...

# 6. 평가
python 05_evaluate.py --result_pkl "outputs/inference_*.pkl"
```

## 비교 실험 매트릭스

| # | 모델 | SFT | RLAIF | 비고 |
|---|------|-----|-------|------|
| 1 | Llama-3.1-8B-Instruct | ✗ | ✗ | Raw baseline |
| 2 | Llama-3.1-8B-Instruct | ✗ | DPO | RLAIF only |
| 3 | Llama-3.1-8B-Instruct | ✗ | SimPO | RLAIF only |
| 4 | Llama-3.1-8B-Instruct | 1ep | ✗ | SFT only |
| 5 | Llama-3.1-8B-Instruct | 3ep | ✗ | SFT only |
| 6 | Llama-3.1-8B-Instruct | 1ep | DPO | SFT+RLAIF |
| 7 | Llama-3.1-8B-Instruct | 3ep | DPO | SFT+RLAIF (최고 성능) |
| 8 | Qwen3-8B | ✗ | ✗ | Raw baseline |
| 9 | Qwen3-8B | 1ep | ✗ | SFT only |
| 10 | Qwen3-8B | 3ep | ✗ | SFT only |
| 11 | Qwen3-8B | 3ep | DPO | SFT+RLAIF |

## 평가 지표

| 지표 | 도구 | 범위 |
|------|------|------|
| Brevity & Relevance | Prometheus-8x7b-v2.0 | 1~5 |
| Critical Focus | Prometheus-8x7b-v2.0 | 1~5 |
| SUM | — | 2~10 |
| Factual Consistency | Flan-T5-large/xl (SCALE) | 0~1 |
| Text Length | 단어 수 | — |

## Threshold 출처 요약

| 항목 | 기준 | 출처 |
|------|------|------|
| HR | 연령별 서맥/빈맥 | 대한소아청소년과학회 |
| SBP 저혈압 | 70+2×age(yr) | PALS/ATLS |
| SBP 고혈압 | 연령별 95th pct | 대한소아청소년학회 2008 |
| MBP 저혈압 | 1.5×age(yr)+40 | PMID 17273118 |
| DBP 저/고 | 연령별 5th/90th pct | AAP 4th Report |
| QTc | <8세 >450ms, ≥8세 >460ms | PMID 16482041 |
| SpO2 | <95% / <90% | —  |
| T1 | <35.5°C / >38.0°C | — |

## 주요 하이퍼파라미터

```python
# LoRA
r=8, lora_alpha=16, lora_dropout=0.1

# SFT
lr=2e-5, batch=2, grad_accum=4, epochs=1 or 3

# RLAIF
lr=5e-6, batch=2, grad_accum=4, epochs=3

# Generation
max_input_len=2048, max_output_len=768
```

## IRB / DRB

- IRB: E-2601-138-1712 (텍스트+바이탈 멀티모달)
- DRB: DRB-E(I)-2026-02-04
- ※ 연구자 명단 수정 필요 (IRB·DRB 모두)
