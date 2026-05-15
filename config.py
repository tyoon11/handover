"""
config.py — 전체 파이프라인 설정
실제 서버 환경 기반으로 수정됨 (check_env_result 확인 후)
"""

from pathlib import Path

# ── 기본 경로 ──────────────────────────────────────────────────────────────
BASE_DIR  = Path("/home/coder/workspace/data/HANDOVER_인계용_다린")
DATA_DIR  = BASE_DIR / "data"
WORK_DIR  = Path("/home/coder/workspace/data/handover/0511/handover")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── EMR / Vital 원본 pkl ───────────────────────────────────────────────────
EMR_PKL      = DATA_DIR / "raw_data/new_irb/all_newirb_260202.pkl"
VITAL_PKL    = DATA_DIR / "preprocessed/vitalsheet_260216.pkl"

# ── 이미 split 완료된 pkl (재사용) ────────────────────────────────────────
GOLD_PKL      = DATA_DIR / "preprocessed/gold_sampled_251008.pkl"   # 22×228, EMR 전체
GOLD_REF_PKL  = DATA_DIR / "preprocessed/gold_sampled_251001.pkl"   # 22×8, Human handoff 포함
SFT_PKL       = DATA_DIR / "preprocessed/jsft_251008.pkl"           # 2736×228
SELFJUDGE_PKL = DATA_DIR / "preprocessed/selfjudge_251008.pkl"      # 2736×228
RLHF_PKL      = DATA_DIR / "preprocessed/rlhf_251008.pkl"           # 150×228
SYNTH_PKL     = DATA_DIR / "synthetic/jsft_score_251213.pkl"        # 2736×249, chosen/rejected

# ── Vital summarizer 출력 경로 ────────────────────────────────────────────
VITAL_MAP_PKL = WORK_DIR / "vital_summary_map.pkl"   # {수술ID: vital_summary_str}

# ── EMR MultiIndex 컬럼 접근 ──────────────────────────────────────────────
# df는 3레벨 MultiIndex. 주요 컬럼 튜플 키:
EMR_SID_COL        = "수술 ID"      # level0, int64, 공백 있음 (Vital의 수술ID와 다름!)
EMR_HANDOFF_COL    = ('마취기록', '기록', '')                    # Human-written handoff (gold_251001)
EMR_PREOP_SUM_COL  = ('마취전평가', '마취 전 상태 평가',
                       '마취 전 상태평가 -> 마취전 환자상태 요약')
EMR_PREMED_COL     = ('마취전평가', '마취 전 상태 평가',
                       '마취 전 상태평가 -> 수술전 준비사항과 Premedication -> Premedication')

# 수술 메타 컬럼 (level0='수술')
EMR_OP_COLS = {
    "수술명":       ('수술', '수술명',       ''),
    "수술진단명":    ('수술', '수술진단명',    ''),
    "ICD9CM명":    ('수술', 'ICD9CM명',     ''),
    "수술진료과":    ('수술', '수술진료과',    ''),
    "수술후퇴실장소": ('수술', '수술후퇴실장소', ''),
    "수술당시나이":  ('수술', '수술당시나이',  ''),
    "마취시작":     ('수술', '마취시작일시(마취기록)', ''),
    "마취종료":     ('수술', '마취종료일시(마취기록)', ''),
    "마취종류":     ('수술', '마취종류',      ''),
}

# TOTALS 컬럼 (level0='마취기록', level1='마취기록TOTALS')
EMR_TOTALS_PREFIX = ('마취기록', '마취기록TOTALS')

# ── Vital DataFrame 컬럼명 ─────────────────────────────────────────────────
VIT_SID_COL  = "수술ID"              # int64, 공백 없음 ← EMR의 "수술 ID"와 다름
VIT_TIME_COL = "기록일시"
VIT_ITEM_COL = "진료서식구성원소명"    # "HR <- HR <- 마취기록" → split(" <- ")[0]
VIT_VAL_COL  = "진료서식구성원소내용"

# ── 선정된 Vital 항목 13개 ────────────────────────────────────────────────
VITAL_ITEMS = [
    "HR",
    "MBP", "DBP", "SBP",
    "IMBP1", "ISBP1", "IDBP1",
    "SpO2",
    "Ppeak",
    "T1",
    "QTc",
    "EBL",
    "UO",
]

BP_ITEMS  = ["MBP", "DBP", "SBP"]
IBP_ITEMS = ["IMBP1", "ISBP1", "IDBP1"]

# ── 모델 경로 ─────────────────────────────────────────────────────────────
MODEL_BASE = Path("/home/coder/workspace/data/local_models")

# 학습 베이스 모델 — 02_sft_train.py --base 인자 기준
SFT_MODELS = {
    "llama":  MODEL_BASE / "Llama-3.1-8B-Instruct",
    "qwen":   MODEL_BASE / "Qwen3-8B",
    "gemma4": MODEL_BASE / "gemma-4-E4B-it",
    "qwen35": MODEL_BASE / "Qwen3.5-9B",
    "hari":   MODEL_BASE / "hari-q3-8b",
}

# 평가 모델 경로
EVAL_MODELS = {
    "prometheus": MODEL_BASE / "prometheus-7b-v2.0",
    "flan-large": MODEL_BASE / "flan-t5-large",
    "flan-xl":    MODEL_BASE / "flan-t5-xl",
}

# ── 실험 run ID (timestamp) — run_all.py가 HANDOVER_RUN_ID 환경변수로 주입 ──
import os as _os
_RUN_ID    = _os.environ.get("HANDOVER_RUN_ID", "default")
OUTPUT_BASE = WORK_DIR / "outputs" / _RUN_ID
SFT_OUT     = OUTPUT_BASE / "sft"
RLAIF_OUT   = OUTPUT_BASE / "rlaif"
INFER_OUT   = OUTPUT_BASE / "inference"
EVAL_OUT    = OUTPUT_BASE / "evaluation"
SAMPLE_OUT  = OUTPUT_BASE / "samples"

# ── LoRA 하이퍼파라미터 ───────────────────────────────────────────────────
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# gemma4: ClippableLinear wrapper 내부 실제 Linear 레이어를 직접 타겟
LORA_TARGET_MODULES_GEMMA4 = ["linear"]

# ── SFT 학습 설정 ─────────────────────────────────────────────────────────
SFT_CONFIG = dict(
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    learning_rate               = 2e-4,
    warmup_ratio                = 0.05,
    lr_scheduler_type           = "cosine",
    bf16                        = True,
    logging_steps               = 10,
    save_strategy               = "epoch",
    dataloader_num_workers      = 4,
)

# ── RLAIF (DPO/SimPO) 설정 ────────────────────────────────────────────────
RLAIF_CONFIG = dict(
    loss_type                   = "sigmoid",   # "dpo" or "simpo"
    num_train_epochs            = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,
    learning_rate               = 5e-5,
    beta                        = 0.1,
    max_length                  = 2048,
    bf16                        = True,
    logging_steps               = 10,
    save_strategy               = "epoch",
)

# ── 추론 설정 ─────────────────────────────────────────────────────────────
# do_sample=False일 때 temperature/top_p 제거 (경고 방지)
INFER_CONFIG = dict(
    max_new_tokens = 512,
    do_sample      = False,
    batch_size     = 4,
)

# ── 평가 설정 ─────────────────────────────────────────────────────────────
EVAL_JUDGE_MODEL = "/home/coder/workspace/data/local_models/prometheus-7b-v2.0"

# ── 프롬프트 ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an anesthesiologist assistant generating structured Korean handoff summaries for pediatric surgery patients.
Generate a concise, clinically accurate handoff summary based on the provided anesthesia EMR data.
Write in Korean. Focus on key clinical information relevant to the receiving care team."""

def build_user_prompt(emr_text: str, vital_summary: str = "") -> str:
    vital_section = f"\n\n[수술 중 바이탈 요약]\n{vital_summary}" if vital_summary else ""
    return f"[마취 EMR 데이터]{vital_section}\n\n{emr_text}\n\n위 데이터를 바탕으로 소아수술 인계요약지를 작성하세요."
