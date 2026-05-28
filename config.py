"""
config.py — 전체 파이프라인 설정
실제 서버 환경 기반으로 수정됨 (check_env_result 확인 후)
"""

from pathlib import Path

# ── 기본 경로 ──────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/coder/workspace/data/handover")
DATA_DIR = BASE_DIR / "data"
WORK_DIR = Path("/home/coder/workspace/data/handover/0526/handover")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── EMR / Vital 원본 pkl ───────────────────────────────────────────────────
EMR_PKL = DATA_DIR / "raw_data/new_irb/all_newirb_260202.pkl"
VITAL_PKL = DATA_DIR / "preprocessed/vitalsheet_260216.pkl"

# ── 이미 split 완료된 pkl (재사용) ────────────────────────────────────────
GOLD_PKL = DATA_DIR / "preprocessed/gold_sampled_251008.pkl"  # 22×228, EMR 전체
GOLD_REF_PKL = (
    DATA_DIR / "preprocessed/gold_sampled_251001.pkl"
)  # 22×8, Human handoff 포함
SFT_PKL = DATA_DIR / "preprocessed/jsft_251008.pkl"  # 2736×228
SELFJUDGE_PKL = DATA_DIR / "preprocessed/selfjudge_251008.pkl"  # 2736×228
RLHF_PKL = DATA_DIR / "preprocessed/rlhf_251008.pkl"  # 150×228
SYNTH_PKL = DATA_DIR / "synthetic/jsft_score_251213.pkl"  # 2736×249, chosen/rejected

# ── Vital summarizer 출력 경로 ────────────────────────────────────────────
VITAL_MAP_PKL = DATA_DIR / "vital_summary_map.pkl"  # {수술ID: vital_summary_str}

# ── EMR MultiIndex 컬럼 접근 ──────────────────────────────────────────────
# df는 3레벨 MultiIndex. 주요 컬럼 튜플 키:
EMR_SID_COL = "수술 ID"  # level0, int64, 공백 있음 (Vital의 수술ID와 다름!)
EMR_HANDOFF_COL = ("마취기록", "기록", "")  # Human-written handoff (gold_251001)
EMR_PREOP_SUM_COL = (
    "마취전평가",
    "마취 전 상태 평가",
    "마취 전 상태평가 -> 마취전 환자상태 요약",
)
EMR_PREMED_COL = (
    "마취전평가",
    "마취 전 상태 평가",
    "마취 전 상태평가 -> 수술전 준비사항과 Premedication -> Premedication",
)

# 수술 메타 컬럼 (level0='수술')
EMR_OP_COLS = {
    "수술명": ("수술", "수술명", ""),
    "수술진단명": ("수술", "수술진단명", ""),
    "ICD9CM명": ("수술", "ICD9CM명", ""),
    "수술진료과": ("수술", "수술진료과", ""),
    "수술후퇴실장소": ("수술", "수술후퇴실장소", ""),
    "수술당시나이": ("수술", "수술당시나이", ""),
    "마취시작": ("수술", "마취시작일시(마취기록)", ""),
    "마취종료": ("수술", "마취종료일시(마취기록)", ""),
    "마취종류": ("수술", "마취종류", ""),
}

# TOTALS 컬럼 (level0='마취기록', level1='마취기록TOTALS')
EMR_TOTALS_PREFIX = ("마취기록", "마취기록TOTALS")

# ── Vital DataFrame 컬럼명 ─────────────────────────────────────────────────
VIT_SID_COL = "수술ID"  # int64, 공백 없음 ← EMR의 "수술 ID"와 다름
VIT_TIME_COL = "기록일시"
VIT_ITEM_COL = "진료서식구성원소명"  # "HR <- HR <- 마취기록" → split(" <- ")[0]
VIT_VAL_COL = "진료서식구성원소내용"

# ── 선정된 Vital 항목 13개 ────────────────────────────────────────────────
VITAL_ITEMS = [
    "HR",
    "MBP",
    "DBP",
    "SBP",
    "IMBP1",
    "ISBP1",
    "IDBP1",
    "SpO2",
    "Ppeak",
    "T1",
    "QTc",
    "EBL",
    "UO",
]

BP_ITEMS = ["MBP", "DBP", "SBP"]
IBP_ITEMS = ["IMBP1", "ISBP1", "IDBP1"]

# ── 모델 경로 ─────────────────────────────────────────────────────────────
MODEL_BASE = Path("/home/coder/workspace/data/local_models")

# 학습 베이스 모델 — 02_sft_train.py --base 인자 기준
SFT_MODELS = {
    "llama": MODEL_BASE / "Llama-3.1-8B-Instruct",
    "qwen": MODEL_BASE / "Qwen3-8B",
    "gemma4": MODEL_BASE / "gemma-4-E4B-it",
    "qwen35": MODEL_BASE / "Qwen3.5-9B",
    "hari": MODEL_BASE / "hari-q3-8b",
}

# 평가 모델 경로
EVAL_MODELS = {
    "prometheus": MODEL_BASE / "prometheus-8x7b-v2.0",
    "flan-large": MODEL_BASE / "flan-t5-large",
    "flan-xl": MODEL_BASE / "flan-t5-xl",
}

# ── 실험 run ID (timestamp) — run_all.py가 HANDOVER_RUN_ID 환경변수로 주입 ──
import os as _os

_RUN_ID = _os.environ.get("HANDOVER_RUN_ID", "default")
OUTPUT_BASE = WORK_DIR / "outputs" / _RUN_ID
SFT_OUT = OUTPUT_BASE / "sft"
RLAIF_OUT = OUTPUT_BASE / "rlaif"
INFER_OUT = OUTPUT_BASE / "inference"
EVAL_OUT = OUTPUT_BASE / "evaluation"
SAMPLE_OUT = OUTPUT_BASE / "samples"

# ── LoRA 하이퍼파라미터 ───────────────────────────────────────────────────
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LORA_TARGET_MODULES_GEMMA4 = r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"

# ── SFT 학습 설정 ─────────────────────────────────────────────────────────
SFT_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # 원본 llama_jsft_3epoch.yaml 기준
    learning_rate=2e-5,             # 원본 llama_jsft_3epoch.yaml 기준
    warmup_steps=50,                # 원본 warmup_ratio=0.03 ≈ 3% (2GPU 기준 ~1539 steps)
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    dataloader_num_workers=4,
)

# ── RLAIF (DPO/SimPO) 설정 ────────────────────────────────────────────────
RLAIF_CONFIG = dict(
    loss_type="sigmoid",
    num_train_epochs=3,
    per_device_train_batch_size=1,    # OOM 방지용 (원본 2 → 1)
    gradient_accumulation_steps=8,    # per_device 절반 보상 (원본 4 → 8, effective batch 동일)
    learning_rate=5e-6,
    beta=0.1,
    max_length=2048,
    lr_scheduler_type="constant_with_warmup",  # 원본 llama_dpo.yaml 기준
    warmup_steps=50,                            # 원본 warmup_ratio=0.1 ≈ 10% (2GPU 기준 ~513 steps)
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# ── 추론 설정 ─────────────────────────────────────────────────────────────
# do_sample=False일 때 temperature/top_p 제거 (경고 방지)
INFER_CONFIG = dict(
    max_new_tokens=768,
    do_sample=False,
    batch_size=4,
)

# thinking 모델(Qwen3 등) 전용 설정:
#   - thinking=True면 <think>...</think> 블록이 수백~수천 토큰을 소비하므로
#     max_new_tokens를 충분히 크게 잡아야 실제 답변이 잘리지 않는다.
#   - 원본 학습은 모두 enable_thinking=False로 진행됐으므로,
#     기본 inference도 enable_thinking=False (INFER_THINKING=False)를 권장.
#   - thinking을 켜고 싶을 때만 --thinking 플래그 사용.
INFER_CONFIG_THINKING = dict(
    max_new_tokens=8192,  # think 블록 ~2000 + 실제 답변 ~768 여유
    do_sample=False,
    batch_size=1,  # thinking 모드는 메모리 소모가 크므로 batch=1
)

# 기본값: 학습과 동일하게 thinking 비활성화 (원본 노트북과 동일)
INFER_ENABLE_THINKING = False

# ── 평가 설정 ─────────────────────────────────────────────────────────────
EVAL_JUDGE_MODEL = "/home/coder/workspace/data/local_models/prometheus-8x7b-v2.0"

# ── EMR 유틸 함수 ─────────────────────────────────────────────────────────
import pandas as _pd


def _safe_get(row, col):
    """MultiIndex Series에서 컬럼 값을 안전하게 추출 (dict/NaN 처리 포함)."""
    try:
        v = row[col]
    except KeyError:
        return ""
    if v is None or (isinstance(v, float) and _pd.isna(v)):
        return ""
    if isinstance(v, dict):
        vals = []
        for vlist in v.values():
            if isinstance(vlist, list):
                vals.extend([str(x) for x in vlist if x is not None])
            else:
                vals.append(str(vlist))
        return " ".join(vals)
    return str(v)


def build_emr_text(row) -> str:
    """
    원본 emr_input_format() 재현: Pre-anesthetic Summary + Premedication + TOTALS + Record.
    row : df.iterrows() 또는 df.iloc[idx] 에서 나온 MultiIndex Series.
    """
    preop = _safe_get(row, EMR_PREOP_SUM_COL)
    premed = _safe_get(row, EMR_PREMED_COL)
    anrec = _safe_get(row, ("마취기록", "기록", ""))

    # TOTALS — ("마취기록", "마취기록TOTALS", item_name) 컬럼 전체 수집
    totals_lines = []
    for k, v in row.items():
        if (
            isinstance(k, tuple)
            and len(k) >= 2
            and k[0] == "마취기록"
            and k[1] == "마취기록TOTALS"
            and v is not None
            and not (isinstance(v, float) and _pd.isna(v))
        ):
            label = k[2] if len(k) > 2 else str(k)
            totals_lines.append(f"{label}: {v}")
    totals_str = "\n".join(totals_lines)

    parts = [
        f"- Pre-anesthetic Patient Conditions Summary\n{preop}",
        f"- Preoperative Preparations and Premedication\n{premed}",
    ]
    if totals_str:
        parts.append(f"- Anesthetic TOTALS\n{totals_str}")
    parts.append(f"- Anesthetic Record\n{anrec}")

    return "\n\n".join(parts)


# ── 프롬프트 ──────────────────────────────────────────────────────────────
# 원본 train_jsft.ipynb / inference_llms.ipynb 와 동일한 system prompt
SYSTEM_PROMPT = """You are an anesthesiologist giving an ultra-brief OR to PACU/ICU handoff AFTER surgery has fully ended.
Do NOT ask for or suggest any intraoperative checks; only summarize key post-op relevant findings from the EMR.

RULES:
- If no clinically meaningful abnormal findings, output EXACTLY and ONLY "특이사항 없음".
- If any exist, output 1-5 VERY short sentences in formal Korean.
- NEVER include normal/stable findings, routine vitals, surgery steps, administrative drug info, or transfer phrases.
- NO lists, NO explanations, NO repetition.

Focus only on abnormal findings relevant after surgery: airway/respiratory status, hemodynamics/bleeding/transfusion,
major or congenital disease, intra-op events already recorded, drug effects, essential lines/devices,
and cooperation/agitation risk. Pediatric airway, fluids, and drug sensitivity are especially important."""


def build_user_prompt(emr_text: str, vital_summary: str = "") -> str:
    """
    원본 user_prompt 포맷 유지 + 바이탈 요약 섹션 추가 (신규 multimodal 파이프라인).
    vital_summary가 있을 때만 ### INTRAOPERATIVE VITAL SUMMARY 섹션 삽입.
    """
    vital_section = (
        f"\n\n### INTRAOPERATIVE VITAL SUMMARY\n{vital_summary}"
        if vital_summary
        else ""
    )
    return f"""Using the EMR below, generate an ultra-brief PACU/ICU handoff.

- If there are NO clinically meaningful issues, output exactly and only "특이사항 없음".
- If there ARE issues, output 1-5 very short sentences in formal Korean.
- Do NOT include normal findings, routine or administrative details, or any request to re-check intraoperative events.

Focus only on post-op relevant abnormalities: airway/respiratory status, hemodynamics/bleeding/transfusion,
major or congenital disease, intra-op events already recorded, drug effects, lines/devices, and cooperation/agitation risk.

### EMR
{emr_text}{vital_section}

### OUTPUT
"""
