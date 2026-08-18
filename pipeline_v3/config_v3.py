"""
config_v3.py — v3 파이프라인 단일 설정 소스

v1 config.py와의 차이:
  - import 부작용 없음 (mkdir/서버경로 강제 없음 → 로컬에서도 import·테스트 가능, B11)
  - 모든 경로가 환경변수로 오버라이드 가능 (폐쇄망 서버 / 로컬 동일 코드)
      HANDOVER_BASE_DIR   기본 /home/coder/workspace/data/handover
      HANDOVER_DATA_DIR   기본 {BASE}/data
      HANDOVER_WORK_DIR   기본 {BASE}/0526/handover
      HANDOVER_MODEL_DIR  기본 /home/coder/workspace/data/local_models
      HANDOVER_RUN_ID     기본 "default" (run_all_v3가 주입)
  - 모델 레지스트리 한 벌 (다운로드 키 = 학습 키 = 리포트 키, B14)
  - repo에 PHI 없음: 수술ID remap 등 실제 식별자는 DATA_DIR 밑 JSON에서 로드
"""

import json
import os
from pathlib import Path

# ── 경로 (env 오버라이드, import 시 mkdir 안 함) ────────────────────────────
BASE_DIR = Path(os.environ.get("HANDOVER_BASE_DIR", "/home/coder/workspace/data/handover"))
DATA_DIR = Path(os.environ.get("HANDOVER_DATA_DIR", str(BASE_DIR / "data")))
WORK_DIR = Path(os.environ.get("HANDOVER_WORK_DIR", str(BASE_DIR / "0526" / "handover")))
MODEL_BASE = Path(os.environ.get("HANDOVER_MODEL_DIR", "/home/coder/workspace/data/local_models"))

RUN_ID = os.environ.get("HANDOVER_RUN_ID", "default")
OUTPUT_BASE = WORK_DIR / "outputs_v3" / RUN_ID

# 산출물 경로 (필요 시점에 mkdir — ensure_dir() 사용)
PAIRS_OUT = OUTPUT_BASE / "pairs"          # 선호쌍/후보 생성 산출물
SFT_OUT = OUTPUT_BASE / "sft"
RLAIF_OUT = OUTPUT_BASE / "rlaif"
INFER_OUT = OUTPUT_BASE / "inference"
EVAL_OUT = OUTPUT_BASE / "evaluation"
REPORT_OUT = OUTPUT_BASE / "report"
PROVENANCE_JSON = OUTPUT_BASE / "provenance_v3.json"


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── 원본 데이터 pkl (v1과 동일 파일 재사용) ─────────────────────────────────
EMR_PKL = DATA_DIR / "raw_data/new_irb/all_newirb_260202.pkl"
VITAL_PKL = DATA_DIR / "preprocessed/vitalsheet_260216.pkl"

GOLD_PKL = DATA_DIR / "preprocessed/gold_sampled_251008.pkl"        # 22×228 — 최종 test (봉인)
GOLD_REF_PKL = DATA_DIR / "preprocessed/gold_sampled_251001.pkl"    # 22×8   — human handoff 포함
SFT_SRC_PKL = DATA_DIR / "preprocessed/jsft_251008.pkl"             # 2736×228 — SFT 데이터 원천
SELFJUDGE_SRC_PKL = DATA_DIR / "preprocessed/selfjudge_251008.pkl"  # 2736×228 — DPO 쌍 생성 원천 (SFT와 분리, T9)
RLHF_PKL = DATA_DIR / "preprocessed/rlhf_251008.pkl"                # 150×228 — few-shot pool + dev set (T1)
VITAL_MAP_PKL = DATA_DIR / "vital_summary_map.pkl"                  # {수술ID: vital_summary}

# v3 생성 산출물 (DATA_DIR가 아니라 run 출력 밑에 둔다 — run 간 오염 방지)
PAIRS_SFT_PKL = PAIRS_OUT / "pairs_sft.pkl"           # SFT 타깃 (chosen만 사용)
PAIRS_DPO_TMPL = "pairs_dpo_{policy_tag}.pkl"         # 정책별 DPO 쌍 (PAIRS_OUT 밑)

# gold checklist (v3): 전문의 gold(KHS c10) 기반, 사람 검수 상태 필드 분리 (E6)
GOLD_CHECKLIST_JSON = DATA_DIR / "preprocessed" / "gold_checklist_v3.json"
GOLD_KHS_XLSX = DATA_DIR / "gold_sampled" / "인계요약지_gold_sampled_251002_KHS.xlsx"
GOLD_SY_XLSX = DATA_DIR / "gold_sampled" / "인계요약지_SY.xlsx"     # 전문의 168행 채점 → calibration 전용

# 수술ID remap (PHI — repo 금지, 데이터 폴더 JSON에서 로드)
KHS_REMAP_JSON = DATA_DIR / "preprocessed" / "khs_gold_remap.json"
# 교수님 재검수본(output_케이스별_*_reviewed.xlsx)에서 확정된 GT — c10 위에 덮어쓴다.
# scripts/import_khs_review.py → scripts/build_gold_override.py 로 생성. PHI.
KHS_GOLD_OVERRIDE_JSON = DATA_DIR / "preprocessed" / "khs_gold_override.json"

# ── 다린(기존 연구) 재추론 경로 (reinfer_darin_on_v3sids.py / report_v3 병기 공용) ──
#   기본값은 handover BASE_DIR 의 형제 폴더 HANDOVER_인계용_다린. env 로 오버라이드 가능.
#   DARIN_EXPERIMENTS: 다린 학습 체크포인트(adapter) 루트 (reinfer --experiments_root)
#   DARIN_INFER_OUT  : v3 sid 재추론 출력 루트 (reinfer --out_root = report_v3 --darin_root)
DARIN_DIR = Path(os.environ.get("HANDOVER_DARIN_DIR",
                                str(BASE_DIR.parent / "HANDOVER_인계용_다린")))
DARIN_EXPERIMENTS = Path(os.environ.get("HANDOVER_DARIN_EXPERIMENTS",
                                        str(DARIN_DIR / "experiments")))
DARIN_INFER_OUT = Path(os.environ.get("HANDOVER_DARIN_INFER_OUT",
                                      str(DARIN_DIR / "data" / "inferenced_v3sids")))


def load_khs_remap() -> dict:
    """{잘못된_sid(str): 실제_sid(str)}. 파일 없으면 빈 dict (경고는 호출부에서)."""
    try:
        return {str(k): str(v)
                for k, v in json.loads(KHS_REMAP_JSON.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


def load_khs_gold_override() -> dict:
    """{sid(str): 확정 GT(str)}. 파일 없으면 빈 dict — 기존 c10 gold 그대로 사용."""
    try:
        raw = json.loads(KHS_GOLD_OVERRIDE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(k): str(v).strip() for k, v in raw.items() if str(v).strip()}


# ── 모델 레지스트리 (단일 소스 — 학습/추론/평가/다운로드가 전부 이걸 사용) ────
# family: judge 순환 방지 규칙에 사용 — 평가 대상과 같은 family의 judge는 배정 금지 (T7)
MODELS = {
    "llama":      dict(dir="Llama-3.1-8B-Instruct",  repo="meta-llama/Llama-3.1-8B-Instruct",
                       family="llama",  thinking=False, gated=True),
    "qwen":       dict(dir="Qwen3-8B",               repo="Qwen/Qwen3-8B",
                       family="qwen",   thinking=True,  gated=False),
    "qwen35":     dict(dir="Qwen3.5-9B",             repo="Qwen/Qwen3.5-9B",
                       family="qwen",   thinking=True,  gated=False),
    "gemma4":     dict(dir="gemma-4-E4B-it",         repo="google/gemma-4-E4B-it",
                       family="gemma",  thinking=False, gated=True),
    "gemma4_31b": dict(dir="gemma-4-31B-it",         repo="google/gemma-4-31B-it",
                       family="gemma",  thinking=False, gated=True),
    "hari":       dict(dir="hari-q3-8b",             repo="snuh/hari-q3-8b",
                       family="qwen",   thinking=True,  gated=False),
    "medgemma27b": dict(dir="medgemma-27b-it",       repo="google/medgemma-27b-it",
                       family="gemma",  thinking=False, gated=True),
    "prometheus": dict(dir="prometheus-8x7b-v2.0",   repo="prometheus-eval/prometheus-8x7b-v2.0",
                       family="mixtral", thinking=False, gated=False),
}

# gemma-4 계열: LoRA 타깃/attn 구현 특수처리 필요
GEMMA4_BASES = {k for k, v in MODELS.items() if v["family"] == "gemma" and k != "medgemma27b"}

# vLLM 로드 불가 모델 → 추론 시 HF 강제 (변형 간 엔진 혼용 방지).
#   근본 원인(공통): 멀티모달/하이브리드 base에 텍스트 LoRA를 merge하면 텍스트 전용
#   체크포인트가 나오는데, vLLM이 그 arch를 '멀티모달'로 로드하려다 실패한다.
#     - qwen35(Qwen3.5): merged config 가 Qwen3_5TextConfig → vLLM(qwen3_5)이 기대하는
#       Qwen3_5Config 와 타입 불일치 → TypeError.
#     - gemma4(Gemma4): vLLM 이 Gemma4ForConditionalGeneration 로 로드하려다 merged 에 없는
#       preprocessor_config.json(비전 프로세서) 을 찾아 실패.
#   두 경우 모두 raw만 vLLM 통과 → raw=vllm / 학습변형=hf 로 섞이면 vs-raw 비교가
#   '학습 효과'가 아니라 '디코딩 엔진 차이'에 오염된다. → 전 변형을 HF로 통일한다.
#   (llama 등 순수 텍스트 base는 merged 도 vLLM 정상 — 목록에서 제외.)
VLLM_INCOMPATIBLE_MODELS = {"qwen35", "gemma4"}


def infer_engine_for(model_key: str) -> str:
    """추론 엔진 선택: vLLM 불가 모델은 'hf' 강제, 그 외 'auto'(vLLM 우선)."""
    return "hf" if model_key in VLLM_INCOMPATIBLE_MODELS else "auto"


def model_path(key: str) -> Path:
    return MODEL_BASE / MODELS[key]["dir"]


def model_family(key: str) -> str:
    return MODELS[key]["family"]


def is_thinking(key: str) -> bool:
    return MODELS[key]["thinking"]


# 학습 대상 후보 (run_all_v3 --models choices)
TRAIN_KEYS = ["llama", "qwen", "qwen35", "gemma4", "gemma4_31b", "hari"]

# ── Judge 배정 (순환 금지, T7) ───────────────────────────────────────────────
# 평가(최종/개발): 2-judge — 대상 모델과 다른 family의 judge만 채점에 반영.
#   양쪽 다 다른 family면 두 judge 평균 + 일치도 보고.
EVAL_JUDGES = ["gemma4_31b", "qwen35"]
# 선호쌍 생성 judge: 평가 judge와도, 학습 대상과도 다른 계열(prometheus=mixtral)
PAIRGEN_JUDGE = "prometheus"


def judges_for(target_model_key: str) -> list:
    """평가 대상 모델에 배정 가능한 judge 목록 (같은 family 제외).
    전부 제외되면 전체 judge를 쓰되 same_family 플래그는 호출부에서 기록."""
    fam = model_family(target_model_key) if target_model_key in MODELS else None
    valid = [j for j in EVAL_JUDGES if fam is None or model_family(j) != fam]
    return valid if valid else list(EVAL_JUDGES)


# ── 토큰 예산 (단일 소스 — 학습/추론/쌍생성 전부 이 값, T4/T5/T6) ────────────
MAX_PROMPT_TOKENS = 2560       # system+user 프롬프트 예산 (EMR-only 좌측절단으로 맞춤)
MAX_COMPLETION_TOKENS = 512    # 인계문은 1~5문장 — 512면 충분
MAX_SEQ_TOKENS = MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS      # 3072 (SFT/DPO/SimPO 공통)
INFER_MAX_MODEL_LEN = 4096     # vLLM/HF 추론 컨텍스트 (프롬프트 2560 + 생성 512 + 여유)
JUDGE_MAX_MODEL_LEN = 8192     # judge 입력 예산 (긴 EMR 케이스 대비, T4)

# ── LoRA (v1과 동일 — 비교 연속성) ──────────────────────────────────────────
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
LORA_TARGET_MODULES_GEMMA4 = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)

# ── 학습 하이퍼파라미터 ─────────────────────────────────────────────────────
SFT_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    dataloader_num_workers=4,
)

RLAIF_CONFIG = dict(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)
DPO_BETA = 0.1
SIMPO_BETA = 2.0
SIMPO_GAMMA_BETA_RATIO = 0.25

# ── 추론 디코딩 (v3 기본 = 생성 위생 ON, 단일 설정) ─────────────────────────
INFER_CONFIG = dict(
    max_new_tokens=MAX_COMPLETION_TOKENS,
    do_sample=False,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,     # HF 전용 (vLLM 미지원 — 결과 레코드에 engine 명시)
    batch_size=4,
)

# ── 선호쌍 생성 (T1/T2/T3/T8) ───────────────────────────────────────────────
PAIRGEN = dict(
    n_fewshot=2,                 # few-shot 예시 수 (held-out pool에서 행별 로테이션)
    fewshot_emr_chars=700,       # 예시 EMR 앞부분 길이(자) — 스타일 앵커 목적이라 짧아도 충분.
                                 # 지시문+few-shot이 MAX_PROMPT_TOKENS를 넘으면 여기부터 줄인다.
    temperature=0.8,             # 후보 다양화 (greedy 1개 + 샘플 K개)
    top_p=0.95,
    n_samples=3,                 # 모델당 temperature 샘플 수 (greedy 포함 후보 = 1+3)
    min_margin=2.0,              # score(chosen) - score(rejected) 최소 마진 (10점 만점 기준)
    max_rows=None,               # 쌍 생성 행 수 상한 (None=전체; 예산 부족 시 1500 권장)
    judge_max_tokens=512,
)

# ── 데이터 split (T1: few-shot 유출 차단 / Phase3: dev-test 분리) ────────────
SPLIT_SEED = 42
N_FEWSHOT_POOL = 40             # RLHF 150 중 few-shot 예시 전용
# 나머지 110 = dev set (모델 선택/하이퍼파라미터 결정은 dev로만; gold 22는 최종 1회)

# ── 평가 v3 ─────────────────────────────────────────────────────────────────
# 3축 가중 composite (calibration 회귀로 근거 부여 — eval_v3/calibrate.py)
V3_WEIGHTS = dict(coverage=0.50, faithfulness=0.30, brevity=0.20)

# judge 실패 허용 상한: 유효 케이스 비율이 이보다 낮으면 평가 전체를 실패 처리 (E3)
MIN_VALID_CASE_RATIO = 0.8

EVAL_GEN = dict(max_new_tokens=1024, temperature=0.0, repetition_penalty=1.05)

# "특이사항 없음" 동의어 — 전체 일치 판정에만 사용 (접두 매칭 금지, E5)
NO_ISSUE_PHRASES = [
    "특이사항 없음", "특이 사항 없음", "특이사항없음",
    "이상 없음", "이상소견 없음", "특이소견 없음",
    "no issues", "no issue", "none", "unremarkable",
]


def is_no_issue_v3(text: str) -> bool:
    """출력 '전체'가 no-issue 문구와 일치할 때만 True (E5: 접두/길이 휴리스틱 폐기).

    '특이사항 없음. intraop VT 발생' → False (실질 내용 있음)
    'None significant except bleeding' → False
    '특이사항 없음.' / '특이사항 없음' → True
    """
    if text is None:
        return False
    compact = "".join(ch for ch in text.strip().lower() if ch.isalnum() or ('가' <= ch <= '힣'))
    if not compact:
        return False
    for p in NO_ISSUE_PHRASES:
        pc = "".join(ch for ch in p.lower() if ch.isalnum() or ('가' <= ch <= '힣'))
        if compact == pc:
            return True
    return False


# ── 통계 (E8) ───────────────────────────────────────────────────────────────
STATS = dict(
    bootstrap_iters=10000,
    permutation_iters=20000,
    seed=42,
    alpha=0.05,
)
