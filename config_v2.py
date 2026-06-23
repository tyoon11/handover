"""
config_v2.py — 파이프라인 v2.0 설정 (v1 config.py 위에 add-on)

개선안 1~6 반영:
  1) 평가 목적함수 재설계 — gold checklist 기반 Coverage(recall) + 3축 분리 + 안전 하드게이트
  2) Judge 개선 — gemma-4-31B 로컬 LLM이 "추출→검증" 2단계 수행
  3) SCALE 교체 — Flan-T5(영어) 대신 동일 LLM의 claim 단위 한국어 entailment(=Faithfulness)
  4) 생성 위생 — repetition_penalty / no_repeat_ngram / 정제 재작성
  5) 실험 위생 — self-consistency, 변형 간 출력 다양성/중복 점검, 체크포인트 provenance
  6) 임상 검수 루프 — checklist 대비 누락/환각 컬럼이 들어간 HTML 리포트

이 파일은 v1 config.py를 import해서 경로/유틸을 재사용하고, v2 전용 상수만 추가한다.
v1 파일(config.py / 05_evaluate.py 등)은 건드리지 않는다.
"""

from pathlib import Path

# v1 설정을 그대로 가져온다 (경로/프롬프트/EMR 유틸 재사용)
from config import (
    DATA_DIR,
    MODEL_BASE,
    OUTPUT_BASE,
    SFT_MODELS,
    GOLD_PKL,
    GOLD_REF_PKL,
    VITAL_MAP_PKL,
    SYSTEM_PROMPT,
    build_emr_text,
    build_user_prompt,
)

# ── v2 평가 엔진 (추출/Coverage/Faithfulness/Brevity 공용 LLM) ──────────────
# 사용자 선택: 기존 gemma-4-31B 재사용 (추가 다운로드 불필요).
# 31B bf16 ≈ 62GB → A6000(48GB) 2장이면 vLLM tensor-parallel로 충분.
# eval GPU 풀(최대 6장) 안에서 보이는 GPU 전체를 tensor-parallel로 사용한다.
EVAL_V2_LLM = SFT_MODELS["gemma4_31b"]  # = MODEL_BASE / "gemma-4-31B-it"

# 엔진 백엔드: "auto"(vLLM 우선, 실패 시 HF) / "vllm" / "hf"
EVAL_V2_BACKEND = "auto"

# 판정은 결정적이어야 하므로 temperature=0 (greedy). 구조화 JSON 출력 강제.
EVAL_V2_GEN = dict(
    max_new_tokens=1024,   # claim 추출 + 항목별 verdict JSON 여유
    temperature=0.0,
    repetition_penalty=1.05,
)

# ── Gold actionable checklist ──────────────────────────────────────────────
# Coverage(recall) 채점의 정답 기준. 케이스별 "PACU/ICU에서 반드시 알아야 할
# 조치 가능 이상소견" 목록 + is_normal_case 플래그.
#
# 소스: 전문의 인계문/모델출력이 들어있는 휴먼평가 엑셀에서 1차 추출 →
#       gemma-4-31B로 구조화 → 전문의가 직접 수정 가능한 JSON으로 저장.
HUMAN_EVAL_XLSX = DATA_DIR / "human_evaluation_modeltype포함_251223.xlsx"
GOLD_CHECKLIST_JSON = DATA_DIR / "preprocessed" / "gold_checklist_v2.json"

# ── 전문의 gold + 평가 엑셀 (인계요약지_SY.xlsx) — v2의 1차 gold 소스 ────────
# 시트 '데이터': 22 케이스 × (llama/qwen × prompt1~4) = 168 채점행.
#   - col6  = 전문의 작성 gold 인계문 (Gold data, Human generated)  ← 케이스별 정답
#   - col8/9 = model 종류 / 생성 인계문
#   - col10~16 = 전문의 평가점수(간략성/중요정보/내용오류/과도설명/설명부족/가독성)
# 시트 '평가 루브릭': 전문의가 직접 쓴 채점 기준 → v2 평가축을 여기에 정렬한다.
GOLD_SY_XLSX = DATA_DIR / "gold_sampled" / "인계요약지_SY.xlsx"
GOLD_SY_XLSX_FALLBACK = Path(__file__).resolve().parent / "인계요약지_SY.xlsx"

# ── KHS 엑셀 — '진짜 gold'(교수님 피드백) 1순위 소스 ────────────────────────
# '데이터' 시트, 다중헤더 3행(데이터는 row3~), 케이스당 1행:
#   - c9  인계요약지_sample_from_LLM   = gemma-3-27b-it 원안 (정답 아님, 피드백 '대상')
#   - c10 인계요약지_sample_from_LLM.1 = 교수님이 c9를 보고 준 피드백/수정본 = 진짜 gold
#   - c11 인계요약지                   = (거의 공란 — 사용 안 함)
# checklist 추출 시 c10(gold) + c9(원안)을 함께 넘긴다(피드백 맥락 보존).
# gold 우선순위: KHS c10 피드백 > SY col6 > EMR-only 부트스트랩.
GOLD_KHS_XLSX = DATA_DIR / "gold_sampled" / "인계요약지_gold_sampled_251002_KHS.xlsx"
KHS_SHEET = "데이터"
KHS_HEADER_ROWS = 3
KHS_COLS = dict(idx=0, pid=1, sid=2, dept_list=3, dept=4, recovery=5,
                anrec=6, preop=7, premed=8, llm=9, feedback=10, newgold=11)

# KHS c10 입력오류 보정: {잘못 들어간_sid: 실제_sid}.
# Crouzon(Fronto-orbital, sid 100515294)의 c10이 옆 행(Craniotomy/DNET, sid 100556518)에
# 잘못 입력됨 → DNET 행의 gold를 진짜 Crouzon으로 이동. (DNET은 c10 없음 → no_gold)
KHS_GOLD_REMAP = {"100556518": "100515294"}

# SY 엑셀 컬럼 위치 (header 3행, 데이터는 row3~). idx·gold는 병합셀이므로 ffill 필요.
SY_COLS = dict(
    idx=0, dept=1, recovery=2, anrec=3, preop=4, premed=5, gold=6,
    prompt=7, model=8, gen=9,
    brevity=10, critical=11, free=12, err=13, over=14, under=15, read=16, comment=17,
)
SY_HEADER_ROWS = 3

# 엑셀 컬럼 자동감지 힌트 (실제 컬럼명이 다르면 여기만 고치면 됨).
# build_gold_checklist.py --inspect 로 실제 컬럼을 먼저 확인할 것.
XLSX_COL_HINTS = dict(
    sid=["수술ID", "수술 ID", "op_id", "surgery_id"],          # 케이스 매칭 키
    idx=["idx", "case_idx", "index", "번호"],                   # gold_df 행 인덱스(있으면)
    human=["전문의", "human", "human_handoff", "인계요약지", "gold", "정답"],  # 전문의 인계문
    opname=["수술명", "op_name", "수술"],
)

# ── 3축 점수 가중치 (composite) ─────────────────────────────────────────────
# coverage(안전 핵심) > faithfulness(환각) > brevity(노이즈).
# 단, 합산으로 brevity가 안전성을 덮지 못하도록 '하드게이트'를 둔다(metrics.py).
V2_WEIGHTS = dict(coverage=0.50, faithfulness=0.30, brevity=0.20)

# 안전 하드게이트: 이상소견이 존재하는 케이스(|checklist|>0)에서 모델이
# "특이사항 없음"류로 응답하면(=놓침) composite를 이 상한으로 강제 절단.
V2_SAFETY_VIOLATION_CAP = 0.15

# "특이사항 없음" 동의어 (안전 위반 판정 + normal-case 정답 판정에 사용)
NO_ISSUE_PHRASES = [
    "특이사항 없음",
    "특이 사항 없음",
    "특이사항없음",
    "이상 없음",
    "이상소견 없음",
    "no issues",
    "none",
    "unremarkable",
]

# ── 생성 위생 (개선안 4) — 추론 v2 디코딩 기본값 ────────────────────────────
# v1 INFER_CONFIG는 그대로 두고, --decode v2 플래그로 opt-in.
INFER_CONFIG_V2 = dict(
    max_new_tokens=512,          # 1~5문장이면 충분 → 폭주/잘림 방지
    do_sample=False,
    repetition_penalty=1.15,     # 반복 루프 억제
    no_repeat_ngram_size=4,      # "same1 same1..." 류 차단
    batch_size=4,
)

# self-consistency (개선안 5): 케이스당 K회 샘플링 후 일치도 측정.
INFER_SELF_CONSISTENCY = dict(
    enabled_by_flag=True,        # --self_consistency 로만 켜짐
    k=5,
    temperature=0.7,
    top_p=0.9,
)

# ── v2 출력 경로 ────────────────────────────────────────────────────────────
EVAL_V2_OUT = OUTPUT_BASE / "evaluation_v2"
REPORT_V2_OUT = OUTPUT_BASE / "report_v2"

# ── GPU 배치 정책 (문서/오케스트레이터 기본값) ──────────────────────────────
#   학습(오래 걸림): 최대 4장
#   eval/judge(추론성, 빨리 끝남): 최대 6장
GPU_POLICY = dict(
    train_gpus="0,1,2,3",
    train_gpus_per_job=1,    # 8B는 1장, 31B 학습은 2~4장 권장
    eval_gpus="0,1,2,3",     # gemma4-31B judge는 4장이면 충분 (tensor-parallel)
)


# ── 전문의 루브릭 → v2 평가축 매핑 (인계요약지_SY '평가 루브릭' 시트 기반) ──
# 전문의가 직접 정의한 채점 기준에 v2 LLM-judge를 정렬한다.
#   I.  내용의 오류      → FAITHFULNESS (환각/창작, r/o→확진 워딩오류, 영한혼용)
#   II. 과도한 설명       → BREVITY/NOISE (진단·수술명 부연, 추론성 권고 "~하니 ~해라",
#                          불필요 내용=약 잔량/이송, 정상지표 과설명)
#   III.설명 부족         → COVERAGE (환자상태 생략, 과도단순화="특이사항없음"인데 이상 있음,
#                          필수 진단명 생략, 핵심 지표/변수 누락)
#   IV. 구성 가독성       → (프롬프트로 해결, 점수화 생략 — 보조 코멘트만)
RUBRIC_AXES = {
    "faithfulness": [
        "없는/거짓 정보 창작(hallucination)",
        "의심(r/o) 환자를 확진처럼 단정하는 워딩 오류",
        "영/한 혼용, 뜬금없는 번역/한자, 영어로만 구성",
    ],
    "brevity": [
        "진단/수술명에 대한 불필요한 부연 설명",
        "추론성 권고('환자가 ~하니 ~해라', 예: 기도 관리 주의 필요)",
        "불필요 내용 포함(약 잔량 반납, PICU 이송, '환자 설명은 다음과 같습니다' 류)",
        "정상 지표에 대한 지나치게 구체적인 설명",
    ],
    "coverage": [
        "현재 환자 상태/우선순위 설명 생략",
        "과도한 단순화('특이사항 없음'인데 실제 이상 있음)",
        "(필수) 독특/중요 진단명 생략",
        "gold에 있는 핵심 지표/변수 누락",
    ],
}


def gold_sy_path():
    """SY gold 엑셀 경로 해결 (data 경로 우선, 없으면 repo 루트 fallback)."""
    if GOLD_SY_XLSX.exists():
        return GOLD_SY_XLSX
    return GOLD_SY_XLSX_FALLBACK


def is_no_issue(text: str) -> bool:
    """출력이 '특이사항 없음'류(=이상소견 없다는 주장)인지 판정."""
    if not text:
        return True
    t = text.strip().lower()
    # 매우 짧고(한 줄) no-issue 문구만 있는 경우
    compact = t.replace(" ", "").replace(".", "").replace("·", "")
    for p in NO_ISSUE_PHRASES:
        pc = p.lower().replace(" ", "")
        if compact == pc or t == p.lower():
            return True
    # 첫 줄이 no-issue 문구이고 그 외 실질 내용이 거의 없을 때
    first = t.splitlines()[0] if t.splitlines() else t
    return any(first.strip().startswith(p.lower()) for p in NO_ISSUE_PHRASES) and len(t) < 40
