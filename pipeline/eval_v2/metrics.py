"""
eval_v2/metrics.py — v2 3축 평가 (전문의 루브릭 정렬)

축 (인계요약지_SY '평가 루브릭' 시트에 정렬):
  COVERAGE     (안전 핵심, III 설명부족) — gold checklist 항목을 얼마나 cover하는가(recall).
                이상소견이 있는데 "특이사항 없음"이면 안전위반 → 하드게이트.
  FAITHFULNESS (I 내용오류)              — 출력 claim이 EMR과 일치하는가(환각/창작 여부).
  BREVITY      (II 과도한 설명)          — 추론성 권고/불필요 내용/정상지표 부연 등 노이즈.

설계 원칙:
  - 합산(sum)으로 brevity가 안전성을 덮지 못하게 한다 → 3축 별도 보고 + 안전 하드게이트.
  - 각 축은 gemma-4-31B 1회 호출(구조화 JSON). 생성실패(빈/반복/잘림)는 LLM 없이 즉시 감점.
  - 누락 항목/환각 claim을 함께 반환 → HTML 리포트(개선안 6)에서 하이라이트.

이 모듈은 (system, user) 프롬프트 빌더 + JSON 파서를 제공한다.
06_evaluate_v2.py가 데이터셋 전체 프롬프트를 모아 엔진을 1회 배치 호출한다.
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))

from config_v2 import (
    V2_WEIGHTS,
    V2_SAFETY_VIOLATION_CAP,
    is_no_issue,
)

_SEV_W = {"high": 3.0, "medium": 2.0, "low": 1.0}


# ══════════════════════════════════════════════════════════════════════════
# COVERAGE
# ══════════════════════════════════════════════════════════════════════════
_COVERAGE_SYSTEM = (
    "You verify whether a PACU/ICU handoff covers each required actionable finding. "
    "Be strict: a finding is 'yes' only if the handoff clearly conveys it; 'partial' if "
    "vaguely/incompletely; 'no' if absent. Paraphrase is fine. Output strict JSON only."
)

_COVERAGE_TMPL = """아래 '필수 인계 항목(gold checklist)'을 모델 인계문이 각각 다루는지 판정하세요.

판정값: "yes"(명확히 전달) / "partial"(모호·불완전) / "no"(없음).

### 모델 인계문
{output}

### 필수 인계 항목
{items}

다음 JSON으로만 출력:
{{"verdicts": [{{"id":"c1","status":"yes|partial|no"}}, ...]}}
"""


def build_coverage_prompt(output, entry):
    items = entry.get("items", [])
    lines = [f'- {it["id"]}: {it["finding"]} (severity={it.get("severity","medium")})'
             for it in items]
    user = _COVERAGE_TMPL.format(output=output, items="\n".join(lines))
    return _COVERAGE_SYSTEM, user


def parse_coverage(pj, entry):
    """LLM verdict → coverage 점수 + 누락항목 + 안전위반 플래그.
    severity 가중 recall: covered/total (yes=1, partial=0.5)."""
    items = entry.get("items", [])
    by_id = {it["id"]: it for it in items}
    verdicts = {}
    if isinstance(pj, dict):
        for v in pj.get("verdicts", []) or []:
            if isinstance(v, dict) and "id" in v:
                verdicts[v["id"]] = str(v.get("status", "no")).lower()

    num = den = 0.0
    missed = []
    for it in items:
        w = _SEV_W.get(it.get("severity", "medium"), 2.0)
        den += w
        st = verdicts.get(it["id"], "no")
        if st == "yes":
            num += w
        elif st == "partial":
            num += 0.5 * w
            missed.append({**it, "status": "partial"})
        else:
            missed.append({**it, "status": "no"})
    coverage = (num / den) if den > 0 else 1.0
    # 고위험 항목을 완전히 놓쳤는지 = 안전 위반
    high_missed = any(m.get("severity") == "high" and m.get("status") == "no" for m in missed)
    return dict(coverage=round(coverage, 4), missed=missed, high_missed=high_missed)


# ══════════════════════════════════════════════════════════════════════════
# FAITHFULNESS  (claim 단위 한국어 entailment — SCALE 대체)
# ══════════════════════════════════════════════════════════════════════════
_FAITH_SYSTEM = (
    "You fact-check a handoff against the source EMR. Split the handoff into atomic claims "
    "and classify each as 'supported' (entailed by EMR), 'contradicted' (conflicts with EMR, "
    "e.g. r/o stated as confirmed), or 'unsupported' (not in EMR). Output strict JSON only."
)

_FAITH_TMPL = """다음 모델 인계문의 각 주장을 원본 EMR과 대조해 분류하세요.
- supported: EMR으로 뒷받침됨
- contradicted: EMR과 모순(예: 의심(r/o)을 확진처럼 단정, 안 한 처치를 했다고 기술)
- unsupported: EMR에 근거 없음(창작/환각)

### 원본 EMR
{premise}

### 모델 인계문
{output}

다음 JSON으로만 출력:
{{"claims": [{{"claim":"...","verdict":"supported|contradicted|unsupported"}}, ...]}}
"""


def build_faithfulness_prompt(premise, output):
    return _FAITH_SYSTEM, _FAITH_TMPL.format(premise=premise, output=output)


def parse_faithfulness(pj):
    claims = []
    if isinstance(pj, dict):
        claims = [c for c in (pj.get("claims", []) or []) if isinstance(c, dict)]
    if not claims:
        # claim을 못 뽑음 → 평가 불가, 중립 처리하되 플래그
        return dict(faithfulness=None, claims=[], hallucinations=[], n_claims=0)
    sup = sum(1 for c in claims if str(c.get("verdict")).lower() == "supported")
    bad = [c for c in claims if str(c.get("verdict")).lower() in ("contradicted", "unsupported")]
    faith = sup / len(claims)
    return dict(faithfulness=round(faith, 4), claims=claims, hallucinations=bad,
                n_claims=len(claims))


# ══════════════════════════════════════════════════════════════════════════
# BREVITY  (전문의 'II 과도한 설명' 루브릭)
# ══════════════════════════════════════════════════════════════════════════
_BREVITY_SYSTEM = (
    "You score a PACU/ICU handoff for conciseness using strict exception-based rules. "
    "Penalize: redundant explanation of diagnosis/procedure names, inferential recommendations "
    "(e.g. 'monitor airway carefully'), administrative noise (drug remainder returns, transfer to "
    "PICU, 'the patient is as follows'), and over-description of normal indices. "
    "A correct 'no issue' note for a stable patient is fully concise. Output strict JSON only."
)

_BREVITY_TMPL = """다음 모델 인계문의 '간결성'을 1~5로 채점하세요(높을수록 간결, 노이즈 없음).
감점 대상(노이즈):
- 진단/수술명 부연 설명, 추론성 권고('~하니 ~해라', 예: 기도 관리 주의 필요)
- 불필요 내용(약 잔량 반납, PICU 이송, "환자 설명은 다음과 같습니다" 류)
- 정상 지표의 지나치게 구체적인 설명

### 모델 인계문
{output}

다음 JSON으로만 출력:
{{"score": <1~5 정수>, "noise": ["감점 사유 짧게", ...]}}
"""


def build_brevity_prompt(output):
    return _BREVITY_SYSTEM, _BREVITY_TMPL.format(output=output)


def parse_brevity(pj):
    if isinstance(pj, dict) and pj.get("score") is not None:
        try:
            s = float(pj["score"])
        except Exception:
            s = 3.0
        s = max(1.0, min(5.0, s))
        noise = [str(x) for x in (pj.get("noise") or [])]
        return dict(brevity=round((s - 1) / 4.0, 4), brevity_raw=s, noise=noise)
    return dict(brevity=None, brevity_raw=None, noise=[])


# ══════════════════════════════════════════════════════════════════════════
# 생성 실패 / normal-case 단축경로 + composite
# ══════════════════════════════════════════════════════════════════════════
def degenerate_scores(status):
    """clean_v2 status가 ok가 아닐 때(빈/반복/잘림) LLM 없이 즉시 0점."""
    return dict(
        coverage=0.0, faithfulness=0.0, brevity=0.0,
        composite=0.0, safety_violation=True,
        missed=[], hallucinations=[], noise=[],
        gen_status=status,
        note=f"생성 실패({status}) — 평가 제외, 0점",
    )


def normal_case_no_llm(output):
    """is_normal_case(정답='특이사항 없음')에서 출력이 no-issue면 즉시 만점 처리.
    (LLM 호출 절약 + 안정 케이스의 올바른 '특이사항 없음'을 보상)"""
    if is_no_issue(output):
        return dict(coverage=1.0, faithfulness=1.0, brevity=1.0, composite=1.0,
                    safety_violation=False, missed=[], hallucinations=[], noise=[],
                    gen_status="ok", note="normal-case + 올바른 '특이사항 없음'")
    return None  # normal인데 뭔가 보고함 → faithfulness/brevity로 정상 채점(false alarm)


def composite(cov, faith, brev, entry, output):
    """3축 → composite. 안전 하드게이트 적용.

    safety_violation = 이상소견 케이스(items>0)인데 모델이 '특이사항 없음'류이거나
                       고위험 항목을 통째로 놓침 → composite 상한 절단."""
    items = entry.get("items", [])
    abnormal = len(items) > 0 and not entry.get("is_normal_case", False)

    cv = cov.get("coverage")
    fa = faith.get("faithfulness")
    br = brev.get("brevity")
    # None(평가불가)은 중립 0.5로 대체하되 별도 기록
    cv_ = 0.0 if cv is None else cv
    fa_ = 0.5 if fa is None else fa
    br_ = 0.5 if br is None else br

    w = V2_WEIGHTS
    comp = w["coverage"] * cv_ + w["faithfulness"] * fa_ + w["brevity"] * br_

    safety = False
    note = ""
    if abnormal and is_no_issue(output):
        safety = True
        note = "안전위반: 이상소견 있는데 '특이사항 없음'으로 응답(놓침)"
    elif abnormal and cov.get("high_missed"):
        safety = True
        note = "안전위반: 고위험 항목 누락"

    if safety:
        comp = min(comp, V2_SAFETY_VIOLATION_CAP)

    return dict(
        coverage=cv, faithfulness=fa, brevity=br,
        composite=round(comp, 4),
        safety_violation=safety,
        missed=cov.get("missed", []),
        hallucinations=faith.get("hallucinations", []),
        noise=brev.get("noise", []),
        n_claims=faith.get("n_claims", 0),
        gen_status="ok",
        note=note,
    )
