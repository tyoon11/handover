"""
eval_v3/checklist.py — gold actionable checklist (Coverage 채점 기준)

v2 대비 수정:
  - `reviewed`는 사람(전문의)만 설정한다. `--accept`는 `accepted_without_review=true`
    별도 플래그로만 기록 — 리포트가 '전문의 검수'로 오표기하지 않는다 (E6).
  - 수술ID remap은 repo 코드가 아니라 DATA_DIR JSON에서 로드 (PHI, P0-3).
  - 추출 실패는 items=[]+coverage 만점이 아니라 source="gold_llm_failed"로 남고,
    평가에서 해당 케이스가 '제외'로 집계된다 (E3 — metrics.py에서 처리).

checklist JSON 구조 (case_key = str(sid)):
{
  "<sid>": {
    "idx": 0, "sid": <int>, "opname": "<수술명>",
    "gold_text": "<교수님 gold 인계문(c10)>",
    "is_normal_case": false,
    "items": [{"id":"c1","finding":"...","category":"...","severity":"...","source":"..."}],
    "source": "gold_llm" | "gold_normal" | "no_gold" | "gold_llm_failed",
    "reviewed": false,                  # 사람 검수 후에만 true (수기)
    "accepted_without_review": false    # --accept 로 잠정 채택 시 true
  }, ...
}
"""

import json
from pathlib import Path

from ..config_v3 import is_no_issue_v3, load_khs_remap
from ..prompt_utils import build_emr_text, get_sid, safe_get


# ── JSON I/O ────────────────────────────────────────────────────────────────
def load_checklist(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checklist(checklist: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)      # atomic


def review_status(checklist: dict) -> dict:
    n_rev = sum(1 for v in checklist.values() if v.get("reviewed"))
    n_acc = sum(1 for v in checklist.values()
                if v.get("accepted_without_review") and not v.get("reviewed"))
    n_prov = len(checklist) - n_rev - n_acc
    return dict(n=len(checklist), reviewed=n_rev,
                accepted_without_review=n_acc, provisional=n_prov)


# ── KHS 엑셀 로더 (교수님 c10 = gold) ───────────────────────────────────────
KHS_SHEET = "데이터"
KHS_HEADER_ROWS = 3
KHS_COLS = dict(idx=0, pid=1, sid=2, dept_list=3, dept=4, recovery=5,
                anrec=6, preop=7, premed=8, llm=9, feedback=10, newgold=11)


def _cell(v) -> str:
    if v is None or isinstance(v, float):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "-") else s


def load_khs_gold(xlsx_path, gold_df):
    """KHS 엑셀 → ({idx: gold(c10)}, {idx: draft(c9)}). remap은 DATA_DIR JSON 적용."""
    import pandas as pd
    remap = load_khs_remap()
    if not remap:
        print("[checklist] ⚠ khs_gold_remap.json 없음/빈 값 — c10 입력오류 보정 미적용. "
              "행 오귀속 케이스가 있으면 gold가 틀린 케이스에 붙는다.")
    raw = pd.read_excel(xlsx_path, sheet_name=KHS_SHEET, header=None)
    data = raw.iloc[KHS_HEADER_ROWS:].reset_index(drop=True)

    sid_to_idx = {}
    for i in range(len(gold_df)):
        s = get_sid(gold_df.iloc[i])
        if s != -1:
            sid_to_idx[s] = i

    gold_by_idx, draft_by_idx = {}, {}
    n_remap = 0
    for r in range(len(data)):
        try:
            row_sid = int(float(data.iloc[r, KHS_COLS["sid"]]))
        except Exception:
            row_sid = None
        gold_sid = int(remap[str(row_sid)]) if str(row_sid) in remap else row_sid
        if str(row_sid) in remap:
            n_remap += 1
        gold_idx = sid_to_idx.get(gold_sid)
        draft_idx = sid_to_idx.get(row_sid)
        fb = _cell(data.iloc[r, KHS_COLS["feedback"]])
        dr = _cell(data.iloc[r, KHS_COLS["llm"]])
        if fb and gold_idx is not None:
            gold_by_idx[gold_idx] = fb
        if dr and draft_idx is not None:
            draft_by_idx[draft_idx] = dr
    print(f"[checklist] KHS gold(c10) {len(gold_by_idx)}건, 원안(c9) {len(draft_by_idx)}건"
          + (f" (remap {n_remap}건)" if n_remap else ""))
    return gold_by_idx, draft_by_idx


# ── LLM 구조화 추출 ──────────────────────────────────────────────────────────
_EXTRACT_SYSTEM = (
    "You convert a senior anesthesiologist's authoritative gold handoff into a structured "
    "checklist of key points a model handoff MUST cover. The gold is the ground truth for WHAT "
    "matters. Use the EMR only to expand abbreviations and to quote a short source span. "
    "CRITICAL: do NOT add any finding the gold did not include — in particular do NOT add QTc "
    "prolongation, transient desaturation, or blood-pressure/heart-rate event counts unless the "
    "gold explicitly mentions them. If the gold says only 'no issue'(특이사항 없음), return "
    "is_normal_case=true with items=[]. Output strict JSON only."
)

_EXTRACT_TMPL = """아래 '교수님 gold 인계문'을 정답으로 삼아, 모델 인계문이 반드시 cover해야 할
핵심 항목(checklist)으로 구조화하세요.

규칙:
- 항목 집합 = '교수님 gold가 담은 내용'만. gold에 없는 소견은 절대 추가 금지
  (특히 QTc 연장, 일시적 SpO2 저하, 혈압/심박 이벤트 횟수 등 vital 파생 항목).
- EMR은 약어 풀이와 source 인용에만 사용.
- gold가 device만 언급하면 그 device를 low 항목 1개로, 나머지는 is_normal_case 판단.
- gold가 사실상 '특이사항 없음'뿐이면 is_normal_case=true, items=[].
- category: airway, respiratory, hemodynamics, bleeding_transfusion, congenital_major_disease,
  intraop_event, drug_effect, lines_devices, cooperation_agitation, other
- severity: high/medium/low. source: EMR 근거 원문(없으면 gold 인용).

JSON만 출력:
{{"is_normal_case": <bool>, "items": [{{"id":"c1","finding":"...","category":"...","severity":"...","source":"..."}}]}}

### 교수님 gold 인계문 (정답)
{gold}

### EMR (약어 풀이/근거 인용용 — 새 소견 추가 금지)
{emr}

### JSON
"""


def build_checklist(engine, gold_df, gold_refs: dict) -> dict:
    """교수님 gold(c10) → 케이스별 checklist. gold 없는 케이스는 no_gold (수기 대상)."""
    idxs, sids, prompts = [], [], []
    checklist = {}
    for idx in range(len(gold_df)):
        sid = get_sid(gold_df.iloc[idx])
        opname = safe_get(gold_df.iloc[idx], ("수술", "수술명", "")) or "-"
        gold = (gold_refs.get(idx) or "").strip()
        base = {"idx": idx, "sid": sid, "opname": opname, "gold_text": gold,
                "reviewed": False, "accepted_without_review": False}

        if not gold:
            checklist[str(sid)] = {**base, "is_normal_case": False, "items": [],
                                   "source": "no_gold", "needs_manual": True}
            continue
        if is_no_issue_v3(gold):
            checklist[str(sid)] = {**base, "is_normal_case": True, "items": [],
                                   "source": "gold_normal"}
            continue

        emr = build_emr_text(gold_df.iloc[idx])
        prompts.append(_EXTRACT_TMPL.format(gold=gold, emr=emr))
        idxs.append(idx)
        sids.append(sid)
        checklist[str(sid)] = base

    parsed = engine.chat_json(prompts, system=_EXTRACT_SYSTEM, want="obj", retries=2) \
        if prompts else []

    for idx, sid, pj in zip(idxs, sids, parsed):
        entry = checklist[str(sid)]
        items, is_normal = [], False
        if isinstance(pj, dict):
            is_normal = bool(pj.get("is_normal_case", False))
            for k, it in enumerate(pj.get("items", []) or []):
                if not isinstance(it, dict):
                    continue
                fnd = str(it.get("finding", "")).strip()
                if not fnd:
                    continue
                items.append({
                    "id": it.get("id") or f"c{k + 1}",
                    "finding": fnd,
                    "category": it.get("category", "other"),
                    "severity": it.get("severity", "medium"),
                    "source": str(it.get("source", "")).strip(),
                })
        entry["items"] = items
        entry["is_normal_case"] = bool(is_normal and not items)
        # 추출 실패(파싱 None) 또는 '항목 0개 + normal 아님'은 실패로 기록 (E3)
        if pj is None or (not items and not entry["is_normal_case"]):
            entry["source"] = "gold_llm_failed"
        else:
            entry["source"] = "gold_llm"
    return checklist


def accept_provisional(checklist: dict, note: str = "") -> dict:
    """전문의 검수가 불가능할 때의 '잠정 채택' — reviewed는 절대 건드리지 않는다 (E6)."""
    for v in checklist.values():
        if v.get("source") in ("gold_llm", "gold_normal") and not v.get("reviewed"):
            v["accepted_without_review"] = True
            if note:
                v["accept_note"] = note
    return checklist
