"""
eval_v2/checklist.py — Gold actionable checklist (Coverage 채점 기준)

개선안 1: 케이스별 "PACU/ICU가 반드시 알아야 할 조치 가능 이상소견" 목록을 만든다.
정답(전문의 인계문)이 들어있는 휴먼평가 엑셀을 1차 소스로,
gemma-4-31B가 구조화 추출 → 전문의가 직접 수정 가능한 JSON으로 저장한다.

checklist JSON 구조 (case_key = str(sid)):
{
  "<sid>": {
    "idx": 0, "sid": 100517749, "opname": "Repair of hypospadias",
    "is_normal_case": false,            # true면 정답이 "특이사항 없음"
    "items": [
       {"id":"c1","finding":"QTc 연장(최대 495ms)","category":"hemodynamics",
        "severity":"high","source":"QTc 연장(132회>450ms, 최대 495ms)"} ...
    ],
    "source": "llm_bootstrap",          # or "xlsx_human" / "manual"
    "reviewed": false                   # 전문의 검수 여부 (검수 후 true로)
  }, ...
}
"""

import json
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import build_emr_text, build_user_prompt  # noqa


# ── JSON I/O ────────────────────────────────────────────────────────────────
def load_checklist(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checklist(checklist, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 엑셀 컬럼 자동감지 ──────────────────────────────────────────────────────
def detect_columns(df, hints):
    """hints={role:[후보키워드,...]} → {role: 실제컬럼명 or None}.
    MultiIndex/단일 컬럼 모두 지원 (컬럼을 문자열로 평탄화해 부분일치)."""
    flat = {}
    for col in df.columns:
        name = " ".join(str(x) for x in col) if isinstance(col, tuple) else str(col)
        flat[col] = name.lower()
    resolved = {}
    for role, kws in hints.items():
        found = None
        for col, name in flat.items():
            if any(kw.lower() in name for kw in kws):
                found = col
                break
        resolved[role] = found
    return resolved


def inspect_xlsx(xlsx_path, hints):
    """--inspect 모드: 컬럼 목록 + 자동감지 결과 출력 (전문의/엔지니어 확인용)."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    print(f"\n[inspect] {xlsx_path}\n  shape={df.shape}")
    print("  ── 컬럼 목록 ──")
    for c in df.columns:
        print(f"    {c!r}")
    print("  ── 자동감지 결과 (config_v2.XLSX_COL_HINTS) ──")
    for role, col in detect_columns(df, hints).items():
        print(f"    {role:8s} → {col!r}")
    return df


# ── LLM 추출 프롬프트 ───────────────────────────────────────────────────────
_EXTRACT_SYSTEM = (
    "You are a pediatric anesthesiologist building a PACU/ICU handoff safety checklist. "
    "From the EMR (and an optional reference handoff), extract ONLY clinically actionable "
    "abnormalities that the receiving team must know after surgery. "
    "Exclude normal/stable findings, routine drug totals, surgery steps, and administrative text. "
    "Output strict JSON only."
)

_EXTRACT_TMPL = """다음 소아 마취 EMR(+vital 요약)과 참고 인계문을 보고, 수술 후 PACU/ICU 인계 시
반드시 전달해야 할 '조치 가능한 이상소견(actionable abnormality)'만 추출하세요.

규칙:
- 정상/안정 소견, 일상적 약물 총량, 수술 단계, 행정 문구는 제외.
- 항목이 하나도 없으면 is_normal_case=true, items=[].
- category는 다음 중 하나: airway, respiratory, hemodynamics, bleeding_transfusion,
  congenital_major_disease, intraop_event, drug_effect, lines_devices, cooperation_agitation, other
- severity는 high/medium/low.
- source는 EMR/vital에서 그 근거가 된 짧은 원문 인용.

아래 JSON 스키마로만 출력:
{{"is_normal_case": <bool>, "items": [{{"id":"c1","finding":"...","category":"...","severity":"...","source":"..."}}]}}

### EMR + VITAL
{premise}

### 참고 인계문 (정답 아님, 보조용)
{reference}

### JSON
"""


def _premise(gold_df, vital_map, idx, sid):
    row = gold_df.iloc[idx] if idx < len(gold_df) else None
    emr = build_emr_text(row) if row is not None else ""
    vital = vital_map.get(sid, "")
    return build_user_prompt(emr, vital)


def _opname(gold_df, idx):
    try:
        v = gold_df.iloc[idx][("수술", "수술명", "")]
        return str(v.iloc[0]) if hasattr(v, "iloc") else str(v)
    except Exception:
        return "-"


def build_checklist(engine, gold_df, vital_map, references=None):
    """gold 전체 케이스에 대해 LLM으로 actionable checklist 부트스트랩.
    references: {idx: 참고인계문str} (xlsx 전문의 인계문 등). 없으면 빈 문자열."""
    references = references or {}
    idxs, sids, prompts = [], [], []
    for idx in range(len(gold_df)):
        try:
            sid = int(gold_df.iloc[idx]["수술 ID"].iloc[0]) if hasattr(
                gold_df.iloc[idx]["수술 ID"], "iloc") else int(gold_df.iloc[idx]["수술 ID"])
        except Exception:
            sid = -1
        premise = _premise(gold_df, vital_map, idx, sid)
        ref = references.get(idx, "") or "(참고 인계문 없음)"
        prompts.append(_EXTRACT_TMPL.format(premise=premise, reference=ref))
        idxs.append(idx)
        sids.append(sid)

    parsed = engine.chat_json(prompts, system=_EXTRACT_SYSTEM, want="obj", retries=2)

    checklist = {}
    for idx, sid, pj in zip(idxs, sids, parsed):
        items = []
        is_normal = False
        if isinstance(pj, dict):
            is_normal = bool(pj.get("is_normal_case", False))
            raw_items = pj.get("items", []) or []
            for k, it in enumerate(raw_items):
                if not isinstance(it, dict):
                    continue
                items.append({
                    "id": it.get("id") or f"c{k+1}",
                    "finding": str(it.get("finding", "")).strip(),
                    "category": it.get("category", "other"),
                    "severity": it.get("severity", "medium"),
                    "source": str(it.get("source", "")).strip(),
                })
            items = [it for it in items if it["finding"]]
        else:
            # 파싱 실패 → 보수적으로 normal=false, 빈 items + 플래그
            is_normal = False
        if not items:
            is_normal = True if (isinstance(pj, dict) and pj.get("is_normal_case")) else is_normal
        checklist[str(sid)] = {
            "idx": idx,
            "sid": sid,
            "opname": _opname(gold_df, idx),
            "is_normal_case": is_normal and not items,
            "items": items,
            "source": "llm_bootstrap" if pj is not None else "llm_failed",
            "reviewed": False,
        }
    return checklist


def load_sy(xlsx_path, sy_cols, header_rows=3):
    """인계요약지_SY.xlsx 전용 로더 (위치 기반, 병합셀 ffill).

    반환:
      gold_by_idx : {idx(int): gold_handoff_text}
      scored_rows : [{idx, model, prompt, gen, brevity, critical, err, over, under, read, comment}]
                    (전문의가 점수를 매긴 생성행. 판정 calibration 용)
    """
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name="데이터", header=None)
    data = df.iloc[header_rows:].reset_index(drop=True)

    def col(role):
        return data.iloc[:, sy_cols[role]]

    inv = {v: k for k, v in sy_cols.items()}
    named = data.rename(columns=inv)
    # 병합셀(케이스 단위)은 ffill
    for role in ("idx", "gold", "dept", "recovery", "anrec", "preop", "premed"):
        named[role] = named[role].ffill()

    gold_by_idx, scored_rows = {}, []
    for _, r in named.iterrows():
        try:
            cidx = int(float(r["idx"]))
        except Exception:
            continue
        if cidx not in gold_by_idx:
            g = r.get("gold")
            gold_by_idx[cidx] = "" if g is None or (isinstance(g, float)) else str(g).strip()
        gen = r.get("gen")
        if gen is None or (isinstance(gen, float)) or not str(gen).strip():
            continue

        def num(role):
            v = r.get(role)
            try:
                return float(v)
            except Exception:
                return None

        scored_rows.append({
            "idx": cidx,
            "model": str(r.get("model", "")).strip(),
            "prompt": str(r.get("prompt", "")).strip(),
            "gen": str(gen).strip(),
            "h_brevity": num("brevity"),
            "h_critical": num("critical"),
            "h_err": num("err"),
            "h_over": num("over"),
            "h_under": num("under"),
            "h_read": num("read"),
            "comment": str(r.get("comment", "") or "").strip(),
        })
    print(f"[checklist] SY 로드: gold {len(gold_by_idx)}건, 채점행 {len(scored_rows)}건")
    return gold_by_idx, scored_rows


def load_khs(xlsx_path, sheet, hints, gold_df):
    """KHS 엑셀의 '진짜 gold'(Feedback) 로드 → {idx: feedback_gold_text}.
    Feedback이 빈 행은 건너뜀. sid 컬럼 있으면 sid로, 없으면 행 순서로 gold_df와 매칭.
    추가로 {idx: llm_sample}(gemma 참고초안)도 반환(참고용)."""
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
    except Exception as e:
        print(f"[checklist] KHS 로드 실패({e})")
        return {}, {}
    cols = detect_columns(df, hints)
    fb_col = cols.get("feedback")
    if fb_col is None:
        print("[checklist] KHS Feedback 컬럼 자동감지 실패 → config_v2.KHS_COL_HINTS 확인")
        return {}, {}
    llm_col = cols.get("llm_sample")
    sid_col = cols.get("sid")

    sid_to_idx = {}
    for i in range(len(gold_df)):
        try:
            s = gold_df.iloc[i]["수술 ID"]
            s = int(s.iloc[0]) if hasattr(s, "iloc") else int(s)
            sid_to_idx[s] = i
        except Exception:
            pass

    feedback, llm_sample = {}, {}
    for r in range(len(df)):
        idx = None
        if sid_col is not None:
            try:
                idx = sid_to_idx.get(int(df.iloc[r][sid_col]))
            except Exception:
                idx = None
        if idx is None:
            idx = r if r < len(gold_df) else None
        if idx is None:
            continue
        fb = df.iloc[r][fb_col]
        if fb is not None and not (isinstance(fb, float)) and str(fb).strip() and str(fb).strip().lower() != "nan":
            feedback[idx] = str(fb).strip()
        if llm_col is not None:
            ls = df.iloc[r][llm_col]
            if ls is not None and not (isinstance(ls, float)) and str(ls).strip():
                llm_sample[idx] = str(ls).strip()
    print(f"[checklist] KHS 진짜 gold(Feedback) {len(feedback)}건 로드")
    return feedback, llm_sample


def merge_references(*ref_dicts):
    """앞쪽 dict 우선으로 {idx: text} 병합 (KHS Feedback > SY gold > ...)."""
    out = {}
    for d in reversed(ref_dicts):
        for k, v in (d or {}).items():
            if v and str(v).strip():
                out[k] = v
    return out


def load_references_from_xlsx(xlsx_path, hints, gold_df):
    """xlsx에서 전문의 인계문을 {idx: text}로 로드. sid 또는 idx로 gold_df와 매칭.
    실패하면 {} 반환(부트스트랩은 EMR만으로 진행)."""
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"[checklist] xlsx 로드 실패({e}) → 참고 인계문 없이 진행")
        return {}
    cols = detect_columns(df, hints)
    human_col = cols.get("human")
    if human_col is None:
        print("[checklist] 전문의 인계문 컬럼 자동감지 실패 → config_v2.XLSX_COL_HINTS 확인 필요")
        return {}

    # gold_df의 sid → idx 매핑
    sid_to_idx = {}
    for i in range(len(gold_df)):
        try:
            s = gold_df.iloc[i]["수술 ID"]
            s = int(s.iloc[0]) if hasattr(s, "iloc") else int(s)
            sid_to_idx[s] = i
        except Exception:
            pass

    refs = {}
    sid_col = cols.get("sid")
    for r in range(len(df)):
        text = df.iloc[r][human_col]
        if text is None or (isinstance(text, float)):
            continue
        text = str(text).strip()
        if not text or text.lower() == "nan":
            continue
        idx = None
        if sid_col is not None:
            try:
                s = int(df.iloc[r][sid_col])
                idx = sid_to_idx.get(s)
            except Exception:
                idx = None
        if idx is None:
            idx = r if r < len(gold_df) else None  # 순서 fallback
        if idx is not None:
            refs[idx] = text
    print(f"[checklist] xlsx 전문의 인계문 {len(refs)}건 로드")
    return refs
