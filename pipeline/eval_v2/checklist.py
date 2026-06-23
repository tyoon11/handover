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


# ── LLM 추출 프롬프트 (교수님 gold=c10 을 '정답'으로 구조화) ─────────────────
# 핵심: checklist 항목은 '교수님 gold가 담은 것'만. EMR은 약어 풀이/근거 인용에만 쓰고
#       gold에 없는 소견(특히 QTc·일시적 desat·혈압 이벤트 카운트 등 vital 파생)은 추가 금지.
_EXTRACT_SYSTEM = (
    "You convert a senior anesthesiologist's authoritative gold handoff into a structured "
    "checklist of key points a model handoff MUST cover. The gold is the ground truth for WHAT "
    "matters. Use the EMR only to expand abbreviations and to quote a short source span. "
    "CRITICAL: do NOT add any finding the gold did not include — in particular do NOT add QTc "
    "prolongation, transient desaturation, or blood-pressure/heart-rate event counts unless the "
    "gold explicitly mentions them. If the gold says only 'no issue'(특이사항 없음), return "
    "is_normal_case=true with items=[] (but keep an airway-device note as a low item if the gold "
    "mentions one). Output strict JSON only."
)

_EXTRACT_TMPL = """아래 '교수님 gold 인계문'을 정답으로 삼아, 모델 인계문이 반드시 cover해야 할
핵심 항목(checklist)으로 구조화하세요.

규칙:
- 항목 집합 = '교수님 gold가 담은 내용'만. gold에 없는 소견은 절대 추가하지 마세요
  (특히 QTc 연장, 일시적 SpO2 저하, 혈압/심박수 이벤트 횟수 등 vital 파생 항목 금지).
- EMR은 약어 풀이와 source 인용에만 사용.
- gold가 'AuraGain 사용, 특이사항 없음'처럼 device만 언급하면 그 device를 low 항목 1개로,
  나머지는 is_normal_case 판단.
- gold가 사실상 '특이사항 없음'뿐이면 is_normal_case=true, items=[].
- category: airway, respiratory, hemodynamics, bleeding_transfusion, congenital_major_disease,
  intraop_event, drug_effect, lines_devices, cooperation_agitation, other
- severity: high/medium/low. source: EMR에서 근거가 된 짧은 원문(없으면 gold 인용).

JSON만 출력:
{{"is_normal_case": <bool>, "items": [{{"id":"c1","finding":"...","category":"...","severity":"...","source":"..."}}]}}

### 교수님 gold 인계문 (정답)
{gold}

### EMR (약어 풀이/근거 인용용 — 새 소견 추가 금지)
{emr}

### JSON
"""


def _opname(gold_df, idx):
    try:
        v = gold_df.iloc[idx][("수술", "수술명", "")]
        return str(v.iloc[0]) if hasattr(v, "iloc") else str(v)
    except Exception:
        return "-"


def build_checklist(engine, gold_df, vital_map, gold_refs, context_refs=None):
    """교수님 gold(c10)를 정답으로 checklist 구조화.
    gold_refs    : {idx: 교수님 gold(c10) text}  ← 항목의 출처(정답)
    context_refs : {idx: gemma 원안(c9) 등}      ← (현재 미사용, 확장 여지)
    gold가 빈 케이스는 source='no_gold'로 표시하고 수기 작성 대상으로 남긴다."""
    gold_refs = gold_refs or {}
    from config_v2 import is_no_issue

    idxs, sids, prompts = [], [], []
    checklist = {}
    for idx in range(len(gold_df)):
        try:
            sid = int(gold_df.iloc[idx]["수술 ID"].iloc[0]) if hasattr(
                gold_df.iloc[idx]["수술 ID"], "iloc") else int(gold_df.iloc[idx]["수술 ID"])
        except Exception:
            sid = -1
        gold = (gold_refs.get(idx) or "").strip()
        base = {"idx": idx, "sid": sid, "opname": _opname(gold_df, idx),
                "gold_text": gold, "reviewed": False}

        if not gold:
            # 교수님 gold 없음 → 수기 작성 필요 (예: idx 12 Crouzon, c10 입력 누락)
            checklist[str(sid)] = {**base, "is_normal_case": False, "items": [],
                                   "source": "no_gold", "needs_manual": True}
            continue
        if is_no_issue(gold):
            checklist[str(sid)] = {**base, "is_normal_case": True, "items": [],
                                   "source": "gold_normal"}
            continue

        row = gold_df.iloc[idx] if idx < len(gold_df) else None
        emr = build_emr_text(row) if row is not None else ""
        prompts.append(_EXTRACT_TMPL.format(gold=gold, emr=emr))
        idxs.append(idx); sids.append(sid)
        checklist[str(sid)] = {**base, "_pending": True}

    parsed = engine.chat_json(prompts, system=_EXTRACT_SYSTEM, want="obj", retries=2) \
        if prompts else []

    for idx, sid, pj in zip(idxs, sids, parsed):
        entry = checklist[str(sid)]
        entry.pop("_pending", None)
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
                    "id": it.get("id") or f"c{k+1}",
                    "finding": fnd,
                    "category": it.get("category", "other"),
                    "severity": it.get("severity", "medium"),
                    "source": str(it.get("source", "")).strip(),
                })
        entry["items"] = items
        entry["is_normal_case"] = bool(is_normal and not items)
        entry["source"] = "gold_llm" if pj is not None else "gold_llm_failed"
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


def _cell(v):
    """엑셀 셀 → 정제 문자열('', 'nan', '-'는 빈값)."""
    if v is None or (isinstance(v, float)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "-") else s


def load_khs(xlsx_path, sheet, cols, header_rows, gold_df):
    """KHS 엑셀(다중헤더, 위치 기반) 로드.
    반환:
      gold_by_idx  : {idx: 교수 피드백(c10) gold text}  — 빈 행 제외
      draft_by_idx : {idx: gemma 원안(c9) — 피드백 '대상' 맥락용}
    매칭: c2 sid → gold_df idx 우선, 실패 시 c0 idx, 그래도 없으면 행순서."""
    import pandas as pd
    try:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    except Exception as e:
        print(f"[checklist] KHS 로드 실패({e})")
        return {}, {}
    data = raw.iloc[header_rows:].reset_index(drop=True)

    sid_to_idx = {}
    for i in range(len(gold_df)):
        try:
            s = gold_df.iloc[i]["수술 ID"]
            s = int(s.iloc[0]) if hasattr(s, "iloc") else int(s)
            sid_to_idx[s] = i
        except Exception:
            pass

    gold_by_idx, draft_by_idx = {}, {}
    for r in range(len(data)):
        idx = None
        try:
            idx = sid_to_idx.get(int(float(data.iloc[r, cols["sid"]])))
        except Exception:
            idx = None
        if idx is None:
            try:
                ci = int(float(data.iloc[r, cols["idx"]]))
                idx = ci if ci < len(gold_df) else None
            except Exception:
                idx = None
        if idx is None:
            idx = r if r < len(gold_df) else None
        if idx is None:
            continue
        fb = _cell(data.iloc[r, cols["feedback"]])   # c10 = gold
        dr = _cell(data.iloc[r, cols["llm"]])         # c9  = gemma 원안
        if fb:
            gold_by_idx[idx] = fb
        if dr:
            draft_by_idx[idx] = dr
    print(f"[checklist] KHS gold(c10 피드백) {len(gold_by_idx)}건, 원안(c9) {len(draft_by_idx)}건 로드")
    return gold_by_idx, draft_by_idx


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
