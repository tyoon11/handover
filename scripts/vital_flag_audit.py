#!/usr/bin/env python3
"""
vital_flag_audit.py — 바이탈 이벤트 판정이 '실제로 인계할 문제'인지 GT로 검증

무엇을 재는가
------------
전문의 GT 인계문이 **실제로 언급한 바이탈 카테고리**를 정답으로 두고,
요약기가 낸 이벤트의 **PPV(언급률)** 를 층별로 낸다.

  PPV(층) = 그 층의 이벤트가 있는 케이스 중 GT가 해당 카테고리를 언급한 비율
  Recall  = GT가 언급한 카테고리 중 요약에 이벤트가 있던 비율

v3.2의 핵심 검증은 **REPORTABLE vs MINOR 의 PPV 격차**다. 설계가 맞다면
REPORTABLE 은 높고 MINOR 는 낮아야 한다. 격차가 없으면 규칙이 무의미하다는 뜻이다.
규칙별(R1 개입 / R2 지속×심도 / R3 종료지속 / R4 위험선) PPV도 함께 낸다.

입력 우선순위
------------
1. `VITAL_AUDIT_PKL` (v3.2 구조화 감사본) — 정확. rebuild_vital_map 이 만든다.
2. 없으면 `VITAL_MAP_PKL` 문자열 파싱 (v3.1 구 포맷 호환 — 재설계 전/후 비교용).

⚠ 유출 주의
-----------
gold test 16건의 GT로 컷오프를 고르면 **입력 표현을 test에 맞추는 것**이다.
결과를 fewshot(6) / test(16) 로 나눠 출력한다.
**컷오프 결정은 fewshot 열(+ 전문의 이진 라벨)로만. test 열은 진단용.**

실행
----
  python scripts/vital_flag_audit.py
  python scripts/vital_flag_audit.py --out /tmp/vital_audit.md   # PHI 없음(sid 미출력)
  python scripts/vital_flag_audit.py --source text               # 구 포맷 강제
"""
import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_v3.config_v3 import (      # noqa: E402
    GOLD_PKL, KHS_GOLD_OVERRIDE_JSON, VITAL_AUDIT_PKL, VITAL_MAP_PKL,
)

CRIT = "[유의]"

# ── 이벤트 카테고리 ↔ GT 언급 키워드 ────────────────────────────────────────
#   GT는 한국어 자유서술이라 표현이 다양하다. '처치'가 언급되면 그 이벤트를 인계한
#   것으로 본다 (예: '승압제 투여' = 저혈압을 인계한 것).
CATEGORY_KEYWORDS = {
    "bradycardia": ["서맥", "brady", "심박수 저하", "hr 저하", "아트로핀", "atropine"],
    "tachycardia": ["빈맥", "tachy", "심박수 상승", "hr 상승"],
    "hypotension": ["저혈압", "hypotens", "혈압 저하", "bp 저하", "승압", "에페드린",
                    "ephedrine", "페닐레프린", "phenylephrine", "노르에피네프린",
                    "norepinephrine", "도파민", "dopamine", "볼루스", "bolus"],
    "hypertension": ["고혈압", "hypertens", "혈압 상승", "bp 상승"],
    "desaturation": ["탈포화", "desat", "산소포화도", "spo2", "저산소", "산소 요구",
                     "o2", "산소 투여"],
    "hypothermia": ["저체온", "체온 저하", "hypotherm", "가온", "warming", "체온 유지"],
    "fever": ["발열", "fever", "고체온", "hypertherm"],
    "qtc": ["qtc", "qt 간격", "qt 연장"],
    "oliguria": ["핍뇨", "소변량", "oliguria", "urine", "uo"],
    "blood_loss": ["실혈", "출혈", "수혈", "transfus", "ebl", "bleed", "rbc", "ffp",
                   "혈액제제", "prbc"],
}

# 구 포맷(문자열) 라벨 → 카테고리
LABEL_TO_CATEGORY = [
    ("서맥", "bradycardia"), ("빈맥", "tachycardia"),
    ("저혈압", "hypotension"), ("고혈압", "hypertension"), ("고(>", "hypertension"),
    ("SpO2<", "desaturation"), ("탈포화", "desaturation"), ("목표미달", "desaturation"),
    ("저체온", "hypothermia"), ("안전범위 초과", "fever"), ("발열", "fever"),
    ("연장", "qtc"),
    ("핍뇨", "oliguria"), ("소변량", "oliguria"),
    ("실혈", "blood_loss"),
    ("정상하한 미만", "bradycardia"), ("정상상한 초과", "tachycardia"),
]

_RE_EVENT = re.compile(
    r"(?P<crit>\[유의\])?"
    r"(?P<label>[^:|]+?)"
    r":\s*(?P<n>\d+)회\s*(?P<dur>[^,]+?)\s*,\s*(?P<extlab>최저|최고)\s*(?P<ext>-?[\d.]+)"
)
_RE_DUR = re.compile(r"(?P<lt1><1분)|(?P<num>[\d.]+)\s*(?P<unit>분|시간)")


def _dur_min(s: str):
    """'20분' / '<1분' / '1.5시간' → 분. 실패 None."""
    m = _RE_DUR.search(s or "")
    if not m:
        return None
    if m.group("lt1"):
        return 0.5
    v = float(m.group("num"))
    return v * 60 if m.group("unit") == "시간" else v


def _category_from_label(label: str):
    for key, cat in LABEL_TO_CATEGORY:
        if key in label:
            return cat
    return None


def parse_vital_summary(text: str):
    """구 포맷(v3.1) 문자열 → (events, n_lines, n_stat_only_lines)."""
    events, n_lines, n_stat_only = [], 0, 0
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("(연령군") or line.startswith("###"):
            continue
        n_lines += 1
        item = line.split(":", 1)[0].strip()

        if ", 이벤트: " in line:
            for part in line.split(", 이벤트: ", 1)[1].split(" | "):
                m = _RE_EVENT.search(part.strip())
                if not m:
                    continue
                label = m.group("label").strip()
                longest = None
                lm = re.search(r"최장\s*([^)]+)\)", m.group("dur"))
                if lm:
                    longest = _dur_min(lm.group(1))
                total = _dur_min(m.group("dur"))
                events.append(dict(
                    item=item, label=label, category=_category_from_label(label),
                    reportable=bool(m.group("crit")), reasons=[], tier2=not m.group("crit"),
                    n=int(m.group("n")), total_min=total,
                    longest_min=longest if longest is not None else total))
        elif CRIT in line:
            for tok in re.findall(r"\[유의\](.*?)(?=\[유의\]|$)", line):
                tok = tok.strip()
                if not tok:
                    continue
                events.append(dict(item=item, label=tok,
                                   category=_category_from_label(tok),
                                   reportable=True, reasons=[], tier2=False,
                                   n=None, total_min=None, longest_min=None))
        else:
            if "이상 없음" in line or "정상" in line:
                n_stat_only += 1
    return events, n_lines, n_stat_only


def events_from_audit(entry: dict):
    """v3.2 구조화 감사본 → 공통 이벤트 형식."""
    out = []
    for bucket, is_rep in (("reportable", True), ("minor", False)):
        for e in entry.get(bucket, []) or []:
            out.append(dict(
                item=e.get("item"), label=e.get("label", ""),
                category=e.get("kind") or _category_from_label(e.get("label", "")),
                reportable=is_rep, reasons=e.get("reasons") or [],
                tier2=bool(e.get("tier2")), n=e.get("n"),
                total_min=e.get("total_min"), longest_min=e.get("longest_min"),
                interventions=e.get("interventions") or []))
    return out


def gt_categories(gt_text: str):
    low = (gt_text or "").lower()
    return {cat for cat, kws in CATEGORY_KEYWORDS.items()
            if any(k.lower() in low for k in kws)}


def bucket_dur(m):
    if m is None:
        return "미상"
    for hi, lab in ((1, "<1분"), (5, "1–5분"), (15, "5–15분"), (30, "15–30분")):
        if m < hi:
            return lab
    return ">30분"


def bucket_n(n):
    if n is None:
        return "미상"
    if n == 1:
        return "1회"
    if n <= 3:
        return "2–3회"
    if n <= 9:
        return "4–9회"
    return "10회+"


def _ppv_table(rows, keyfn, title):
    """rows: [(event, mentioned, subset)] → 버킷별 PPV 표(md)."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for ev, mentioned, subset in rows:
        for k in keyfn(ev) if isinstance(keyfn(ev), list) else [keyfn(ev)]:
            for sc in (subset, "all"):
                agg[k][sc][0] += 1
                agg[k][sc][1] += int(mentioned)
    out = [f"\n### {title}\n",
           "| 버킷 | fewshot n/언급/PPV | test n/언급/PPV | all PPV (n) |",
           "|---|---|---|---|"]
    for k in sorted(agg, key=lambda x: -agg[x]["all"][0]):
        cells = []
        for sc in ("fewshot", "test"):
            n, hit = agg[k][sc]
            cells.append(f"{n}/{hit}/{hit / n:.0%}" if n else "—")
        n, hit = agg[k]["all"]
        cells.append(f"{hit / n:.0%} ({n})" if n else "—")
        out.append(f"| {k} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="바이탈 이벤트 판정 감사 (GT 대조)")
    ap.add_argument("--out", type=str, default=None, help="md 저장 경로")
    ap.add_argument("--source", choices=["auto", "audit", "text"], default="auto")
    args = ap.parse_args()

    try:
        gt_map = {str(k): str(v) for k, v in
                  json.loads(Path(KHS_GOLD_OVERRIDE_JSON).read_text(encoding="utf-8")).items()}
    except Exception as e:
        sys.exit(f"GT override 로드 실패: {KHS_GOLD_OVERRIDE_JSON} ({e})")

    audit_map, vital_map, source = None, None, None
    if args.source in ("auto", "audit") and Path(VITAL_AUDIT_PKL).exists():
        with open(VITAL_AUDIT_PKL, "rb") as f:
            audit_map = pickle.load(f)
        source = "audit(v3.2 구조화)"
    if audit_map is None:
        if not Path(VITAL_MAP_PKL).exists():
            sys.exit(f"입력 없음: {VITAL_AUDIT_PKL} 도 {VITAL_MAP_PKL} 도 없다")
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)
        source = "text(구 포맷 파싱)"
    print(f"[source] {source}")

    # fewshot / test 라벨 (gold 6/16 분할과 동일 규칙)
    subset_of = {}
    try:
        import pandas as pd
        from pipeline_v3.data_splits import split_gold
        from pipeline_v3.prompt_utils import get_sid
        gold_df = pd.read_pickle(GOLD_PKL)
        fs, te = split_gold(gold_df)
        for df, name in ((fs, "fewshot"), (te, "test")):
            for i in range(len(df)):
                subset_of[str(get_sid(df.iloc[i]))] = name
    except Exception as e:
        print(f"⚠ gold 분할 라벨 실패({e}) — 'unknown' 으로 집계")

    rows, per_case = [], []
    recall_hit = recall_tot = 0
    missed = defaultdict(int)
    for ci, (sid, gt) in enumerate(sorted(gt_map.items())):
        if audit_map is not None:
            entry = audit_map.get(int(sid)) or audit_map.get(sid) or {}
            events = events_from_audit(entry)
            n_lines = n_stat = 0
        else:
            vs = vital_map.get(int(sid)) or vital_map.get(sid) or ""
            events, n_lines, n_stat = parse_vital_summary(vs)
        gcats = gt_categories(gt)
        subset = subset_of.get(sid, "unknown")
        for ev in events:
            rows.append((ev, ev["category"] in gcats, subset))
        rep_cats = {e["category"] for e in events if e["category"] and e["reportable"]}
        all_cats = {e["category"] for e in events if e["category"]}
        for c in gcats:
            recall_tot += 1
            if c in rep_cats:
                recall_hit += 1
            else:
                missed[c] += 1
        per_case.append(dict(case=ci, subset=subset, n_lines=n_lines,
                             n_stat_only=n_stat, n_events=len(events),
                             n_rep=sum(1 for e in events if e["reportable"]),
                             n_gt=len(gcats), n_hit=len(rep_cats & gcats),
                             n_hit_any=len(all_cats & gcats)))

    n_rep = sum(1 for e, _, _ in rows if e["reportable"])
    md = ["# 바이탈 이벤트 판정 감사 (GT 대조)", "",
          f"source: **{source}** · 케이스 {len(per_case)}건 · 이벤트 {len(rows)}개 "
          f"(REPORTABLE {n_rep} / MINOR {len(rows) - n_rep})", "",
          "> PPV = 그 층의 이벤트가 있을 때 전문의 GT가 해당 카테고리를 언급한 비율.",
          "> **컷오프 결정은 fewshot 열로만** — test 열로 고르면 입력 표현을 test에 맞추는 것이다.",
          "> GT는 1~5문장 제약이 있어 '언급 안 됨 ≠ 안 중요함' — PPV는 하한이다.",
          ]
    md.append(_ppv_table(rows, lambda e: "REPORTABLE" if e["reportable"] else "MINOR",
                         "층별 (v3.2 설계의 핵심 검증 — 격차가 커야 한다)"))
    md.append(_ppv_table(rows, lambda e: e["reasons"] or ["(없음=MINOR)"],
                         "판정 규칙별 (이벤트가 여러 규칙에 걸리면 중복 계수)"))
    md.append(_ppv_table(rows, lambda e: e["category"] or "기타", "카테고리별"))
    md.append(_ppv_table(rows, lambda e: bucket_dur(e["longest_min"]),
                         "최장 episode 길이별"))
    md.append(_ppv_table(rows, lambda e: bucket_n(e["n"]), "반복 횟수별"))
    md.append(_ppv_table(rows, lambda e: ("tier1(임상기준)" if not e["tier2"]
                                          else "tier2(범위이탈)"), "tier별"))

    md += ["", "### 케이스별 부담", "",
           "| 케이스 | subset | 이벤트 | REPORTABLE | GT 카테고리 | 적중(REP) | 적중(전체) |",
           "|---|---|---|---|---|---|---|"]
    for r in per_case:
        md.append(f"| {r['case']} | {r['subset']} | {r['n_events']} | {r['n_rep']} | "
                  f"{r['n_gt']} | {r['n_hit']} | {r['n_hit_any']} |")

    md += ["", "### Recall (GT가 언급했는데 REPORTABLE 에 없던 것)", ""]
    md.append(f"- GT 언급 카테고리 {recall_tot}개 중 REPORTABLE 에 존재 {recall_hit}개 "
              f"({recall_hit / recall_tot:.0%})" if recall_tot else "- GT 언급 없음")
    for c, n in sorted(missed.items(), key=lambda x: -x[1]):
        md.append(f"- 미검출 `{c}`: {n}건")
    md += ["", "> 미검출이 많으면 규칙이 과하게 막고 있다는 뜻이다(특히 R2 지속 컷오프).",
           "> MINOR 의 PPV가 REPORTABLE 과 비슷하면 규칙이 정보를 못 담고 있다는 뜻이다."]

    text = "\n".join(md)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n저장: {args.out}  (PHI 없음 — sid·GT 원문 미포함)")


if __name__ == "__main__":
    main()
