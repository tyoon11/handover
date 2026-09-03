#!/usr/bin/env python3
"""
vital_flag_audit.py — 바이탈 이벤트 표기가 '실제로 인계할 문제'인지 GT로 검증

문제
----
현재 vital_summary는 임계값을 넘은 표본이 **1개만 있어도** 이벤트로 등재하고,
이벤트가 하나도 없는 항목까지 중앙값·범위를 출력한다. 그래서 인계 가치가 없는
"저혈압 1회 <1분" 류가 프롬프트에 대량으로 실리고, 생성문이 바이탈 쪽으로 쏠린다.

이 스크립트가 하는 일
-------------------
gold 케이스의 **전문의 GT 인계문이 실제로 언급한 바이탈 카테고리**를 정답으로 삼아,
우리 요약이 등재한 이벤트의 **PPV(언급률)** 를 tier·지속시간·반복횟수 버킷별로 낸다.
→ "어떤 버킷을 이벤트로 올릴 가치가 있는가"를 감으로가 아니라 데이터로 정한다.

  PPV(bucket) = (그 버킷 이벤트가 있는 케이스 중 GT가 해당 카테고리를 언급한 비율)
  Recall       = (GT가 언급한 카테고리 중 우리 요약에 이벤트가 있던 비율)

⚠ 유출 주의
-----------
gold test 16건의 GT로 컷오프를 고르면 **입력 표현을 test에 맞추는 것**이 된다.
그래서 결과를 fewshot(6) / test(16) 로 나눠 출력한다.
**컷오프 결정은 fewshot 열(+ 별도 전문의 라벨링)로만 하고, test 열은 진단용으로만 본다.**

실행
----
  python scripts/vital_flag_audit.py                 # 표 출력
  python scripts/vital_flag_audit.py --out audit.md  # md 저장 (PHI 없음: sid 미출력)
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
    GOLD_PKL, VITAL_MAP_PKL, KHS_GOLD_OVERRIDE_JSON,
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

# 요약 이벤트 라벨 → 카테고리
LABEL_TO_CATEGORY = [
    ("서맥", "bradycardia"), ("빈맥", "tachycardia"),
    ("저혈압", "hypotension"), ("고혈압", "hypertension"), ("고(>", "hypertension"),
    ("SpO2<", "desaturation"), ("목표미달", "desaturation"),
    ("저체온", "hypothermia"), ("안전범위 초과", "fever"), ("발열", "fever"),
    ("연장", "qtc"), ("정상상한 초과(>4", "qtc"),
    ("핍뇨", "oliguria"), ("실혈", "blood_loss"),
    ("정상하한 미만", "bradycardia"), ("정상상한 초과", "tachycardia"),
]

_RE_EVENT = re.compile(
    r"(?P<crit>\[유의\])?"
    r"(?P<label>[^:|]+?)"
    r":\s*(?P<n>\d+)회\s*(?P<dur>[^,]+?)\s*,\s*(?P<extlab>최저|최고)\s*(?P<ext>-?[\d.]+)"
)
_RE_DUR = re.compile(r"(?P<lt1><1분)|(?P<num>[\d.]+)\s*(?P<unit>분|시간)")


def _dur_min(s: str):
    """'20분' / '<1분' / '1.5시간' → 분. 파싱 실패 None."""
    m = _RE_DUR.search(s or "")
    if not m:
        return None
    if m.group("lt1"):
        return 0.5
    v = float(m.group("num"))
    return v * 60 if m.group("unit") == "시간" else v


def _category(label: str):
    for key, cat in LABEL_TO_CATEGORY:
        if key in label:
            return cat
    return None


def parse_vital_summary(text: str):
    """요약 문자열 → (events, n_lines, n_stat_only_lines).

    events: [{item, label, category, crit, n, total_min, longest_min}]
    """
    events, n_lines, n_stat_only = [], 0, 0
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("(연령군"):
            continue
        n_lines += 1
        item = line.split(":", 1)[0].strip()

        if ", 이벤트: " in line:
            seg = line.split(", 이벤트: ", 1)[1]
            for part in seg.split(" | "):
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
                    item=item, label=label, category=_category(label),
                    crit=bool(m.group("crit")), n=int(m.group("n")),
                    total_min=total, longest_min=longest if longest is not None else total,
                ))
        elif CRIT in line:
            # EBL/UO 처럼 _ev 포맷이 아닌 유의 표기 (대량 실혈·핍뇨)
            for tok in re.findall(r"\[유의\](.*?)(?=\[유의\]|$)", line):
                tok = tok.strip()
                if not tok:
                    continue
                events.append(dict(item=item, label=tok, category=_category(tok),
                                   crit=True, n=None, total_min=None, longest_min=None))
        else:
            if "이상 없음" in line or "정상" in line or ":" in line:
                n_stat_only += 1
    return events, n_lines, n_stat_only


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
    """rows: [(key_src_event, mentioned, subset)] → 버킷별 PPV 표(md)."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # key → subset → [n, hit]
    for ev, mentioned, subset in rows:
        k = keyfn(ev)
        for s in (subset, "all"):
            agg[k][s][0] += 1
            agg[k][s][1] += int(mentioned)
    out = [f"\n### {title}\n",
           "| 버킷 | fewshot n / 언급 / PPV | test n / 언급 / PPV | all PPV |",
           "|---|---|---|---|"]
    for k in sorted(agg, key=lambda x: -agg[x]["all"][0]):
        cells = []
        for s in ("fewshot", "test"):
            n, hit = agg[k][s]
            cells.append(f"{n} / {hit} / {hit / n:.0%}" if n else "—")
        n, hit = agg[k]["all"]
        cells.append(f"{hit / n:.0%} ({n})" if n else "—")
        out.append(f"| {k} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="바이탈 이벤트 표기 감사 (GT 대조)")
    ap.add_argument("--out", type=str, default=None, help="md 저장 경로")
    args = ap.parse_args()

    # GT (교수님 확정본)
    try:
        gt_map = {str(k): str(v) for k, v in
                  json.loads(Path(KHS_GOLD_OVERRIDE_JSON).read_text(encoding="utf-8")).items()}
    except Exception as e:
        sys.exit(f"GT override 로드 실패: {KHS_GOLD_OVERRIDE_JSON} ({e})")

    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    # fewshot / test 라벨 (gold 6/16 분할과 동일 규칙)
    subset_of = {}
    try:
        import pandas as pd
        from pipeline_v3.data_splits import split_gold
        gold_df = pd.read_pickle(GOLD_PKL)
        fs, te = split_gold(gold_df)
        from pipeline_v3.prompt_utils import get_sid
        for df, name in ((fs, "fewshot"), (te, "test")):
            for i in range(len(df)):
                subset_of[str(get_sid(df.iloc[i]))] = name
    except Exception as e:
        print(f"⚠ gold 분할 라벨 실패({e}) — 전부 'all' 로만 집계한다")

    rows, per_case, recall_hit, recall_tot, missed = [], [], 0, 0, defaultdict(int)
    for ci, (sid, gt) in enumerate(sorted(gt_map.items())):
        vs = vital_map.get(int(sid)) or vital_map.get(sid) or ""
        events, n_lines, n_stat = parse_vital_summary(vs)
        gcats = gt_categories(gt)
        subset = subset_of.get(sid, "unknown")
        for ev in events:
            rows.append((ev, ev["category"] in gcats, subset))
        ev_cats = {e["category"] for e in events if e["category"]}
        crit_cats = {e["category"] for e in events if e["category"] and e["crit"]}
        for c in gcats:
            recall_tot += 1
            if c in ev_cats:
                recall_hit += 1
            else:
                missed[c] += 1
        per_case.append(dict(case=ci, subset=subset, n_lines=n_lines,
                             n_stat_only=n_stat, n_events=len(events),
                             n_crit=sum(1 for e in events if e["crit"]),
                             n_gt_cats=len(gcats), n_hit=len(ev_cats & gcats),
                             n_crit_hit=len(crit_cats & gcats)))

    md = ["# 바이탈 이벤트 표기 감사 (GT 대조)", "",
          f"케이스 {len(per_case)}건 · 등재 이벤트 {len(rows)}개 "
          f"(유의 {sum(1 for e,_,_ in rows if e['crit'])}개)", "",
          "> PPV = 그 버킷 이벤트가 있을 때 전문의 GT가 해당 카테고리를 언급한 비율.",
          "> **컷오프 결정은 fewshot 열로만** — test 열로 고르면 입력 표현을 test에 맞추는 것이다.",
          ]
    md.append(_ppv_table(rows, lambda e: f"{'유의' if e['crit'] else '범위이탈'}", "tier별"))
    md.append(_ppv_table(rows, lambda e: e["category"] or "기타", "카테고리별"))
    md.append(_ppv_table(rows, lambda e: bucket_dur(e["longest_min"]), "최장 episode 길이별"))
    md.append(_ppv_table(rows, lambda e: bucket_n(e["n"]), "반복 횟수별"))
    md.append(_ppv_table(rows, lambda e: f"{'유의' if e['crit'] else '범위이탈'} · "
                                         f"{bucket_dur(e['longest_min'])}", "tier × 길이"))

    md += ["", "### 케이스별 부담 (프롬프트에 실리는 양)", "",
           "| 케이스 | subset | 요약 줄수 | 이벤트0 항목줄 | 이벤트 | 유의 | GT 카테고리 | 적중 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in per_case:
        md.append(f"| {r['case']} | {r['subset']} | {r['n_lines']} | {r['n_stat_only']} | "
                  f"{r['n_events']} | {r['n_crit']} | {r['n_gt_cats']} | {r['n_hit']} |")

    md += ["", "### Recall (GT가 언급했는데 요약이 못 잡은 것)", "",
           f"- GT 언급 카테고리 {recall_tot}개 중 요약에 이벤트 존재 {recall_hit}개 "
           f"({recall_hit / recall_tot:.0%})" if recall_tot else "- GT 언급 없음",
           ""]
    for c, n in sorted(missed.items(), key=lambda x: -x[1]):
        md.append(f"- 미검출 `{c}`: {n}건")

    text = "\n".join(md)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n저장: {args.out}  (PHI 없음 — sid·GT 원문 미포함)")


if __name__ == "__main__":
    main()
