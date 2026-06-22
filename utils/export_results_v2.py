"""
export_results_v2.py — v2 평가결과 → 임상검토용 HTML (개선안 6)

v1 export_results.py는 그대로 보존. 이 리포트는 v2의 3축 + 안전위반 + 누락/환각을
한눈에 보여주고, '안전축(coverage·safety) 우선'으로 정렬한다.

실행:
  python utils/export_results_v2.py --run_id <RUN_ID>
"""

import sys, os, argparse, json, html
from pathlib import Path


def _early_run_id():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run_id", type=str, default=None)
    rid = p.parse_known_args()[0].run_id
    if rid:
        os.environ["HANDOVER_RUN_ID"] = rid
    return rid


_RID = _early_run_id()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pickle
from config import GOLD_PKL, VITAL_MAP_PKL, build_emr_text, build_user_prompt
from config_v2 import EVAL_V2_OUT, REPORT_V2_OUT, GOLD_CHECKLIST_JSON


def esc(x):
    return html.escape(str(x)) if x is not None else ""


def load_scores():
    """EVAL_V2_OUT/<tag>/*_scores_v2.jsonl 전부 로드 → {tag: [recs]}."""
    out = {}
    for f in sorted(EVAL_V2_OUT.rglob("*_scores_v2.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if recs:
            out[f.parent.name] = recs
    return out


# v1 export의 _split_tag와 동일 의도 (model/exp 분리). 알려진 모델키 우선.
_MODEL_KEYS = ["gemma4_31b", "gemma4", "qwen35", "llama", "qwen", "hari"]


def split_tag(tag):
    for mk in _MODEL_KEYS:
        if tag == mk:
            return mk, "?"
        if tag.startswith(mk + "_"):
            return mk, tag[len(mk) + 1:]
    parts = tag.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "?")


CSS = """
body{font-family:-apple-system,'Segoe UI',sans-serif;max-width:1280px;margin:30px auto;padding:0 20px;line-height:1.5;color:#222}
h1{border-bottom:2px solid #333;padding-bottom:8px}
h2{background:#f0f4f8;padding:12px 16px;border-left:4px solid #2563eb;margin-top:40px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border:1px solid #d1d5db;padding:5px 8px;text-align:left}
th{background:#f3f4f6}
.viol{background:#fee2e2}
.best{background:#dcfce7;font-weight:700}
.miss{color:#b91c1c}.halu{color:#b45309}.noise{color:#6b7280}
pre{background:#f8f9fa;border:1px solid #e1e4e8;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:13px}
details summary{cursor:pointer;padding:6px 10px;background:#f9fafb;border-radius:4px;font-weight:600}
.meta{color:#6b7280;font-size:13px}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:700}
.b-viol{background:#dc2626;color:#fff}.b-ok{background:#16a34a;color:#fff}
"""


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default=_RID)
    ap.add_argument("--out", type=str, default=None)
    cli = ap.parse_args()

    scores = load_scores()
    if not scores:
        print(f"[export_v2] 점수 파일 없음: {EVAL_V2_OUT}")
        return

    checklist = json.loads(GOLD_CHECKLIST_JSON.read_text(encoding="utf-8")) \
        if GOLD_CHECKLIST_JSON.exists() else {}
    gold_df = pd.read_pickle(GOLD_PKL)
    try:
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)
    except Exception:
        vital_map = {}

    # case(idx) → [(tag, rec)]
    cases = {}
    for tag, recs in scores.items():
        for r in recs:
            cases.setdefault(r.get("idx", 0), []).append((tag, r))

    # 케이스 정렬: 평균 composite 낮은(=문제 많은) 케이스를 위로
    def case_avg(idx):
        rs = [r for _, r in cases[idx]]
        cs = [r.get("composite", 0) for r in rs]
        return sum(cs) / len(cs) if cs else 0
    ordered = sorted(cases.keys(), key=case_avg)

    H = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>인계요약 v2 평가 — {esc(_RID)}</title>",
         f"<style>{CSS}</style></head><body>",
         "<h1>인계요약 모델 평가 v2.0 — 임상 검토용</h1>",
         f"<p class='meta'>Run: <code>{esc(_RID)}</code> · 케이스 {len(cases)} · "
         f"모델 {len(scores)} · 정렬: 안전축 우선(composite ↑, 안전위반 강조)</p>",
         "<p><b>축 안내</b>: COVERAGE(필수 이상소견 recall·안전핵심) · "
         "FAITHFULNESS(환각/사실일치) · BREVITY(노이즈 없음). "
         "<span class='b-viol badge'>안전위반</span>=이상소견 있는데 '특이사항 없음' 또는 고위험 누락.</p>"]

    n_reviewed = sum(1 for v in checklist.values() if v.get("reviewed"))
    if n_reviewed == 0 and checklist:
        H.append("<p class='meta'>⚠ checklist 전문의 검수 0건 — LLM 부트스트랩 잠정 기준입니다.</p>")

    for idx in ordered:
        rows = cases[idx]
        sid = rows[0][1].get("sid", -1)
        entry = checklist.get(str(sid), {})
        opname = entry.get("opname", "-")
        H.append(f"<h2 id='case-{idx}'>Case {idx} — {esc(opname)} "
                 f"<span class='meta'>(sid {sid}, 필수항목 {entry.get('items', []) and len(entry['items']) or 0}개"
                 f"{', NORMAL' if entry.get('is_normal_case') else ''})</span></h2>")

        # gold checklist 표시
        if entry.get("items"):
            H.append("<details><summary>Gold 필수 인계 항목</summary><pre>")
            for it in entry["items"]:
                H.append(f"- [{esc(it.get('severity'))}] {esc(it.get('finding'))} "
                         f"<span class='meta'>({esc(it.get('category'))})</span>")
            H.append("</pre></details>")

        # 점수표 (composite 내림차순, 안전위반은 하단/강조)
        rows_sorted = sorted(rows, key=lambda tr: (
            tr[1].get("safety_violation", False), -(tr[1].get("composite") or 0)))
        H.append("<table><thead><tr><th>모델</th><th>학습</th><th>COVERAGE</th>"
                 "<th>FAITH</th><th>BREVITY</th><th>COMPOSITE</th><th>안전</th>"
                 "<th>누락</th><th>환각</th><th>상태</th></tr></thead><tbody>")
        _safe_comps = [(r.get("composite") or 0) for _, r in rows
                       if not r.get("safety_violation")]
        best_comp = max(_safe_comps) if _safe_comps else None
        for tag, r in rows_sorted:
            model, exp = split_tag(tag)
            viol = r.get("safety_violation", False)
            cls = " class='viol'" if viol else (
                " class='best'" if (best_comp is not None and r.get("composite") == best_comp) else "")
            badge = "<span class='b-viol badge'>위반</span>" if viol else "<span class='b-ok badge'>ok</span>"
            H.append(
                f"<tr{cls}><td>{esc(model)}</td><td>{esc(exp)}</td>"
                f"<td>{fmt(r.get('coverage'))}</td><td>{fmt(r.get('faithfulness'))}</td>"
                f"<td>{fmt(r.get('brevity_v2'))}</td><td><b>{fmt(r.get('composite'))}</b></td>"
                f"<td>{badge}</td><td>{len(r.get('missed_items', []))}</td>"
                f"<td>{len(r.get('hallucinations', []))}</td>"
                f"<td class='meta'>{esc(r.get('gen_status', 'ok'))}</td></tr>")
        H.append("</tbody></table>")

        # 모델별 출력 + 누락/환각 상세
        H.append("<details><summary>모델 출력 + 누락/환각 상세 (클릭)</summary>")
        for tag, r in rows_sorted:
            model, exp = split_tag(tag)
            H.append(f"<h4>{esc(model)} · {esc(exp)} — composite={fmt(r.get('composite'))}"
                     f"{'  ⚠안전위반' if r.get('safety_violation') else ''}</h4>")
            H.append(f"<pre>{esc(r.get('generated_v2') or r.get('generated', ''))}</pre>")
            miss = r.get("missed_items", [])
            halu = r.get("hallucinations", [])
            noise = r.get("noise", [])
            if miss:
                H.append("<div class='miss'>누락: " +
                         "; ".join(esc(m.get("finding", "")) for m in miss) + "</div>")
            if halu:
                H.append("<div class='halu'>환각/모순: " +
                         "; ".join(esc(h.get("claim", "")) for h in halu) + "</div>")
            if noise:
                H.append("<div class='noise'>노이즈: " + "; ".join(esc(n) for n in noise) + "</div>")
        H.append("</details>")

    H.append("</body></html>")

    REPORT_V2_OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(cli.out) if cli.out else REPORT_V2_OUT / "results_cases_v2.html"
    out_path.write_text("".join(H), encoding="utf-8")
    print(f"[export_v2] 저장: {out_path}")


if __name__ == "__main__":
    main()
