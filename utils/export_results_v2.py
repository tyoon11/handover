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
body{font-family:-apple-system,'Segoe UI',sans-serif;max-width:1280px;margin:30px auto;padding:0 20px;line-height:1.55;color:#222}
h1{border-bottom:2px solid #333;padding-bottom:8px}
h2{background:#f0f4f8;padding:12px 16px;border-left:4px solid #2563eb;margin-top:46px;scroll-margin-top:10px}
h4{color:#374151;margin:14px 0 4px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border:1px solid #d1d5db;padding:5px 8px;text-align:left}
th{background:#f3f4f6}
.viol{background:#fee2e2}
.best{background:#dcfce7;font-weight:700}
.miss{color:#b91c1c}.halu{color:#b45309}.noise{color:#6b7280}
pre{background:#f8f9fa;border:1px solid #e1e4e8;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:13px}
details{margin:6px 0}
details summary{cursor:pointer;padding:6px 10px;background:#f9fafb;border-radius:4px;font-weight:600}
.meta{color:#6b7280;font-size:13px}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:700}
.b-viol{background:#dc2626;color:#fff}.b-ok{background:#16a34a;color:#fff}
/* 목차 */
.toc{columns:2;column-gap:30px;background:#fafafa;border:1px solid #eee;border-radius:8px;padding:14px 20px;font-size:13px}
.toc a{text-decoration:none;color:#2563eb}
.toc li{margin:3px 0;break-inside:avoid}
.toc .v{color:#dc2626;font-weight:700}
/* 메트릭 가이드 */
.guide{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px 20px;margin:16px 0}
.guide h3{margin:0 0 8px}
.guide table{font-size:13px;margin:6px 0}
.guide td:first-child{font-weight:700;white-space:nowrap}
.gold{background:#ecfdf5;border-left:4px solid #10b981;padding:10px 14px;margin:12px 0;border-radius:4px}
.casebar{font-size:13px;color:#374151;margin:2px 0 10px}
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

    # 케이스별 집계 (평균 composite)
    def case_stats(idx):
        rs = [r for _, r in cases[idx]]
        cs = [r.get("composite", 0) or 0 for r in rs]
        return (sum(cs) / len(cs) if cs else 0, len(rs))
    ordered = sorted(cases.keys(), key=lambda i: case_stats(i)[0])  # 문제 많은 케이스 위로

    from pipeline.eval_v2.checklist import gold_note_from_items

    H = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>인계요약 v2 평가 — {esc(_RID)}</title>",
         f"<style>{CSS}</style></head><body>",
         "<h1>인계요약 모델 평가 v2.0 — 임상 검토용</h1>",
         f"<p class='meta'>Run: <code>{esc(_RID)}</code> · 케이스 {len(cases)}건 · "
         f"모델×실험 {len(scores)}개 · 정렬: 평균 composite 낮은(문제 많은) 케이스부터</p>"]

    n_reviewed = sum(1 for v in checklist.values() if v.get("reviewed"))
    H.append(f"<p class='meta'>Gold 기준: 교수님 인계 피드백(c10)에서 추출한 case별 필수 항목. "
             f"채택(reviewed) {n_reviewed}/{len(checklist)}건"
             + ("" if n_reviewed else " — 잠정") + ".</p>")

    # ── 메트릭 가이드 ──
    H.append("""<div class='guide'><h3>📐 점수 읽는 법</h3>
<p class='meta'>세 축 모두 0~1, <b>높을수록 좋음</b>. 합산(sum)으로 뭉치지 않고 따로 봅니다.</p>
<table>
<tr><td>COVERAGE</td><td>Gold 필수 항목 중 모델이 <b>담은 비율</b>(단순 recall). <b>가장 중요한 축</b> — 놓치면 낮음.</td></tr>
<tr><td>FAITHFULNESS</td><td>출력 문장이 EMR <b>사실과 일치</b>하는 비율. 창작/모순(환각)이 많으면 낮음.</td></tr>
<tr><td>BREVITY</td><td>불필요한 설명·추론성 권고('~하니 ~해라')·행정 문구(약 잔량/이송) 같은 <b>노이즈가 없는</b> 정도.</td></tr>
<tr><td>COMPOSITE</td><td>0.5·COV + 0.3·FAITH + 0.2·BREV. coverage 비중이 가장 커서 항목을 놓치면 자연히 낮아짐.</td></tr>
</table>
<p class='meta'>표에서 <span class='best badge' style='background:#dcfce7;color:#166534'>초록</span>=그 케이스 최고 composite. 'NORMAL' 케이스는 정답이 '특이사항 없음'이라 그렇게 답하면 만점. (안전위반 같은 임의 게이트는 사용하지 않음 — coverage 점수로 직접 판단)</p>
</div>""")

    # ── 목차 ──
    H.append("<h2 style='background:none;border:0;padding:0'>케이스 목차</h2><ul class='toc'>")
    for idx in ordered:
        avg, n = case_stats(idx)
        sid = cases[idx][0][1].get("sid", -1)
        entry = checklist.get(str(sid), {})
        op = entry.get("opname", "-")
        tag = " · NORMAL" if entry.get("is_normal_case") else (
            " · gold없음" if entry.get("source") == "no_gold" else "")
        H.append(f"<li><a href='#case-{idx}'>Case {idx} — {esc(op)}</a> "
                 f"<span class='meta'>(avg {avg:.2f}{tag})</span></li>")
    H.append("</ul>")

    for idx in ordered:
        rows = cases[idx]
        sid = rows[0][1].get("sid", -1)
        entry = checklist.get(str(sid), {})
        opname = entry.get("opname", "-")
        avg, n = case_stats(idx)
        nitems = len(entry.get("items", []))
        flag = " · NORMAL(정답=특이사항 없음)" if entry.get("is_normal_case") else (
            " · gold없음" if entry.get("source") == "no_gold" else "")
        H.append(f"<h2 id='case-{idx}'>Case {idx} — {esc(opname)}</h2>")
        H.append(f"<div class='casebar'>sid {sid} · 필수항목 {nitems}개{flag} · "
                 f"평균 composite {avg:.2f} · 모델 {n}개</div>")

        # ── 입력 (EMR + Vital) — v1처럼 ──
        try:
            emr = build_emr_text(gold_df.iloc[idx]) if idx < len(gold_df) else ""
        except Exception:
            emr = ""
        vital = vital_map.get(sid, "") or vital_map.get(str(sid), "")
        H.append(f"<details><summary>입력 1) EMR (모델에게 준 원문)</summary><pre>{esc(emr)}</pre></details>")
        H.append("<details><summary>입력 2) Vital 요약</summary><pre>"
                 f"{esc(vital) if vital else '(없음)'}</pre></details>")

        # ── Gold 인계지 (정답) — 크게 표시 ──
        gold_note = entry.get("gold_note")
        if gold_note is None:
            gold_note = gold_note_from_items(entry)
        if entry.get("source") == "no_gold":
            gold_html = "<span class='meta'>(교수님 gold 없음 — coverage 평가 제외)</span>"
        else:
            gold_html = esc(gold_note) or "<span class='meta'>(없음)</span>"
        H.append(f"<div class='gold'><b>📋 Gold 인계지 (정답)</b><br>"
                 f"<span style='font-size:15px'>{gold_html}</span></div>")

        # gold checklist(항목) — 접어서 보조 표시
        if entry.get("items"):
            H.append("<details><summary>Gold 필수 인계 항목 (coverage 채점 기준)</summary><pre>")
            for it in entry["items"]:
                H.append(f"- [{esc(it.get('severity'))}] {esc(it.get('finding'))} "
                         f"<span class='meta'>({esc(it.get('category'))})</span>")
            H.append("</pre></details>")

        # 점수표 (composite 내림차순)
        rows_sorted = sorted(rows, key=lambda tr: -(tr[1].get("composite") or 0))
        H.append("<table><thead><tr><th>모델</th><th>학습</th><th>COVERAGE</th>"
                 "<th>FAITH</th><th>BREVITY</th><th>COMPOSITE</th>"
                 "<th>누락</th><th>환각</th><th>상태</th></tr></thead><tbody>")
        _comps = [(r.get("composite") or 0) for _, r in rows]
        best_comp = max(_comps) if _comps else None
        for tag, r in rows_sorted:
            model, exp = split_tag(tag)
            cls = " class='best'" if (best_comp is not None and r.get("composite") == best_comp) else ""
            H.append(
                f"<tr{cls}><td>{esc(model)}</td><td>{esc(exp)}</td>"
                f"<td>{fmt(r.get('coverage'))}</td><td>{fmt(r.get('faithfulness'))}</td>"
                f"<td>{fmt(r.get('brevity_v2'))}</td><td><b>{fmt(r.get('composite'))}</b></td>"
                f"<td>{len(r.get('missed_items', []))}</td>"
                f"<td>{len(r.get('hallucinations', []))}</td>"
                f"<td class='meta'>{esc(r.get('gen_status', 'ok'))}</td></tr>")
        H.append("</tbody></table>")

        # 모델별 출력 + 누락/환각 상세
        H.append("<details><summary>모델 출력 + 누락/환각 상세 (클릭)</summary>")
        for tag, r in rows_sorted:
            model, exp = split_tag(tag)
            H.append(f"<h4>{esc(model)} · {esc(exp)} — composite={fmt(r.get('composite'))}</h4>")
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
