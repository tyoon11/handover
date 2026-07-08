"""
report_v3.py — 결과 요약 (통계 포함) + HTML 리포트

E8 반영:
  - 변형별 3축 평균 + bootstrap 95% CI
  - baseline({model}_raw) 대비 paired permutation test (공통 유효 케이스만) + Holm 보정
  - macro/micro coverage 병기, 제외 케이스(judge 실패/no_gold) 별도 표
  - gold checklist 검수 상태('전문의 검수 n / 잠정채택 n / 잠정 n') 명시 (E6)

출력: {OUTPUT_BASE}/report/results_{split}_v3.{csv,md,html}
※ HTML에는 EMR 원문 미포함 (PHI 최소화) — 모델 출력·checklist 항목만.
"""

import argparse
import json

from .config_v3 import EVAL_OUT, REPORT_OUT, RUN_ID, STATS, ensure_dir
from .eval_v3.stats import holm_correction, paired_tests


def _load(split: str):
    """EVAL_OUT/*/ {split}_results_summary_v3.json + scores jsonl 로드."""
    rows, scores = {}, {}
    if not EVAL_OUT.exists():
        return rows, scores
    for d in sorted(EVAL_OUT.iterdir()):
        sf = d / f"{split}_results_summary_v3.json"
        jf = d / f"{split}_results_scores_v3.jsonl"
        if not sf.exists():
            continue
        summary = json.loads(sf.read_text(encoding="utf-8"))
        rows[summary["tag"]] = summary
        if jf.exists():
            scores[summary["tag"]] = [
                json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    return rows, scores


def _valid_map(recs):
    """{sid: composite} — official이 유효(excluded 아님)한 케이스만."""
    out = {}
    for r in recs:
        off = r.get("official") or {}
        if not off.get("excluded") and off.get("composite") is not None:
            out[r["sid"]] = off["composite"]
    return out


def pairwise_vs_raw(rows, scores):
    """모델별 {model}_raw 대비 각 변형의 paired permutation + Holm."""
    results = {}
    models = sorted({s["model"] for s in rows.values()})
    for m in models:
        base_tag = f"{m}_raw"
        if base_tag not in scores:
            continue
        base = _valid_map(scores[base_tag])
        pvals, details = {}, {}
        for tag, recs in scores.items():
            if not tag.startswith(m + "_") or tag == base_tag:
                continue
            comp = _valid_map(recs)
            common = sorted(set(base) & set(comp))
            if len(common) < 5:
                pvals[tag] = None
                details[tag] = dict(n_common=len(common))
                continue
            t = paired_tests([comp[s] for s in common], [base[s] for s in common],
                             iters=STATS["permutation_iters"], seed=STATS["seed"])
            pvals[tag] = t["permutation"]["p"]
            details[tag] = dict(n_common=len(common), mean_diff=t["mean_diff"],
                                permutation_p=t["permutation"]["p"],
                                t=t["t"], wilcoxon=t["wilcoxon"])
        holm = holm_correction(pvals, alpha=STATS["alpha"])
        for tag in details:
            details[tag]["holm"] = holm.get(tag)
        results[m] = details
    return results


def _fmt_ci(x):
    if not x or x.get("mean") is None:
        return "-"
    if x.get("lo") is None:
        return f"{x['mean']:.3f}"
    return f"{x['mean']:.3f} [{x['lo']:.3f},{x['hi']:.3f}]"


def write_outputs(split, rows, pw):
    out_dir = ensure_dir(REPORT_OUT)
    # CSV
    import csv
    csv_path = out_dir / f"results_{split}_v3.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tag", "model", "experiment", "n_valid", "n_excluded",
                    "composite", "composite_lo", "composite_hi",
                    "coverage", "micro_coverage", "faithfulness", "brevity",
                    "gate_missed_abnormal", "gate_degenerate", "no_issue_rate",
                    "judge_agreement_rho", "vs_raw_diff", "vs_raw_p_holm",
                    "significant"])
        for tag, s in sorted(rows.items()):
            d = pw.get(s["model"], {}).get(tag, {})
            holm = d.get("holm") or {}
            w.writerow([
                tag, s["model"], s["experiment"], s["n_valid"], s["n_excluded"],
                s["composite"]["mean"], s["composite"]["lo"], s["composite"]["hi"],
                s["coverage"]["mean"], s["micro"]["micro_coverage"],
                s["faithfulness"]["mean"], s["brevity"]["mean"],
                s["gates"]["missed_abnormal"], s["gates"]["degenerate"],
                s["no_issue_rate"], s["judge_agreement"]["spearman"],
                d.get("mean_diff"), holm.get("p_adj"), holm.get("significant"),
            ])

    # MD
    md = [f"# v3 결과 요약 — split={split}\n",
          f"run_id: `{RUN_ID}`\n"]
    rev = next(iter(rows.values()), {}).get("checklist_review")
    if rev:
        md.append(f"\n**gold checklist 상태**: 전문의검수 {rev['reviewed']} / "
                  f"잠정채택 {rev['accepted_without_review']} / 잠정 {rev['provisional']} "
                  f"(검수 0이면 결과는 잠정치)\n")
    md.append("\n| tag | n(유효/제외) | composite [95% CI] | coverage(macro/micro) "
              "| faith | brev | 안전게이트 | vs raw Δ (Holm p) |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for tag, s in sorted(rows.items(), key=lambda kv: (kv[1]["model"],
                                                       -(kv[1]["composite"]["mean"] or 0))):
        d = pw.get(s["model"], {}).get(tag, {})
        holm = d.get("holm") or {}
        sig = " **✓**" if holm.get("significant") else ""
        vs = (f"{d.get('mean_diff'):+0.3f} (p={holm.get('p_adj')}){sig}"
              if d.get("mean_diff") is not None else "-")
        micro = s["micro"]["micro_coverage"]
        md.append(
            f"| {tag} | {s['n_valid']}/{s['n_excluded']} | {_fmt_ci(s['composite'])} "
            f"| {s['coverage']['mean']}/{micro} | {s['faithfulness']['mean']} "
            f"| {s['brevity']['mean']} | 누락게이트 {s['gates']['missed_abnormal']}건 "
            f"| {vs} |\n")
    md.append("\n- composite = 0.5·coverage + 0.3·faithfulness + 0.2·brevity "
              "(교차 judge 평균; 대상 모델과 같은 family judge는 제외)\n"
              "- '제외'는 judge 실패/gold 부재 케이스 — 점수가 아니라 인프라 문제로 따로 센다\n"
              "- vs raw: 공통 유효 케이스 paired permutation test, Holm 보정\n")
    (out_dir / f"results_{split}_v3.md").write_text("".join(md), encoding="utf-8")

    # HTML (간이 — PHI 없는 요약만)
    html = ["<meta charset='utf-8'><title>v3 results</title>",
            "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;"
            "padding:4px 8px;font-size:13px}th{background:#f0f0f0}</style>",
            f"<h1>v3 결과 — split={split} (run {RUN_ID})</h1>"]
    if rev:
        html.append(f"<p><b>gold checklist</b>: 전문의검수 {rev['reviewed']} / "
                    f"잠정채택 {rev['accepted_without_review']} / 잠정 {rev['provisional']}"
                    "</p>")
    html.append("<table><tr><th>tag</th><th>n(유효/제외)</th><th>composite [CI]</th>"
                "<th>coverage</th><th>micro</th><th>faith</th><th>brev</th>"
                "<th>누락게이트</th><th>붕괴</th><th>vs raw (Holm)</th></tr>")
    for tag, s in sorted(rows.items(), key=lambda kv: (kv[1]["model"],
                                                       -(kv[1]["composite"]["mean"] or 0))):
        d = pw.get(s["model"], {}).get(tag, {})
        holm = d.get("holm") or {}
        vs = (f"{d.get('mean_diff'):+0.3f} (p={holm.get('p_adj')})"
              + ("✓" if holm.get("significant") else "")
              if d.get("mean_diff") is not None else "-")
        html.append(
            f"<tr><td>{tag}</td><td>{s['n_valid']}/{s['n_excluded']}</td>"
            f"<td>{_fmt_ci(s['composite'])}</td><td>{s['coverage']['mean']}</td>"
            f"<td>{s['micro']['micro_coverage']}</td><td>{s['faithfulness']['mean']}</td>"
            f"<td>{s['brevity']['mean']}</td><td>{s['gates']['missed_abnormal']}</td>"
            f"<td>{s['gates']['degenerate']}</td><td>{vs}</td></tr>")
    html.append("</table>")
    (out_dir / f"results_{split}_v3.html").write_text("".join(html), encoding="utf-8")

    print(f"[report_v3] 저장: {out_dir}/results_{split}_v3.(csv|md|html)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["gold", "dev"], default="dev")
    args = ap.parse_args()
    rows, scores = _load(args.split)
    if not rows:
        print(f"[report_v3] {args.split} 평가 요약 없음 — evaluate 먼저 실행")
        return
    pw = pairwise_vs_raw(rows, scores)
    write_outputs(args.split, rows, pw)


if __name__ == "__main__":
    main()
