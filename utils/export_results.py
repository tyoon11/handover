"""
utils/export_results.py — 평가 결과를 Excel/Notion 포맷으로 정리

실행:
  HANDOVER_RUN_ID=20260528_023711 python utils/export_results.py
  # 또는
  python utils/export_results.py --run_id 20260528_023711

산출물 (outputs/<run_id>/ 아래):
  - results.xlsx        — 다중 시트 (Summary, PerModelBest, <model>_details ...)
  - results_notion.md   — 노션에 붙여넣기 좋은 마크다운
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import pandas as pd

from config import OUTPUT_BASE, EVAL_OUT, INFER_OUT


def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def collect_results(eval_out: Path):
    """EVAL_OUT의 모든 *_scores.jsonl + (있으면) *_scores_scale.jsonl을 수집."""
    summary_rows = []        # model × exp 집계
    detail_records = {}      # {model: [모든 샘플 dict]}

    for score_file in sorted(eval_out.rglob("*_scores.jsonl")):
        # *_scores_scale.jsonl 은 별도 처리, 여기선 제외
        if score_file.name.endswith("_scores_scale.jsonl"):
            continue

        tag = score_file.parent.name           # "llama_sft_1ep_dpo"
        parts = tag.split("_", 1)
        model = parts[0]
        exp_key = parts[1] if len(parts) > 1 else "?"

        # judge 점수 로드
        lines = [json.loads(l) for l in score_file.read_text().splitlines() if l.strip()]
        if not lines:
            continue

        # scale 파일도 있으면 merge
        scale_file = score_file.with_name(score_file.name.replace("_scores.jsonl", "_scores_scale.jsonl"))
        if not scale_file.exists():
            # 대안 위치: 같은 폴더의 gold_results_scale.jsonl
            scale_file = score_file.parent / "gold_results_scale.jsonl"
        if scale_file.exists():
            scale_lines = [json.loads(l) for l in scale_file.read_text().splitlines() if l.strip()]
            scale_by_idx = {r["idx"]: r for r in scale_lines}
            for r in lines:
                sx = scale_by_idx.get(r["idx"], {})
                r["scale_large"] = sx.get("scale_large")
                r["scale_xl"] = sx.get("scale_xl")

        # 샘플별 상세 보관
        detail_records.setdefault(model, [])
        for r in lines:
            detail_records[model].append({
                "experiment": exp_key,
                "idx": r.get("idx"),
                "sid": r.get("sid"),
                "brevity": r.get("brevity_score"),
                "critical": r.get("critical_score"),
                "sum": r.get("sum_score"),
                "scale_large": r.get("scale_large"),
                "scale_xl": r.get("scale_xl"),
                "generated": (r.get("generated") or "")[:500],
            })

        # 집계
        summary_rows.append({
            "model": model,
            "experiment": exp_key,
            "n": len(lines),
            "brevity": _avg([r.get("brevity_score") for r in lines]),
            "critical": _avg([r.get("critical_score") for r in lines]),
            "sum": _avg([r.get("sum_score") for r in lines]),
            "scale_large": _avg([r.get("scale_large") for r in lines]),
            "scale_xl": _avg([r.get("scale_xl") for r in lines]),
        })

    return summary_rows, detail_records


def export_excel(summary_df: pd.DataFrame, detail_records: dict, best_df: pd.DataFrame, xlsx_path: Path):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[Excel] openpyxl 미설치 — pip install openpyxl 필요")
        return False

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        best_df.to_excel(writer, sheet_name="PerModelBest", index=False)
        for model, recs in detail_records.items():
            df = pd.DataFrame(recs).sort_values(["experiment", "idx"])
            sheet = f"{model[:28]}_details"
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"  [Excel] 저장: {xlsx_path}")
    return True


def export_notion_md(summary_df: pd.DataFrame, detail_records: dict, best_df: pd.DataFrame,
                     md_path: Path, run_id: str):
    lines = []
    lines.append(f"# 인계요약 모델 평가 결과\n")
    lines.append(f"- **Run ID**: `{run_id}`\n")
    lines.append(f"- **생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"- **총 실험**: {len(summary_df)}개\n\n")

    # ── 전체 요약 ────────────────────────────────────────────────────────
    lines.append("## 전체 요약\n\n")
    cols = [c for c in ["model", "experiment", "n", "brevity", "critical", "sum",
                        "scale_large", "scale_xl"] if c in summary_df.columns]
    has_scale = "scale_large" in summary_df.columns and summary_df["scale_large"].notna().any()
    sorted_df = summary_df.sort_values(["model", "sum"], ascending=[True, False])
    lines.append("| " + " | ".join(cols) + " |\n")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
    for _, r in sorted_df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |\n")

    # ── 모델별 best ──────────────────────────────────────────────────────
    lines.append("\n## 모델별 Best 실험\n\n")
    best_cols = [c for c in best_df.columns if c in ("model", "experiment", "sum", "brevity",
                                                      "critical", "scale_large", "scale_xl")]
    lines.append("| " + " | ".join(best_cols) + " |\n")
    lines.append("|" + "|".join(["---"] * len(best_cols)) + "|\n")
    for _, r in best_df.iterrows():
        cells = []
        for c in best_cols:
            v = r[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |\n")

    # ── 모델별 실험 상세 평균 ────────────────────────────────────────────
    lines.append("\n## 모델별 실험 평균\n\n")
    for model, g in sorted_df.groupby("model"):
        lines.append(f"### {model}\n\n")
        lines.append("| experiment | n | brevity | critical | sum |")
        if has_scale:
            lines.append(" scale_large | scale_xl |")
        lines.append("\n")
        sep = "|---|---|---|---|---|"
        if has_scale:
            sep += "---|---|"
        lines.append(sep + "\n")
        for _, r in g.iterrows():
            sl = f"{r['scale_large']:.4f}" if has_scale and pd.notna(r.get('scale_large')) else "-"
            sx = f"{r['scale_xl']:.4f}" if has_scale and pd.notna(r.get('scale_xl')) else "-"
            base = f"| {r['experiment']} | {r['n']} | {r['brevity']:.4f} | {r['critical']:.4f} | **{r['sum']:.4f}** |"
            if has_scale:
                base += f" {sl} | {sx} |"
            lines.append(base + "\n")
        lines.append("\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"  [Notion MD] 저장: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="평가 결과 Excel/Notion 정리")
    parser.add_argument("--run_id", type=str, default=None,
                        help="HANDOVER_RUN_ID 환경변수 또는 인자로 지정")
    args = parser.parse_args()

    if args.run_id:
        import os
        os.environ["HANDOVER_RUN_ID"] = args.run_id
        # config 다시 로드 (이미 로드됐을 수 있음 — 그래서 BASE 경로 직접 계산)
        from config import WORK_DIR
        eval_out = WORK_DIR / "outputs" / args.run_id / "evaluation"
        output_base = WORK_DIR / "outputs" / args.run_id
    else:
        eval_out = EVAL_OUT
        output_base = OUTPUT_BASE

    print(f"[Export] EVAL_OUT = {eval_out}")
    if not eval_out.exists():
        print(f"  ✗ 폴더 없음: {eval_out}")
        sys.exit(1)

    summary_rows, detail_records = collect_results(eval_out)
    if not summary_rows:
        print("  ✗ 평가 결과 파일 없음")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    print(f"  수집: {len(summary_df)}개 실험, 모델 {len(detail_records)}개")

    # 모델별 best (sum 기준)
    best_df = (
        summary_df.sort_values(["model", "sum"], ascending=[True, False])
        .groupby("model", as_index=False)
        .first()
    )

    # ── 산출 ────────────────────────────────────────────────────────────
    run_id = args.run_id or (eval_out.parent.name)
    output_base.mkdir(parents=True, exist_ok=True)

    csv_path = output_base / "results.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [CSV] 저장: {csv_path}")

    xlsx_path = output_base / "results.xlsx"
    export_excel(summary_df, detail_records, best_df, xlsx_path)

    md_path = output_base / "results_notion.md"
    export_notion_md(summary_df, detail_records, best_df, md_path, run_id)

    print("\n[모델별 Best]")
    for _, r in best_df.iterrows():
        scale_str = f" scale_xl={r['scale_xl']:.4f}" if pd.notna(r.get('scale_xl')) else ""
        print(f"  {r['model']:<10} {r['experiment']:<16} sum={r['sum']:.4f}{scale_str}")


if __name__ == "__main__":
    main()
