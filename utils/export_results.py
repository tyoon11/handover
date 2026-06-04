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

import pickle
import pandas as pd

from config import (
    OUTPUT_BASE, EVAL_OUT, INFER_OUT,
    GOLD_PKL, GOLD_REF_PKL, VITAL_MAP_PKL,
    build_emr_text, build_user_prompt,
)


def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _load_inputs():
    """gold_df + vital_map + human ref. idx → (op_name, emr, vital, human) 매핑 반환."""
    inputs = {}
    try:
        gold_df = pd.read_pickle(GOLD_PKL)
    except Exception as e:
        print(f"  [경고] GOLD_PKL 로드 실패: {e}")
        return inputs

    try:
        ref_df = pd.read_pickle(GOLD_REF_PKL)
    except Exception:
        ref_df = None

    try:
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)
    except Exception:
        vital_map = {}

    for idx in range(len(gold_df)):
        row = gold_df.iloc[idx]
        # 수술명
        try:
            v = row[("수술", "수술명", "")]
            op_name = str(v.iloc[0]) if hasattr(v, "iloc") else str(v)
        except Exception:
            op_name = "-"
        # 수술 ID
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        # EMR + vital
        emr = build_emr_text(row)
        vital = vital_map.get(sid, "")
        # human reference
        human = "-"
        if ref_df is not None and idx < len(ref_df):
            try:
                h = ref_df.iloc[idx][("마취기록", "기록", "")]
                human = str(h)
                if human == "nan":
                    human = "-"
            except Exception:
                pass
        inputs[idx] = {
            "op_name": op_name,
            "sid": sid,
            "emr": emr,
            "vital": vital,
            "human_ref": human,
        }
    return inputs


def collect_results(eval_out: Path):
    """EVAL_OUT의 모든 *_scores.jsonl + (있으면) *_scores_scale.jsonl을 수집.
    각 샘플에 입력(EMR/vital/human ref/수술명)도 함께 담음."""
    inputs = _load_inputs()  # {idx: {op_name, sid, emr, vital, human_ref}}

    summary_rows = []
    detail_records = {}            # {model: [샘플 dict ...]}
    cases_long = []                # long-format: 1 row per (idx, model, exp)

    for score_file in sorted(eval_out.rglob("*_scores.jsonl")):
        if score_file.name.endswith("_scores_scale.jsonl"):
            continue

        tag = score_file.parent.name
        parts = tag.split("_", 1)
        model = parts[0]
        exp_key = parts[1] if len(parts) > 1 else "?"

        lines = [json.loads(l) for l in score_file.read_text().splitlines() if l.strip()]
        if not lines:
            continue

        # scale 머지
        scale_file = score_file.with_name(score_file.name.replace("_scores.jsonl", "_scores_scale.jsonl"))
        if not scale_file.exists():
            scale_file = score_file.parent / "gold_results_scale.jsonl"
        if scale_file.exists():
            scale_lines = [json.loads(l) for l in scale_file.read_text().splitlines() if l.strip()]
            scale_by_idx = {r["idx"]: r for r in scale_lines}
            for r in lines:
                sx = scale_by_idx.get(r["idx"], {})
                r["scale_large"] = sx.get("scale_large")
                r["scale_xl"] = sx.get("scale_xl")

        detail_records.setdefault(model, [])
        for r in lines:
            idx = r.get("idx")
            inp = inputs.get(idx, {})
            rec = {
                "experiment": exp_key,
                "idx": idx,
                "sid": r.get("sid") if r.get("sid") not in (None, -1) else inp.get("sid"),
                "op_name": inp.get("op_name", "-"),
                "brevity": r.get("brevity_score"),
                "critical": r.get("critical_score"),
                "sum": r.get("sum_score"),
                "scale_large": r.get("scale_large"),
                "scale_xl": r.get("scale_xl"),
                "EMR": inp.get("emr", ""),
                "vital": inp.get("vital", ""),
                "human_ref": inp.get("human_ref", "-"),
                "generated": r.get("generated") or "",
            }
            detail_records[model].append(rec)
            cases_long.append({"model": model, **rec})

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

    return summary_rows, detail_records, cases_long


def export_excel(summary_df: pd.DataFrame, detail_records: dict,
                 best_df: pd.DataFrame, cases_long: list, xlsx_path: Path):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[Excel] openpyxl 미설치 — pip install openpyxl 필요")
        return False

    detail_cols = [
        "experiment", "idx", "sid", "op_name",
        "brevity", "critical", "sum", "scale_large", "scale_xl",
        "EMR", "vital", "human_ref", "generated",
    ]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        best_df.to_excel(writer, sheet_name="PerModelBest", index=False)

        # Cases (long format): 1 row per (idx, model, experiment) — 가장 유용한 분석 시트
        if cases_long:
            cases_df = pd.DataFrame(cases_long)
            ordered = ["model"] + [c for c in detail_cols if c in cases_df.columns]
            cases_df = cases_df[ordered].sort_values(["idx", "model", "experiment"])
            cases_df.to_excel(writer, sheet_name="Cases", index=False)

        # 모델별 상세
        for model, recs in detail_records.items():
            df = pd.DataFrame(recs)
            df = df[[c for c in detail_cols if c in df.columns]]
            df = df.sort_values(["idx", "experiment"])
            sheet = f"{model[:28]}_details"
            df.to_excel(writer, sheet_name=sheet, index=False)

        # 셀 폭/줄바꿈 정리 (입력/출력 컬럼은 길어서 가독성 위해)
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                header = col[0].value
                if header in ("EMR", "vital", "human_ref", "generated"):
                    ws.column_dimensions[col[0].column_letter].width = 60
                    for cell in col[1:]:
                        cell.alignment = openpyxl.styles.Alignment(wrap_text=True, vertical="top")
                elif header in ("op_name",):
                    ws.column_dimensions[col[0].column_letter].width = 24
                else:
                    ws.column_dimensions[col[0].column_letter].width = 14

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


def export_cases_md(detail_records: dict, best_df: pd.DataFrame, md_path: Path, run_id: str,
                    n_top: int = 5, n_bottom: int = 2):
    """케이스별 입력+출력 비교 MD — 각 케이스마다 EMR/vital/human + 모든 모델 출력 나란히."""
    # idx 기준 모든 모델·실험을 모으기
    by_idx = {}
    for model, recs in detail_records.items():
        for r in recs:
            idx = r["idx"]
            by_idx.setdefault(idx, []).append({"model": model, **r})

    # 모든 case 정렬 (sum 평균 내림차순으로 좋은 케이스부터)
    case_avg_sum = {idx: _avg([r["sum"] for r in rs]) for idx, rs in by_idx.items()}
    sorted_idxs = sorted(case_avg_sum.keys(), key=lambda i: case_avg_sum[i] or 0, reverse=True)

    # 상위 N + 하위 K만 표시 (전체는 너무 길어짐)
    pick = sorted_idxs[:n_top]
    if n_bottom > 0:
        pick += sorted_idxs[-n_bottom:]

    lines = [
        f"# 케이스별 입력·출력 비교 ({len(pick)}건 발췌)\n",
        f"- **Run ID**: `{run_id}`\n",
        f"- **전체**: {len(sorted_idxs)} 케이스 중 상위 {n_top} + 하위 {n_bottom}\n\n",
    ]

    for idx in pick:
        rs = by_idx[idx]
        first = rs[0]
        lines.append(f"\n---\n\n## Case idx={idx} · sid={first.get('sid')} · {first.get('op_name', '-')}\n")
        lines.append(f"- **평균 SUM**: {case_avg_sum[idx]:.3f}\n\n")

        # 입력
        emr = (first.get("EMR") or "").strip()
        vital = (first.get("vital") or "").strip()
        human = (first.get("human_ref") or "-").strip()
        lines.append("### 입력 (EMR)\n```\n" + emr + "\n```\n\n")
        if vital and vital != "-":
            lines.append("### Vital 요약\n```\n" + vital + "\n```\n\n")
        lines.append("### Human Reference (정답)\n```\n" + human + "\n```\n\n")

        # 모델별 출력 (모델 × 실험 정렬, sum 내림차순)
        lines.append("### 모델 출력\n\n")
        rs_sorted = sorted(rs, key=lambda r: r.get("sum") or 0, reverse=True)
        for r in rs_sorted:
            score_str = (f"brev={r['brevity']} crit={r['critical']} **sum={r['sum']}**"
                         + (f"  scale_xl={r['scale_xl']:.3f}" if r.get("scale_xl") is not None else ""))
            lines.append(f"**{r['model']} / {r['experiment']}** — {score_str}\n")
            lines.append("```\n" + (r.get("generated") or "").strip() + "\n```\n\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"  [Cases MD] 저장: {md_path}")


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

    summary_rows, detail_records, cases_long = collect_results(eval_out)
    if not summary_rows:
        print("  ✗ 평가 결과 파일 없음")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    print(f"  수집: {len(summary_df)}개 실험, 모델 {len(detail_records)}개, 샘플 {len(cases_long)}건")

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
    export_excel(summary_df, detail_records, best_df, cases_long, xlsx_path)

    md_path = output_base / "results_notion.md"
    export_notion_md(summary_df, detail_records, best_df, md_path, run_id)

    cases_md = output_base / "results_cases.md"
    export_cases_md(detail_records, best_df, cases_md, run_id, n_top=5, n_bottom=2)

    print("\n[모델별 Best]")
    for _, r in best_df.iterrows():
        scale_str = f" scale_xl={r['scale_xl']:.4f}" if pd.notna(r.get('scale_xl')) else ""
        print(f"  {r['model']:<10} {r['experiment']:<16} sum={r['sum']:.4f}{scale_str}")


if __name__ == "__main__":
    main()
