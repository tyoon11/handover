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
    GOLD_PKL, VITAL_MAP_PKL, GOLD_REF_XLSX,
    SFT_MODELS,
    build_emr_text, build_user_prompt,
)

# 폴더명(f"{model}_{exp_key}")에서 model/exp 분리.
# model 키 자체에 '_'가 있을 수 있어(gemma4_31b) 단순 split이 아니라
# 알려진 모델 키 중 긴 것부터 prefix 매칭한다.
_MODEL_KEYS = sorted(SFT_MODELS.keys(), key=len, reverse=True)


def _split_tag(tag: str):
    for mk in _MODEL_KEYS:
        if tag == mk:
            return mk, "?"
        if tag.startswith(mk + "_"):
            return mk, tag[len(mk) + 1:]
    parts = tag.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "?")


def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _load_gemma_ref_map():
    """KHS 엑셀에서 {수술ID: gemma-3-27b 참고용 인계요약 초안} 매핑 로드.
    주의: 사람이 쓴 정답이 아니라 gemma-3-27b 생성 초안(참고용)."""
    ref_map = {}
    try:
        gx = pd.read_excel(GOLD_REF_XLSX, header=[0, 1, 2])
    except Exception as e:
        print(f"  [경고] 참고 인계요약 엑셀 로드 실패: {e}")
        return ref_map

    # 수술 ID 컬럼
    sid_col = next((c for c in gx.columns if c[0] == "수술 ID"), None)
    # '참고용' (gemma-3-27b) 컬럼 — 그룹 '인계요약지_sample_from_LLM' 중 채워진 게 많은 것
    cand = [c for c in gx.columns if c[0] == "인계요약지_sample_from_LLM"]
    ref_col = None
    best = -1
    for c in cand:
        n = gx[c].apply(lambda v: isinstance(v, str) and v.strip() not in ("", "-", "nan")).sum()
        if n > best:
            best, ref_col = n, c
    if sid_col is None or ref_col is None:
        print("  [경고] 참고 인계요약 컬럼을 찾지 못함")
        return ref_map

    for i in range(len(gx)):
        sid_v = gx[sid_col].iloc[i]
        sid = None
        try:
            sid = int(sid_v.iloc[0]) if hasattr(sid_v, "iloc") else int(sid_v)
        except Exception:
            try:
                sid = int(str(sid_v).strip())
            except Exception:
                continue
        v = gx[ref_col].iloc[i]
        if isinstance(v, str) and v.strip() not in ("", "-", "nan"):
            ref_map[sid] = v.strip()
    print(f"  참고 인계요약(gemma-3-27b 초안): {len(ref_map)}건 로드")
    return ref_map


def _load_inputs():
    """gold_df + vital_map + 참고 인계요약(gemma 초안). idx → 매핑 반환.
    원본 마취기록은 입력 EMR에 이미 포함되므로 별도 표시하지 않음."""
    inputs = {}
    try:
        gold_df = pd.read_pickle(GOLD_PKL)
    except Exception as e:
        print(f"  [경고] GOLD_PKL 로드 실패: {e}")
        return inputs

    try:
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)
    except Exception:
        vital_map = {}

    gemma_ref = _load_gemma_ref_map()

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
        emr = build_emr_text(row)
        vital = vital_map.get(sid, "")
        ref_draft = gemma_ref.get(sid, "-")
        inputs[idx] = {
            "op_name": op_name,
            "sid": sid,
            "emr": emr,
            "vital": vital,
            "ref_draft": ref_draft,
        }
    return inputs


def collect_results(eval_out: Path):
    """EVAL_OUT의 모든 *_scores.jsonl + (있으면) *_scores_scale.jsonl을 수집.
    각 샘플에 입력(EMR/vital/human ref/수술명)도 함께 담음."""
    inputs = _load_inputs()  # {idx: {op_name, sid, emr, vital, ref_draft}}

    summary_rows = []
    detail_records = {}            # {model: [샘플 dict ...]}
    cases_long = []                # long-format: 1 row per (idx, model, exp)

    for score_file in sorted(eval_out.rglob("*_scores.jsonl")):
        if score_file.name.endswith("_scores_scale.jsonl"):
            continue

        tag = score_file.parent.name
        model, exp_key = _split_tag(tag)

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
                "ref_draft": inp.get("ref_draft", "-"),
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
        "EMR", "vital", "ref_draft", "generated",
    ]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        best_df.to_excel(writer, sheet_name="PerModelBest", index=False)

        # Cases (long format): 1 row per (idx, model, experiment) — 가장 유용한 분석 시트
        if cases_long:
            cases_df = pd.DataFrame(cases_long)
            ordered = ["model"] + [c for c in detail_cols if c in cases_df.columns]
            cases_df = cases_df[ordered].sort_values(["idx", "model", "experiment"])
            if "generated" in cases_df.columns:
                cases_df["generated"] = cases_df["generated"].map(_collapse_repeats)
            cases_df.to_excel(writer, sheet_name="Cases", index=False)

        # 모델별 상세
        for model, recs in detail_records.items():
            df = pd.DataFrame(recs)
            df = df[[c for c in detail_cols if c in df.columns]]
            df = df.sort_values(["idx", "experiment"])
            if "generated" in df.columns:
                df["generated"] = df["generated"].map(_collapse_repeats)
            sheet = f"{model[:28]}_details"
            df.to_excel(writer, sheet_name=sheet, index=False)

        # 셀 폭/줄바꿈 정리 (입력/출력 컬럼은 길어서 가독성 위해)
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                header = col[0].value
                if header in ("EMR", "vital", "ref_draft", "generated"):
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


# ── 학습 방식 한글 이름 ──────────────────────────────────────────────────
EXP_LABELS = {
    "raw":          "Raw (no training)",
    "sft_1ep":      "SFT 1 epoch",
    "sft_3ep":      "SFT 3 epoch",
    "rlaif_dpo":    "DPO (from raw)",
    "rlaif_simpo":  "SimPO (from raw)",
    "sft_1ep_dpo":  "SFT 1ep → DPO",
    "sft_3ep_dpo":  "SFT 3ep → DPO",
}


def _exp_label(exp_key: str) -> str:
    return EXP_LABELS.get(exp_key, exp_key)


def _fmt_score(v, digits=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


import re as _re

# 반복 단위 탐지: 1~40자 유닛이 4회 이상(원본 + 3회) 연속
_RE_REPEAT = _re.compile(r"(.{1,40}?)(?:\1){3,}", _re.DOTALL)
# 단일 문자 긴 런 (예: "......", "----")
_RE_CHAR_RUN = _re.compile(r"(.)\1{14,}", _re.DOTALL)


def _collapse_repeats(text: str) -> str:
    """degenerate 반복 출력(hari 등)을 보기 쉽게 축약.
    예: '(34.5°C) (34.5°C) (34.5°C) ...' → '(34.5°C)  …(×38회 반복 생략)'"""
    if not text:
        return text

    def _rep_sub(m):
        unit = m.group(1)
        total = len(m.group(0))
        count = total // len(unit) if unit else 0
        if count < 4:
            return m.group(0)
        return f"{unit} …(×{count}회 반복 생략)"

    text = _RE_REPEAT.sub(_rep_sub, text)

    def _char_sub(m):
        ch = m.group(1)
        n = len(m.group(0))
        return f"{ch * 3}…(×{n})"

    text = _RE_CHAR_RUN.sub(_char_sub, text)
    return text


def _rank_key(r):
    """정렬 키: sum 우선, 동점이면 scale_xl로 2차 정렬.
    judge 점수가 5/5로 포화되는 케이스가 많아 scale_xl이 실질 구분자."""
    return (r.get("sum") or 0, r.get("scale_xl") or 0)


def _group_by_idx(detail_records):
    """{idx: [{model, experiment, ...}, ...]} 형태로 재정리."""
    by_idx = {}
    for model, recs in detail_records.items():
        for r in recs:
            idx = r["idx"]
            by_idx.setdefault(idx, []).append({"model": model, **r})
    return by_idx


def export_cases_md(detail_records: dict, md_path: Path, run_id: str,
                    only_idxs=None):
    """케이스별 입력·출력 전체 비교 MD — 임상 검토용 템플릿.
    각 케이스마다 EMR/vital/human + 모든 모델·실험 출력 + 점수 표.
    only_idxs로 일부 케이스만 선택 가능 (기본: 전체)."""
    by_idx = _group_by_idx(detail_records)
    case_avg = {idx: _avg([r["sum"] for r in rs]) for idx, rs in by_idx.items()}
    all_idxs = sorted(by_idx.keys(), key=lambda i: case_avg[i] or 0, reverse=True)
    pick = [i for i in all_idxs if (only_idxs is None or i in only_idxs)]

    lines = [
        f"# 인계요약 모델 출력 비교 — 임상 검토용\n",
        f"- **Run ID**: `{run_id}`\n",
        f"- **케이스 수**: {len(pick)}\n",
        f"- **점수 안내**: `brevity`(간결성), `critical`(중요정보 포함도), `sum`(합), `scale_xl`(factual consistency 0~1)\n\n",
        "## 케이스 목차\n\n",
    ]
    for idx in pick:
        first = by_idx[idx][0]
        lines.append(f"- [Case {idx} — {first.get('op_name', '-')}](#case-{idx}) (평균 SUM: {case_avg[idx]:.2f})\n")
    lines.append("\n")

    for idx in pick:
        rs = by_idx[idx]
        first = rs[0]
        op = first.get("op_name", "-")
        sid = first.get("sid", "-")
        lines.append(f"\n---\n\n## <a id=\"case-{idx}\"></a>Case {idx} — {op}\n")
        lines.append(f"- **수술 ID (sid)**: `{sid}`  \n")
        lines.append(f"- **평균 SUM**: **{case_avg[idx]:.2f}**\n\n")

        # 입력
        emr = (first.get("EMR") or "").strip()
        vital = (first.get("vital") or "").strip()
        ref = (first.get("ref_draft") or "-").strip()

        lines.append("### 입력 1) EMR\n```\n" + emr + "\n```\n\n")
        if vital and vital != "-":
            lines.append("### 입력 2) Vital 요약\n```\n" + vital + "\n```\n\n")
        lines.append("### 참고 인계요약 (gemma-3-27b 초안, 정답 아님)\n```\n" + ref + "\n```\n\n")

        # 점수 표 (모델 × 학습방식)
        lines.append("### 점수 표 (sum 내림차순, 동점 시 scale_xl 우선)\n\n")
        lines.append("| 모델 | 학습 방식 | brevity | critical | **sum** | scale_xl |\n")
        lines.append("|---|---|---|---|---|---|\n")
        rs_sorted = sorted(rs, key=_rank_key, reverse=True)
        for r in rs_sorted:
            lines.append(
                f"| {r['model']} | {_exp_label(r['experiment'])} | "
                f"{_fmt_score(r['brevity'])} | {_fmt_score(r['critical'])} | "
                f"**{_fmt_score(r['sum'])}** | {_fmt_score(r.get('scale_xl'), 3)} |\n"
            )
        lines.append("\n")

        # 모델별 출력 본문 (각 출력 <details> 토글 — 노션/깃헙 렌더 지원)
        lines.append("### 모델 출력 (점수 높은 순, 클릭하여 펼치기)\n\n")
        for rank, r in enumerate(rs_sorted, 1):
            score = (f"sum=**{_fmt_score(r['sum'])}** "
                     f"(brev={_fmt_score(r['brevity'])}, "
                     f"crit={_fmt_score(r['critical'])}"
                     + (f", scale_xl={_fmt_score(r.get('scale_xl'), 3)}" if r.get('scale_xl') is not None else "")
                     + ")")
            gen = _collapse_repeats((r.get("generated") or "").strip())
            lines.append(
                f"<details><summary>{rank}위. {r['model']} · {_exp_label(r['experiment'])} — {score}</summary>\n\n"
                + "```\n" + gen + "\n```\n\n</details>\n\n"
            )

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"  [Cases MD] 저장: {md_path}  ({len(pick)} 케이스)")


def export_cases_html(detail_records: dict, html_path: Path, run_id: str):
    """검토용 단일 HTML — 토글/스타일 적용으로 브라우저에서 보기 쉬움."""
    by_idx = _group_by_idx(detail_records)
    case_avg = {idx: _avg([r["sum"] for r in rs]) for idx, rs in by_idx.items()}
    all_idxs = sorted(by_idx.keys(), key=lambda i: case_avg[i] or 0, reverse=True)

    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    css = """
    body { font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 1200px; margin: 30px auto; padding: 0 20px; line-height: 1.5; color: #222; }
    h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
    h2 { background: #f0f4f8; padding: 12px 16px; border-left: 4px solid #2563eb; margin-top: 40px; }
    h3 { color: #1e40af; margin-top: 20px; }
    h4 { color: #4b5563; margin-bottom: 4px; }
    pre { background: #f8f9fa; border: 1px solid #e1e4e8; padding: 12px; border-radius: 6px;
          white-space: pre-wrap; word-wrap: break-word; font-size: 13px; line-height: 1.5; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
    th, td { border: 1px solid #d1d5db; padding: 6px 10px; text-align: left; }
    th { background: #f3f4f6; font-weight: 600; }
    .score-best { background: #dcfce7; font-weight: 700; }
    .toc a { text-decoration: none; color: #2563eb; }
    .meta { color: #6b7280; font-size: 14px; }
    details { margin: 8px 0; }
    details summary { cursor: pointer; padding: 6px 10px; background: #f9fafb; border-radius: 4px; font-weight: 600; }
    details[open] summary { background: #e0f2fe; }
    """

    out = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>인계요약 모델 비교 — {run_id}</title>",
           f"<style>{css}</style></head><body>",
           f"<h1>인계요약 모델 출력 비교 — 임상 검토용</h1>",
           f"<p class='meta'>Run ID: <code>{run_id}</code> · 총 {len(all_idxs)} 케이스 · "
           f"생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
           "<p><b>점수 안내</b>: brevity(간결성 1~5), critical(중요정보 1~5), sum(합 2~10), "
           "scale_xl(factual consistency 0~1)</p>",
           "<h2>케이스 목차</h2><ul class='toc'>"]
    for idx in all_idxs:
        first = by_idx[idx][0]
        out.append(f"<li><a href='#case-{idx}'>Case {idx} — {esc(first.get('op_name', '-'))}</a> "
                   f"<span class='meta'>(평균 SUM: {case_avg[idx]:.2f})</span></li>")
    out.append("</ul>")

    for idx in all_idxs:
        rs = by_idx[idx]
        first = rs[0]
        out.append(f"<h2 id='case-{idx}'>Case {idx} — {esc(first.get('op_name', '-'))}</h2>")
        out.append(f"<p class='meta'>수술 ID: <code>{first.get('sid')}</code> · "
                   f"평균 SUM: <b>{case_avg[idx]:.2f}</b></p>")

        # 입력 (토글)
        emr = (first.get("EMR") or "").strip()
        vital = (first.get("vital") or "").strip()
        ref = (first.get("ref_draft") or "-").strip()

        out.append(f"<details open><summary>입력 1) EMR</summary><pre>{esc(emr)}</pre></details>")
        if vital and vital != "-":
            out.append(f"<details><summary>입력 2) Vital 요약</summary><pre>{esc(vital)}</pre></details>")
        out.append("<details open><summary>참고 인계요약 (gemma-3-27b 초안 · 정답 아님)</summary>"
                   f"<pre>{esc(ref)}</pre></details>")

        # 점수 표
        out.append("<h3>점수 표 (sum 내림차순, 동점 시 scale_xl 우선)</h3>")
        out.append("<table><thead><tr><th>모델</th><th>학습 방식</th><th>brevity</th>"
                   "<th>critical</th><th>sum</th><th>scale_xl</th></tr></thead><tbody>")
        rs_sorted = sorted(rs, key=_rank_key, reverse=True)
        best_sum = rs_sorted[0].get("sum") if rs_sorted else None
        for r in rs_sorted:
            cls = " class='score-best'" if r.get("sum") == best_sum else ""
            out.append(
                f"<tr{cls}><td>{esc(r['model'])}</td><td>{esc(_exp_label(r['experiment']))}</td>"
                f"<td>{_fmt_score(r['brevity'])}</td><td>{_fmt_score(r['critical'])}</td>"
                f"<td><b>{_fmt_score(r['sum'])}</b></td>"
                f"<td>{_fmt_score(r.get('scale_xl'), 3)}</td></tr>"
            )
        out.append("</tbody></table>")

        # 출력 본문 (각 출력 토글, 상위 3개는 기본 펼침)
        out.append("<h3>모델 출력 (클릭하여 펼치기)</h3>")
        for rank, r in enumerate(rs_sorted, 1):
            gen = _collapse_repeats((r.get("generated") or "").strip())
            open_attr = " open" if rank <= 3 else ""
            summary = (f"{rank}위. {esc(r['model'])} · {esc(_exp_label(r['experiment']))} "
                       f"— sum={_fmt_score(r['sum'])}, "
                       f"brev={_fmt_score(r['brevity'])}, crit={_fmt_score(r['critical'])}"
                       + (f", scale_xl={_fmt_score(r.get('scale_xl'), 3)}" if r.get('scale_xl') is not None else ""))
            out.append(f"<details{open_attr}><summary>{summary}</summary>"
                       f"<pre>{esc(gen)}</pre></details>")

    out.append("</body></html>")
    html_path.write_text("".join(out), encoding="utf-8")
    print(f"  [HTML] 저장: {html_path}  ({len(all_idxs)} 케이스)")


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

    # 임상 검토용 — 전체 케이스 입력/출력/학습방식/점수 비교
    cases_md = output_base / "results_cases.md"
    export_cases_md(detail_records, cases_md, run_id)

    cases_html = output_base / "results_cases.html"
    export_cases_html(detail_records, cases_html, run_id)

    print("\n[모델별 Best]")
    for _, r in best_df.iterrows():
        scale_str = f" scale_xl={r['scale_xl']:.4f}" if pd.notna(r.get('scale_xl')) else ""
        print(f"  {r['model']:<10} {r['experiment']:<16} sum={r['sum']:.4f}{scale_str}")


if __name__ == "__main__":
    main()
