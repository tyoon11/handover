"""
run_all.py — 전체 파이프라인 오케스트레이션
outputs/{timestamp}/ 아래에 모든 결과 저장

실행:
  python run_all.py --models llama qwen35 --gpus 4,5,6,7
  python run_all.py --models llama --gpus 4,5,6,7 --skip_done
  python run_all.py --models llama --gpus 4,5,6,7 --only_eval
  python run_all.py --models llama --gpus 4,5,6,7 --experiments raw sft_3ep
  python run_all.py --run_id 20260513_032935 --summarize   # 기존 결과 요약만
"""

import os, sys, argparse, subprocess, time, json, pickle
from pathlib import Path
from datetime import datetime


# ── GPU 조기 파싱 ─────────────────────────────────────────────────────────
def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    return p.parse_known_args()[0]


_early = _early_gpus()
if _early.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early.gpus
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_early.gpus}")

# RUN_ID: 인자 > 기존 환경변수 > 새 timestamp
_RUN_ID = (
    _early.run_id
    or os.environ.get("HANDOVER_RUN_ID")
    or datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.environ["HANDOVER_RUN_ID"] = _RUN_ID
print(f"[RUN] HANDOVER_RUN_ID={_RUN_ID}")

import pandas as pd
from config import (
    SFT_MODELS,
    SFT_OUT,
    RLAIF_OUT,
    INFER_OUT,
    EVAL_OUT,
    SAMPLE_OUT,
    OUTPUT_BASE,
    WORK_DIR,
    GOLD_PKL,
    GOLD_REF_PKL,
    VITAL_MAP_PKL,
    EMR_PREOP_SUM_COL,
    EMR_PREMED_COL,
)

# ── 실험 조합 ──────────────────────────────────────────────────────────────
# (키, sft_epochs, rlaif_loss, sft_ckpt_for_rlaif)
EXPERIMENTS = [
    ("raw", None, None, None),
    ("rlaif_dpo", None, "dpo", None),
    ("rlaif_simpo", None, "simpo", None),
    ("sft_1ep", 1, None, None),
    ("sft_3ep", 3, None, None),
    ("sft_1ep_dpo", 1, "dpo", "sft_1ep"),
    ("sft_3ep_dpo", 3, "dpo", "sft_3ep"),
]

LOG_FILE = OUTPUT_BASE / "run_all.log"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd: list, desc: str) -> bool:
    env = os.environ.copy()
    log(f"  실행: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], env=env)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    log(f"  {'✓' if ok else '✗'} {desc}  ({elapsed/60:.1f}분)")
    return ok


# ── 경로 헬퍼 ─────────────────────────────────────────────────────────────


def _infer_path(model: str, exp_key: str) -> Path:
    return INFER_OUT / f"{model}_{exp_key}" / "gold_results.jsonl"


def _eval_path(model: str, exp_key: str) -> Path:
    return EVAL_OUT / f"{model}_{exp_key}" / "gold_results_scores.jsonl"


def _ckpt(model: str, exp_key: str) -> Path:
    if exp_key == "raw":
        return SFT_MODELS[model]
    if exp_key.startswith("sft_") and "dpo" not in exp_key:
        ep = exp_key.split("_")[1].replace("ep", "")
        return SFT_OUT / f"{model}_{ep}ep" / "final"
    if exp_key == "rlaif_dpo":
        return RLAIF_OUT / f"{model}_dpo" / "final"
    if exp_key == "rlaif_simpo":
        return RLAIF_OUT / f"{model}_simpo" / "final"
    if exp_key == "sft_1ep_dpo":
        return RLAIF_OUT / f"{model}_sft_dpo" / "final"
    if exp_key == "sft_3ep_dpo":
        return RLAIF_OUT / f"{model}_sft_dpo" / "final"
    return SFT_MODELS[model]


# ── 샘플 MD 생성 ──────────────────────────────────────────────────────────


def _get(row, col):
    try:
        v = row[col]
    except KeyError:
        return ""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, dict):
        vals = []
        for vlist in v.values():
            vals.extend(
                [str(x) for x in vlist] if isinstance(vlist, list) else [str(vlist)]
            )
        return " ".join(vals)
    return str(v)


def make_sample_md(model: str, exp_key: str, n_samples: int = 5):
    """inference + score 결과에서 샘플 N개를 MD 표로 저장."""
    score_file = _eval_path(model, exp_key)
    if not score_file.exists():
        log(f"  [MD SKIP] score 파일 없음: {score_file.name}")
        return

    lines = [json.loads(l) for l in score_file.read_text().splitlines() if l.strip()]
    if not lines:
        return

    # Gold EMR / Human ref 로드
    try:
        gold_df = pd.read_pickle(GOLD_PKL)
        ref_df = pd.read_pickle(GOLD_REF_PKL)
    except Exception:
        gold_df = ref_df = None

    try:
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)
    except Exception:
        vital_map = {}

    SAMPLE_OUT.mkdir(parents=True, exist_ok=True)
    md_path = SAMPLE_OUT / f"{model}_{exp_key}.md"

    # 점수순 정렬 후 상위 n_samples
    lines_sorted = sorted(lines, key=lambda x: x.get("sum_score", 0), reverse=True)
    samples = lines_sorted[:n_samples]

    md = [f"# {model} / {exp_key} — 샘플 인계요약지 (상위 {n_samples}건)\n"]
    md.append(f"> Run ID: `{_RUN_ID}`  |  총 {len(lines)}건 중 sum_score 상위\n")

    # 평균 점수
    avg_b = sum(l.get("brevity_score", 0) for l in lines) / len(lines)
    avg_c = sum(l.get("critical_score", 0) for l in lines) / len(lines)
    md.append(
        f"**평균 점수**: Brevity={avg_b:.3f}  Critical={avg_c:.3f}  SUM={avg_b+avg_c:.3f}\n"
    )

    for rec in samples:
        idx = rec["idx"]
        sid = rec.get("sid", -1)
        gen = rec["generated"]
        b = rec.get("brevity_score", "-")
        c = rec.get("critical_score", "-")
        s = rec.get("sum_score", "-")

        # 수술명
        opname = "-"
        if gold_df is not None and idx < len(gold_df):
            try:
                v2 = gold_df.iloc[idx][("수술", "수술명", "")]
                opname = str(v2.iloc[0]) if hasattr(v2, "iloc") else str(v2)
            except Exception:
                pass

        # Human ref
        human = "-"
        if ref_df is not None and idx < len(ref_df):
            try:
                human = str(ref_df.iloc[idx][("마취기록", "기록", "")])
                if human == "nan":
                    human = "-"
            except Exception:
                pass

        # Vital
        vital = vital_map.get(sid, "-")

        md.append(f"\n---\n")
        md.append(f"## 케이스 {idx+1} — {opname}\n")
        md.append(f"| 항목 | 내용 |\n|------|------|\n")
        md.append(f"| 수술ID | {sid} |\n")
        md.append(f"| 수술명 | {opname} |\n")

        vital_cell = vital.replace("\n", "<br>") if vital != "-" else "-"
        md.append(f"| 바이탈 요약 | {vital_cell} |\n")

        gen_cell = gen.replace("\n", "<br>")
        human_cell = human.replace("\n", "<br>") if human != "-" else "-"
        md.append(f"| **모델 생성** | {gen_cell} |\n")
        md.append(f"| **Human 정답** | {human_cell} |\n")
        md.append(f"| Brevity | {b} |\n")
        md.append(f"| Critical | {c} |\n")
        md.append(f"| **SUM** | **{s}** |\n")

    md_path.write_text("".join(md), encoding="utf-8")
    log(f"  [MD] 저장: {md_path.name}")


# ── 결과 요약 ─────────────────────────────────────────────────────────────


def summarize_results():
    rows = []
    for score_file in sorted(EVAL_OUT.rglob("*_scores.jsonl")):
        tag = score_file.parent.name  # e.g. llama_raw
        parts = tag.split("_", 1)
        model = parts[0]
        exp_key = parts[1] if len(parts) > 1 else "?"
        lines = [
            json.loads(l) for l in score_file.read_text().splitlines() if l.strip()
        ]
        if not lines:
            continue
        avg_b = sum(l.get("brevity_score", 0) for l in lines) / len(lines)
        avg_c = sum(l.get("critical_score", 0) for l in lines) / len(lines)
        rows.append(
            dict(
                model=model,
                experiment=exp_key,
                n=len(lines),
                brevity=round(avg_b, 3),
                critical=round(avg_c, 3),
                sum=round(avg_b + avg_c, 3),
            )
        )

    if not rows:
        print("\n[결과 없음] 평가 완료된 실험이 없습니다.")
        return

    df = pd.DataFrame(rows).sort_values(["model", "sum"], ascending=[True, False])

    print("\n" + "=" * 72)
    print(f" 실험 결과 요약  (run_id: {_RUN_ID})")
    print("=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)

    # CSV
    csv_path = OUTPUT_BASE / "results_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # MD 요약
    md_path = OUTPUT_BASE / "results_summary.md"
    md = [
        f"# 실험 결과 요약\n\n",
        f"**Run ID**: `{_RUN_ID}`  \n",
        f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n",
        "| model | experiment | n | brevity | critical | sum |\n",
        "|-------|-----------|---|---------|----------|-----|\n",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {r.model} | {r.experiment} | {r.n} | {r.brevity} | {r.critical} | **{r['sum']}** |\n"
        )

    md.append("\n## 모델별 Best\n\n")
    md.append("| model | best_experiment | sum | brevity | critical |\n")
    md.append("|-------|----------------|-----|---------|----------|\n")
    for model, g in df.groupby("model"):
        best = g.iloc[0]
        md.append(
            f"| {model} | {best['experiment']} | **{best['sum']}** | {best['brevity']} | {best['critical']} |\n"
        )

    md_path.write_text("".join(md), encoding="utf-8")

    print(f"\n  CSV: {csv_path}")
    print(f"  MD:  {md_path}")

    print(f"\n[모델별 Best]")
    for model, g in df.groupby("model"):
        best = g.iloc[0]
        print(f"  {model:<12} {best['experiment']:<16} sum={best['sum']}")


# ── 단일 모델 파이프라인 ───────────────────────────────────────────────────


def run_pipeline(model: str, gpus: str, skip_done: bool, only_eval: bool, exps: list):
    log(f"\n{'='*60}")
    log(f"모델: {model}  |  run_id: {_RUN_ID}")
    log(f"{'='*60}")

    py = sys.executable
    gpu_args = ["--gpus", gpus] if gpus else []

    for exp_key, sft_ep, rlaif_loss, sft_for_rlaif in exps:
        log(f"\n[{model}] 실험: {exp_key}")

        # ── SFT ────────────────────────────────────────────────────────
        if not only_eval and sft_ep is not None:
            sft_final = SFT_OUT / f"{model}_{sft_ep}ep" / "final"
            if skip_done and (sft_final / "config.json").exists():
                log(f"  [SKIP] SFT {sft_ep}ep")
            else:
                ok = run_cmd(
                    [py, "02_sft_train.py", "--base", model, "--epochs", str(sft_ep)]
                    + gpu_args,
                    f"SFT {sft_ep}ep",
                )
                if not ok:
                    log(f"  [ERROR] SFT 실패 → {exp_key} 스킵")
                    continue

        # ── RLAIF ──────────────────────────────────────────────────────
        if not only_eval and rlaif_loss is not None:
            if sft_for_rlaif:
                ep = sft_for_rlaif.split("_")[1].replace("ep", "")
                sft_ckpt = str(SFT_OUT / f"{model}_{ep}ep" / "final")
                rlaif_tag = f"{model}_sft_{rlaif_loss}"
            else:
                sft_ckpt = None
                rlaif_tag = f"{model}_{rlaif_loss}"

            rlaif_final = RLAIF_OUT / rlaif_tag / "final"
            if skip_done and (rlaif_final / "config.json").exists():
                log(f"  [SKIP] RLAIF {rlaif_loss}")
            else:
                cmd = [
                    py,
                    "03_rlaif_train.py",
                    "--base",
                    model,
                    "--loss",
                    rlaif_loss,
                ] + gpu_args
                if sft_ckpt:
                    cmd += ["--sft_ckpt", sft_ckpt]
                ok = run_cmd(cmd, f"RLAIF {rlaif_loss}")
                if not ok:
                    log(f"  [ERROR] RLAIF 실패 → {exp_key} 스킵")
                    continue

        # ── Inference ──────────────────────────────────────────────────
        infer_file = _infer_path(model, exp_key)
        if skip_done and infer_file.exists():
            log(f"  [SKIP] Inference")
        else:
            ckpt = str(_ckpt(model, exp_key))
            cmd = [
                py,
                "04_inference.py",
                "--model_path",
                ckpt,
                "--split",
                "gold",
                "--out_tag",
                f"{model}_{exp_key}",
            ] + gpu_args
            if exp_key != "raw":
                cmd += ["--base_model", str(SFT_MODELS[model])]
            ok = run_cmd(cmd, "Inference")
            if not ok:
                log(f"  [ERROR] Inference 실패 → evaluate 스킵")
                continue

        # ── Evaluate ───────────────────────────────────────────────────
        eval_file = _eval_path(model, exp_key)
        if skip_done and eval_file.exists():
            log(f"  [SKIP] Evaluate")
        else:
            if not infer_file.exists():
                log(f"  [SKIP] inference 결과 없음")
                continue
            ok = run_cmd(
                [
                    py,
                    "05_evaluate.py",
                    "--result_file",
                    str(infer_file),
                    "--out_tag",
                    f"{model}_{exp_key}",
                ]
                + gpu_args,
                "Evaluate",
            )

        # ── Sample MD ──────────────────────────────────────────────────
        make_sample_md(model, exp_key)


# ── 진입점 ────────────────────────────────────────────────────────────────


def main():
    global EXPERIMENTS
    parser = argparse.ArgumentParser(description="전체 파이프라인 실행")
    parser.add_argument(
        "--models", nargs="+", choices=list(SFT_MODELS.keys()), default=["llama"]
    )
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="실험 ID (기본: timestamp). 기존 결과 이어받을 때 사용",
    )
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument(
        "--only_eval", action="store_true", help="Inference + Evaluate만 실행"
    )
    parser.add_argument("--summarize", action="store_true", help="결과 요약만 출력")
    parser.add_argument(
        "--experiments", nargs="+", choices=[e[0] for e in EXPERIMENTS], default=None
    )
    parser.add_argument(
        "--n_samples", type=int, default=5, help="샘플 MD에 포함할 케이스 수 (기본 5)"
    )
    args = parser.parse_args()

    if args.summarize:
        summarize_results()
        return

    active_exps = EXPERIMENTS
    if args.experiments:
        active_exps = [e for e in EXPERIMENTS if e[0] in args.experiments]

    log(f"\n실험 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"run_id:    {_RUN_ID}")
    log(f"출력경로:  {OUTPUT_BASE}")
    log(f"모델:      {args.models}")
    log(f"실험:      {[e[0] for e in active_exps]}")
    log(f"GPU:       {args.gpus or '전체'}")

    for model in args.models:
        run_pipeline(model, args.gpus, args.skip_done, args.only_eval, active_exps)

    log("\n모든 파이프라인 완료.")
    summarize_results()


if __name__ == "__main__":
    main()
