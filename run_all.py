"""
run_all.py — 전체 파이프라인 오케스트레이션 (병렬 실행)

GPU pool 기반으로 여러 실험을 동시에 실행합니다.
의존성(SFT → RLAIF)은 Future로 자동 처리됩니다.

실행 예시:
  python run_all.py --models llama qwen --gpus 0,1,2,3
  python run_all.py --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2
  python run_all.py --models llama --gpus 0,1,2,3 --skip_done
  python run_all.py --models llama --gpus 0,1,2,3 --only_eval
  python run_all.py --models llama --gpus 0,1,2,3 --experiments raw sft_3ep sft_3ep_dpo
  python run_all.py --run_id 20260526_032935 --summarize
"""

import os, sys, argparse, subprocess, time, json, pickle, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# ── GPU 조기 파싱 ─────────────────────────────────────────────────────────
def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus",   type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    return p.parse_known_args()[0]

_early = _early_gpus()
_RUN_ID = (
    _early.run_id
    or os.environ.get("HANDOVER_RUN_ID")
    or datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.environ["HANDOVER_RUN_ID"] = _RUN_ID
print(f"[RUN] HANDOVER_RUN_ID={_RUN_ID}")

import pandas as pd
from config import (
    SFT_MODELS, SFT_OUT, RLAIF_OUT, INFER_OUT, EVAL_OUT, SAMPLE_OUT,
    OUTPUT_BASE, GOLD_PKL, GOLD_REF_PKL, VITAL_MAP_PKL,
)

# ── 실험 정의 ─────────────────────────────────────────────────────────────
# (key, sft_epochs, rlaif_loss, sft_dep_key)
# sft_dep_key: 이 실험이 완료돼야 시작 가능한 선행 SFT 실험 키
EXPERIMENTS = [
    ("raw",         None, None,    None),
    ("rlaif_dpo",   None, "dpo",   None),
    ("rlaif_simpo", None, "simpo", None),
    ("sft_1ep",     1,    None,    None),
    ("sft_3ep",     3,    None,    None),
    ("sft_1ep_dpo", 1,    "dpo",   "sft_1ep"),
    ("sft_3ep_dpo", 3,    "dpo",   "sft_3ep"),
]

LOG_FILE = OUTPUT_BASE / "run_all.log"
_log_lock = threading.Lock()


# ── 로그 (스레드 안전) ────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


# ── GPU Pool ──────────────────────────────────────────────────────────────
class GpuPool:
    """Thread-safe pool of GPU-group strings.

    예: gpus="0,1,2,3", gpus_per_job=2 → ["0,1", "2,3"] 두 슬롯
    """
    def __init__(self, gpus_str: str, gpus_per_job: int):
        ids = [g.strip() for g in gpus_str.split(",")]
        groups = [
            ",".join(ids[i:i + gpus_per_job])
            for i in range(0, len(ids), gpus_per_job)
        ]
        self._q: Queue = Queue()
        for g in groups:
            self._q.put(g)
        self.n_workers = len(groups)
        log(f"[GpuPool] {self.n_workers}개 GPU 그룹: {groups}")

    def acquire(self) -> str:
        return self._q.get(block=True)

    def release(self, g: str):
        self._q.put(g)


# ── 경로 헬퍼 ─────────────────────────────────────────────────────────────
def _infer_path(model: str, exp_key: str) -> Path:
    return INFER_OUT / f"{model}_{exp_key}" / "gold_results.jsonl"


def _eval_path(model: str, exp_key: str) -> Path:
    return EVAL_OUT / f"{model}_{exp_key}" / "gold_results_scores.jsonl"


def _ckpt(model: str, exp_key: str) -> Path:
    """추론에 사용할 체크포인트 경로 (03_rlaif_train.py 출력 경로와 일치)."""
    if exp_key == "raw":
        return SFT_MODELS[model]
    if exp_key.startswith("sft_") and "dpo" not in exp_key and "simpo" not in exp_key:
        ep = exp_key.split("_")[1]          # "1ep" or "3ep"
        return SFT_OUT / f"{model}_{ep}" / "final"
    if exp_key == "rlaif_dpo":
        return RLAIF_OUT / f"{model}_raw_dpo" / "final"
    if exp_key == "rlaif_simpo":
        return RLAIF_OUT / f"{model}_raw_simpo" / "final"
    if exp_key == "sft_1ep_dpo":
        return RLAIF_OUT / f"{model}_sft1ep_dpo" / "final"
    if exp_key == "sft_3ep_dpo":
        return RLAIF_OUT / f"{model}_sft3ep_dpo" / "final"
    return SFT_MODELS[model]


# ── subprocess 실행 ───────────────────────────────────────────────────────
def run_cmd(cmd: list, desc: str, gpus: str, tag: str = "") -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    prefix = f"[{tag} GPU:{gpus}]" if tag else f"[GPU:{gpus}]"
    log(f"  {prefix} {desc}: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], env=env)
    ok = result.returncode == 0
    log(f"  {'✓' if ok else '✗'} {prefix} {desc} ({(time.time()-t0)/60:.1f}분)")
    return ok


# ── 단일 실험 실행 ────────────────────────────────────────────────────────
def _execute_exp(
    model: str,
    exp: tuple,
    gpus: str,
    skip_done: bool,
    only_eval: bool,
) -> bool:
    exp_key, sft_ep, rlaif_loss, sft_dep_key = exp
    py = sys.executable
    tag = f"{model}/{exp_key}"

    def _run(cmd, desc):
        return run_cmd(cmd, desc, gpus, tag)

    # ── SFT ──────────────────────────────────────────────────────────────
    if not only_eval and sft_ep is not None:
        sft_final = SFT_OUT / f"{model}_{sft_ep}ep" / "final"
        # 체크포인트가 이미 있으면 항상 스킵 (병렬 dep 처리 포함)
        if (sft_final / "config.json").exists():
            log(f"  [SKIP] [{tag}] SFT {sft_ep}ep (체크포인트 존재)")
        elif skip_done:
            log(f"  [SKIP] [{tag}] SFT {sft_ep}ep")
        else:
            ok = _run(
                [py, "02_sft_train.py", "--base", model, "--epochs", str(sft_ep)],
                f"SFT {sft_ep}ep",
            )
            if not ok:
                return False

    # ── RLAIF ────────────────────────────────────────────────────────────
    if not only_eval and rlaif_loss is not None:
        if sft_dep_key:
            ep = sft_dep_key.split("_")[1].replace("ep", "")  # "sft_1ep" → "1"
            sft_ckpt = str(SFT_OUT / f"{model}_{ep}ep" / "final")
            rlaif_tag = f"{model}_sft{ep}ep_{rlaif_loss}"      # "llama_sft1ep_dpo"
        else:
            sft_ckpt = None
            rlaif_tag = f"{model}_raw_{rlaif_loss}"             # "llama_raw_dpo"

        rlaif_final = RLAIF_OUT / rlaif_tag / "final"
        if skip_done and (rlaif_final / "config.json").exists():
            log(f"  [SKIP] [{tag}] RLAIF {rlaif_loss}")
        else:
            cmd = [py, "03_rlaif_train.py", "--base", model, "--loss", rlaif_loss]
            if sft_ckpt:
                cmd += ["--sft_ckpt", sft_ckpt]
            ok = _run(cmd, f"RLAIF {rlaif_loss}")
            if not ok:
                return False

    # ── Inference ────────────────────────────────────────────────────────
    infer_file = _infer_path(model, exp_key)
    if skip_done and infer_file.exists():
        log(f"  [SKIP] [{tag}] Inference")
    else:
        ckpt = str(_ckpt(model, exp_key))
        cmd = [
            py, "04_inference.py",
            "--model_path", ckpt,
            "--split", "gold",
            "--out_tag", f"{model}_{exp_key}",
        ]
        if exp_key != "raw":
            cmd += ["--base_model", str(SFT_MODELS[model])]
        ok = _run(cmd, "Inference")
        if not ok:
            return False

    # ── Evaluate ─────────────────────────────────────────────────────────
    eval_file = _eval_path(model, exp_key)
    if skip_done and eval_file.exists():
        log(f"  [SKIP] [{tag}] Evaluate")
    elif infer_file.exists():
        _run(
            [py, "05_evaluate.py",
             "--result_file", str(infer_file),
             "--out_tag", f"{model}_{exp_key}"],
            "Evaluate",
        )
        make_sample_md(model, exp_key)

    return True


# ── 병렬 오케스트레이터 ───────────────────────────────────────────────────
def _run_exp_with_dep(
    model: str,
    exp: tuple,
    gpu_pool: GpuPool,
    dep_future,          # Future | None
    skip_done: bool,
    only_eval: bool,
) -> bool:
    """의존성 Future가 완료된 후 GPU를 획득해서 실험 실행."""
    if dep_future is not None:
        try:
            dep_ok = dep_future.result()
        except Exception:
            dep_ok = False
        if not dep_ok:
            log(f"  [SKIP] {model}/{exp[0]}: 선행 실험 실패")
            return False

    gpus = gpu_pool.acquire()
    try:
        return _execute_exp(model, exp, gpus, skip_done, only_eval)
    finally:
        gpu_pool.release(gpus)


def run_parallel(
    models: list,
    gpus_str: str,
    gpus_per_job: int,
    skip_done: bool,
    only_eval: bool,
    exps: list,
):
    gpu_pool = GpuPool(gpus_str, gpus_per_job)
    n_total = len(models) * len(exps)
    log(f"총 {n_total}개 실험 (모델 {len(models)}개 × 실험 {len(exps)}개)")

    # 모든 잡을 한꺼번에 제출 (GPU pool이 동시 실행 수를 제어)
    with ThreadPoolExecutor(max_workers=n_total) as executor:
        futures = {}  # (model, exp_key) → Future

        for model in models:
            for exp in exps:
                exp_key = exp[0]
                dep_key = exp[3]
                dep_future = futures.get((model, dep_key)) if dep_key else None

                f = executor.submit(
                    _run_exp_with_dep,
                    model, exp, gpu_pool, dep_future, skip_done, only_eval,
                )
                futures[(model, exp_key)] = f

        # 완료 대기 및 결과 수집
        for (model, exp_key), f in futures.items():
            try:
                ok = f.result()
                log(f"[완료] {model}/{exp_key}: {'✓' if ok else '✗'}")
            except Exception as e:
                log(f"[오류] {model}/{exp_key}: {e}")


# ── 샘플 MD 생성 ──────────────────────────────────────────────────────────
def make_sample_md(model: str, exp_key: str, n_samples: int = 5):
    score_file = _eval_path(model, exp_key)
    if not score_file.exists():
        log(f"  [MD SKIP] score 파일 없음: {score_file.name}")
        return

    lines = [json.loads(l) for l in score_file.read_text().splitlines() if l.strip()]
    if not lines:
        return

    try:
        gold_df = pd.read_pickle(GOLD_PKL)
        ref_df  = pd.read_pickle(GOLD_REF_PKL)
    except Exception:
        gold_df = ref_df = None

    try:
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)
    except Exception:
        vital_map = {}

    SAMPLE_OUT.mkdir(parents=True, exist_ok=True)
    md_path = SAMPLE_OUT / f"{model}_{exp_key}.md"

    lines_sorted = sorted(lines, key=lambda x: x.get("sum_score", 0), reverse=True)
    samples = lines_sorted[:n_samples]

    avg_b = sum(l.get("brevity_score", 0) for l in lines) / len(lines)
    avg_c = sum(l.get("critical_score", 0) for l in lines) / len(lines)

    md = [
        f"# {model} / {exp_key} — 샘플 인계요약지 (상위 {n_samples}건)\n",
        f"> Run ID: `{_RUN_ID}`  |  총 {len(lines)}건 중 sum_score 상위\n",
        f"**평균 점수**: Brevity={avg_b:.3f}  Critical={avg_c:.3f}  SUM={avg_b+avg_c:.3f}\n",
    ]

    for rec in samples:
        idx = rec["idx"]
        sid = rec.get("sid", -1)
        gen = rec["generated"]
        b, c, s = rec.get("brevity_score","-"), rec.get("critical_score","-"), rec.get("sum_score","-")

        opname = "-"
        if gold_df is not None and idx < len(gold_df):
            try:
                v = gold_df.iloc[idx][("수술", "수술명", "")]
                opname = str(v.iloc[0]) if hasattr(v, "iloc") else str(v)
            except Exception:
                pass

        human = "-"
        if ref_df is not None and idx < len(ref_df):
            try:
                human = str(ref_df.iloc[idx][("마취기록", "기록", "")])
                if human == "nan": human = "-"
            except Exception:
                pass

        vital = vital_map.get(sid, "-")
        vital_cell  = vital.replace("\n", "<br>") if vital != "-" else "-"
        gen_cell    = gen.replace("\n", "<br>")
        human_cell  = human.replace("\n", "<br>") if human != "-" else "-"

        md += [
            f"\n---\n",
            f"## 케이스 {idx+1} — {opname}\n",
            f"| 항목 | 내용 |\n|------|------|\n",
            f"| 수술ID | {sid} |\n",
            f"| 수술명 | {opname} |\n",
            f"| 바이탈 요약 | {vital_cell} |\n",
            f"| **모델 생성** | {gen_cell} |\n",
            f"| **Human 정답** | {human_cell} |\n",
            f"| Brevity | {b} |\n",
            f"| Critical | {c} |\n",
            f"| **SUM** | **{s}** |\n",
        ]

    md_path.write_text("".join(md), encoding="utf-8")
    log(f"  [MD] 저장: {md_path.name}")


# ── 결과 요약 ─────────────────────────────────────────────────────────────
def summarize_results():
    rows = []
    for score_file in sorted(EVAL_OUT.rglob("*_scores.jsonl")):
        tag   = score_file.parent.name
        parts = tag.split("_", 1)
        model = parts[0]
        exp_key = parts[1] if len(parts) > 1 else "?"
        lines = [json.loads(l) for l in score_file.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        avg_b = sum(l.get("brevity_score", 0) for l in lines) / len(lines)
        avg_c = sum(l.get("critical_score", 0) for l in lines) / len(lines)
        rows.append(dict(
            model=model, experiment=exp_key, n=len(lines),
            brevity=round(avg_b,3), critical=round(avg_c,3), sum=round(avg_b+avg_c,3),
        ))

    if not rows:
        print("\n[결과 없음] 평가 완료된 실험이 없습니다.")
        return

    df = pd.DataFrame(rows).sort_values(["model","sum"], ascending=[True,False])

    print("\n" + "="*72)
    print(f" 실험 결과 요약  (run_id: {_RUN_ID})")
    print("="*72)
    print(df.to_string(index=False))
    print("="*72)

    csv_path = OUTPUT_BASE / "results_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = OUTPUT_BASE / "results_summary.md"
    md = [
        f"# 실험 결과 요약\n\n",
        f"**Run ID**: `{_RUN_ID}`  \n",
        f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n",
        "| model | experiment | n | brevity | critical | sum |\n",
        "|-------|-----------|---|---------|----------|-----|\n",
    ]
    for _, r in df.iterrows():
        md.append(f"| {r.model} | {r.experiment} | {r.n} | {r.brevity} | {r.critical} | **{r['sum']}** |\n")

    md.append("\n## 모델별 Best\n\n")
    md.append("| model | best_experiment | sum | brevity | critical |\n")
    md.append("|-------|----------------|-----|---------|----------|\n")
    for model, g in df.groupby("model"):
        best = g.iloc[0]
        md.append(f"| {model} | {best['experiment']} | **{best['sum']}** | {best['brevity']} | {best['critical']} |\n")

    md_path.write_text("".join(md), encoding="utf-8")
    print(f"\n  CSV: {csv_path}")
    print(f"  MD:  {md_path}")
    print(f"\n[모델별 Best]")
    for model, g in df.groupby("model"):
        best = g.iloc[0]
        print(f"  {model:<12} {best['experiment']:<16} sum={best['sum']}")


def compare_models(file_a: str, file_b: str):
    import numpy as np
    from scipy import stats

    def load_scores(path):
        return pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8")])

    df_a = load_scores(file_a)
    df_b = load_scores(file_b)

    print("\n" + "="*70)
    print(f"Model A: {file_a}\nModel B: {file_b}")
    print("="*70)

    cols = [c for c in ["brevity_score","critical_score","sum_score","scale_large","scale_xl"]
            if c in df_a.columns]
    for col in cols:
        A, B = np.array(df_a[col]), np.array(df_b[col])
        d = A - B
        t_stat, p_t = stats.ttest_rel(A, B)
        dz = d.mean() / d.std(ddof=1)
        se = d.std(ddof=1) / (len(d)**0.5)
        ci = (d.mean() - 1.984*se, d.mean() + 1.984*se)
        w_stat, p_w = stats.wilcoxon(A, B, zero_method="wilcox")
        wins = int((d>0).sum()); losses = int((d<0).sum())
        rb = (wins-losses)/(wins+losses) if (wins+losses) > 0 else float("nan")
        p_sign = stats.binomtest(wins, wins+losses, p=0.5).pvalue
        sig_t = " !!!" if p_t < 0.05 else ""
        sig_w = " !!!" if p_w < 0.05 else ""
        sig_s = " !!!" if p_sign < 0.05 else ""
        print(f"\n── {col} ──")
        print(f"  Paired t  : t={t_stat:.3f}, p={p_t:.4g}, Δ={d.mean():.4f}, CI={ci}, dz={dz:.3f}{sig_t}")
        print(f"  Wilcoxon  : W={w_stat}, p={p_w:.4g}, rank-biserial={rb:.3f}{sig_w}")
        print(f"  Sign test : wins={wins}, losses={losses}, p={p_sign:.4g}{sig_s}")
    print("\n" + "="*70)


# ── 진입점 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="전체 파이프라인 병렬 실행")
    parser.add_argument("--models", nargs="+", choices=list(SFT_MODELS.keys()), default=["llama"])
    parser.add_argument("--gpus",          type=str, default="0,1,2,3", help="사용할 GPU (예: 0,1,2,3)")
    parser.add_argument("--gpus_per_job",  type=int, default=1,
                        help="잡당 GPU 수 (기본 1 — 48GB GPU는 8B 모델에 충분, 2로 늘리면 더 안전)")
    parser.add_argument("--run_id",        type=str, default=None)
    parser.add_argument("--skip_done",     action="store_true", help="완료된 단계 건너뜀")
    parser.add_argument("--only_eval",     action="store_true", help="Inference+Evaluate만")
    parser.add_argument("--summarize",     action="store_true", help="결과 요약만 출력")
    parser.add_argument("--experiments",   nargs="+", choices=[e[0] for e in EXPERIMENTS], default=None)
    parser.add_argument("--n_samples",     type=int, default=5, help="샘플 MD 케이스 수")
    parser.add_argument("--compare",       nargs=2, metavar=("FILE_A","FILE_B"), default=None)
    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare[0], args.compare[1])
        return

    if args.summarize:
        summarize_results()
        return

    active_exps = EXPERIMENTS
    if args.experiments:
        active_exps = [e for e in EXPERIMENTS if e[0] in args.experiments]

    log(f"\n실험 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"run_id:       {_RUN_ID}")
    log(f"출력경로:     {OUTPUT_BASE}")
    log(f"모델:         {args.models}")
    log(f"실험:         {[e[0] for e in active_exps]}")
    log(f"GPU:          {args.gpus}  ({args.gpus_per_job}개/잡)")

    run_parallel(args.models, args.gpus, args.gpus_per_job, args.skip_done, args.only_eval, active_exps)

    log("\n모든 파이프라인 완료.")
    summarize_results()


if __name__ == "__main__":
    main()
