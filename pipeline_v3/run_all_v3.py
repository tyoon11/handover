"""
run_all_v3.py — v3 전체 오케스트레이터

v1/v2 오케스트레이터 대비:
  B1  provenance = 체크포인트 '내용' 해시 (파일명+크기 아님)
  B2  완료 판정 = .done 마커(+해시) — 크래시 잔해를 '완료'로 스킵하지 않음.
      모든 jsonl은 tmp→atomic rename.
  T9  DPO 쌍은 정책별 on-policy 생성 (SFT 데이터 재사용 금지)
  Phase3  gold(22) 봉인: 기본 dev(110)로만 평가. --final 에서만 gold 추론+평가.

실행 순서 (권장):
  # 0) 최초 1회 (judge GPU)
  python -m pipeline_v3.make_fewshot_bank --gpus 0,1,2,3
  python -m pipeline_v3.build_gold_checklist_v3 --gpus 0,1,2,3   # (--final 전 필요)
  python -m pipeline_v3.eval_v3.calibrate --gpus 0,1,2,3         # judge↔전문의 정렬 검증
  # 1) SFT 타깃 생성 (1회 공유)
  python -m pipeline_v3.gen_pairs --split sft --models llama qwen --gpus 0,1,2,3
  # 2) 학습+추론+dev 평가
  python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --skip_done
  # 3) 최종 1회 (gold 개봉 — 모델/하이퍼 선택이 끝난 뒤에만!)
  python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --skip_done --final
  # 3') 다린 병기 최종 리포트: 먼저 다린 재추론(gold sid) → 그 다음 --final 리포트에 병기
  python reinfer_darin_on_v3sids.py --gpus 0,1 --split gold --skip_done
  python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --skip_done \
      --final --include_source --include_darin
  # (--darin_root 미지정 시 config DARIN_INFER_OUT 에서 다린 pkl을 읽음)
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue


def _early():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run_id", type=str, default=None)
    return p.parse_known_args()[0]


_RID = (_early().run_id or os.environ.get("HANDOVER_RUN_ID")
        or datetime.now().strftime("%Y%m%d_%H%M%S"))
os.environ["HANDOVER_RUN_ID"] = _RID
print(f"[RUN v3] HANDOVER_RUN_ID={_RID}")

from .config_v3 import (       # noqa: E402
    EVAL_OUT, GOLD_CHECKLIST_JSON, INFER_OUT, OUTPUT_BASE, PAIRS_OUT,
    PAIRS_SFT_PKL, PROVENANCE_JSON, RLAIF_OUT, SFT_OUT, TRAIN_KEYS, ensure_dir,
    infer_engine_for, model_path,
)
from .provenance import (      # noqa: E402
    ckpt_valid, dir_hash, jsonl_rows, read_done_marker, stage_done,
)
from .make_fewshot_bank import FEWSHOT_BANK_JSON      # noqa: E402

# (key, sft_epochs, rlaif_loss, sft_dep_key)
EXPERIMENTS = [
    ("raw", None, None, None),
    ("rlaif_dpo", None, "dpo", None),
    ("rlaif_simpo", None, "simpo", None),
    ("sft_1ep", 1, None, None),
    ("sft_3ep", 3, None, None),
    ("sft_1ep_dpo", 1, "dpo", "sft_1ep"),
    ("sft_3ep_dpo", 3, "dpo", "sft_3ep"),
]

LOG_FILE = OUTPUT_BASE / "run_all_v3.log"
_log_lock = threading.Lock()


def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    with _log_lock:
        print(line, flush=True)
        ensure_dir(OUTPUT_BASE)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


class GpuPool:
    def __init__(self, gpus_str: str, gpus_per_job: int):
        ids = [g.strip() for g in gpus_str.split(",")]
        groups = [",".join(ids[i:i + gpus_per_job])
                  for i in range(0, len(ids), gpus_per_job)]
        self._q = Queue()
        for g in groups:
            self._q.put(g)
        self.n = len(groups)
        log(f"[GpuPool] {self.n}개 그룹: {groups}")

    def acquire(self):
        return self._q.get(block=True)

    def release(self, g):
        self._q.put(g)


def _libstdcxx_preload_env(env: dict) -> dict:
    """vLLM/zmq의 GLIBCXX 요구를 위해 conda libstdc++를 자식 env의 LD_PRELOAD에 주입.
    LD_PRELOAD는 프로세스 시작 시점에만 유효하므로, 여기서 자식에게 넘겨야 먹힌다
    (부모가 in-process로 os.environ만 바꾸면 소급 적용 안 됨). 사용자가 셸에서 이미
    export 했으면 os.environ.copy()에 들어와 있어 그대로 유지된다."""
    prefix = env.get("CONDA_PREFIX")
    if not prefix:
        return env
    lib = os.path.join(prefix, "lib", "libstdc++.so.6")
    if os.path.exists(lib) and lib not in env.get("LD_PRELOAD", "").split(":"):
        cur = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = lib + (":" + cur if cur else "")
    return env


def run_cmd(cmd: list, desc: str, gpus: str, tag: str = "", extra_env: dict = None) -> bool:
    env = _libstdcxx_preload_env(os.environ.copy())
    env["CUDA_VISIBLE_DEVICES"] = gpus
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    prefix = f"[{tag} GPU:{gpus}]" if tag else f"[GPU:{gpus}]"
    log(f"  {prefix} {desc}: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    ok = subprocess.run([str(c) for c in cmd], env=env).returncode == 0
    log(f"  {'✓' if ok else '✗'} {prefix} {desc} ({(time.time() - t0) / 60:.1f}분)")
    return ok


PY = sys.executable


def _train_env(model):
    """학습 자식 프로세스에 CUDA_LAUNCH_BLOCKING=1 주입 (결과 동일, 속도만 손해).

    비동기 CUDA 스트림 race(AccumulateGrad stream mismatch)로 loss 계산 중
    `cudaErrorLaunchFailure: unspecified launch failure`가 **간헐** 발생한다.
    torch 2.11/transformers 5.8 + device_map + grad-checkpoint 조합 이슈.

    처음엔 gemma4에서만 관측돼 GEMMA4_BASES 로 한정했는데, 260824 run 에서
    llama·qwen SFT 가 같은 예외로 전멸(21변형 중 13종만 산출)했다. gemma4 만
    살아남은 이유가 이 env 주입이었다 → **전 모델로 확대**한다.

    끄고 싶으면 HANDOVER_NO_LAUNCH_BLOCKING=1 (재현성 대신 속도 선택).
    """
    if os.environ.get("HANDOVER_NO_LAUNCH_BLOCKING") == "1":
        return None
    return {"CUDA_LAUNCH_BLOCKING": "1"}


def _sft_dir(model, ep):
    return SFT_OUT / f"{model}_{ep}ep"


def _rlaif_dir(model, exp_key, loss):
    if exp_key.startswith("rlaif_"):
        return RLAIF_OUT / f"{model}_raw_{loss}"
    ep = exp_key.split("_")[1]      # "1ep"/"3ep"
    return RLAIF_OUT / f"{model}_sft{ep}_{loss}"


def _ckpt(model, exp_key):
    if exp_key == "raw":
        return model_path(model)
    if exp_key.startswith("sft_") and not exp_key.endswith(("dpo", "simpo")):
        return _sft_dir(model, exp_key.split("_")[1].replace("ep", "")) / "final"
    loss = "simpo" if exp_key.endswith("simpo") else "dpo"
    return _rlaif_dir(model, exp_key, loss) / "final"


def _pairs_dpo_pkl(model, exp_key):
    if exp_key.startswith("rlaif_"):
        return PAIRS_OUT / f"pairs_dpo_{model}_raw.pkl"
    ep = exp_key.split("_")[1]
    return PAIRS_OUT / f"pairs_dpo_{model}_sft{ep}.pkl"


def _infer_file(model, exp_key, split):
    return INFER_OUT / f"{model}_{exp_key}" / f"{split}_results.jsonl"


# ── 단계 실행기 (skip은 .done + 유효성 기준, B2) ─────────────────────────────
def ensure_sft(model, ep, gpus, skip_done) -> bool:
    out = _sft_dir(model, ep)
    if skip_done and stage_done(out) and ckpt_valid(out / "final"):
        log(f"  [SKIP] {model} SFT {ep}ep (.done+유효 체크포인트)")
        return True
    return run_cmd([PY, "-m", "pipeline_v3.sft_train", "--base", model,
                    "--epochs", str(ep)], f"SFT {ep}ep", gpus, f"{model}",
                   extra_env=_train_env(model))


def gen_dpo_pairs(model, kind, ep, gpus, skip_done) -> bool:
    """정책별 on-policy DPO 쌍 생성 (Phase B — prometheus 47B judge라 gpus=전체).
    kind='raw'(base 정책) / 'sft'(SFT ep 정책). 같은 정책이면 1회만 생성(여러 loss가 공유)."""
    if kind == "raw":
        out_tag = f"{model}_raw"
        pkl = PAIRS_OUT / f"pairs_dpo_{model}_raw.pkl"
        extra = []
    else:
        out_tag = f"{model}_sft{ep}ep"
        pkl = PAIRS_OUT / f"pairs_dpo_{model}_sft{ep}ep.pkl"
        sft_final = _sft_dir(model, ep) / "final"
        if not ckpt_valid(sft_final):
            log(f"  [실패] {model} sft{ep}ep 체크포인트 없음 → DPO쌍 생성 불가")
            return False
        extra = ["--policy_ckpt", str(sft_final)]
    if skip_done and pkl.exists():
        log(f"  [SKIP] pairs {pkl.name}")
        return True
    cmd = [PY, "-m", "pipeline_v3.gen_pairs", "--split", "dpo", "--base", model,
           "--out_tag", out_tag] + extra
    return run_cmd(cmd, f"gen_pairs dpo({out_tag})", gpus, model)


def _needed_sft_eps(exps):
    """활성 실험이 요구하는 SFT epoch 집합 (sft_* 변형 + dpo-on-sft의 선행)."""
    eps = set()
    for (_key, sft_ep, _loss, dep) in exps:
        if sft_ep is not None:
            eps.add(sft_ep)
        if dep:
            eps.add(int(dep.split("_")[1].replace("ep", "")))
    return sorted(eps)


def _needed_policies(exps):
    """DPO 쌍이 필요한 (정책종류, ep) 유니크 목록. 여러 loss(dpo/simpo)가 같은 정책을 공유."""
    pol = set()
    for (_key, _sft_ep, loss, dep) in exps:
        if loss is None:
            continue
        if dep is None:
            pol.add(("raw", None))
        else:
            pol.add(("sft", int(dep.split("_")[1].replace("ep", ""))))
    return sorted(pol, key=lambda p: (p[0], p[1] or 0))


def ensure_rlaif(model, exp_key, loss, gpus, skip_done) -> bool:
    out = _rlaif_dir(model, exp_key, loss)
    if skip_done and stage_done(out) and ckpt_valid(out / "final"):
        log(f"  [SKIP] {model} {exp_key} (.done+유효 체크포인트)")
        return True
    pkl = _pairs_dpo_pkl(model, exp_key)
    if not pkl.exists():
        log(f"  [실패] {model}/{exp_key}: 선호쌍 없음 ({pkl.name})")
        return False
    cmd = [PY, "-m", "pipeline_v3.dpo_train", "--base", model, "--loss", loss,
           "--pairs", str(pkl)]
    if exp_key.startswith("sft_"):
        ep = exp_key.split("_")[1].replace("ep", "")
        cmd += ["--sft_ckpt", str(_sft_dir(model, ep) / "final")]
    return run_cmd(cmd, f"RLAIF {exp_key}", gpus, f"{model}",
                   extra_env=_train_env(model))


def ensure_infer(model, exp_key, split, gpus, skip_done, allow_gold=False) -> bool:
    out_file = _infer_file(model, exp_key, split)
    engine = infer_engine_for(model)     # vLLM 불가 모델은 'hf' 강제 (엔진 혼용 방지)
    marker = read_done_marker(out_file.parent)
    if skip_done and marker and marker.get("split") == split and \
            jsonl_rows(out_file) > 0:
        # 엔진 강제 대상인데 기존 산출물이 다른 엔진이면 재실행 (변형 간 엔진 통일)
        engine_ok = (engine == "auto") or (marker.get("engine") == engine)
        # 체크포인트가 그 후 갱신됐으면 재실행 (B3)
        ck = _ckpt(model, exp_key)
        if engine_ok and (exp_key == "raw" or marker.get("ckpt_hash") == dir_hash(Path(ck))):
            log(f"  [SKIP] {model}/{exp_key} infer({split})")
            return True
        if not engine_ok:
            log(f"  [재실행] {model}/{exp_key}: 엔진 불일치 "
                f"(기존 {marker.get('engine')} → {engine} 강제)")
        else:
            log(f"  [재실행] {model}/{exp_key}: 체크포인트 갱신 감지")
    ck = _ckpt(model, exp_key)
    if exp_key != "raw" and not ckpt_valid(Path(ck)):
        log(f"  [실패] {model}/{exp_key}: 추론용 체크포인트 없음 {ck}")
        return False
    cmd = [PY, "-m", "pipeline_v3.inference", "--model_path", str(ck),
           "--split", split, "--out_tag", f"{model}_{exp_key}"]
    if exp_key != "raw":
        cmd += ["--base", model]
    if split == "gold" and allow_gold:
        cmd += ["--allow_gold"]
    if engine != "auto":
        cmd += ["--engine", engine]
    return run_cmd(cmd, f"Inference({split})", gpus, f"{model}/{exp_key}")


def _slot_run(fn, pool: GpuPool):
    gpus = pool.acquire()
    try:
        return fn(gpus)
    finally:
        pool.release(gpus)


def run_phase_parallel(title, jobs, pool: GpuPool) -> dict:
    """jobs: [(label, fn(gpus)->bool)]. GPU 슬롯 풀로 병렬 실행 (학습/추론용)."""
    if not jobs:
        return {}
    log(f"\n{'=' * 60}\n=== {title} (병렬 {pool.n}슬롯, {len(jobs)}잡) ===\n{'=' * 60}")
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, pool.n)) as ex:
        futs = {label: ex.submit(_slot_run, fn, pool) for label, fn in jobs}
        for label, f in futs.items():
            try:
                results[label] = f.result()
            except Exception as e:
                log(f"  [{label}] 오류: {e}")
                results[label] = False
            log(f"  [{label}] {'✓' if results.get(label) else '✗'}")
    return results


def provenance(models, exps):
    """내용 해시 provenance + 변형 간 동일출력 경고 (B1)."""
    import hashlib
    report = {"run_id": _RID, "checkpoints": {}, "pairs": {},
              "identical_output_groups": []}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        report["git_rev"] = rev
    except Exception:
        pass
    for m in models:
        for e in exps:
            ck = _ckpt(m, e[0])
            report["checkpoints"][f"{m}/{e[0]}"] = {
                "path": str(ck), "content_hash": dir_hash(Path(ck))}
    for p in sorted(PAIRS_OUT.glob("pairs_*.pkl")) if PAIRS_OUT.exists() else []:
        report["pairs"][p.name] = dict(
            sha1=hashlib.sha1(p.read_bytes()).hexdigest()[:16], bytes=p.stat().st_size)
    for m in models:
        sig_map = {}
        for e in exps:
            inf = _infer_file(m, e[0], "dev")
            if not inf.exists():
                continue
            gens = [json.loads(l).get("generated", "")
                    for l in inf.read_text(encoding="utf-8").splitlines() if l.strip()]
            sig = hashlib.sha1("".join(gens).encode()).hexdigest()[:12]
            sig_map.setdefault(sig, []).append(e[0])
        for sig, group in sig_map.items():
            if len(group) > 1:
                report["identical_output_groups"].append({"model": m, "variants": group})
                log(f"  ⚠ [{m}] 출력 동일 변형군 {group} — 학습 효과 없음/체크포인트 의심")
    ensure_dir(OUTPUT_BASE)
    PROVENANCE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    log(f"  provenance 저장: {PROVENANCE_JSON}")


def phase_eval(models, exps, eval_gpus, split, skip_done):
    files = []
    for m in models:
        for e in exps:
            inf = _infer_file(m, e[0], split)
            sc = EVAL_OUT / f"{m}_{e[0]}" / f"{split}_results_scores_v3.jsonl"
            if not inf.exists():
                continue
            if skip_done and sc.exists():
                log(f"  [SKIP eval] {m}/{e[0]} ({split})")
                continue
            files.append(str(inf))
    if not files:
        log(f"  평가 대상 없음 ({split})")
        return True
    mode = "gold" if split == "gold" else "dev"
    log(f"\n=== 평가({mode}): {len(files)}개 파일, GPU {eval_gpus} ===")
    return run_cmd([PY, "-m", "pipeline_v3.eval_v3.evaluate", "--mode", mode,
                    "--result_files"] + files, f"Evaluate v3({mode})", eval_gpus, "all")


def phase_stats_report(split, include_source=False, include_darin=False,
                       darin_root=None):
    cmd = [PY, "-m", "pipeline_v3.report_v3", "--split", split]
    if include_source:
        cmd += ["--include_source"]
    if include_darin:
        cmd += ["--include_darin"]
        if darin_root:
            cmd += ["--darin_root", str(darin_root)]
    return run_cmd(cmd, f"Stats+Report({split})", "", "cpu")


def main():
    ap = argparse.ArgumentParser(description="v3 오케스트레이터")
    ap.add_argument("--models", nargs="+", choices=TRAIN_KEYS, default=["llama"])
    ap.add_argument("--experiments", nargs="+",
                    choices=[e[0] for e in EXPERIMENTS], default=None)
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--gpus_per_job", type=int, default=2)
    ap.add_argument("--eval_gpus", type=str, default=None)
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--only_eval", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="gold(22) 개봉 — dev로 모델 선택이 끝난 뒤 최종 1회만!")
    ap.add_argument("--include_source", action="store_true",
                    help="리포트에 비식별 EMR·GT 병기(별도 *_source.html, PHI 포함 — 외부공유 금지).")
    ap.add_argument("--include_darin", action="store_true",
                    help="리포트에 다린(기존 연구) 재추론 출력 병기(*_darin.html). "
                         "다린 pkl은 reinfer_darin_on_v3sids.py 로 미리 생성해 둬야 함.")
    ap.add_argument("--darin_root", type=str, default=None,
                    help="다린 재추론 출력 루트. 미지정 시 config DARIN_INFER_OUT 사용.")
    args = ap.parse_args()

    exps = EXPERIMENTS if not args.experiments else \
        [e for e in EXPERIMENTS if e[0] in args.experiments]
    eval_gpus = args.eval_gpus or args.gpus

    # ── preflight ──
    log(f"run_id={_RID} models={args.models} exps={[e[0] for e in exps]} "
        f"final={args.final}")
    from .data_splits import load_splits, write_split_manifest
    splits = load_splits(need=("gold", "fewshot", "dev"))     # 유출 assert 포함
    write_split_manifest(OUTPUT_BASE, splits)
    del splits

    if not FEWSHOT_BANK_JSON.exists():
        log(f"[오류] fewshot bank 없음 — 먼저: python -m pipeline_v3.make_fewshot_bank "
            f"--gpus {eval_gpus}")
        sys.exit(2)
    if not Path(PAIRS_SFT_PKL).exists() and any(e[1] is not None for e in exps):
        log(f"[오류] SFT 쌍 없음({PAIRS_SFT_PKL}) — 먼저: "
            f"python -m pipeline_v3.gen_pairs --split sft --models llama qwen "
            f"--gpus {eval_gpus}")
        sys.exit(2)
    if args.final and not GOLD_CHECKLIST_JSON.exists():
        log(f"[오류] --final 인데 gold checklist 없음 — 먼저: "
            f"python -m pipeline_v3.build_gold_checklist_v3 --gpus {eval_gpus}")
        sys.exit(2)

    # ── 학습·추론은 2-GPU 병렬(A/C/D), DPO 쌍 생성만 전체 GPU 직렬(B, prometheus 47B) ──
    if not args.only_eval:
        pool = GpuPool(args.gpus, args.gpus_per_job)
        sd = args.skip_done

        # Phase A — SFT 학습 (병렬)
        run_phase_parallel(
            "Phase A: SFT 학습",
            [(f"{m}/sft{ep}ep",
              (lambda g, m=m, ep=ep: ensure_sft(m, ep, g, sd)))
             for m in args.models for ep in _needed_sft_eps(exps)],
            pool)

        # Phase B — DPO 선호쌍 생성 (직렬, 전체 GPU: prometheus 47B judge가 ≥3장 필요)
        policies = _needed_policies(exps)
        if policies:
            log(f"\n{'=' * 60}\n=== Phase B: DPO 선호쌍 (직렬, 전체 GPU {args.gpus} — "
                f"prometheus 47B) ===\n{'=' * 60}")
            for m in args.models:
                for kind, ep in policies:
                    ok = gen_dpo_pairs(m, kind, ep, args.gpus, sd)
                    log(f"  [pairs {m}/{kind}{ep or ''}] {'✓' if ok else '✗'}")

        # Phase C — RLAIF 학습 (병렬)
        run_phase_parallel(
            "Phase C: RLAIF 학습",
            [(f"{m}/{e[0]}",
              (lambda g, m=m, e=e: ensure_rlaif(m, e[0], e[2], g, sd)))
             for m in args.models for e in exps if e[2] is not None],
            pool)

        # Phase D — 추론 (병렬): dev(+gold if --final)
        d_jobs = []
        for m in args.models:
            for e in exps:
                d_jobs.append((f"{m}/{e[0]}/dev",
                               (lambda g, m=m, e=e: ensure_infer(m, e[0], "dev", g, sd))))
                if args.final:
                    d_jobs.append((f"{m}/{e[0]}/gold",
                                   (lambda g, m=m, e=e: ensure_infer(
                                       m, e[0], "gold", g, sd, allow_gold=True))))
        run_phase_parallel("Phase D: 추론", d_jobs, pool)

        provenance(args.models, exps)

    # ── Phase 2: 평가 ──
    ok = phase_eval(args.models, exps, eval_gpus, "dev", args.skip_done)
    if args.final:
        ok = phase_eval(args.models, exps, eval_gpus, "gold", args.skip_done) and ok

    # ── Phase 3: 통계 + 리포트 ──
    #   병기 옵션(source/darin)은 gold(최종) 리포트에만 적용 — dev 자동 리포트는 PHI-free 유지.
    phase_stats_report("dev")
    if args.final:
        phase_stats_report("gold", include_source=args.include_source,
                           include_darin=args.include_darin,
                           darin_root=args.darin_root)

    log("\nv3 파이프라인 완료." if ok else "\nv3 파이프라인 완료 (일부 평가 실패 — 로그 확인).")


if __name__ == "__main__":
    main()
