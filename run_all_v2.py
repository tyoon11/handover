"""
run_all_v2.py — 파이프라인 v2.0 오케스트레이터

차이(v1 run_all.py 대비):
  - 추론: 생성 위생 ON (--decode v2; repetition_penalty/no_repeat_ngram + clean_v2)
  - 평가: 06_evaluate_v2.py (gold checklist 3축 + 안전 하드게이트)
  - 개선안 5(실험 위생): 체크포인트 provenance 로깅 + 학습변형 간 출력 다양성 점검
  - GPU 정책: 학습(오래걸림)=train_gpus(≤4), 평가(빠름)=eval_gpus(≤6, 31B judge TP)

전제: gold_checklist_v2.json 이 있어야 함. 없으면 --build_checklist 로 먼저 생성.

실행:
  # 1) (최초 1회) checklist 빌드 — gemma4-31B judge는 4장이면 충분
  python run_all_v2.py --build_checklist --eval_gpus 0,1,2,3
  # 2) 전체: 학습+추론(4GPU) → v2 평가(4GPU) → HTML
  python run_all_v2.py --models llama qwen35 gemma4_31b \
      --train_gpus 0,1,2,3 --eval_gpus 0,1,2,3 --skip_done
"""

import os, sys, argparse, json, hashlib, subprocess, time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


def _early():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run_id", type=str, default=None)
    return p.parse_known_args()[0]


_RID = _early().run_id or os.environ.get("HANDOVER_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["HANDOVER_RUN_ID"] = _RID
# 추론 생성 위생 ON (run_all 의 inference cmd가 이 env를 읽어 --decode v2 주입)
os.environ["HANDOVER_INFER_DECODE"] = "v2"
print(f"[RUN v2] HANDOVER_RUN_ID={_RID}  (decode=v2)")

# v1 오케스트레이션 기본기 재사용 (GpuPool/Phase1 실행기/경로헬퍼)
import run_all as R
from config_v2 import EVAL_V2_OUT, GOLD_CHECKLIST_JSON


def _ckpt_hash(path: Path):
    """체크포인트 provenance: adapter/safetensors의 크기+mtime 해시 (학습변형 동일성 점검)."""
    path = Path(path)
    h = hashlib.sha1()
    files = sorted(list(path.glob("*.safetensors")) + list(path.glob("adapter_*.json")))
    for f in files:
        try:
            st = f.stat()
            h.update(f.name.encode()); h.update(str(st.st_size).encode())
        except Exception:
            pass
    return h.hexdigest()[:12] if files else "no-ckpt"


def phase1_train_infer(models, exps, train_gpus, gpus_per_job, skip_done):
    """학습+추론 병렬 (run_all 의 Phase1과 동일 로직, eval은 v2에서 따로)."""
    pool = R.GpuPool(train_gpus, gpus_per_job)
    R.log(f"\n=== Phase 1 (v2): 학습+추론 — GPU {train_gpus} ===")
    with ThreadPoolExecutor(max_workers=len(models) * len(exps)) as ex:
        futures = {}
        for m in models:
            for e in exps:
                dep = futures.get((m, e[3])) if e[3] else None
                futures[(m, e[0])] = ex.submit(
                    R._run_exp_with_dep, m, e, pool, dep, skip_done, False)
        for (m, k), f in futures.items():
            try:
                R.log(f"[Phase1 v2] {m}/{k}: {'✓' if f.result() else '✗'}")
            except Exception as ex_:
                R.log(f"[Phase1 v2 오류] {m}/{k}: {ex_}")


def provenance_and_diversity(models, exps):
    """개선안 5: 체크포인트 해시 + 추론 출력 다양성 점검 → JSON 리포트."""
    report = {"checkpoints": {}, "identical_output_groups": []}
    # 체크포인트 해시
    for m in models:
        for e in exps:
            ck = R._ckpt(m, e[0])
            report["checkpoints"][f"{m}/{e[0]}"] = {"path": str(ck), "hash": _ckpt_hash(ck)}
    # 동일 출력 점검: 모델별로 변형 간 generated가 완전 동일하면 경고
    for m in models:
        sig_map = {}
        for e in exps:
            inf = R._infer_path(m, e[0])
            if not inf.exists():
                continue
            gens = [json.loads(l).get("generated", "")
                    for l in inf.read_text(encoding="utf-8").splitlines() if l.strip()]
            sig = hashlib.sha1("".join(gens).encode()).hexdigest()[:12]
            sig_map.setdefault(sig, []).append(e[0])
        for sig, group in sig_map.items():
            if len(group) > 1:
                report["identical_output_groups"].append({"model": m, "variants": group})
                R.log(f"  ⚠ [{m}] 출력 동일 변형군: {group} "
                      f"(학습 효과 없음/체크포인트 의심 — provenance 해시 확인)")
    out = R.OUTPUT_BASE / "provenance_v2.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    R.log(f"  provenance 저장: {out}")


def phase2_eval_v2(models, exps, eval_gpus, skip_done):
    infer_files = []
    for m in models:
        for e in exps:
            inf = R._infer_path(m, e[0])
            sc = EVAL_V2_OUT / f"{m}_{e[0]}" / "gold_results_scores_v2.jsonl"
            if not inf.exists():
                continue
            if skip_done and sc.exists():
                R.log(f"  [SKIP eval_v2] {m}/{e[0]}"); continue
            infer_files.append(str(inf))
    if not infer_files:
        R.log("  v2 평가 대상 없음"); return
    R.log(f"\n=== Phase 2 (v2): checklist 3축 평가 — GPU {eval_gpus}, {len(infer_files)}개 ===")
    cmd = [sys.executable, "pipeline/06_evaluate_v2.py", "--gpus", eval_gpus,
           "--result_files"] + infer_files
    R.run_cmd(cmd, "Evaluate v2", eval_gpus, "all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama"])
    ap.add_argument("--experiments", nargs="+", default=None)
    ap.add_argument("--train_gpus", type=str, default="0,1,2,3")
    ap.add_argument("--gpus_per_job", type=int, default=1)
    ap.add_argument("--eval_gpus", type=str, default="0,1,2,3",
                    help="gemma4-31B judge는 4장이면 충분(tensor-parallel)")
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--only_eval", action="store_true", help="추론 결과로 v2 평가+리포트만")
    ap.add_argument("--build_checklist", action="store_true",
                    help="gold_checklist_v2.json 생성 후 종료 (eval_gpus 사용)")
    args = ap.parse_args()

    exps = R.EXPERIMENTS if not args.experiments else \
        [e for e in R.EXPERIMENTS if e[0] in args.experiments]

    if args.build_checklist:
        R.run_cmd([sys.executable, "build_gold_checklist.py", "--gpus", args.eval_gpus],
                  "Build checklist", args.eval_gpus, "all")
        return

    if not GOLD_CHECKLIST_JSON.exists():
        R.log(f"[오류] checklist 없음: {GOLD_CHECKLIST_JSON}\n"
              f"  먼저: python run_all_v2.py --build_checklist --eval_gpus {args.eval_gpus}")
        return

    if not args.only_eval:
        phase1_train_infer(args.models, exps, args.train_gpus, args.gpus_per_job, args.skip_done)
        provenance_and_diversity(args.models, exps)

    phase2_eval_v2(args.models, exps, args.eval_gpus, args.skip_done)

    # HTML 리포트
    R.run_cmd([sys.executable, "utils/export_results_v2.py", "--run_id", _RID],
              "Export v2 HTML", args.eval_gpus, "all")
    R.log("\nv2 파이프라인 완료.")


if __name__ == "__main__":
    main()
