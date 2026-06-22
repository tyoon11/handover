"""
06_evaluate_v2.py — 평가 파이프라인 v2.0 (gold checklist 기반 3축 + 안전 하드게이트)

v1(05_evaluate.py)은 그대로 보존. 이 스크립트는 별도 산출물(*_scores_v2.jsonl)을 만든다.

실행:
  # 단일/다중 추론결과 평가 (엔진 1회 로드)
  python pipeline/06_evaluate_v2.py --result_files outputs/<run>/inference/*/gold_results.jsonl --gpus 0,1,2,3,4,5

요건: gold_checklist_v2.json 이 먼저 생성돼 있어야 함 (build_gold_checklist.py).
"""

import sys as _sys, os, argparse, json, pickle
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


def _early_parse_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    return p.parse_known_args()[0].gpus


_gpus = _early_parse_gpus()
if _gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_gpus}")

import pandas as pd
from config import GOLD_PKL, VITAL_MAP_PKL, build_emr_text, build_user_prompt
from config_v2 import (
    EVAL_V2_LLM, EVAL_V2_BACKEND, EVAL_V2_GEN,
    GOLD_CHECKLIST_JSON, EVAL_V2_OUT,
)
from pipeline.eval_v2.engine import EvalEngine, clean_v2
from pipeline.eval_v2 import metrics as M
from pipeline.eval_v2.checklist import load_checklist
from config_v2 import is_no_issue


def _sid_of(gold_df, idx):
    try:
        s = gold_df.iloc[idx]["수술 ID"]
        return int(s.iloc[0]) if hasattr(s, "iloc") else int(s)
    except Exception:
        return -1


def _premise(gold_df, vital_map, idx, sid):
    row = gold_df.iloc[idx] if idx < len(gold_df) else None
    emr = build_emr_text(row) if row is not None else ""
    return build_user_prompt(emr, vital_map.get(sid, ""))


def evaluate_files(result_files, checklist, gold_df, vital_map, engine):
    """여러 추론결과 파일을 한 번에 평가. 모든 LLM 프롬프트를 모아 축별로 배치 호출."""
    # 1) 레코드 로드 + 분류
    records = []  # 각 원소: dict(file, rec, idx, sid, gen, status, entry, fast(score or None))
    for rf in result_files:
        rf = _Path(rf)
        recs = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
        for rec in recs:
            idx = rec.get("idx", 0)
            sid = rec.get("sid", _sid_of(gold_df, idx))
            raw = rec.get("generated_raw") or rec.get("generated", "")
            gen, status = clean_v2(raw)
            entry = checklist.get(str(sid)) or {"idx": idx, "sid": sid, "items": [],
                                                 "is_normal_case": True, "opname": "-"}
            rinfo = dict(file=str(rf), rec=rec, idx=idx, sid=sid, gen=gen,
                         status=status, entry=entry, fast=None,
                         need_cov=False, need_faith=False, need_brev=False)

            if status != "ok":
                rinfo["fast"] = M.degenerate_scores(status)
            elif entry.get("is_normal_case") and is_no_issue(gen):
                rinfo["fast"] = M.normal_case_no_llm(gen)
            elif (len(entry.get("items", [])) > 0 and not entry.get("is_normal_case")) and is_no_issue(gen):
                # 이상소견 케이스인데 '특이사항 없음' → 안전위반, LLM 불필요
                rinfo["fast"] = dict(
                    coverage=0.0, faithfulness=1.0, brevity=1.0,
                    composite=min(M.V2_WEIGHTS["faithfulness"] + M.V2_WEIGHTS["brevity"],
                                  M.V2_SAFETY_VIOLATION_CAP),
                    safety_violation=True,
                    missed=entry.get("items", []), hallucinations=[], noise=[],
                    n_claims=0, gen_status="ok",
                    note="안전위반: 이상소견 있는데 '특이사항 없음'(놓침)")
            else:
                # LLM 채점 필요
                rinfo["need_faith"] = True
                rinfo["need_brev"] = True
                rinfo["need_cov"] = len(entry.get("items", [])) > 0
            records.append(rinfo)

    # 2) 프롬프트 수집 (축별)
    cov_idx, cov_prompts = [], []
    fa_idx, fa_prompts = [], []
    br_idx, br_prompts = [], []
    premise_cache = {}
    for i, r in enumerate(records):
        if r["fast"] is not None:
            continue
        key = (r["idx"], r["sid"])
        if key not in premise_cache:
            premise_cache[key] = _premise(gold_df, vital_map, r["idx"], r["sid"])
        prem = premise_cache[key]
        if r["need_cov"]:
            s, u = M.build_coverage_prompt(r["gen"], r["entry"])
            cov_idx.append(i); cov_prompts.append(u); r["_cov_sys"] = s
        if r["need_faith"]:
            s, u = M.build_faithfulness_prompt(prem, r["gen"])
            fa_idx.append(i); fa_prompts.append(u); r["_fa_sys"] = s
        if r["need_brev"]:
            s, u = M.build_brevity_prompt(r["gen"])
            br_idx.append(i); br_prompts.append(u); r["_br_sys"] = s

    # 3) 배치 LLM 호출 (system은 축별 동일)
    def run(prompts, idxs, sys_attr):
        if not prompts:
            return {}
        system = records[idxs[0]][sys_attr]
        out = engine.chat_json(prompts, system=system, want="obj", retries=2)
        return {idxs[k]: out[k] for k in range(len(idxs))}

    print(f"[eval_v2] LLM 호출: coverage={len(cov_prompts)}, "
          f"faithfulness={len(fa_prompts)}, brevity={len(br_prompts)}")
    cov_res = run(cov_prompts, cov_idx, "_cov_sys")
    fa_res = run(fa_prompts, fa_idx, "_fa_sys")
    br_res = run(br_prompts, br_idx, "_br_sys")

    # 4) 조립
    for i, r in enumerate(records):
        if r["fast"] is not None:
            r["scores"] = r["fast"]
            continue
        entry = r["entry"]
        cov = M.parse_coverage(cov_res.get(i), entry) if r["need_cov"] else \
            dict(coverage=1.0, missed=[], high_missed=False)
        fa = M.parse_faithfulness(fa_res.get(i))
        br = M.parse_brevity(br_res.get(i))
        r["scores"] = M.composite(cov, fa, br, entry, r["gen"])

    # 5) 파일별 저장
    by_file = {}
    for r in records:
        by_file.setdefault(r["file"], []).append(r)

    for rf, rs in by_file.items():
        rf = _Path(rf)
        tag = rf.parent.name
        out_dir = EVAL_V2_OUT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / rf.name.replace(".jsonl", "_scores_v2.jsonl")
        covs, fas, brs, comps, viol = [], [], [], [], 0
        with open(out_file, "w", encoding="utf-8") as f:
            for r in sorted(rs, key=lambda x: x["idx"]):
                sc = r["scores"]
                out_rec = {
                    **r["rec"],
                    "generated_v2": r["gen"],
                    "gen_status": sc.get("gen_status", r["status"]),
                    "coverage": sc.get("coverage"),
                    "faithfulness": sc.get("faithfulness"),
                    "brevity_v2": sc.get("brevity"),
                    "composite": sc.get("composite"),
                    "safety_violation": sc.get("safety_violation", False),
                    "missed_items": sc.get("missed", []),
                    "hallucinations": sc.get("hallucinations", []),
                    "noise": sc.get("noise", []),
                    "v2_note": sc.get("note", ""),
                    "checklist_n": len(r["entry"].get("items", [])),
                    "is_normal_case": r["entry"].get("is_normal_case", False),
                }
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                if sc.get("coverage") is not None: covs.append(sc["coverage"])
                if sc.get("faithfulness") is not None: fas.append(sc["faithfulness"])
                if sc.get("brevity") is not None: brs.append(sc["brevity"])
                comps.append(sc.get("composite", 0.0))
                viol += int(sc.get("safety_violation", False))

        def _m(x): return (sum(x) / len(x)) if x else float("nan")
        print(f"\n[{tag}] n={len(rs)}  "
              f"COV={_m(covs):.3f}  FAITH={_m(fas):.3f}  BREV={_m(brs):.3f}  "
              f"COMPOSITE={_m(comps):.3f}  안전위반={viol}건")
        print(f"  저장: {out_file}")


def main():
    ap = argparse.ArgumentParser(description="평가 v2.0 (checklist 3축)")
    ap.add_argument("--result_files", nargs="+", required=True)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--checklist", type=str, default=str(GOLD_CHECKLIST_JSON))
    ap.add_argument("--backend", type=str, default=EVAL_V2_BACKEND)
    args = ap.parse_args()

    checklist = load_checklist(args.checklist)
    if not checklist:
        print(f"[오류] checklist 없음: {args.checklist}\n"
              f"  먼저 실행: python build_gold_checklist.py --gpus {args.gpus or '0,1'}")
        return
    n_reviewed = sum(1 for v in checklist.values() if v.get("reviewed"))
    print(f"[eval_v2] checklist {len(checklist)}건 로드 (전문의 검수 {n_reviewed}건)")
    if n_reviewed == 0:
        print("  ⚠ 전문의 검수 0건 — LLM 부트스트랩 checklist입니다. 결과는 잠정치로 해석하세요.")

    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    engine = EvalEngine(EVAL_V2_LLM, backend=args.backend, gen_cfg=EVAL_V2_GEN)
    evaluate_files(args.result_files, checklist, gold_df, vital_map, engine)


if __name__ == "__main__":
    main()
