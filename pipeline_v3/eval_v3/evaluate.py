"""
eval_v3/evaluate.py — 평가 CLI (교차 judge + fail-loud 집계)

핵심 규칙:
  - 2-judge(EVAL_JUDGES) 전부로 채점하되, '공식 점수'는 평가 대상 모델과
    다른 family의 judge 점수만 평균한다 (T7 순환 차단). 두 judge 일치도는 항상 보고.
  - sid가 split에 없으면 하드 에러 (조용한 normal-case fallback 금지, E3).
  - judge 실패/gold 부재 케이스는 '제외'로 따로 세고, 유효 비율 <
    MIN_VALID_CASE_RATIO 면 종료코드 1 (성적표가 아니라 인프라 문제라는 뜻).
  - 산출물은 tmp → atomic rename (B2).

실행:
  # gold(최종) — gold checklist 필요
  python -m pipeline_v3.eval_v3.evaluate --mode gold \
      --result_files outputs_v3/<run>/inference/*/gold_results.jsonl --gpus 0,1,2,3
  # dev(개발) — checklist 없이 faithfulness/brevity/위생만
  python -m pipeline_v3.eval_v3.evaluate --mode dev \
      --result_files outputs_v3/<run>/inference/*/dev_results.jsonl --gpus 0,1,2,3
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    g = p.parse_known_args()[0].gpus
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = g
        print(f"[GPU] CUDA_VISIBLE_DEVICES={g}")


_early_gpus()

from ..config_v3 import (      # noqa: E402
    EVAL_JUDGES, EVAL_OUT, GOLD_CHECKLIST_JSON, MIN_VALID_CASE_RATIO, MODELS,
    STATS, ensure_dir, is_no_issue_v3, judges_for, model_family, model_path,
)
from ..prompt_utils import build_emr_text, get_sid       # noqa: E402
from . import metrics as M                                # noqa: E402
from .checklist import load_checklist, review_status      # noqa: E402
from .cleaning import clean_v3                             # noqa: E402
from .stats import bootstrap_ci, judge_agreement, micro_coverage   # noqa: E402

# 모델 키 4벌 문제(B14) 해소: 태그 파싱은 이 함수 한 벌만 사용
_MODEL_KEYS = sorted(MODELS.keys(), key=len, reverse=True)


def split_tag(tag: str):
    """'gemma4_31b_sft_3ep_dpo' → ('gemma4_31b', 'sft_3ep_dpo'). 긴 키 우선 매칭."""
    for mk in _MODEL_KEYS:
        if tag == mk:
            return mk, "?"
        if tag.startswith(mk + "_"):
            return mk, tag[len(mk) + 1:]
    parts = tag.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "?")


def _atomic_write_jsonl(path: Path, records):
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_split_df(mode: str):
    from ..data_splits import load_splits
    if mode == "gold":
        return load_splits(need=("gold",))["gold"], None
    splits = load_splits(need=("dev",))
    return splits["dev"], splits


def _sid_row_map(df):
    m = {}
    for i in range(len(df)):
        s = get_sid(df.iloc[i])
        if s != -1:
            m[s] = i
    return m


def _collect_records(result_files, split_df, vital_map, checklist, mode):
    """레코드 로드 + 정제 + fast-path. 반환: list[dict] (need_* 플래그 포함)."""
    sid_map = _sid_row_map(split_df)
    records = []
    for rf in result_files:
        rf = Path(rf)
        for line in rf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("sid")
            if sid is None or int(sid) not in sid_map:
                raise RuntimeError(
                    f"[E3] sid={sid}가 {mode} split에 없음 ({rf}) — "
                    "추론 결과와 split이 어긋났다. normal-case fallback은 하지 않는다."
                )
            sid = int(sid)
            row = split_df.iloc[sid_map[sid]]
            gen, status, flags = clean_v3(rec.get("generated_raw") or rec.get("generated", ""))

            if mode == "gold":
                entry = checklist.get(str(sid))
                if entry is None:
                    raise RuntimeError(
                        f"[E3] checklist에 sid={sid} 없음 — build_gold_checklist_v3 재실행 필요."
                    )
            else:
                entry = {"items": [], "is_normal_case": False, "source": "dev"}

            r = dict(file=str(rf), tag=rf.parent.name, rec=rec, sid=sid,
                     gen=gen, gen_status=status, gen_flags=flags, entry=entry,
                     emr=build_emr_text(row), vital=vital_map.get(sid, ""),
                     fast=None, no_issue=is_no_issue_v3(gen))

            if mode == "gold":
                r["fast"] = M.fast_path(gen, status, entry)
            else:
                # dev: 생성 붕괴만 fast 0점. no-issue는 정오 판단 불가 → 별도 집계.
                if status != "ok":
                    r["fast"] = M._base(coverage=None, faithfulness=0.0, brevity=0.0,
                                        composite=0.0, gate="degenerate",
                                        note=f"생성 실패({status})")
                elif r["no_issue"]:
                    r["fast"] = M._base(excluded=True, exclude_reason="dev_no_issue",
                                        note="dev: no-issue 출력 — 정오 판단 불가, 비율만 집계")
            records.append(r)
    return records


def _judge_pass(judge_key, records, backend, mode):
    """단일 judge로 전체 레코드 채점 → {record_idx: score_dict}."""
    from .engine import JudgeEngine
    engine = JudgeEngine(model_path(judge_key), backend=backend)

    cov_i, cov_p = [], []
    fa_i, fa_p = [], []
    br_i, br_p = [], []
    for i, r in enumerate(records):
        if r["fast"] is not None:
            continue
        if mode == "gold" and len(r["entry"].get("items", [])) > 0:
            s, u = M.build_coverage_prompt(r["gen"], r["entry"])
            cov_i.append(i); cov_p.append(u); r["_cov_sys"] = s
        s, u = M.build_faithfulness_prompt(r["emr"], r["vital"], r["gen"])
        fa_i.append(i); fa_p.append(u); r["_fa_sys"] = s
        s, u = M.build_brevity_prompt(r["gen"])
        br_i.append(i); br_p.append(u); r["_br_sys"] = s

    print(f"[eval_v3·{judge_key}] LLM 호출: cov={len(cov_p)} faith={len(fa_p)} brev={len(br_p)}")

    def run(prompts, idxs, sys_attr):
        if not prompts:
            return {}
        system = records[idxs[0]][sys_attr]
        out = engine.chat_json(prompts, system=system, want="obj", retries=2)
        return {idxs[k]: out[k] for k in range(len(idxs))}

    cov_res = run(cov_p, cov_i, "_cov_sys")
    fa_res = run(fa_p, fa_i, "_fa_sys")
    br_res = run(br_p, br_i, "_br_sys")
    engine.unload()

    scores = {}
    for i, r in enumerate(records):
        if r["fast"] is not None:
            scores[i] = r["fast"]
            continue
        cov = M.parse_coverage(cov_res.get(i), r["entry"]) if i in cov_res else \
            dict(coverage=None, covered=[], partial=[], missed=[],
                 category_coverage={}, missed_categories=[],
                 judge_failed=False, reason="no_items")
        fa = M.parse_faithfulness(fa_res.get(i))
        br = M.parse_brevity(br_res.get(i))
        scores[i] = M.composite_from_axes(cov, fa, br, r["entry"])
    return scores


def _official(per_judge: dict, allowed: list):
    """허용 judge들의 축 평균으로 공식 점수 산출. 하나라도 excluded면 제외."""
    use = {j: per_judge[j] for j in allowed if j in per_judge}
    if not use:
        return None
    if any(s.get("excluded") for s in use.values()):
        reasons = {j: s.get("exclude_reason") for j, s in use.items() if s.get("excluded")}
        return dict(M._base(excluded=True, exclude_reason=json.dumps(reasons, ensure_ascii=False)),
                    judges_used=list(use))

    def avg(key):
        vals = [s[key] for s in use.values() if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    first = next(iter(use.values()))
    out = M._base(coverage=avg("coverage"), faithfulness=avg("faithfulness"),
                  brevity=avg("brevity"), composite=avg("composite"),
                  gate=first.get("gate", ""), note=first.get("note", ""))
    # 상세 리스트(missed 등)는 첫 허용 judge 기준으로 첨부 (표시용)
    for k in ("covered", "partial", "missed", "hallucinations", "noise",
              "category_coverage", "missed_categories"):
        out[k] = first.get(k, [])
    out["judges_used"] = list(use)
    return out


def evaluate(result_files, mode, backend, checklist_path, out_root):
    import pickle
    checklist = {}
    if mode == "gold":
        checklist = load_checklist(checklist_path)
        if not checklist:
            print(f"[오류] checklist 없음: {checklist_path} — build_gold_checklist_v3 먼저 실행")
            sys.exit(2)
        st = review_status(checklist)
        print(f"[eval_v3] checklist {st['n']}건 — 전문의검수 {st['reviewed']}건 / "
              f"잠정채택 {st['accepted_without_review']}건 / 잠정 {st['provisional']}건")
        if st["reviewed"] == 0:
            print("  ⚠ 전문의 검수 0건 — 모든 결과는 '잠정치'로 보고할 것 (reviewed는 사람만 설정)")

    split_df, _ = _load_split_df(mode)
    from ..config_v3 import VITAL_MAP_PKL
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    records = _collect_records(result_files, split_df, vital_map, checklist, mode)
    print(f"[eval_v3] 레코드 {len(records)}건 로드 ({len(result_files)}개 파일, mode={mode})")

    # judge별 순차 채점 (GPU 재사용)
    per_judge_all = {}
    for jk in EVAL_JUDGES:
        per_judge_all[jk] = _judge_pass(jk, records, backend, mode)

    # 파일(tag)별 조립 + 저장
    by_file = {}
    for i, r in enumerate(records):
        by_file.setdefault(r["file"], []).append(i)

    exit_bad = False
    for rf, idxs in by_file.items():
        rf = Path(rf)
        tag = rf.parent.name
        model_key, exp_key = split_tag(tag)
        allowed = judges_for(model_key)
        same_family = [j for j in EVAL_JUDGES
                       if model_key in MODELS and model_family(j) == model_family(model_key)]

        out_records, official_scores = [], []
        ja, jb = [], []
        for i in idxs:
            r = records[i]
            pj = {jk: per_judge_all[jk][i] for jk in EVAL_JUDGES}
            off = _official(pj, allowed)
            official_scores.append(off)
            ja.append(pj[EVAL_JUDGES[0]])
            jb.append(pj[EVAL_JUDGES[1]])
            out_records.append({
                **{k: r["rec"].get(k) for k in ("idx", "sid")},
                "generated": r["gen"],
                "gen_status": r["gen_status"],
                "gen_flags": r["gen_flags"],
                "no_issue": r["no_issue"],
                "official": off,
                "per_judge": {jk: {a: pj[jk].get(a) for a in
                                   ("coverage", "faithfulness", "brevity",
                                    "composite", "gate", "excluded", "exclude_reason")}
                              for jk in EVAL_JUDGES},
                "judges_allowed": allowed,
                "same_family_judges_excluded": same_family,
            })

        out_dir = ensure_dir(Path(out_root) / tag)
        out_file = out_dir / rf.name.replace(".jsonl", f"_scores_v3.jsonl")
        _atomic_write_jsonl(out_file, out_records)

        # 요약 (fail-loud 집계)
        valid = [s for s in official_scores if s and not s.get("excluded")]
        excluded = [s for s in official_scores if s is None or s.get("excluded")]
        summary = dict(
            tag=tag, model=model_key, experiment=exp_key, mode=mode,
            n_total=len(official_scores), n_valid=len(valid), n_excluded=len(excluded),
            exclude_reasons={},
            judges=EVAL_JUDGES, judges_allowed=allowed,
            composite=bootstrap_ci([s["composite"] for s in valid],
                                   iters=STATS["bootstrap_iters"], seed=STATS["seed"]),
            coverage=bootstrap_ci([s["coverage"] for s in valid],
                                  iters=STATS["bootstrap_iters"], seed=STATS["seed"]),
            faithfulness=bootstrap_ci([s["faithfulness"] for s in valid],
                                      iters=STATS["bootstrap_iters"], seed=STATS["seed"]),
            brevity=bootstrap_ci([s["brevity"] for s in valid],
                                 iters=STATS["bootstrap_iters"], seed=STATS["seed"]),
            micro=micro_coverage(valid),
            gates=dict(degenerate=sum(1 for s in valid if s.get("gate") == "degenerate"),
                       missed_abnormal=sum(1 for s in valid
                                           if s.get("gate") == "missed_abnormal")),
            no_issue_rate=round(sum(1 for i2 in idxs if records[i2]["no_issue"])
                                / max(1, len(idxs)), 4),
            judge_agreement=judge_agreement(ja, jb, axis="composite"),
            checklist_review=review_status(checklist) if mode == "gold" else None,
        )
        for s in excluded:
            key = (s or {}).get("exclude_reason", "unknown") or "unknown"
            key = key.split("(")[0]
            summary["exclude_reasons"][key] = summary["exclude_reasons"].get(key, 0) + 1

        sum_file = out_dir / rf.name.replace(".jsonl", f"_summary_v3.json")
        tmp = sum_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(sum_file)

        comp = summary["composite"]
        print(f"\n[{tag}] n={summary['n_valid']}/{summary['n_total']} 유효  "
              f"COMPOSITE={comp['mean']} CI=[{comp['lo']},{comp['hi']}]  "
              f"COV={summary['coverage']['mean']} FAITH={summary['faithfulness']['mean']} "
              f"BREV={summary['brevity']['mean']}  "
              f"judge일치 rho={summary['judge_agreement']['spearman']}")
        print(f"  저장: {out_file}")

        if mode == "gold" and summary["n_total"] > 0 and \
                summary["n_valid"] / summary["n_total"] < MIN_VALID_CASE_RATIO:
            print(f"  ✗ 유효 비율 {summary['n_valid']}/{summary['n_total']} < "
                  f"{MIN_VALID_CASE_RATIO} — judge/checklist 인프라 문제. 결과 사용 금지.")
            exit_bad = True

    if exit_bad:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="평가 v3 (3축 + 교차 judge + fail-loud)")
    ap.add_argument("--result_files", nargs="+", required=True)
    ap.add_argument("--mode", choices=["gold", "dev"], default="dev")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--checklist", type=str, default=str(GOLD_CHECKLIST_JSON))
    ap.add_argument("--out", type=str, default=str(EVAL_OUT))
    args = ap.parse_args()
    evaluate(args.result_files, args.mode, args.backend, args.checklist, args.out)


if __name__ == "__main__":
    main()
