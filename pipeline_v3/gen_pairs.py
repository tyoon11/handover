"""
gen_pairs.py — 선호쌍/SFT타깃 생성 (v1 01_synthetic_gen.py 전면 대체)

v1 결함 → v3 수정:
  T1  few-shot이 평가셋 22케이스에서 나옴(유출)
      → held-out fewshot bank(RLHF 40건, 모델 초안)에서 행별 로테이션(random_state=row_idx)
  T2  기준답안이 '남의 환자 인계문 1개' 고정
      → reference-free: 해당 케이스 EMR 대비 채점 (기준답안 필드 제거)
  T3  '특이사항 없음' 편향 + 예외가 chosen으로 둔갑 + 파싱실패 준랜덤 점수
      → coverage 루브릭 추가(놓치면 1점), 생성실패/붕괴 후보는 제외, 파싱실패는 drop+로깅
  T4  judge 입력 4096 우측절단(응답·루브릭 잘림)
      → 8192 예산 + EMR-only 좌측절단 (fit_text_prompt)
  T8  greedy 동일후보/동점랜덤/마진없음
      → greedy+temperature 샘플로 후보 확대, dedup, margin >= PAIRGEN.min_margin 강제
  T9  SFT와 DPO가 같은 데이터 재사용
      → --split sft (jsft原) / --split dpo (selfjudge原, 정책 자체 출력 = on-policy)
  T7  judge 순환
      → 쌍 생성 judge = prometheus (평가 judge gemma4_31b/qwen35 와 다른 계열)

실행:
  # SFT 타깃: base 모델 2종의 후보 → judge 선별 (chosen만 SFT 타깃으로 사용)
  python -m pipeline_v3.gen_pairs --split sft --models llama qwen --gpus 0,1,2,3
  # DPO 쌍: SFT 정책의 on-policy 후보 → judge 선별
  python -m pipeline_v3.gen_pairs --split dpo --policy_ckpt outputs_v3/<run>/sft/llama_3ep/final \
      --base llama --out_tag llama_sft3ep --gpus 0,1,2,3
"""

import argparse
import json
import os
import re
from pathlib import Path


def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    g = p.parse_known_args()[0].gpus
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = g
        print(f"[GPU] CUDA_VISIBLE_DEVICES={g}")


_early_gpus()

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

from .config_v3 import (                # noqa: E402
    JUDGE_MAX_MODEL_LEN, MAX_PROMPT_TOKENS, PAIRGEN, PAIRGEN_JUDGE, PAIRS_OUT,
    SPLIT_SEED, ensure_dir, model_path,
)
from .data_splits import load_splits, write_split_manifest      # noqa: E402
from .eval_v3.cleaning import clean_v3                           # noqa: E402
from .make_fewshot_bank import load_fewshot_bank                 # noqa: E402
from .merging import merge_lora_for_vllm                         # noqa: E402
from .prompt_utils import (                                      # noqa: E402
    SYSTEM_PROMPT, build_emr_text, fit_chat_prompt, fit_text_prompt, get_sid,
)


# ── few-shot 블록 (행별 로테이션 — T1) ──────────────────────────────────────
def fewshot_block_for_row(bank: list, row_idx: int, n_shot: int) -> str:
    if not bank or n_shot <= 0:
        return ""
    rng = np.random.RandomState(SPLIT_SEED + row_idx)     # 행마다 다른 예시 조합
    picks = rng.choice(len(bank), size=min(n_shot, len(bank)), replace=False)
    parts = ["### EXAMPLES (follow style, do NOT copy content)"]
    for i, bi in enumerate(picks, 1):
        ex = bank[int(bi)]
        # 예시 EMR은 앞부분만 (예산 절약 — 스타일 앵커 목적이므로 충분)
        emr_head = ex["emr"][:1500]
        parts.append(f"\nExample {i}\n**EMR(요약)**\n{emr_head}\n"
                     f"**PACU/ICU Handoff**\n{ex['handoff']}\n--")
    return "\n".join(parts)


# ── 후보 생성 (vLLM: greedy 1 + temperature K — T8) ─────────────────────────
def generate_candidates(gen_model_dir: str, gen_key: str, rows: list, bank: list):
    """rows: [{row_idx, sid, emr, vital}] → {row_idx: [candidate_text, ...]} (정제·dedup 전)"""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(gen_model_dir, trust_remote_code=True)

    prompts = []
    for r in rows:
        fs = fewshot_block_for_row(bank, r["row_idx"], PAIRGEN["n_fewshot"])
        prompts.append(fit_chat_prompt(
            tok, r["emr"], r["vital"], system=SYSTEM_PROMPT,
            budget=MAX_PROMPT_TOKENS, enable_thinking=False, fewshot_block=fs))

    import torch
    llm = LLM(model=gen_model_dir, tensor_parallel_size=max(1, torch.cuda.device_count()),
              dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=MAX_PROMPT_TOKENS + 640, trust_remote_code=True)

    out = {r["row_idx"]: [] for r in rows}
    # (1) greedy
    sp_g = SamplingParams(max_tokens=512, temperature=0.0)
    for r, o in zip(rows, llm.generate(prompts, sp_g)):
        out[r["row_idx"]].append(o.outputs[0].text.strip())
    # (2) temperature 샘플 K개 (한 요청에 n개 — vLLM이 병렬 처리)
    sp_s = SamplingParams(max_tokens=512, temperature=PAIRGEN["temperature"],
                          top_p=PAIRGEN["top_p"], n=PAIRGEN["n_samples"],
                          seed=SPLIT_SEED)
    for r, o in zip(rows, llm.generate(prompts, sp_s)):
        out[r["row_idx"]].extend(c.text.strip() for c in o.outputs)

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def dedup_clean_candidates(cands: list):
    """clean_v3 통과(ok) + 중복 제거. 붕괴 후보는 학습 타깃 후보에서 제외 (T3)."""
    seen, out = set(), []
    n_drop = 0
    for c in cands:
        text, status, _flags = clean_v3(c)
        if status != "ok":
            n_drop += 1
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out, n_drop


# ── Judge (prometheus 계열 — reference-free, EMR 대비 채점: T2/T3/T4/T7) ────
_JUDGE_TMPL = """###Task Description:
An EMR-grounded evaluation. A source EMR (with intraoperative vital summary), a handoff
response to evaluate, and a score rubric are given.
1. Write brief feedback strictly based on the score rubric, judging ONLY against the EMR.
2. After feedback, write a score (integer 1-5).
3. Output format: "(feedback) [RESULT] (score)"
4. No other opening or closing.

###Source EMR (ground truth for this patient):
{emr}

###Intraoperative vital summary:
{vital}

###Response to evaluate:
{response}

###Score Rubric:
{rubric}

###Feedback:"""

# T3 교정 핵심: '전부 놓치고 특이사항 없음'이 최고점이 되지 않도록 coverage 루브릭 신설
_COVERAGE_RUBRIC = (
    "Clinical coverage (recall of actionable abnormal findings). "
    "Score 5: mentions ALL clinically actionable abnormal findings present in the EMR/vitals "
    "(airway events, hemodynamic instability, bleeding/transfusion, major or congenital disease, "
    "intra-op events, critical lines/devices, drug effects). "
    "Score 3: mentions the most critical findings but misses some. "
    "Score 1: says '특이사항 없음' despite clear abnormal findings in the EMR, or misses most. "
    "If the EMR truly has no meaningful abnormal findings, an exact '특이사항 없음' scores 5."
)

_FIDELITY_BREVITY_RUBRIC = (
    "Fidelity and exception-based brevity. "
    "Score 5: every statement is supported by the EMR/vitals, only abnormal findings are "
    "mentioned, in 1-5 short formal Korean sentences. "
    "Score 3: minor unsupported details or mild verbosity. "
    "Score 1: fabricates findings not in the EMR, states r/o as confirmed, or is dominated by "
    "normal/routine/administrative content."
)

_RE_RESULT = re.compile(r"\[RESULT\]\s*\(?\s*([1-5])\s*\)?")


def _parse_result(text: str):
    """[RESULT] n 만 신뢰. '아무 숫자나' fallback 금지 (T3). 실패 → None."""
    m = _RE_RESULT.search(text or "")
    return float(m.group(1)) if m else None


def judge_candidates(rows: list, cand_map: dict, backend: str):
    """모든 (row, candidate)를 두 루브릭으로 배치 채점.
    반환 {row_idx: [ {text, cov, fid, total} ... ]} — 파싱실패 후보는 drop."""
    from transformers import AutoTokenizer
    from .eval_v3.engine import JudgeEngine

    judge_dir = str(model_path(PAIRGEN_JUDGE))
    engine = JudgeEngine(judge_dir, backend=backend,
                         gen_cfg=dict(max_new_tokens=PAIRGEN["judge_max_tokens"],
                                      temperature=0.0),
                         max_model_len=JUDGE_MAX_MODEL_LEN)
    jtok = AutoTokenizer.from_pretrained(judge_dir, trust_remote_code=True)
    budget = JUDGE_MAX_MODEL_LEN - PAIRGEN["judge_max_tokens"] - 64

    tasks = []      # (row_idx, cand_i, rubric_name, prompt)
    for r in rows:
        for ci, cand in enumerate(cand_map.get(r["row_idx"], [])):
            for name, rubric in (("cov", _COVERAGE_RUBRIC),
                                 ("fid", _FIDELITY_BREVITY_RUBRIC)):
                def render(emr, _v=r["vital"], _c=cand, _rub=rubric):
                    return _JUDGE_TMPL.format(emr=emr, vital=_v or "(없음)",
                                              response=_c, rubric=_rub)
                prompt = fit_text_prompt(jtok, render, r["emr"], budget)
                tasks.append((r["row_idx"], ci, name, prompt))

    print(f"[gen_pairs] judge 호출 {len(tasks)}건 (rubric 2 × 후보)")
    texts = engine.chat([t[3] for t in tasks], system=None)
    engine.unload()

    scores = {}
    n_fail = 0
    for (row_idx, ci, name, _), text in zip(tasks, texts):
        s = _parse_result(text)
        if s is None:
            n_fail += 1
            continue
        scores.setdefault(row_idx, {}).setdefault(ci, {})[name] = s

    if n_fail:
        print(f"  ⚠ judge 파싱 실패 {n_fail}건 → 해당 (후보,루브릭) drop (준랜덤 점수 주입 금지)")

    out = {}
    for r in rows:
        row_idx = r["row_idx"]
        cands = cand_map.get(row_idx, [])
        entries = []
        for ci, cand in enumerate(cands):
            sc = scores.get(row_idx, {}).get(ci, {})
            if "cov" not in sc or "fid" not in sc:
                continue        # 두 루브릭 모두 성공한 후보만 사용
            entries.append(dict(text=cand, cov=sc["cov"], fid=sc["fid"],
                                total=2.0 * sc["cov"] + sc["fid"]))   # cov 가중 2배 (만점 15)
        out[row_idx] = entries
    return out


def select_pairs(rows: list, judged: dict):
    """chosen/rejected 선정: dedup·마진·동일성 검사 (T8). 반환 (records, 통계)."""
    stats = dict(kept=0, too_few=0, no_margin=0, identical=0)
    records = []
    for r in rows:
        entries = judged.get(r["row_idx"], [])
        if len(entries) < 2:
            stats["too_few"] += 1
            continue
        entries.sort(key=lambda e: e["total"], reverse=True)
        chosen, rejected = entries[0], entries[-1]
        if _normalize(chosen["text"]) == _normalize(rejected["text"]):
            stats["identical"] += 1
            continue
        if chosen["total"] - rejected["total"] < PAIRGEN["min_margin"]:
            stats["no_margin"] += 1
            continue
        records.append(dict(
            row_idx=r["row_idx"], sid=r["sid"],
            chosen=chosen["text"], rejected=rejected["text"],
            chosen_score=chosen["total"], rejected_score=rejected["total"],
            margin=round(chosen["total"] - rejected["total"], 2),
            n_candidates=len(entries),
            all_scores=[dict(total=e["total"], cov=e["cov"], fid=e["fid"])
                        for e in entries],
        ))
        stats["kept"] += 1
    return records, stats


def main():
    ap = argparse.ArgumentParser(description="선호쌍/SFT타깃 생성 v3")
    ap.add_argument("--split", choices=["sft", "dpo"], required=True)
    ap.add_argument("--models", nargs="+", default=["llama", "qwen"],
                    help="--split sft: 후보 생성 base 모델 키")
    ap.add_argument("--policy_ckpt", type=str, default=None,
                    help="--split dpo: on-policy 후보를 만들 정책 체크포인트 (없으면 --base raw)")
    ap.add_argument("--base", type=str, default=None,
                    help="--split dpo: 정책의 base 모델 키 (merge/템플릿용)")
    ap.add_argument("--out_tag", type=str, default=None,
                    help="출력 파일 태그 (dpo: pairs_dpo_<tag>.pkl)")
    ap.add_argument("--max_rows", type=int, default=PAIRGEN["max_rows"])
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--debug", action="store_true", help="10행만")
    args = ap.parse_args()

    splits = load_splits(need=("gold", args.split, "fewshot"))
    src_df = splits[args.split]
    vital_map = splits["vital_map"]
    bank = load_fewshot_bank()
    if not bank:
        raise FileNotFoundError(
            "fewshot bank 없음 — 먼저 실행: python -m pipeline_v3.make_fewshot_bank")

    # 행 준비 (+ 결정적 서브샘플)
    rows = []
    for i in range(len(src_df)):
        row = src_df.iloc[i]
        sid = get_sid(row)
        rows.append(dict(row_idx=i, sid=sid, emr=build_emr_text(row),
                         vital=vital_map.get(sid, "")))
    if args.debug:
        rows = rows[:10]
    elif args.max_rows and len(rows) > args.max_rows:
        rng = np.random.RandomState(SPLIT_SEED)
        keep = set(rng.choice(len(rows), size=args.max_rows, replace=False).tolist())
        rows = [r for r in rows if r["row_idx"] in keep]
        print(f"[gen_pairs] 서브샘플 {len(rows)}행 (seed={SPLIT_SEED})")

    ensure_dir(PAIRS_OUT)
    write_split_manifest(PAIRS_OUT, splits)

    # 후보 생성
    cand_map = {r["row_idx"]: [] for r in rows}
    if args.split == "sft":
        gen_specs = [(k, str(model_path(k))) for k in args.models]
    else:
        if args.policy_ckpt:
            if not args.base:
                raise ValueError("--split dpo에 --policy_ckpt를 주면 --base 필수")
            merged = merge_lora_for_vllm(args.policy_ckpt, str(model_path(args.base)))
            gen_specs = [(args.base, merged)]
        else:
            if not args.base:
                raise ValueError("--split dpo는 --policy_ckpt 또는 --base 필요")
            gen_specs = [(args.base, str(model_path(args.base)))]

    n_drop_total = 0
    for gen_key, gen_dir in gen_specs:
        print(f"\n[gen_pairs] 후보 생성: {gen_key} ({gen_dir})")
        raw_map = generate_candidates(gen_dir, gen_key, rows, bank)
        for ridx, cands in raw_map.items():
            cand_map[ridx].extend(cands)

    for ridx in list(cand_map):
        cand_map[ridx], nd = dedup_clean_candidates(cand_map[ridx])
        n_drop_total += nd
    print(f"[gen_pairs] 붕괴/중복 제거 후 후보: 평균 "
          f"{np.mean([len(v) for v in cand_map.values()]):.1f}개/행 (붕괴 drop {n_drop_total})")

    # judge 채점 + 쌍 선정
    judged = judge_candidates(rows, cand_map, args.backend)
    records, stats = select_pairs(rows, judged)
    print(f"[gen_pairs] 선정: {stats}")

    # 저장: 원본 행 + chosen/rejected (학습 스크립트가 EMR 재구성에 사용)
    out_rows = []
    for rec in records:
        row = src_df.iloc[rec["row_idx"]].copy()
        for k in ("chosen", "rejected", "chosen_score", "rejected_score", "margin"):
            row[k] = rec[k]
        out_rows.append(row)
    pairs_df = pd.DataFrame(out_rows)

    tag = args.out_tag or (args.split if args.split == "sft"
                           else f"{args.base}_raw")
    out_pkl = PAIRS_OUT / ("pairs_sft.pkl" if args.split == "sft"
                           else f"pairs_dpo_{tag}.pkl")
    audit = PAIRS_OUT / (out_pkl.stem + "_audit.jsonl")
    pairs_df.to_pickle(out_pkl)
    with open(audit, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({k: rec[k] for k in
                                ("row_idx", "sid", "chosen_score", "rejected_score",
                                 "margin", "n_candidates", "all_scores")},
                               ensure_ascii=False) + "\n")
    meta = dict(split=args.split, tag=tag, n_pairs=len(pairs_df), stats=stats,
                pairgen=dict(PAIRGEN), judge=PAIRGEN_JUDGE,
                generators=[g for g, _ in gen_specs])
    (PAIRS_OUT / (out_pkl.stem + "_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen_pairs] 저장: {out_pkl} ({len(pairs_df)}쌍) / 감사로그: {audit.name}")


if __name__ == "__main__":
    main()
