"""
verify_output.py — inference 결과 샘플 출력 & 빠른 검증

실행:
  python verify_output.py --model_path /path/to/model --gpus 4 --n 5
  python verify_output.py --result_file outputs/inference/llama_raw/gold_results.jsonl
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import sys, os, argparse, json, pickle
from pathlib import Path

def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    return p.parse_known_args()[0].gpus

_gpus = _early_gpus()
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import (
    GOLD_PKL, GOLD_REF_PKL, VITAL_MAP_PKL,
    SYSTEM_PROMPT, build_user_prompt,
    EMR_PREOP_SUM_COL, EMR_PREMED_COL,
)


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
            vals.extend([str(x) for x in vlist] if isinstance(vlist, list) else [str(vlist)])
        return " ".join(vals)
    return str(v)


def _emr_text(row):
    parts = []
    preop  = _get(row, EMR_PREOP_SUM_COL)
    premed = _get(row, EMR_PREMED_COL)
    anrec  = _get(row, ("마취기록", "기록", ""))
    if preop:  parts.append(f"[마취전 환자상태 요약]\n{preop}")
    if premed: parts.append(f"[수술전 준비사항 및 Premedication]\n{premed}")
    if anrec:  parts.append(f"[마취기록]\n{anrec}")
    return "\n\n".join(parts)


def infer_samples(model_path: str, n: int, gpus: str):
    """모델로 gold 샘플 n개 inference 후 출력."""
    print(f"\n모델 로드: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    mdl.eval()

    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    # gold ref (Human handoff)
    try:
        ref_df = pd.read_pickle(GOLD_REF_PKL)
    except Exception:
        ref_df = None

    print(f"Gold 샘플 {min(n, len(gold_df))}건 inference 중...\n")

    for i, (_, row) in enumerate(gold_df.iloc[:n].iterrows()):
        emr = _emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vital_map.get(sid, "")
        user  = build_user_prompt(emr, vital)

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(mdl.device)

        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Human reference
        human = ""
        if ref_df is not None and i < len(ref_df):
            try:
                human = str(ref_df.iloc[i][("마취기록", "기록", "")])
            except Exception:
                pass

        # 수술명
        try:
            v2 = row[("수술", "수술명", "")]
            opname = str(v2.iloc[0]) if hasattr(v2, "iloc") else str(v2)
        except Exception:
            opname = "N/A"

        print(f"{'─'*70}")
        print(f"[{i+1}] 수술ID={sid}  수술명: {opname}")
        if vital:
            print(f"\n  [바이탈 요약]\n  {vital.replace(chr(10), chr(10)+'  ')}")
        print(f"\n  [모델 생성]\n  {generated}")
        if human and human != "nan":
            print(f"\n  [Human 정답]\n  {human}")
        print()

    print(f"{'─'*70}")
    print(f"  GPU 메모리: {torch.cuda.memory_allocated()/1e9:.1f}GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")


def show_results(result_file: str, n: int):
    """저장된 jsonl 결과 파일 출력 (inference 없이)."""
    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    try:
        ref_df = pd.read_pickle(GOLD_REF_PKL)
    except Exception:
        ref_df = None

    lines = [json.loads(l) for l in Path(result_file).read_text().splitlines() if l.strip()]
    print(f"\n결과 파일: {result_file}  ({len(lines)}건)")

    has_score = "brevity_score" in lines[0] if lines else False

    for rec in lines[:n]:
        i   = rec["idx"]
        sid = rec.get("sid", -1)
        gen = rec["generated"]

        row = gold_df.iloc[i] if i < len(gold_df) else None
        vital = vital_map.get(sid, "")
        try:
            v2 = row[("수술", "수술명", "")]
            opname = str(v2.iloc[0]) if hasattr(v2, "iloc") else str(v2)
        except Exception:
            opname = "N/A"

        human = ""
        if ref_df is not None and i < len(ref_df):
            try:
                human = str(ref_df.iloc[i][("마취기록", "기록", "")])
            except Exception:
                pass

        print(f"{'─'*70}")
        print(f"[{i+1}] 수술ID={sid}  {opname}")
        if vital:
            print(f"\n  [바이탈]\n  {vital.replace(chr(10), chr(10)+'  ')}")
        print(f"\n  [생성]\n  {gen}")
        if human and human != "nan":
            print(f"\n  [Human]\n  {human}")
        if has_score:
            b = rec.get("brevity_score", "-")
            c = rec.get("critical_score", "-")
            print(f"\n  [점수]  brevity={b}  critical={c}  sum={rec.get('sum_score','-')}")
        print()


def main():
    parser = argparse.ArgumentParser(description="inference 결과 샘플 검증")
    parser.add_argument("--model_path",   type=str, default=None,
                        help="모델 경로 (직접 inference)")
    parser.add_argument("--result_file",  type=str, default=None,
                        help="저장된 jsonl 결과 파일 경로")
    parser.add_argument("--n",    type=int, default=5,
                        help="출력할 샘플 수 (기본 5)")
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    if args.result_file:
        show_results(args.result_file, args.n)
    elif args.model_path:
        infer_samples(args.model_path, args.n, args.gpus)
    else:
        parser.error("--model_path 또는 --result_file 중 하나 필요")


if __name__ == "__main__":
    main()
