"""
check_seq_length.py — 학습 데이터 토큰 길이 분포 확인
실행: python check_seq_length.py --base llama
"""
import argparse, os, sys, pickle
import pandas as pd
import numpy as np

def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    args, _ = p.parse_known_args()
    return args.gpus

_gpus = _early_parse()
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

from transformers import AutoTokenizer
from config import (
    SYNTH_PKL, VITAL_MAP_PKL, SFT_MODELS,
    SYSTEM_PROMPT, build_user_prompt, build_emr_text,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=list(SFT_MODELS.keys()), default="llama")
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    model_id = str(SFT_MODELS[args.base])
    print(f"토크나이저 로드: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("데이터 로드 중...")
    synth_df = pd.read_pickle(SYNTH_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  합성데이터: {len(synth_df)}건")

    lengths = []
    for _, row in synth_df.iterrows():
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital   = vital_map.get(sid, "")
        chosen  = str(row.get("chosen", ""))
        user    = build_user_prompt(emr, vital)
        msgs = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": chosen},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        toks = tokenizer(text, return_tensors="pt").input_ids.shape[1]
        lengths.append(toks)

    lengths = np.array(lengths)
    print(f"\n{'='*50}")
    print(f"[토큰 길이 분포]  n={len(lengths)}")
    print(f"  min   : {lengths.min()}")
    print(f"  p25   : {int(np.percentile(lengths, 25))}")
    print(f"  p50   : {int(np.percentile(lengths, 50))}")
    print(f"  p75   : {int(np.percentile(lengths, 75))}")
    print(f"  p90   : {int(np.percentile(lengths, 90))}")
    print(f"  p95   : {int(np.percentile(lengths, 95))}")
    print(f"  p99   : {int(np.percentile(lengths, 99))}")
    print(f"  max   : {lengths.max()}")
    print(f"  mean  : {lengths.mean():.0f}")
    print()

    for thresh in [1024, 2048, 3072, 4096, 6144, 8192]:
        n_over = (lengths > thresh).sum()
        pct    = n_over / len(lengths) * 100
        print(f"  > {thresh:5d} 토큰: {n_over:4d}건 ({pct:.1f}%)  → 이 케이스 loss 무시됨")

    # 권장 max_seq_length
    p99 = int(np.percentile(lengths, 99))
    recommended = min(8192, int(np.ceil(p99 / 512) * 512))
    print(f"\n  권장 max_seq_length: {recommended}  (p99 기준 올림)")
    print(f"  현재 설정:           2048")
    print(f"  → config.py의 MAX_SEQ_LEN = {recommended} 으로 변경 권장")

if __name__ == "__main__":
    main()
