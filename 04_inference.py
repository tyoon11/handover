"""
04_inference.py — 배치 Inference

실행 예시:
  python 04_inference.py --model_path outputs/sft/llama_3ep/final --split eval --gpus 0
  python 04_inference.py --model_path outputs/rlaif/llama_dpo/final --split sft --gpus 2,3
"""

import sys, os, argparse

def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    args, _ = p.parse_known_args()
    return args.gpus

_gpus = _early_parse()
if _gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_gpus}")
elif "CUDA_VISIBLE_DEVICES" in os.environ:
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (환경변수)")
else:
    print("[GPU] CUDA_VISIBLE_DEVICES 미지정 → 전체 GPU 사용")

import pickle
import json
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    SFT_PKL, SELFJUDGE_PKL, GOLD_PKL, RLHF_PKL,
    VITAL_MAP_PKL, INFER_OUT,
    SYSTEM_PROMPT, build_user_prompt,
    INFER_CONFIG,
    EMR_PREOP_SUM_COL, EMR_PREMED_COL,
)

SPLIT_MAP = {
    "sft":       SFT_PKL,
    "selfjudge": SELFJUDGE_PKL,
    "gold":      GOLD_PKL,
    "rlhf":      RLHF_PKL,
}


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
            if isinstance(vlist, list):
                vals.extend([str(x) for x in vlist if x is not None])
            else:
                vals.append(str(vlist))
        return " ".join(vals)
    return str(v)


def _emr_text(row):
    preop  = _get(row, EMR_PREOP_SUM_COL)
    premed = _get(row, EMR_PREMED_COL)
    anrec  = _get(row, ("마취기록", "기록", ""))
    parts = []
    if preop:  parts.append(f"[마취전 환자상태 요약]\n{preop}")
    if premed: parts.append(f"[수술전 준비사항 및 Premedication]\n{premed}")
    if anrec:  parts.append(f"[마취기록]\n{anrec}")
    return "\n\n".join(parts)


def load_model(model_path: str, base_model_id: str = None):
    """모델 로드. PEFT 체크포인트면 base + adapter 방식."""
    adapter_config = Path(model_path) / "adapter_config.json"
    is_peft = adapter_config.exists()

    if is_peft and base_model_id:
        print(f"  PEFT 모델 로드: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print(f"  전체 모델 로드: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def run_inference(args):
    model_path = args.model_path
    split_pkl  = SPLIT_MAP[args.split]

    tag = args.out_tag if args.out_tag else Path(model_path).name
    out_dir = INFER_OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.split}_results.jsonl"

    n_gpu = torch.cuda.device_count()
    print(f"\n[Inference]")
    print(f"  모델:   {model_path}")
    print(f"  Split:  {args.split}  ({split_pkl.name})")
    print(f"  출력:   {out_file}")
    print(f"  GPU:    {n_gpu}개")

    # 데이터 로드
    df = pd.read_pickle(split_pkl)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  데이터: {len(df)}건")

    # 모델 로드
    model, tokenizer = load_model(model_path, args.base_model)

    cfg = INFER_CONFIG
    results = []

    with open(out_file, "w", encoding="utf-8") as fout:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
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
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=cfg["max_new_tokens"],
                    do_sample=cfg.get("do_sample", False),
                    pad_token_id=tokenizer.eos_token_id,
                )
                if cfg.get("temperature") is not None:
                    gen_kwargs["temperature"] = cfg["temperature"]
                if cfg.get("top_p") is not None and cfg.get("do_sample"):
                    gen_kwargs["top_p"] = cfg["top_p"]
                output_ids = model.generate(**inputs, **gen_kwargs)

            gen = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            rec = {
                "idx": i,
                "sid": sid,
                "generated": gen,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)

    print(f"\n[완료] {len(results)}건 저장: {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="모델 경로 (full model or PEFT adapter 폴더)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="PEFT 체크포인트일 때 베이스 모델 경로")
    parser.add_argument("--split", choices=list(SPLIT_MAP.keys()), default="gold",
                        help="평가 데이터 split")
    parser.add_argument("--out_tag", type=str, default=None,
                        help="출력 폴더 태그 (예: llama_raw). 없으면 모델명으로 자동 결정")
    parser.add_argument("--gpus", type=str, default=None,
                        help="사용할 GPU 번호. 예: '0' 또는 '0,1'")
    run_inference(parser.parse_args())
