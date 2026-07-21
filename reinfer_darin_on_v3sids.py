#!/usr/bin/env python3
"""
reinfer_darin_on_v3sids.py — 다린 체크포인트를 v3 sid(gold/dev)에 재추론 (폐쇄망, GPU)

왜 필요한가
  다린 기존 inferenced 출력은 evaluation_251008(100건) 전용이고, v3 gold(22)·dev(110)와
  sid가 0 겹침(완전 disjoint split). 그래서 기존 출력으론 sid 병기가 불가능하다.
  → 다린이 저장한 체크포인트(experiments/)를 v3의 '바로 그 sid'에 다시 추론해야
    같은 입력으로 '옛 모델 vs v3' 나란히 비교가 성립한다.

입력 프롬프트 주의 (중요)
  - SYSTEM + user의 EMR 부분은 다린=v3 동일.
  - 다린은 '바이탈 요약'을 입력에 넣지 않았다(EMR only). v3는 ### INTRAOPERATIVE
    VITAL SUMMARY 블록을 추가로 넣는다. 이 스크립트는 다린 make_self_judge_format을
    verbatim 재현하므로 **바이탈을 넣지 않는다**(다린 모델은 학습 때 바이탈을 본 적이
    없어 넣으면 오히려 왜곡). 따라서 병기는 'EMR-only 옛 모델' vs 'EMR+바이탈 v3'로,
    v3 개편(바이탈 입력 추가 포함)의 총효과를 보는 비교다.

무엇을 하는가
  - v3 split 로드(pipeline_v3.data_splits.load_splits) → gold/dev df (권위 있는 sid).
  - 다린 model_L 각 변형: base + experiments/{setting}/{model} adapter 로드(다린과 동일),
    make_self_judge_format(다린 코드 verbatim)으로 query 생성 → greedy 생성.
  - {out_root}/{raw|jsft|self_judge}/{model}.pkl 로 저장('수술 ID','인계요약지').
    → 이 경로/컬럼은 report_v3 --include_darin 이 그대로 읽는다.

이후
  HANDOVER_RUN_ID=v3_20260709 python -m pipeline_v3.report_v3 \
      --split gold --include_source --include_darin --darin_root {out_root}
  (dev도 함께 생성했다면 --split dev 로도 병기됨)

사용 (v3 repo 루트에서 실행 — pipeline_v3 import 되는 곳)
  CUDA_VISIBLE_DEVICES=6 python reinfer_darin_on_v3sids.py \
      --split gold \
      --experiments_root ~/workspace/data/HANDOVER_인계용_다린/experiments \
      --out_root         ~/workspace/data/HANDOVER_인계용_다린/data/inferenced_v3sids \
      --cache_dir        /home/coder/workspace/data/share/_hf_models/ \
      --skip_done
"""
import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from pipeline_v3.data_splits import load_splits
from pipeline_v3.prompt_utils import get_sid

# 다린 model_L (evaluation.ipynb cell 47과 동일). (setting, model_type, is_raw)
MODEL_L = [
    ("jsft", "llama", True),          # = raw llama (base only)
    ("jsft", "llama", False),         # sft_1ep
    ("jsft", "llama_3epoch", False),  # sft_3ep
    ("self_judge", "llama_dpo", False),
    ("self_judge", "llama_simpo", False),
    ("self_judge", "llama_dpo_jsft_1epoch", False),
    ("self_judge", "llama_dpo_jsft_3epoch", False),
    ("jsft", "qwen", True),           # = raw qwen (base only)
    ("jsft", "qwen", False),
    ("jsft", "qwen_3epoch", False),
    ("self_judge", "qwen_dpo", False),
    ("self_judge", "qwen_simpo", False),
    ("self_judge", "qwen_dpo_jsft_1epoch", False),
    ("self_judge", "qwen_dpo_jsft_3epoch", False),
]


# ── 다린 프롬프트/EMR 포맷 (inference_llms.ipynb verbatim, tokenizer/model_type 인자화) ──
def totals_process(row):
    row = row[row.notnull()]
    return "\n".join(f"{k}: {v}" for k, v in row.items())


def emr_input_format(df, idx):
    return f"""- Pre-anesthetic Patient Conditions Summary
{df['마취전평가']['마취 전 상태 평가']['마취 전 상태평가 -> 마취전 환자상태 요약'].iloc[idx]}

- Preoperative Preparations and Premedication
{df['마취전평가']['마취 전 상태 평가']['마취 전 상태평가 -> 수술전 준비사항과 Premedication -> Premedication'].iloc[idx]}

- Anesthetic TOTALS
{totals_process(df['마취기록']['마취기록TOTALS'].iloc[idx])}

- Anesthetic Record
{df['마취기록']['기록'].iloc[idx]}"""


SYSTEM_PROMPT = """You are an anesthesiologist giving an ultra-brief OR to PACU/ICU handoff AFTER surgery has fully ended.
Do NOT ask for or suggest any intraoperative checks; only summarize key post-op relevant findings from the EMR.

RULES:
- If no clinically meaningful abnormal findings, output EXACTLY and ONLY "특이사항 없음".
- If any exist, output 1-5 VERY short sentences in formal Korean.
- NEVER include normal/stable findings, routine vitals, surgery steps, administrative drug info, or transfer phrases.
- NO lists, NO explanations, NO repetition.

Focus only on abnormal findings relevant after surgery: airway/respiratory status, hemodynamics/bleeding/transfusion,
major or congenital disease, intra-op events already recorded, drug effects, essential lines/devices,
and cooperation/agitation risk. Pediatric airway, fluids, and drug sensitivity are especially important."""


def make_self_judge_format(df, tokenizer, model_type):
    data = []
    for idx in range(len(df)):
        user_prompt = f"""Using the EMR below, generate an ultra-brief PACU/ICU handoff.

- If there are NO clinically meaningful issues, output exactly and only "특이사항 없음".
- If there ARE issues, output 1-5 very short sentences in formal Korean.
- Do NOT include normal findings, routine or administrative details, or any request to re-check intraoperative events.

Focus only on post-op relevant abnormalities: airway/respiratory status, hemodynamics/bleeding/transfusion,
major or congenital disease, intra-op events already recorded, drug effects, lines/devices, and cooperation/agitation risk.

### EMR
{emr_input_format(df, idx)}

### OUTPUT
"""
        if "System role not supported" in (tokenizer.chat_template or ""):  # gemma 계열
            messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_prompt}]
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}]
        if "qwen" in model_type:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        data.append({"query": input_text})
    return Dataset.from_pandas(pd.DataFrame(data))


def out_path(out_root, setting_type, model_type, is_raw):
    if is_raw:
        return os.path.join(out_root, "raw", f"{model_type.split('_')[0]}.pkl")
    return os.path.join(out_root, setting_type, f"{model_type}.pkl")


def load_model(setting_type, model_type, is_raw, experiments_root, cache_dir, max_gb):
    base = ("meta-llama/Meta-Llama-3.1-8B-Instruct" if "llama" in model_type
            else "Qwen/Qwen3-8B")
    print(f"  base={base}  raw={is_raw}")
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory=({0: f"{max_gb}GB"} if max_gb else None),
        cache_dir=cache_dir or None)
    if not is_raw:
        peft_dir = os.path.join(experiments_root, setting_type, model_type)
        print(f"  adapter={peft_dir}")
        model = PeftModel.from_pretrained(model, peft_dir)
        tok = AutoTokenizer.from_pretrained(peft_dir, use_fast=True)
    else:
        tok = AutoTokenizer.from_pretrained(base, use_fast=True)
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model.eval()
    return model, tok


def generate_for_df(model, tok, df, model_type):
    ds = make_self_judge_format(df, tok, model_type)
    gens, sids = [], []
    for idx in tqdm(range(len(df)), leave=False):
        prompt = ds[idx]["query"]
        inputs = tok(prompt, padding=True, truncation=True, max_length=2048,
                     return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=768, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gens.append(tok.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True).strip())
        sids.append(get_sid(df.iloc[idx]))
    return pd.DataFrame({"수술 ID": sids, "인계요약지": gens})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["gold", "dev", "both"], default="gold")
    ap.add_argument("--experiments_root", required=True,
                    help=".../HANDOVER_인계용_다린/experiments")
    ap.add_argument("--out_root", required=True,
                    help="저장 루트(=report_v3 --darin_root). 예 .../data/inferenced_v3sids")
    ap.add_argument("--cache_dir", default=None, help="base 모델 HF 캐시 경로")
    ap.add_argument("--max_gb", type=int, default=0, help="GPU0 max_memory GB(0=제한없음)")
    ap.add_argument("--skip_raw", action="store_true", help="raw(base) 변형 건너뛰기")
    ap.add_argument("--skip_done", action="store_true", help="출력 pkl 있으면 건너뛰기")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    need = ("gold", "dev") if args.split == "both" else (args.split,)
    sp = load_splits(need=need)
    targets = {s: sp[s] for s in need}
    for s, df in targets.items():
        print(f"[target] {s}: {len(df)}건")

    todo = [m for m in MODEL_L if not (args.skip_raw and m[2])]
    for setting_type, model_type, is_raw in todo:
        op = out_path(args.out_root, setting_type, model_type, is_raw)
        if args.skip_done and os.path.exists(op):
            print(f"[skip_done] {op}")
            continue
        print(f"\n=== {setting_type}/{model_type} (raw={is_raw}) ===")
        model, tok = load_model(setting_type, model_type, is_raw,
                                args.experiments_root, args.cache_dir, args.max_gb)
        frames = [generate_for_df(model, tok, df, model_type) for df in targets.values()]
        os.makedirs(os.path.dirname(op), exist_ok=True)
        pd.concat(frames, ignore_index=True).drop_duplicates("수술 ID").to_pickle(op)
        print(f"[saved] {op}")
        del model
        torch.cuda.empty_cache()

    print("\n완료. 리포트 재생성:")
    print(f"  HANDOVER_RUN_ID=$RUN_ID python -m pipeline_v3.report_v3 "
          f"--split {'gold' if args.split!='dev' else 'dev'} "
          f"--include_source --include_darin --darin_root {args.out_root}")


if __name__ == "__main__":
    main()
