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
  # GPU 2장 병렬 (GPU당 모델 하나씩; 각 모델은 1 GPU에 통째로 — 모델을 쪼개지 않음)
  # 기본 --split both → gold(22)+dev(110) 모두 생성(한 pkl에 함께 저장).
  # base 모델은 local_models 하위에서 로드(Llama-3.1-8B-Instruct/, Qwen3-8B/).
  # --local_models 미지정 시 config MODEL_BASE(=HANDOVER_MODEL_DIR) 사용.
  python reinfer_darin_on_v3sids.py --gpus 6,7 \
      --experiments_root ~/workspace/data/HANDOVER_인계용_다린/experiments \
      --out_root         ~/workspace/data/HANDOVER_인계용_다린/data/inferenced_v3sids \
      --skip_done
  # 단일 GPU
  CUDA_VISIBLE_DEVICES=6 python reinfer_darin_on_v3sids.py \
      --experiments_root ... --out_root ... --skip_done
"""
import argparse
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

# 폐쇄망: HF 허브 온라인 조회 차단 → 로컬 캐시만 사용 (transformers import 전에 설정해야 함).
# 외부에서 export 로 0을 주면 존중(온라인 환경 호환).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from pipeline_v3.data_splits import load_splits
from pipeline_v3.prompt_utils import get_sid
from pipeline_v3.config_v3 import (  # 로컬 base 모델 루트 + 다린 경로 기본값
    MODEL_BASE, DARIN_EXPERIMENTS, DARIN_INFER_OUT)

# 다린 base → local_models 하위 디렉토리명 (config_v3.MODELS 와 동일)
BASE_DIR = {"llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen3-8B"}

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


def load_model(setting_type, model_type, is_raw, experiments_root, models_root):
    key = "llama" if "llama" in model_type else "qwen"
    base = str(models_root / BASE_DIR[key])   # local_models 하위 로컬 경로
    print(f"  base={base}  raw={is_raw}")
    # 모델 전체를 '보이는 GPU 1장'(cuda:0)에 올린다. device_map='auto'와 달리
    # 여러 GPU가 보여도 한 모델을 쪼개지 않는다 (GPU당 모델 하나씩 병렬용).
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True)
    if not is_raw:
        peft_dir = os.path.join(experiments_root, setting_type, model_type)
        print(f"  adapter={peft_dir}")
        model = PeftModel.from_pretrained(model, peft_dir, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(peft_dir, use_fast=True,
                                            local_files_only=True)
    else:
        tok = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=True)
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


def launch_parallel(args, gpus):
    """GPU별로 이 스크립트를 --shard i/n 워커로 띄워 모델을 나눠 추론."""
    n = len(gpus)
    base = [sys.executable, os.path.abspath(__file__),
            "--split", args.split,
            "--experiments_root", args.experiments_root,
            "--out_root", args.out_root]
    if args.local_models:
        base += ["--local_models", args.local_models]
    if args.skip_raw:
        base += ["--skip_raw"]
    if args.skip_done:
        base += ["--skip_done"]

    procs = []
    for i, g in enumerate(gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = g
        cmd = base + ["--shard", f"{i}/{n}"]
        print(f"[launch] GPU {g} ← shard {i}/{n}")
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        print(f"[launch] 일부 워커 실패 rc={rcs}", file=sys.stderr)
        sys.exit(1)
    print("\n전체 완료. 리포트 재생성:")
    for s in (("gold", "dev") if args.split == "both" else (args.split,)):
        print(f"  HANDOVER_RUN_ID=$RUN_ID python -m pipeline_v3.report_v3 "
              f"--split {s} --include_source --include_darin --darin_root {args.out_root}")


def run_inference(args):
    """단일 프로세스(1 GPU) 추론. --shard i/n 이면 MODEL_L[i::n]만 담당."""
    torch.manual_seed(42); np.random.seed(42)
    need = ("gold", "dev") if args.split == "both" else (args.split,)
    sp = load_splits(need=need)
    targets = {s: sp[s] for s in need}
    tag = f"[shard {args.shard}] " if args.shard else ""
    for s, df in targets.items():
        print(f"{tag}[target] {s}: {len(df)}건")

    todo = [m for m in MODEL_L if not (args.skip_raw and m[2])]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo[i::n]
    print(f"{tag}담당 변형 {len(todo)}개 (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES','?')})")

    for setting_type, model_type, is_raw in todo:
        op = out_path(args.out_root, setting_type, model_type, is_raw)
        if args.skip_done and os.path.exists(op):
            print(f"{tag}[skip_done] {op}")
            continue
        print(f"\n{tag}=== {setting_type}/{model_type} (raw={is_raw}) ===")
        models_root = Path(args.local_models) if args.local_models else MODEL_BASE
        model, tok = load_model(setting_type, model_type, is_raw,
                                args.experiments_root, models_root)
        frames = [generate_for_df(model, tok, df, model_type) for df in targets.values()]
        os.makedirs(os.path.dirname(op), exist_ok=True)
        pd.concat(frames, ignore_index=True).drop_duplicates("수술 ID").to_pickle(op)
        print(f"{tag}[saved] {op}")
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["gold", "dev", "both"], default="both",
                    help="both=gold(22)+dev(110) 모두 생성(기본). 한 pkl에 함께 저장돼 "
                         "report_v3 --split gold/dev 양쪽에서 읽힘.")
    ap.add_argument("--experiments_root", default=str(DARIN_EXPERIMENTS),
                    help=f".../HANDOVER_인계용_다린/experiments (기본 {DARIN_EXPERIMENTS})")
    ap.add_argument("--out_root", default=str(DARIN_INFER_OUT),
                    help="저장 루트(=report_v3 --darin_root). "
                         f"기본 {DARIN_INFER_OUT}")
    ap.add_argument("--local_models", default=None,
                    help="base 모델 로컬 루트(하위에 Llama-3.1-8B-Instruct/, Qwen3-8B/). "
                         f"미지정 시 config MODEL_BASE({MODEL_BASE}) 사용.")
    ap.add_argument("--gpus", default=None,
                    help="예 '6,7' — GPU별 프로세스로 모델 병렬 추론(모델당 1 GPU). "
                         "미지정 시 현재 CUDA_VISIBLE_DEVICES로 단일 실행.")
    ap.add_argument("--skip_raw", action="store_true", help="raw(base) 변형 건너뛰기")
    ap.add_argument("--skip_done", action="store_true", help="출력 pkl 있으면 건너뛰기")
    ap.add_argument("--shard", default=None,
                    help="내부용 'i/n' — launcher가 GPU별로 자동 설정(직접 줄 필요 없음)")
    args = ap.parse_args()

    if args.gpus and not args.shard:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if len(gpus) > 1:
            launch_parallel(args, gpus)
            return
        if gpus:                       # 1장만 준 경우: env 세팅해 단일 실행
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    run_inference(args)


if __name__ == "__main__":
    main()
