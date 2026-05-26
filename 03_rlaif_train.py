"""
03_rlaif_train.py — Self-Judge RLAIF (DPO / SimPO)

실행 예시:
  python 03_rlaif_train.py --base llama --loss dpo  --gpus 0,1,2,3
  python 03_rlaif_train.py --base qwen  --loss simpo --gpus 4,5
  python 03_rlaif_train.py --base llama --loss dpo --sft_ckpt outputs/sft/llama_3ep/final --gpus 0,1
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
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

from config import (
    SYNTH_PKL,
    VITAL_MAP_PKL,
    SFT_MODELS,
    RLAIF_OUT,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LORA_TARGET_MODULES_GEMMA4,
    RLAIF_CONFIG,
    SYSTEM_PROMPT,
    build_user_prompt,
    EMR_PREOP_SUM_COL,
    EMR_PREMED_COL,
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
            if isinstance(vlist, list):
                vals.extend([str(x) for x in vlist if x is not None])
            else:
                vals.append(str(vlist))
        return " ".join(vals)
    return str(v)


def _emr_text(row):
    preop = _get(row, EMR_PREOP_SUM_COL)
    premed = _get(row, EMR_PREMED_COL)
    anrec = _get(row, ("마취기록", "기록", ""))
    parts = []
    if preop:
        parts.append(f"[마취전 환자상태 요약]\n{preop}")
    if premed:
        parts.append(f"[수술전 준비사항 및 Premedication]\n{premed}")
    if anrec:
        parts.append(f"[마취기록]\n{anrec}")
    return "\n\n".join(parts)


def build_dpo_dataset(df, vital_map, tokenizer) -> Dataset:
    """chosen / rejected 쌍 → DPO dataset."""
    prompts, chosens, rejecteds = [], [], []
    no_vital = 0

    for _, row in df.iterrows():
        emr = _emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vital_map.get(sid, "")
        if not vital:
            no_vital += 1

        user = build_user_prompt(emr, vital)
        sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}]
        prompt_str = tokenizer.apply_chat_template(
            sys_msg + [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_str)
        chosens.append(str(row.get("chosen", "")))
        rejecteds.append(str(row.get("rejected", "")))

    print(f"  DPO 샘플: {len(prompts)}건  (vital 없음: {no_vital}건)")
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }
    )


def train(args):
    # 기본 모델 or SFT 체크포인트
    if args.sft_ckpt:
        base_model_id = args.sft_ckpt
        print(f"[RLAIF] SFT 체크포인트에서 시작: {base_model_id}")
    else:
        base_model_id = str(SFT_MODELS[args.base])
        print(f"[RLAIF] Raw 모델에서 시작: {base_model_id}")

    output_dir = RLAIF_OUT / f"{args.base}_{args.loss}"
    if args.sft_ckpt:
        output_dir = RLAIF_OUT / f"{args.base}_sft_{args.loss}"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpu = torch.cuda.device_count()
    print(f"  Loss:   {args.loss}")
    print(f"  출력:   {output_dir}")
    print(f"  GPU:    {n_gpu}개")

    # 데이터
    df = pd.read_pickle(SYNTH_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  데이터: {len(df)}건")

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO는 left padding

    # 모델
    print("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA (SFT 체크포인트면 이미 PEFT 모델일 수 있음)
    if not args.sft_ckpt:
        _lora_targets = (
            LORA_TARGET_MODULES_GEMMA4 if args.base == "gemma4" else LORA_TARGET_MODULES
        )
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=_lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # 데이터셋
    dataset = build_dpo_dataset(df, vital_map, tokenizer)

    # DPOConfig
    cfg = dict(RLAIF_CONFIG)
    loss_type = cfg.pop("loss_type", args.loss)
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        loss_type={"dpo": "sigmoid", "simpo": "ipo"}.get(args.loss, args.loss),
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        **{
            k: v
            for k, v in cfg.items()
            if k
            not in (
                "loss_type",
                "num_train_epochs",
                "save_strategy",
                "dataloader_num_workers",
            )
        },
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # SimPO/온라인 DPO는 ref_model 없이도 동작
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n학습 시작...")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n[완료] 저장: {output_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Judge RLAIF (DPO/SimPO)")
    parser.add_argument(
        "--base", choices=["llama", "qwen", "gemma4", "qwen35", "hari"], default="llama"
    )
    parser.add_argument("--loss", choices=["dpo", "simpo"], default="dpo")
    parser.add_argument(
        "--sft_ckpt",
        type=str,
        default=None,
        help="SFT 완료 체크포인트 경로 (없으면 raw 모델 사용)",
    )
    parser.add_argument(
        "--gpus", type=str, default=None, help="사용할 GPU 번호. 예: '0' 또는 '0,1,2,3'"
    )
    train(parser.parse_args())
