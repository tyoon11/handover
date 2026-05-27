"""
03_rlaif_train.py — Self-Judge RLAIF (DPO / SimPO)

실행 예시:
  python 03_rlaif_train.py --base llama --loss dpo  --gpus 0,1,2,3
  python 03_rlaif_train.py --base qwen  --loss simpo --gpus 4,5
  python 03_rlaif_train.py --base llama --loss dpo --sft_ckpt outputs/sft/llama_3ep/final --gpus 0,1
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import sys, os, argparse
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


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
from pathlib import Path
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
    build_emr_text,
)


def build_dpo_dataset(df, vital_map, tokenizer) -> Dataset:
    """chosen / rejected 쌍 → DPO dataset."""
    prompts, chosens, rejecteds = [], [], []
    no_vital = 0
    # 원본과 동일: Qwen3 계열은 학습 시 thinking 비활성화
    is_qwen = "qwen" in str(getattr(tokenizer, "name_or_path", "")).lower()

    for _, row in df.iterrows():
        emr = build_emr_text(row)
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
        user_msg = [{"role": "user", "content": user}]

        # Qwen3: enable_thinking=False (원본 train_self_judge.ipynb와 동일)
        if is_qwen:
            try:
                prompt_str = tokenizer.apply_chat_template(
                    sys_msg + user_msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_str = tokenizer.apply_chat_template(
                    sys_msg + user_msg, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt_str = tokenizer.apply_chat_template(
                sys_msg + user_msg, tokenize=False, add_generation_prompt=True
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
    raw_model_id = str(SFT_MODELS[args.base])

    # 출력 경로 결정
    if args.sft_ckpt:
        # sft_epochs 정보를 폴더명에 포함 (예: llama_sft1ep_dpo)
        ep_tag = Path(args.sft_ckpt).parent.name  # e.g. "llama_1ep"
        ep_num = ep_tag.split("_")[-1] if "_" in ep_tag else "sft"
        output_dir = RLAIF_OUT / f"{args.base}_sft{ep_num}_{args.loss}"
    else:
        output_dir = RLAIF_OUT / f"{args.base}_raw_{args.loss}"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpu = torch.cuda.device_count()
    print(f"\n[RLAIF 시작]")
    print(f"  Base:   {args.base}  ({raw_model_id})")
    print(f"  Loss:   {args.loss}")
    print(f"  출력:   {output_dir}")
    print(f"  GPU:    {n_gpu}개")

    # 데이터
    df = pd.read_pickle(SYNTH_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  데이터: {len(df)}건")

    # ── 모델 / 토크나이저 로드 ────────────────────────────────────────────
    # gemma4(MoE)는 expert weight concat 시 단일 GPU에 메모리 집중 → 8GB 여유 확보
    _max_mem = (
        {i: "40GiB" for i in range(torch.cuda.device_count())}
        if args.base == "gemma4"
        else None
    )
    if args.sft_ckpt:
        sft_path = Path(args.sft_ckpt)
        is_peft = (sft_path / "adapter_config.json").exists()

        if is_peft:
            # SFT 저장본이 LoRA adapter → 원본 노트북과 동일하게 merge_and_unload
            print(f"  SFT 체크포인트(PEFT): {sft_path}")
            print("  → base 모델 로드 후 LoRA merge...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(sft_path), trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                raw_model_id,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory=_max_mem,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, str(sft_path), is_trainable=False)
            model = model.merge_and_unload()
            print("  → merge 완료")
        else:
            # 이미 merged full model로 저장된 경우
            print(f"  SFT 체크포인트(full): {sft_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                str(sft_path), trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(sft_path),
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory=_max_mem,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
    else:
        print(f"  Raw 모델: {raw_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(raw_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            raw_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=_max_mem,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO는 left padding
    model.config.use_cache = False

    # ── LoRA 적용 (항상 새로 붙임 — merge 후이므로) ───────────────────────
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

    cfg = dict(RLAIF_CONFIG)
    cfg.pop("loss_type", None)
    shared_kwargs = {k: v for k, v in cfg.items() if k != "dataloader_num_workers"}

    if args.loss == "simpo":
        from trainers.simpo_config import SimPOConfig
        from trainers.simpo_trainer import SimPOTrainer

        simpo_cfg = SimPOConfig(
            output_dir=str(output_dir),
            beta=2.0,  # SimPO 논문 기본값 (DPO의 0.1과 다름)
            gamma_beta_ratio=0.25,
            max_length=shared_kwargs.pop("max_length", 2048),
            max_prompt_length=1792,
            seed=42,
            report_to="none",
            gradient_checkpointing=True,
            **{k: v for k, v in shared_kwargs.items() if k != "beta"},  # beta 중복 방지
        )
        trainer = SimPOTrainer(
            model=model,
            args=simpo_cfg,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
    else:
        dpo_config = DPOConfig(
            output_dir=str(output_dir),
            loss_type="sigmoid",
            seed=42,
            report_to="none",
            gradient_checkpointing=True,
            **shared_kwargs,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
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
