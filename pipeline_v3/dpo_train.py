"""
dpo_train.py — RLAIF (DPO / SimPO) v3

v1 03_rlaif_train.py 대비:
  T5  DPO max_prompt_length 미설정(TRL 기본 512 → EMR 소실) 수정:
      DPO/SimPO 모두 max_prompt_length=MAX_PROMPT_TOKENS, max_length=MAX_SEQ_TOKENS
  T9  데이터 분리: 기본 입력이 gen_pairs --split dpo 산출물(pairs_dpo_*.pkl).
      SFT가 외운 SYNTH를 재사용하지 않는다.
  T6  프롬프트 절단: prompt_utils 한 벌 (EMR-only 좌측절단)
  B2  저장 후 provenance 해시 + .done 마커

실행:
  python -m pipeline_v3.dpo_train --base llama --loss dpo \
      --pairs outputs_v3/<run>/pairs/pairs_dpo_llama_sft3ep.pkl \
      --sft_ckpt outputs_v3/<run>/sft/llama_3ep/final --gpus 0,1
"""

import argparse
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    g = p.parse_known_args()[0].gpus
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = g
        print(f"[GPU] CUDA_VISIBLE_DEVICES={g}")


_early_parse()

import pickle                  # noqa: E402
from pathlib import Path       # noqa: E402

import pandas as pd            # noqa: E402
import torch                   # noqa: E402
from datasets import Dataset   # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model     # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

from .config_v3 import (       # noqa: E402
    DPO_BETA, GEMMA4_BASES, LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_MODULES,
    LORA_TARGET_MODULES_GEMMA4, MAX_PROMPT_TOKENS, MAX_SEQ_TOKENS, RLAIF_CONFIG,
    RLAIF_OUT, SIMPO_BETA, SIMPO_GAMMA_BETA_RATIO, TRAIN_KEYS, VITAL_MAP_PKL,
    ensure_dir, is_thinking, model_path,
)
from .prompt_utils import (    # noqa: E402
    SYSTEM_PROMPT, apply_chat_template, build_emr_text, fit_user_content, get_sid,
)
from .provenance import ckpt_valid, dir_hash, write_done_marker   # noqa: E402


def build_dpo_dataset(df, vital_map, tokenizer, base: str) -> Dataset:
    thinking = is_thinking(base)
    prompts, chosens, rejecteds = [], [], []
    n_skip = 0
    for _, row in df.iterrows():
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if not chosen or not rejected or chosen == rejected:
            n_skip += 1      # 퇴화 쌍은 학습에 넣지 않는다 (T8 방어선)
            continue
        emr = build_emr_text(row)
        sid = get_sid(row)
        user = fit_user_content(tokenizer, emr, vital_map.get(sid, ""),
                                system=SYSTEM_PROMPT, budget=MAX_PROMPT_TOKENS,
                                enable_thinking=thinking)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        prompts.append(apply_chat_template(tokenizer, msgs, add_generation_prompt=True,
                                           enable_thinking=thinking))
        chosens.append(chosen)
        rejecteds.append(rejected)
    if n_skip:
        print(f"  퇴화/빈 쌍 제외: {n_skip}건")
    print(f"  DPO 샘플: {len(prompts)}건")
    if not prompts:
        raise RuntimeError("유효한 선호쌍이 없다 — gen_pairs 산출물 확인")
    return Dataset.from_dict({"prompt": prompts, "chosen": chosens,
                              "rejected": rejecteds})


def train(args):
    raw_model_id = str(model_path(args.base))
    if args.sft_ckpt:
        ep_tag = Path(args.sft_ckpt).parent.name          # e.g. "llama_3ep"
        ep_num = ep_tag.split("_")[-1] if "_" in ep_tag else "sft"
        output_dir = RLAIF_OUT / f"{args.base}_sft{ep_num}_{args.loss}"
    else:
        output_dir = RLAIF_OUT / f"{args.base}_raw_{args.loss}"
    ensure_dir(output_dir)

    print(f"\n[RLAIF v3] base={args.base} loss={args.loss} pairs={args.pairs}")
    df = pd.read_pickle(args.pairs)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  데이터: {len(df)}건")

    _max_mem = {i: "40GiB" for i in range(torch.cuda.device_count())}
    if args.sft_ckpt:
        sft_path = Path(args.sft_ckpt)
        if not ckpt_valid(sft_path):
            raise RuntimeError(f"선행 SFT 체크포인트 무효: {sft_path} (B2)")
        tokenizer = AutoTokenizer.from_pretrained(str(sft_path), trust_remote_code=True)
        if (sft_path / "adapter_config.json").exists():
            print(f"  SFT(PEFT) merge: {sft_path}")
            model = AutoModelForCausalLM.from_pretrained(
                raw_model_id, dtype=torch.bfloat16, device_map="auto",
                max_memory=_max_mem, low_cpu_mem_usage=True, trust_remote_code=True)
            model = PeftModel.from_pretrained(model, str(sft_path), is_trainable=False)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(sft_path), dtype=torch.bfloat16, device_map="auto",
                max_memory=_max_mem, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(raw_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            raw_model_id, dtype=torch.bfloat16, device_map="auto",
            max_memory=_max_mem, low_cpu_mem_usage=True, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=(LORA_TARGET_MODULES_GEMMA4 if args.base in GEMMA4_BASES
                        else LORA_TARGET_MODULES),
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = build_dpo_dataset(df, vital_map, tokenizer, args.base)

    cfg = dict(RLAIF_CONFIG)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
        cfg.pop("num_train_epochs", None)
    shared = dict(output_dir=str(output_dir), seed=42, report_to="none",
                  gradient_checkpointing=True,
                  gradient_checkpointing_kwargs={"use_reentrant": False},
                  # T5 핵심: 두 트레이너 모두 동일 프롬프트/시퀀스 예산
                  max_prompt_length=MAX_PROMPT_TOKENS,
                  max_length=MAX_SEQ_TOKENS,
                  **cfg)

    if args.loss == "simpo":
        from trainers.simpo_config import SimPOConfig
        from trainers.simpo_trainer import SimPOTrainer
        simpo_cfg = SimPOConfig(beta=SIMPO_BETA,
                                gamma_beta_ratio=SIMPO_GAMMA_BETA_RATIO, **shared)
        trainer = SimPOTrainer(model=model, args=simpo_cfg,
                               train_dataset=dataset, tokenizer=tokenizer)
    else:
        from trl import DPOConfig, DPOTrainer
        dpo_cfg = DPOConfig(loss_type="sigmoid", beta=DPO_BETA, **shared)
        trainer = DPOTrainer(model=model, ref_model=None, args=dpo_cfg,
                             train_dataset=dataset, processing_class=tokenizer)

    print("\n학습 시작...")
    trainer.train()

    final_dir = output_dir / "final"
    if hasattr(trainer.model, "peft_config"):
        trainer.model.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if not ckpt_valid(final_dir):
        raise RuntimeError(f"[저장 실패] {final_dir}: 유효 가중치 없음")
    h = dir_hash(final_dir)
    write_done_marker(output_dir, dict(stage="rlaif", base=args.base, loss=args.loss,
                                       pairs=str(args.pairs),
                                       sft_ckpt=str(args.sft_ckpt), ckpt_hash=h))
    print(f"\n[완료] 저장: {final_dir} (hash={h})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RLAIF v3 (DPO/SimPO)")
    p.add_argument("--base", choices=TRAIN_KEYS, default="llama")
    p.add_argument("--loss", choices=["dpo", "simpo"], default="dpo")
    p.add_argument("--pairs", type=str, required=True,
                   help="gen_pairs --split dpo 산출 pkl (SFT 데이터 재사용 금지, T9)")
    p.add_argument("--sft_ckpt", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--gpus", type=str, default=None)
    train(p.parse_args())
