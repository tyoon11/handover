"""
sft_train.py — Synthetic SFT (LoRA) v3

v1 02_sft_train.py 대비:
  - 데이터: gen_pairs --split sft 산출물(pairs_sft.pkl) — 평가셋 유출 없는 타깃 (T1/T2)
  - 프롬프트/절단: prompt_utils 한 벌 (EMR-only 좌측절단, 예산 MAX_SEQ_TOKENS) (T6)
  - judge-augmented 샘플(A/B)은 --judge_aug 로 유지 가능 (기본 ON — self-judge 방법 보존)
  - 저장 후 provenance 내용해시 + .done 마커 기록 (B1/B2)

실행:
  python -m pipeline_v3.sft_train --base llama --epochs 3 --gpus 0,1
"""

import argparse
import os


def _early_parse_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    g = p.parse_known_args()[0].gpus
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = g
        print(f"[GPU] CUDA_VISIBLE_DEVICES={g}")


_early_parse_gpus()

import importlib.util          # noqa: E402
import pickle                  # noqa: E402
import warnings                # noqa: E402
from dataclasses import dataclass          # noqa: E402
from typing import Any, List, Optional     # noqa: E402

import pandas as pd            # noqa: E402
import torch                   # noqa: E402
from datasets import Dataset   # noqa: E402
from peft import LoraConfig, get_peft_model               # noqa: E402
from transformers import (     # noqa: E402
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
)

from .config_v3 import (       # noqa: E402
    GEMMA4_BASES, LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_MODULES,
    LORA_TARGET_MODULES_GEMMA4, MAX_PROMPT_TOKENS, MAX_SEQ_TOKENS, PAIRS_SFT_PKL,
    SFT_CONFIG, SFT_OUT, TRAIN_KEYS, VITAL_MAP_PKL, ensure_dir, is_thinking, model_path,
)
from .prompt_utils import (    # noqa: E402
    SYSTEM_PROMPT, apply_chat_template, build_emr_text, fit_user_content, get_sid,
)
from .provenance import dir_hash, write_done_marker       # noqa: E402


@dataclass
class JudgeAugmentedSFTCollator:
    """좌측 패딩 + 좌측 절단 collator (원본 self-judge collator와 동일 동작)."""
    tokenizer: Any
    max_length: Optional[int] = None
    ignore_index: int = -100

    def __post_init__(self):
        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_length
            warnings.warn(f"max_length 미지정 → {self.max_length}")

    def __call__(self, examples: List[dict]) -> dict:
        input_ids = [ex["input_ids"] for ex in examples]
        labels = [ex["labels"] for ex in examples]
        padded = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
        padded.input_ids = padded.input_ids[:, -self.max_length:]
        padded.attention_mask = padded.attention_mask[:, -self.max_length:]
        padded_labels = torch.full_like(padded.input_ids, self.ignore_index)
        for i, lbl in enumerate(labels):
            lbl = lbl[-self.max_length:]
            padded_labels[i, -len(lbl):] = torch.tensor(lbl, dtype=torch.long)
        return {"input_ids": padded.input_ids,
                "attention_mask": padded.attention_mask,
                "labels": padded_labels}


def _make_example(system, user, assistant, tokenizer, thinking: bool) -> dict:
    """프롬프트 길이 기반 라벨 마스킹 — 응답 토큰에만 loss (v1 검증 로직 유지)."""
    sys_user = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    full = sys_user + [{"role": "assistant", "content": assistant}]

    prompt_text = apply_chat_template(tokenizer, sys_user, add_generation_prompt=True,
                                      enable_thinking=thinking)
    full_text = apply_chat_template(tokenizer, full, add_generation_prompt=False,
                                    enable_thinking=thinking)
    pids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    fids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    plen = len(pids)
    if fids[:plen] != pids:
        plen = 0
        for a, b in zip(pids, fids):
            if a != b:
                break
            plen += 1
    labels = [-100] * plen + fids[plen:]
    return {"input_ids": fids, "labels": labels}


def build_dataset(pairs_df, vital_map, tokenizer, base: str, judge_aug: bool) -> Dataset:
    """1행 → generation 샘플 (+ judge_aug면 judge A/B 2샘플).

    user 프롬프트는 fit_user_content로 예산 내 절단 — 학습·추론 동일 규칙 (T6).
    """
    thinking = is_thinking(base)
    examples = []
    no_vital = 0
    for _, row in pairs_df.iterrows():
        emr = build_emr_text(row)
        sid = get_sid(row)
        vital = vital_map.get(sid, "")
        if not vital:
            no_vital += 1
        user = fit_user_content(tokenizer, emr, vital, system=SYSTEM_PROMPT,
                                budget=MAX_PROMPT_TOKENS, enable_thinking=thinking)
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))

        examples.append(_make_example(SYSTEM_PROMPT, user, chosen, tokenizer, thinking))
        if judge_aug and rejected:
            ju1 = ("Evaluate which of the following PACU/ICU handoffs is better. "
                   f"No reasoning.\n\nEMR:\n{user}\n\nAssistant A: {chosen}\n\n"
                   f"Assistant B: {rejected}")
            examples.append(_make_example(SYSTEM_PROMPT, ju1, "A", tokenizer, thinking))
            ju2 = ("Evaluate which of the following PACU/ICU handoffs is better. "
                   f"No reasoning.\n\nEMR:\n{user}\n\nAssistant A: {rejected}\n\n"
                   f"Assistant B: {chosen}")
            examples.append(_make_example(SYSTEM_PROMPT, ju2, "B", tokenizer, thinking))

    print(f"  총 샘플: {len(examples)}개 (vital 없음: {no_vital}건)")
    n_trainable = sum(sum(1 for l in ex["labels"] if l != -100) for ex in examples)
    if n_trainable == 0:
        raise RuntimeError("[라벨 마스킹 실패] 학습 토큰 0개 — chat template 확인 (loss=0 방지)")
    print(f"  학습 토큰: {n_trainable}개")
    return Dataset.from_list(examples)


def train(args):
    model_id = str(model_path(args.base))
    output_dir = ensure_dir(SFT_OUT / f"{args.base}_{args.epochs}ep")
    pairs_pkl = args.pairs or str(PAIRS_SFT_PKL)

    print(f"\n[SFT v3] base={args.base} epochs={args.epochs} pairs={pairs_pkl}")
    pairs_df = pd.read_pickle(pairs_pkl)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  데이터: {len(pairs_df)}건")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    _attn = {}
    if args.base not in GEMMA4_BASES and importlib.util.find_spec("flash_attn"):
        _attn["attn_implementation"] = "flash_attention_2"
    _max_mem = {i: "40GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", max_memory=_max_mem,
        low_cpu_mem_usage=True, trust_remote_code=True, **_attn)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=(LORA_TARGET_MODULES_GEMMA4 if args.base in GEMMA4_BASES
                        else LORA_TARGET_MODULES),
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = build_dataset(pairs_df, vital_map, tokenizer, args.base,
                            judge_aug=not args.no_judge_aug)
    collator = JudgeAugmentedSFTCollator(tokenizer=tokenizer, max_length=MAX_SEQ_TOKENS)

    cfg = dict(SFT_CONFIG)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
        cfg.pop("num_train_epochs", None)
    else:
        cfg["num_train_epochs"] = args.epochs
    train_args = TrainingArguments(
        output_dir=str(output_dir), seed=42, report_to="none",
        remove_unused_columns=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, **cfg)

    import inspect
    tok_kw = ("processing_class"
              if "processing_class" in inspect.signature(Trainer.__init__).parameters
              else "tokenizer")
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset,
                      data_collator=collator, **{tok_kw: tokenizer})
    print("\n학습 시작...")
    trainer.train()

    final_dir = output_dir / "final"
    if hasattr(trainer.model, "peft_config"):
        trainer.model.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    from .provenance import ckpt_valid
    if not ckpt_valid(final_dir):
        raise RuntimeError(f"[저장 실패] {final_dir}: 유효 가중치 없음")
    h = dir_hash(final_dir)
    write_done_marker(output_dir, dict(stage="sft", base=args.base, epochs=args.epochs,
                                       pairs=str(pairs_pkl), ckpt_hash=h))
    print(f"\n[완료] 저장: {final_dir} (hash={h})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SFT v3 (LoRA)")
    p.add_argument("--base", choices=TRAIN_KEYS, default="llama")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--pairs", type=str, default=None, help="pairs pkl (기본 PAIRS_SFT_PKL)")
    p.add_argument("--no_judge_aug", action="store_true",
                   help="judge(A/B) 샘플 제외 — generation-only SFT")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--gpus", type=str, default=None)
    train(p.parse_args())
