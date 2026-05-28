"""
02_sft_train.py — Synthetic SFT (LoRA)

실행 예시:
  python 02_sft_train.py --base llama --epochs 3 --gpus 0,1,2,3
  python 02_sft_train.py --base qwen  --epochs 1 --gpus 4,5
  CUDA_VISIBLE_DEVICES=0,1 python 02_sft_train.py --base llama
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# ── GPU 지정: torch import 이전에 환경변수 설정 ──────────────────────────
import sys, os, argparse


def _early_parse_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    args, _ = p.parse_known_args()
    return args.gpus


_gpus = _early_parse_gpus()
if _gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_gpus}")
elif "CUDA_VISIBLE_DEVICES" in os.environ:
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (환경변수)")
else:
    print("[GPU] CUDA_VISIBLE_DEVICES 미지정 → 전체 GPU 사용")

# ── 나머지 import ─────────────────────────────────────────────────────────
import importlib.util
import pickle
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from config import (
    SYNTH_PKL,
    VITAL_MAP_PKL,
    SFT_MODELS,
    SFT_OUT,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LORA_TARGET_MODULES_GEMMA4,
    SFT_CONFIG,
    SYSTEM_PROMPT,
    build_user_prompt,
    build_emr_text,
)

MAX_SEQ_LEN = 2048


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JudgeAugmentedSFTCollator  (원본 git_new/self-judge/src/self_judge/collators.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class JudgeAugmentedSFTCollator:
    tokenizer: Any
    max_length: Optional[int] = None
    ignore_index: int = -100

    def __post_init__(self):
        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_length
            warnings.warn(
                f"max_length 미지정 → tokenizer.model_max_length={self.max_length} 사용"
            )

    def __call__(self, examples: List[dict]) -> dict:
        input_ids = [ex["input_ids"] for ex in examples]
        labels = [ex["labels"] for ex in examples]

        # tokenizer.pad()로 패딩 (padding_side="left" 이면 왼쪽 패딩)
        padded = self.tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
        # 오른쪽에서 max_length 토큰만 유지 (왼쪽 truncation)
        padded.input_ids = padded.input_ids[:, -self.max_length :]
        padded.attention_mask = padded.attention_mask[:, -self.max_length :]

        padded_labels = torch.full_like(padded.input_ids, self.ignore_index)
        for i, lbl in enumerate(labels):
            lbl = lbl[-self.max_length :]
            padded_labels[i, -len(lbl) :] = torch.tensor(lbl, dtype=torch.long)

        return {
            "input_ids": padded.input_ids,
            "attention_mask": padded.attention_mask,
            "labels": padded_labels,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 포맷
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _chat(system, user, assistant, tokenizer, is_qwen: bool = False) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if is_qwen:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, enable_thinking=False
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(msgs, tokenize=False)


def _tokenize_with_labels(
    text: str,
    tokenizer,
    resp_template_ids: List[int],
) -> dict:
    """전체 텍스트를 토크나이즈하고, assistant 응답 토큰에만 label을 부여.

    assistant template 마지막 출현 이후 토큰만 loss에 포함(label≠-100).
    judge 샘플의 경우 응답이 "A" 또는 "B" 단일 토큰이므로 해당 토큰만 학습됨.
    """
    input_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]

    # resp_template_ids의 마지막 출현 위치 탐색
    n = len(resp_template_ids)
    response_start = None
    for i in range(len(input_ids) - n, -1, -1):
        if input_ids[i : i + n] == resp_template_ids:
            response_start = i + n
            break

    if response_start is None or response_start >= len(input_ids):
        # template을 찾지 못하면 전체 마스킹 (loss 미반영)
        labels = [-100] * len(input_ids)
    else:
        labels = [-100] * response_start + input_ids[response_start:]

    return {"input_ids": input_ids, "labels": labels}


def _get_resp_template_ids(base: str, tokenizer) -> List[int]:
    """모델별 assistant 응답 시작 template 토큰 ID 목록."""
    templates = {
        "llama": "<|start_header_id|>assistant<|end_header_id|>",
        "qwen": "<|im_start|>assistant\n",
        "qwen35": "<|im_start|>assistant\n",
        "hari": "<|im_start|>assistant\n",
        "gemma4": "<start_of_turn>model\n",
    }
    tmpl = templates.get(base, "<|im_start|>assistant\n")
    return tokenizer.encode(tmpl, add_special_tokens=False)


def build_dataset(synth_df, vital_map, tokenizer, base: str) -> Dataset:
    """1행 → 3샘플: generation + judge(A>B) + judge(B>A)

    원본 train_jsft.ipynb와 동일:
    - generation 샘플: chosen 인계문 전체에 loss
    - judge 샘플: "A" 또는 "B" 단일 토큰에만 loss (원본 JudgeAugmented 방식)
    """
    resp_template_ids = _get_resp_template_ids(base, tokenizer)
    is_qwen = base in ("qwen", "qwen35", "hari")
    examples = []
    no_vital = 0

    for _, row in synth_df.iterrows():
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vital_map.get(sid, "")
        if not vital:
            no_vital += 1
        user_base = build_user_prompt(emr, vital)

        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))

        # (1) generation: chosen 인계문 전체에 loss
        text = _chat(SYSTEM_PROMPT, user_base, chosen, tokenizer, is_qwen)
        examples.append(_tokenize_with_labels(text, tokenizer, resp_template_ids))

        # (2) judge: A=chosen, B=rejected → 정답 "A"
        judge_user = (
            "Evaluate which of the following PACU/ICU handoffs is better. No reasoning.\n\n"
            f"EMR:\n{user_base}\n\nAssistant A: {chosen}\n\nAssistant B: {rejected}"
        )
        text = _chat(SYSTEM_PROMPT, judge_user, "A", tokenizer, is_qwen)
        examples.append(_tokenize_with_labels(text, tokenizer, resp_template_ids))

        # (3) judge: A=rejected, B=chosen → 정답 "B"
        judge_user2 = (
            "Evaluate which of the following PACU/ICU handoffs is better. No reasoning.\n\n"
            f"EMR:\n{user_base}\n\nAssistant A: {rejected}\n\nAssistant B: {chosen}"
        )
        text = _chat(SYSTEM_PROMPT, judge_user2, "B", tokenizer, is_qwen)
        examples.append(_tokenize_with_labels(text, tokenizer, resp_template_ids))

    print(f"  총 샘플: {len(examples)}개  (vital 없는 케이스: {no_vital}건)")
    return Dataset.from_list(examples)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train(args):
    model_id = str(SFT_MODELS[args.base])
    output_dir = SFT_OUT / f"{args.base}_{args.epochs}ep"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpu = torch.cuda.device_count()
    print(f"\n[SFT 시작]")
    print(f"  모델:  {model_id}")
    print(f"  Epoch: {args.epochs}")
    print(f"  출력:  {output_dir}")
    print(f"  GPU:   {n_gpu}개  ({torch.cuda.get_device_name(0) if n_gpu else 'CPU'})")

    # 데이터
    print("\n데이터 로드 중...")
    synth_df = pd.read_pickle(SYNTH_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  합성데이터: {len(synth_df)}건  |  Vital map: {len(vital_map)}건")

    # 토크나이저 — 원본과 동일: left padding + left truncation
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # 모델
    print("모델 로드 중...")
    _attn_kwargs = {}
    if args.base != "gemma4" and importlib.util.find_spec("flash_attn") is not None:
        _attn_kwargs["attn_implementation"] = "flash_attention_2"

    _max_mem = {i: "40GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=_max_mem,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **_attn_kwargs,
    )
    model.config.use_cache = False

    # LoRA
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

    # 데이터셋 (pre-tokenized input_ids + labels)
    print("데이터셋 구성 중...")
    dataset = build_dataset(synth_df, vital_map, tokenizer, args.base)

    # Collator — 원본 JudgeAugmentedSFTCollator
    collator = JudgeAugmentedSFTCollator(tokenizer=tokenizer, max_length=MAX_SEQ_LEN)

    # TrainingArguments (원본은 Trainer + TrainingArguments 사용)
    cfg = dict(SFT_CONFIG)
    if getattr(args, "max_steps", None) is not None:
        cfg["max_steps"] = args.max_steps
        cfg.pop("num_train_epochs", None)
    else:
        cfg["num_train_epochs"] = args.epochs
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **cfg,
    )

    # Trainer — TRL>=0.9 renamed tokenizer→processing_class
    import inspect as _inspect
    _trainer_init_params = _inspect.signature(Trainer.__init__).parameters
    _tok_kwarg = "processing_class" if "processing_class" in _trainer_init_params else "tokenizer"
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        **{_tok_kwarg: tokenizer},
        data_collator=collator,
    )
    print("\n학습 시작...")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n[완료] 저장: {output_dir / 'final'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic SFT with LoRA")
    parser.add_argument(
        "--base", choices=["llama", "qwen", "gemma4", "qwen35", "hari"], default="llama"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None, help="smoke test용 최대 step 수 (지정 시 epochs 무시)")
    parser.add_argument(
        "--gpus", type=str, default=None, help="사용할 GPU 번호. 예: '0' 또는 '0,1,2,3'"
    )
    train(parser.parse_args())
