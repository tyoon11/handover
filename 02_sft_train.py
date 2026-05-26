"""
02_sft_train.py — Synthetic SFT (LoRA)

실행 예시:
  python 02_sft_train.py --base llama --epochs 3 --gpus 0,1,2,3
  python 02_sft_train.py --base qwen  --epochs 1 --gpus 4,5
  CUDA_VISIBLE_DEVICES=0,1 python 02_sft_train.py --base llama
"""

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
import pickle
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer import DataCollatorForCompletionOnlyLM
    except ImportError:
        DataCollatorForCompletionOnlyLM = None

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
# 데이터 포맷
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _chat(system, user, assistant, tokenizer, is_qwen: bool = False) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    # 원본 train_jsft.ipynb와 동일: Qwen3는 학습 시 thinking 비활성화
    # enable_thinking=False 없으면 <think>\n\n</think> 빈 블록이 학습 데이터에 삽입됨
    if is_qwen:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, enable_thinking=False
            )
        except TypeError:
            pass  # 구버전 tokenizer → fallback
    return tokenizer.apply_chat_template(msgs, tokenize=False)


def build_dataset(synth_df, vital_map, tokenizer) -> Dataset:
    """1행 → 3샘플: generation + judge(A>B) + judge(B>A)"""
    texts = []
    no_vital = 0
    # 원본과 동일: Qwen3 계열은 학습 시 thinking 비활성화
    is_qwen = "qwen" in str(getattr(tokenizer, "name_or_path", "")).lower()

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

        # (1) generation
        texts.append(_chat(SYSTEM_PROMPT, user_base, chosen, tokenizer, is_qwen))

        # (2) judge A=chosen
        judge_user = (
            "Evaluate which of the following PACU/ICU handoffs is better. No reasoning.\n\n"
            f"EMR:\n{user_base}\n\nAssistant A: {chosen}\n\nAssistant B: {rejected}"
        )
        texts.append(_chat(SYSTEM_PROMPT, judge_user, "Winner:\nA", tokenizer, is_qwen))

        # (3) judge B=chosen
        judge_user2 = (
            "Evaluate which of the following PACU/ICU handoffs is better. No reasoning.\n\n"
            f"EMR:\n{user_base}\n\nAssistant A: {rejected}\n\nAssistant B: {chosen}"
        )
        texts.append(
            _chat(SYSTEM_PROMPT, judge_user2, "Winner:\nB", tokenizer, is_qwen)
        )

    print(f"  총 샘플: {len(texts)}개  (vital 없는 케이스: {no_vital}건)")
    return Dataset.from_dict({"text": texts})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def train(args):
    model_id = str(SFT_MODELS[args.base])
    output_dir = SFT_OUT / f"{args.base}_{args.epochs}ep"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpu = torch.cuda.device_count()
    print(f"\n[SFT 시작]")
    print(f"  모델:     {model_id}")
    print(f"  Epoch:    {args.epochs}")
    print(f"  출력:     {output_dir}")
    print(
        f"  GPU:      {n_gpu}개  ({torch.cuda.get_device_name(0) if n_gpu else 'CPU'})"
    )

    # 데이터
    print("\n데이터 로드 중...")
    synth_df = pd.read_pickle(SYNTH_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  합성데이터: {len(synth_df)}건  |  Vital map: {len(vital_map)}건")

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 모델
    print("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA
    _lora_targets = (
        LORA_TARGET_MODULES_GEMMA4 if args.base == "gemma4" else LORA_TARGET_MODULES
    )
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
    print("데이터셋 구성 중...")
    dataset = build_dataset(synth_df, vital_map, tokenizer)

    # Response-only collator
    if args.base in ("llama",):
        resp_tok = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>",
            add_special_tokens=False,
        )
    elif args.base in ("qwen", "qwen35", "hari"):
        resp_tok = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    elif args.base in ("gemma4",):
        resp_tok = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
    else:
        resp_tok = None
    if DataCollatorForCompletionOnlyLM and resp_tok:
        collator = DataCollatorForCompletionOnlyLM(resp_tok, tokenizer=tokenizer)
    else:
        collator = None

    cfg = dict(SFT_CONFIG)
    cfg["num_train_epochs"] = args.epochs
    train_args = SFTConfig(
        output_dir=str(output_dir),
        seed=42,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        **cfg,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
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
    parser.add_argument(
        "--gpus", type=str, default=None, help="사용할 GPU 번호. 예: '0' 또는 '0,1,2,3'"
    )
    train(parser.parse_args())
