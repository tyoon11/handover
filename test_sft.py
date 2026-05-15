"""
test_sft.py — SFT 파이프라인 Smoke Test
실제 학습 전 설정/데이터/모델 로드가 정상인지 빠르게 확인

실행:
  python test_sft.py --base llama --gpus 4
  python test_sft.py --base gemma4 --gpus 5 --steps 3 --samples 8
"""

import sys, os, argparse

def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    return p.parse_known_args()[0].gpus

_gpus = _early_gpus()
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_gpus}")

import pickle, traceback
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
        print("[WARN] DataCollatorForCompletionOnlyLM 없음 → collator 미사용")

from config import (
    SYNTH_PKL, VITAL_MAP_PKL, SFT_MODELS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    SYSTEM_PROMPT, build_user_prompt,
    EMR_PREOP_SUM_COL, EMR_PREMED_COL,
)

MAX_SEQ_LEN = 2048
PASS = "✓"
FAIL = "✗"

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}")
        return result
    except Exception as e:
        print(f"  {FAIL} {label}")
        print(f"      → {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

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
            vals.extend([str(x) for x in vlist] if isinstance(vlist, list) else [str(vlist)])
        return " ".join(vals)
    return str(v)

def _emr_text(row):
    parts = []
    preop  = _get(row, EMR_PREOP_SUM_COL)
    premed = _get(row, EMR_PREMED_COL)
    anrec  = _get(row, ("마취기록", "기록", ""))
    if preop:  parts.append(f"[마취전 환자상태 요약]\n{preop}")
    if premed: parts.append(f"[수술전 준비사항 및 Premedication]\n{premed}")
    if anrec:  parts.append(f"[마취기록]\n{anrec}")
    return "\n\n".join(parts)


def run_test(args):
    model_id = str(SFT_MODELS[args.base])
    n        = args.samples
    steps    = args.steps

    print(f"\n{'='*60}")
    print(f" SFT Smoke Test")
    print(f"  모델:    {model_id}")
    print(f"  샘플수:  {n}건 → {n*3}샘플")
    print(f"  steps:   {steps}")
    print(f"  GPU:     {torch.cuda.device_count()}개")
    print(f"{'='*60}\n")

    results = {}

    # ── 1. 데이터 로드 ───────────────────────────────────────────────
    print("[1] 데이터 로드")
    synth_df = check("합성데이터 pkl 로드",
                     lambda: pd.read_pickle(SYNTH_PKL))
    vital_map = check("Vital map pkl 로드",
                      lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if synth_df is None or vital_map is None:
        print("\n[ABORT] 데이터 로드 실패"); return

    # ── 2. 토크나이저 ────────────────────────────────────────────────
    print("\n[2] 토크나이저")
    tokenizer = check("AutoTokenizer 로드",
                      lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    if tokenizer is None:
        print("\n[ABORT] 토크나이저 로드 실패"); return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    check("pad_token 설정", lambda: tokenizer.pad_token)

    # ── 3. 데이터셋 구성 ─────────────────────────────────────────────
    print("\n[3] 데이터셋 구성")
    sub_df = synth_df.iloc[:n]
    texts = []
    for _, row in sub_df.iterrows():
        emr = _emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital   = vital_map.get(sid, "")
        chosen  = str(row.get("chosen", "특이사항 없음"))
        rejected = str(row.get("rejected", "특이사항 없음"))
        user = build_user_prompt(emr, vital)

        def _chat(assistant):
            msgs = [{"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":user},
                    {"role":"assistant","content":assistant}]
            return tokenizer.apply_chat_template(msgs, tokenize=False)

        texts.append(_chat(chosen))
        judge_user = (f"EMR:\n{user}\n\nAssistant A: {chosen}\n\nAssistant B: {rejected}")
        judge_msgs = [{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":judge_user},
                      {"role":"assistant","content":"Winner:\nA"}]
        texts.append(tokenizer.apply_chat_template(judge_msgs, tokenize=False))
        judge_msgs2 = [{"role":"system","content":SYSTEM_PROMPT},
                       {"role":"user","content":judge_user.replace(f"A: {chosen}","A: TMP").replace(f"B: {rejected}",f"B: {chosen}").replace("A: TMP",f"A: {rejected}")},
                       {"role":"assistant","content":"Winner:\nB"}]
        texts.append(tokenizer.apply_chat_template(judge_msgs2, tokenize=False))

    dataset = check("Dataset 생성", lambda: Dataset.from_dict({"text": texts}))
    if dataset is None:
        print("\n[ABORT] 데이터셋 생성 실패"); return

    # 토큰 길이 확인
    tok_lens = [tokenizer(t, return_tensors="pt").input_ids.shape[1] for t in texts[:6]]
    print(f"  {PASS} 샘플 토큰 길이 (처음 6개): {tok_lens}")
    over = sum(1 for l in tok_lens if l > MAX_SEQ_LEN)
    if over:
        print(f"  ⚠ {over}개 샘플이 {MAX_SEQ_LEN} 초과 (loss 무시됨)")

    # ── 4. Response-only collator ─────────────────────────────────────
    print("\n[4] Response collator")
    if args.base in ("llama",):
        resp_tok = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>",
                                    add_special_tokens=False)
    elif args.base in ("qwen", "qwen35", "hari"):
        resp_tok = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    elif args.base in ("gemma4",):
        resp_tok = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
    else:
        resp_tok = None

    if DataCollatorForCompletionOnlyLM and resp_tok:
        collator = check("DataCollatorForCompletionOnlyLM",
            lambda: DataCollatorForCompletionOnlyLM(resp_tok, tokenizer=tokenizer))
    else:
        collator = None
        print(f"  ⚠ collator 미사용 (DataCollatorForCompletionOnlyLM={'없음' if not DataCollatorForCompletionOnlyLM else 'resp_tok 없음'})")

    # ── 5. 모델 로드 ─────────────────────────────────────────────────
    print("\n[5] 모델 로드")
    model = check("AutoModelForCausalLM 로드",
        lambda: AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True))
    if model is None:
        print("\n[ABORT] 모델 로드 실패"); return

    mem = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  {PASS} GPU 메모리: {mem:.1f}GB / {total:.0f}GB")
    model.config.use_cache = False

    # ── 6. LoRA ──────────────────────────────────────────────────────
    print("\n[6] LoRA")
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          target_modules=LORA_TARGET_MODULES,
                          bias="none", task_type="CAUSAL_LM")
    model = check("LoRA 적용", lambda: get_peft_model(model, lora_cfg))
    if model is None:
        print("\n[ABORT] LoRA 실패"); return
    model.print_trainable_parameters()

    # ── 7. SFTConfig + SFTTrainer ────────────────────────────────────
    print("\n[7] SFTConfig + SFTTrainer")
    sft_config = check("SFTConfig 생성",
        lambda: SFTConfig(
            output_dir="/tmp/sft_test",
            max_steps=steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=1,
            max_seq_length=MAX_SEQ_LEN,
            dataset_text_field="text",
            report_to="none",
            seed=42,
        ))

    trainer = check("SFTTrainer 초기화",
        lambda: SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        ))
    if trainer is None:
        print("\n[ABORT] SFTTrainer 초기화 실패"); return

    # ── 8. 실제 학습 (N steps) ───────────────────────────────────────
    print(f"\n[8] 학습 실행 ({steps} steps)")
    result = check(f"{steps} steps 학습", lambda: trainer.train())
    if result is None:
        print("\n[ABORT] 학습 실패"); return

    mem2 = torch.cuda.memory_allocated() / 1e9
    print(f"  {PASS} 학습 후 GPU 메모리: {mem2:.1f}GB / {total:.0f}GB")

    # ── 결과 요약 ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" ✓ Smoke Test 통과!  모델={args.base}  steps={steps}")
    print(f"   → 02_sft_train.py 전체 학습 실행 가능")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Smoke Test")
    parser.add_argument("--base",    choices=list(SFT_MODELS.keys()), default="llama")
    parser.add_argument("--gpus",    type=str, default=None)
    parser.add_argument("--steps",   type=int, default=5,  help="학습 스텝 수 (기본 5)")
    parser.add_argument("--samples", type=int, default=10, help="사용할 데이터 샘플 수 (기본 10)")
    run_test(parser.parse_args())
