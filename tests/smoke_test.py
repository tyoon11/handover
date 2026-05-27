"""
tests/smoke_test.py — 단계별 파이프라인 빠른 동작 확인

각 단계를 소량 샘플로 돌려 "돌아가는지" 확인합니다.
--stage data 는 GPU 없이도 동작합니다.

실행 예시:
  python tests/smoke_test.py --stage data
  python tests/smoke_test.py --stage sft    --gpus 4   --base llama
  python tests/smoke_test.py --stage rlaif  --gpus 4,5 --base llama
  python tests/smoke_test.py --stage infer  --gpus 4   --base llama
  python tests/smoke_test.py --stage eval   --gpus 4
  python tests/smoke_test.py --all          --gpus 4,5,6,7 --base llama
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── GPU 설정 (import 전에) ────────────────────────────────────────────────
def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    return p.parse_known_args()[0].gpus


_gpus = _early_parse()
if _gpus:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_gpus}")

import os, pickle, traceback
import pandas as pd

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def ok(label, fn):
    try:
        r = fn()
        print(f"  {PASS} {label}")
        return r
    except Exception as e:
        print(f"  {FAIL} {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def header(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


# ── Stage 0: 데이터 & 설정 확인 (GPU 불필요) ──────────────────────────────
def stage_data():
    header("Stage: data — 데이터 & 설정 확인")
    from config import (
        GOLD_PKL, GOLD_REF_PKL, SFT_PKL, SYNTH_PKL,
        VITAL_MAP_PKL, SFT_MODELS, RLAIF_CONFIG, SFT_CONFIG,
    )

    # 파일 존재 확인
    print("\n[1] pkl 파일 존재 여부")
    for name, path in [
        ("GOLD_PKL", GOLD_PKL),
        ("GOLD_REF_PKL", GOLD_REF_PKL),
        ("SFT_PKL", SFT_PKL),
        ("SYNTH_PKL", SYNTH_PKL),
        ("VITAL_MAP_PKL", VITAL_MAP_PKL),
    ]:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1e6
            print(f"  {PASS} {name} ({size_mb:.0f}MB)")
        else:
            print(f"  {FAIL} {name} 없음: {path}")

    # 데이터 로드 & 구조 확인
    print("\n[2] 데이터 로드 & 구조")
    synth = ok("SYNTH_PKL 로드", lambda: pd.read_pickle(SYNTH_PKL))
    if synth is not None:
        print(f"       rows={len(synth)}  cols={len(synth.columns)}")
        has_chosen = ok("'chosen' 컬럼 존재", lambda: synth["chosen"].notna().sum())
        has_rejected = ok("'rejected' 컬럼 존재", lambda: synth["rejected"].notna().sum())
        if has_chosen and has_rejected:
            print(f"       chosen 비율: {has_chosen}/{len(synth)}  rejected: {has_rejected}/{len(synth)}")

    vmap = ok("VITAL_MAP_PKL 로드", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if vmap:
        print(f"       vital entries: {len(vmap)}")

    # 모델 경로 확인
    print("\n[3] 모델 경로 확인")
    for name, path in SFT_MODELS.items():
        cfg_path = Path(path) / "config.json"
        if cfg_path.exists():
            print(f"  {PASS} {name}: {path}")
        else:
            print(f"  {WARN} {name}: config.json 없음 ({path})")

    # 설정값 출력
    print("\n[4] RLAIF_CONFIG")
    for k, v in RLAIF_CONFIG.items():
        print(f"       {k}: {v}")

    print(f"\n{PASS} stage_data 완료")


# ── Stage 1: SFT 3 steps ──────────────────────────────────────────────────
def stage_sft(base: str, steps: int = 3, n_samples: int = 8):
    header(f"Stage: sft — {base} · {steps} steps · {n_samples} samples")
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from config import (
        SYNTH_PKL, VITAL_MAP_PKL, SFT_MODELS,
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, LORA_TARGET_MODULES_GEMMA4,
        SYSTEM_PROMPT, build_user_prompt, build_emr_text,
    )

    model_id = str(SFT_MODELS[base])
    print(f"\n  모델: {model_id}")
    print(f"  GPU: {torch.cuda.device_count()}개")

    # 데이터
    print("\n[1] 데이터")
    df = ok("SYNTH_PKL", lambda: pd.read_pickle(SYNTH_PKL).iloc[:n_samples])
    vmap = ok("VITAL_MAP_PKL", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if df is None or vmap is None:
        return False

    tokenizer = ok("토크나이저", lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    if tokenizer is None:
        return False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    for _, row in df.iterrows():
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vmap.get(sid, "")
        chosen = str(row.get("chosen", "특이사항 없음"))
        user = build_user_prompt(emr, vital)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": chosen},
        ]
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))

    dataset = ok("Dataset", lambda: Dataset.from_dict({"text": texts}))
    if dataset is None:
        return False

    # 모델
    print("\n[2] 모델 & LoRA")
    model = ok("모델 로드", lambda: AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ))
    if model is None:
        return False

    _targets = LORA_TARGET_MODULES_GEMMA4 if base == "gemma4" else LORA_TARGET_MODULES
    model = ok("LoRA", lambda: get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=_targets, bias="none", task_type="CAUSAL_LM",
    )))
    if model is None:
        return False
    model.config.use_cache = False
    model.print_trainable_parameters()

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU 메모리(로드 후): {mem:.1f}GB")

    # 학습
    print(f"\n[3] 학습 {steps} steps")
    cfg = ok("SFTConfig", lambda: SFTConfig(
        output_dir="/tmp/smoke_sft",
        max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=2e-4,
        bf16=True, logging_steps=1, max_seq_length=512,
        dataset_text_field="text", report_to="none", seed=42,
    ))
    trainer = ok("SFTTrainer", lambda: SFTTrainer(
        model=model, args=cfg, train_dataset=dataset, processing_class=tokenizer,
    ))
    if trainer is None:
        return False

    result = ok(f"{steps} steps 학습", lambda: trainer.train())
    if result is None:
        return False

    mem2 = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU 메모리(학습 후): {mem2:.1f}GB")
    print(f"\n{PASS} stage_sft 완료 → 02_sft_train.py 실행 가능")
    return True


# ── Stage 2: RLAIF (DPO) 3 steps ─────────────────────────────────────────
def stage_rlaif(base: str, loss: str = "dpo", steps: int = 3, n_samples: int = 8):
    header(f"Stage: rlaif — {base} · {loss} · {steps} steps")
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from config import (
        SYNTH_PKL, VITAL_MAP_PKL, SFT_MODELS,
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, LORA_TARGET_MODULES_GEMMA4,
        RLAIF_CONFIG, SYSTEM_PROMPT, build_user_prompt, build_emr_text,
    )

    model_id = str(SFT_MODELS[base])
    print(f"\n  모델: {model_id}\n  GPU: {torch.cuda.device_count()}개")

    df = ok("SYNTH_PKL", lambda: pd.read_pickle(SYNTH_PKL).iloc[:n_samples])
    vmap = ok("VITAL_MAP_PKL", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    tokenizer = ok("토크나이저", lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    if None in (df, vmap, tokenizer):
        return False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts, chosens, rejecteds = [], [], []
    for _, row in df.iterrows():
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vmap.get(sid, "")
        user = build_user_prompt(emr, vital)
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)
        chosens.append(str(row.get("chosen", "특이사항 없음")))
        rejecteds.append(str(row.get("rejected", "특이사항 없음")))

    dataset = ok("Dataset", lambda: Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds}))

    _max_mem = {i: "40GiB" for i in range(torch.cuda.device_count())} if base == "gemma4" else None
    model = ok("모델 로드", lambda: AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        max_memory=_max_mem, low_cpu_mem_usage=True, trust_remote_code=True,
    ))
    if model is None:
        return False

    _targets = LORA_TARGET_MODULES_GEMMA4 if base == "gemma4" else LORA_TARGET_MODULES
    model = ok("LoRA", lambda: get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=_targets, bias="none", task_type="CAUSAL_LM",
    )))
    if model is None:
        return False
    model.config.use_cache = False
    model.print_trainable_parameters()

    if loss == "simpo":
        from trainers.simpo_config import SimPOConfig
        from trainers.simpo_trainer import SimPOTrainer
        cfg = SimPOConfig(
            output_dir="/tmp/smoke_rlaif", max_steps=steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            max_length=512, max_prompt_length=384, bf16=True,
            logging_steps=1, report_to="none", seed=42,
        )
        trainer = ok("SimPOTrainer", lambda: SimPOTrainer(
            model=model, args=cfg, train_dataset=dataset, tokenizer=tokenizer,
        ))
    else:
        from trl import DPOTrainer, DPOConfig
        cfg = ok("DPOConfig", lambda: DPOConfig(
            output_dir="/tmp/smoke_rlaif", loss_type="sigmoid", max_steps=steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            max_length=512, bf16=True, logging_steps=1,
            gradient_checkpointing=True, report_to="none", seed=42,
        ))
        trainer = ok("DPOTrainer", lambda: DPOTrainer(
            model=model, ref_model=None, args=cfg,
            train_dataset=dataset, processing_class=tokenizer,
        ))

    if trainer is None:
        return False

    result = ok(f"{steps} steps 학습", lambda: trainer.train())
    if result is None:
        return False

    print(f"\n{PASS} stage_rlaif 완료 → 03_rlaif_train.py 실행 가능")
    return True


# ── Stage 3: Inference 샘플 2개 ──────────────────────────────────────────
def stage_infer(base: str, n: int = 2):
    header(f"Stage: infer — {base} · {n} samples")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from config import (
        GOLD_PKL, VITAL_MAP_PKL, SFT_MODELS,
        SYSTEM_PROMPT, build_user_prompt, build_emr_text,
    )

    model_id = str(SFT_MODELS[base])
    print(f"\n  모델: {model_id}")

    tokenizer = ok("토크나이저", lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    model = ok("모델 로드", lambda: AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ))
    if None in (tokenizer, model):
        return False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    gold_df = ok("GOLD_PKL", lambda: pd.read_pickle(GOLD_PKL))
    vmap = ok("VITAL_MAP_PKL", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if None in (gold_df, vmap):
        return False

    print(f"\n  Inference {n}건")
    generated_ok = 0
    for i, (_, row) in enumerate(gold_df.iloc[:n].iterrows()):
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vmap.get(sid, "")
        user = build_user_prompt(emr, vital)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        is_ok = len(gen) >= 2
        print(f"  {'✓' if is_ok else '✗'} [{i+1}] sid={sid}  출력: {gen[:80]!r}")
        if is_ok:
            generated_ok += 1

    if generated_ok == n:
        print(f"\n{PASS} stage_infer 완료 → 04_inference.py 실행 가능")
        return True
    else:
        print(f"\n{FAIL} 출력 실패 {n - generated_ok}/{n}건")
        return False


# ── Stage 4: Evaluate (Judge) 샘플 2개 ───────────────────────────────────
def stage_eval(n: int = 2):
    header(f"Stage: eval — judge {n} samples")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from config import GOLD_PKL, VITAL_MAP_PKL, EVAL_JUDGE_MODEL, build_user_prompt, build_emr_text
    from pipeline.evaluate_helpers import judge_score, BREVITY_RUBRIC

    tokenizer = ok("Judge 토크나이저", lambda: AutoTokenizer.from_pretrained(EVAL_JUDGE_MODEL, trust_remote_code=True))
    model = ok("Judge 모델 로드", lambda: AutoModelForCausalLM.from_pretrained(
        EVAL_JUDGE_MODEL, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ))
    if None in (tokenizer, model):
        return False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    gold_df = ok("GOLD_PKL", lambda: pd.read_pickle(GOLD_PKL))
    vmap = ok("VITAL_MAP_PKL", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if None in (gold_df, vmap):
        return False

    DUMMY_RESP = "SpO2 93% → O2 보충 유지. EBL 800mL → pRBC 2U 수혈."
    print(f"\n  Scoring {n}건 (dummy response: '{DUMMY_RESP}')")
    scored = 0
    for i, (_, row) in enumerate(gold_df.iloc[:n].iterrows()):
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vmap.get(sid, "")
        instruction = build_user_prompt(emr, vital)
        score = ok(f"[{i+1}] judge_score", lambda: judge_score(model, tokenizer, instruction, DUMMY_RESP, BREVITY_RUBRIC))
        if score is not None:
            print(f"       score: {score}")
            scored += 1

    if scored == n:
        print(f"\n{PASS} stage_eval 완료 → 05_evaluate.py 실행 가능")
        return True
    return False


# ── helpers module 없이 eval 돌릴 수 있도록 인라인 가져오기 ──────────────
def _import_judge():
    """05_evaluate.py에서 judge 함수 직접 임포트."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate", Path(__file__).resolve().parent.parent / "pipeline" / "05_evaluate.py"
    )
    mod = importlib.util.load_from_spec = spec
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def stage_eval_inline(n: int = 2):
    """05_evaluate.py의 judge_score를 직접 import해서 테스트."""
    header(f"Stage: eval — judge {n} samples")
    import torch
    from config import GOLD_PKL, VITAL_MAP_PKL, EVAL_JUDGE_MODEL, build_user_prompt, build_emr_text
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ev05",
        Path(__file__).resolve().parent.parent / "pipeline" / "05_evaluate.py",
    )
    ev05 = importlib.util.module_from_spec(spec)
    ok("05_evaluate.py import", lambda: spec.loader.exec_module(ev05))

    tokenizer = ok("Judge 토크나이저", lambda: ev05.load_judge_model(EVAL_JUDGE_MODEL)[1])
    model_and_tok = ok("Judge 모델", lambda: ev05.load_judge_model(EVAL_JUDGE_MODEL))
    if model_and_tok is None:
        return False
    judge_model, judge_tok = model_and_tok

    gold_df = ok("GOLD_PKL", lambda: pd.read_pickle(GOLD_PKL))
    vmap = ok("VITAL_MAP_PKL", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if None in (gold_df, vmap):
        return False

    DUMMY = "SpO2 93% → O2 보충. EBL 800mL → pRBC 2U."
    scored = 0
    for i, (_, row) in enumerate(gold_df.iloc[:n].iterrows()):
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vmap.get(sid, "")
        instr = build_user_prompt(emr, vital)
        score = ok(f"[{i+1}] judge_score", lambda instr=instr: ev05.judge_score(judge_model, judge_tok, instr, DUMMY, ev05.BREVITY_RUBRIC))
        if score is not None:
            print(f"       score: {score}")
            scored += 1

    if scored == n:
        print(f"\n{PASS} stage_eval 완료 → 05_evaluate.py 실행 가능")
        return True
    return False


# ── main ──────────────────────────────────────────────────────────────────
ALL_MODELS = ["llama", "qwen", "gemma4", "qwen35", "hari"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="파이프라인 단계별 smoke test")
    parser.add_argument("--stage", choices=["data", "sft", "rlaif", "infer", "eval"], default=None)
    parser.add_argument("--all", action="store_true", help="모든 단계 순서대로 실행")
    parser.add_argument(
        "--models", nargs="+",
        choices=ALL_MODELS + ["all"], default=["llama"],
        help="테스트할 모델 (여러 개 가능, 'all'로 전체 선택)",
    )
    parser.add_argument("--loss", choices=["dpo", "simpo"], default="dpo")
    parser.add_argument("--steps", type=int, default=3, help="학습 steps (sft/rlaif용)")
    parser.add_argument("--n", type=int, default=2, help="샘플 수 (infer/eval용)")
    parser.add_argument("--samples", type=int, default=8, help="학습용 데이터 샘플 수")
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    models = ALL_MODELS if "all" in args.models else args.models

    if not (args.stage or args.all):
        parser.print_help()
        sys.exit(0)

    # {model: {stage: bool}}
    results: dict = {}

    # data 단계는 모델 무관 — 한 번만
    if args.all or args.stage == "data":
        stage_data()
        results.setdefault("_common", {})["data"] = True

    for model in models:
        results.setdefault(model, {})

        if args.all or args.stage == "sft":
            results[model]["sft"] = stage_sft(model, args.steps, args.samples)

        if args.all or args.stage == "rlaif":
            results[model]["rlaif"] = stage_rlaif(model, args.loss, args.steps, args.samples)

        if args.all or args.stage == "infer":
            results[model]["infer"] = stage_infer(model, args.n)

    # eval 단계는 모델 무관 (judge 모델 고정) — 한 번만
    if args.all or args.stage == "eval":
        results.setdefault("_common", {})["eval"] = stage_eval_inline(args.n)

    # ── 결과 요약 테이블 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" Smoke Test 결과 요약")
    print(f"{'='*60}")

    common = results.pop("_common", {})
    for stage, flag in common.items():
        print(f"  {'✓' if flag else '✗'} [공통] {stage}")

    if results:
        model_stages = sorted({s for v in results.values() for s in v})
        header_row = f"  {'모델':<12}" + "".join(f" {s:<8}" for s in model_stages)
        print(header_row)
        print(f"  {'-'*12}" + "".join(f" {'-'*8}" for _ in model_stages))
        for model, stages in results.items():
            row = f"  {model:<12}" + "".join(
                f" {'✓' if stages.get(s) else '✗':<8}" for s in model_stages
            )
            print(row)
    print(f"{'='*60}")
