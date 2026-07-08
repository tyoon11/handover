"""
inference.py — 배치 추론 v3 (vLLM 우선, HF fallback)

v1 04_inference.py 대비:
  T6  HF 경로 좌측 절단 + 프롬프트는 prompt_utils 한 벌 (### OUTPUT 생존 보장)
  B3  LoRA merge 캐시가 adapter '내용 해시' 키 — 재학습 감지
  B2  출력은 tmp → atomic rename + .done 마커(입력·체크포인트 해시 포함)
  -   레코드에 engine("vllm"/"hf")·정제 status·flags 기록 — 엔진 혼용이 조용히 섞이지 않음
  -   split: gold(봉인 — --allow_gold 필요) / dev

실행:
  python -m pipeline_v3.inference --model_path outputs_v3/<run>/sft/llama_3ep/final \
      --base llama --split dev --out_tag llama_sft_3ep --gpus 0,1
"""

import argparse
import json
import os
import sys


def _ensure_conda_libstdcxx():
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return
    libpath = os.path.join(prefix, "lib", "libstdc++.so.6")
    if os.path.exists(libpath):
        cur = os.environ.get("LD_PRELOAD", "")
        if libpath not in cur:
            os.environ["LD_PRELOAD"] = (libpath + ":" + cur).rstrip(":")


_ensure_conda_libstdcxx()


def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    g = p.parse_known_args()[0].gpus
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = g
        print(f"[GPU] CUDA_VISIBLE_DEVICES={g}")


_early_parse()

from pathlib import Path       # noqa: E402

from .config_v3 import (       # noqa: E402
    INFER_CONFIG, INFER_MAX_MODEL_LEN, INFER_OUT, MAX_PROMPT_TOKENS,
    ensure_dir, model_path,
)
from .data_splits import load_splits                       # noqa: E402
from .eval_v3.cleaning import clean_v3                      # noqa: E402
from .merging import merge_lora_for_vllm                    # noqa: E402
from .prompt_utils import (                                 # noqa: E402
    SYSTEM_PROMPT, build_emr_text, fit_chat_prompt, get_sid,
)
from .provenance import dir_hash, write_done_marker         # noqa: E402


def _build_prompts(tokenizer, df, vital_map):
    prompts, metas = [], []
    for i in range(len(df)):
        row = df.iloc[i]
        sid = get_sid(row)
        if sid == -1:
            raise RuntimeError(f"idx={i}: 수술ID 추출 실패 — 추론 불가 (조용한 -1 금지)")
        emr = build_emr_text(row)
        prompts.append(fit_chat_prompt(tokenizer, emr, vital_map.get(sid, ""),
                                       system=SYSTEM_PROMPT, budget=MAX_PROMPT_TOKENS,
                                       enable_thinking=False))
        metas.append((i, sid))
    return prompts, metas


def _write_results(out_file: Path, metas, raws, engine_name: str):
    ensure_dir(out_file.parent)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for (i, sid), raw in zip(metas, raws):
            cleaned, status, flags = clean_v3(raw)
            rec = dict(idx=i, sid=sid, generated_raw=raw, generated=cleaned,
                       gen_status=status, gen_flags=flags, engine=engine_name)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out_file)


def run_vllm(model_dir: str, prompts) -> list:
    import torch
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_dir, tensor_parallel_size=max(1, torch.cuda.device_count()),
              dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=INFER_MAX_MODEL_LEN, trust_remote_code=True)
    sp = SamplingParams(max_tokens=INFER_CONFIG["max_new_tokens"], temperature=0.0,
                        repetition_penalty=INFER_CONFIG["repetition_penalty"])
    outs = llm.generate(prompts, sp)
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return [o.outputs[0].text.strip() for o in outs]


def run_hf(model_dir_or_ckpt: str, base_dir: str, prompts) -> list:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    is_peft = (Path(model_dir_or_ckpt) / "adapter_config.json").exists()
    tok = AutoTokenizer.from_pretrained(model_dir_or_ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"        # T6: 우측절단으로 ### OUTPUT이 사라지던 버그 제거
    _max_mem = {i: "40GiB" for i in range(torch.cuda.device_count())}
    if is_peft:
        if not base_dir:
            raise ValueError("PEFT 체크포인트인데 base 모델 미지정")
        model = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype=torch.bfloat16, device_map="auto", max_memory=_max_mem,
            low_cpu_mem_usage=True, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, model_dir_or_ckpt)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir_or_ckpt, dtype=torch.bfloat16, device_map="auto",
            max_memory=_max_mem, low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()

    raws = []
    bs = INFER_CONFIG["batch_size"]
    for i in tqdm(range(0, len(prompts), bs)):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=INFER_MAX_MODEL_LEN - INFER_CONFIG["max_new_tokens"]
                  ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=INFER_CONFIG["max_new_tokens"],
                do_sample=False,
                repetition_penalty=INFER_CONFIG["repetition_penalty"],
                no_repeat_ngram_size=INFER_CONFIG["no_repeat_ngram_size"],
                pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            gen = out[j][enc["input_ids"].shape[1]:]
            raws.append(tok.decode(gen, skip_special_tokens=True).strip())
    return raws


def main():
    ap = argparse.ArgumentParser(description="배치 추론 v3")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--base", type=str, default=None,
                    help="PEFT 체크포인트일 때 base 모델 키")
    ap.add_argument("--split", choices=["gold", "dev"], default="dev")
    ap.add_argument("--allow_gold", action="store_true",
                    help="gold(최종 test)는 봉인 — 이 플래그 없이는 실행 거부 (Phase 3)")
    ap.add_argument("--out_tag", type=str, default=None)
    ap.add_argument("--engine", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--gpus", type=str, default=None)
    args = ap.parse_args()

    if args.split == "gold" and not args.allow_gold:
        print("[봉인] gold split은 최종 1회 비교 전 사용 금지. "
              "정말 최종 평가면 --allow_gold 를 명시하라.")
        sys.exit(3)

    splits = load_splits(need=(args.split,))
    df = splits[args.split]
    vital_map = splits["vital_map"]

    tag = args.out_tag or Path(args.model_path).name
    out_dir = INFER_OUT / tag
    out_file = out_dir / f"{args.split}_results.jsonl"

    base_dir = str(model_path(args.base)) if args.base else None
    is_peft = (Path(args.model_path) / "adapter_config.json").exists()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts, metas = _build_prompts(tokenizer, df, vital_map)
    print(f"[infer v3] {tag} split={args.split} n={len(prompts)}")

    engine_used = None
    raws = None
    if args.engine in ("auto", "vllm"):
        try:
            model_dir = merge_lora_for_vllm(args.model_path, base_dir) if is_peft \
                else args.model_path
            raws = run_vllm(model_dir, prompts)
            engine_used = "vllm"
        except Exception as e:
            print(f"[infer v3] vLLM 실패({type(e).__name__}: {e})")
            if args.engine == "vllm":
                sys.exit(1)
    if raws is None:
        raws = run_hf(args.model_path, base_dir, prompts)
        engine_used = "hf"
        print("  ⚠ HF fallback 사용 — 최종 비교는 전 변형 동일 엔진으로 다시 돌릴 것")

    _write_results(out_file, metas, raws, engine_used)
    write_done_marker(out_dir, dict(stage="inference", split=args.split, tag=tag,
                                    engine=engine_used, n=len(prompts),
                                    ckpt_hash=dir_hash(Path(args.model_path))))
    print(f"[완료] {len(prompts)}건 저장: {out_file} (engine={engine_used})")


if __name__ == "__main__":
    main()
