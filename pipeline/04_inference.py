"""
04_inference.py — 배치 Inference

실행 예시:
  python 04_inference.py --model_path outputs/sft/llama_3ep/final --split eval --gpus 0
  python 04_inference.py --model_path outputs/rlaif/llama_dpo/final --split sft --gpus 2,3
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import sys, os, argparse, re


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
import json
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    SFT_PKL,
    SELFJUDGE_PKL,
    GOLD_PKL,
    RLHF_PKL,
    VITAL_MAP_PKL,
    INFER_OUT,
    SYSTEM_PROMPT,
    build_user_prompt,
    build_emr_text,
    INFER_CONFIG,
    INFER_CONFIG_THINKING,
    INFER_ENABLE_THINKING,
)

SPLIT_MAP = {
    "sft": SFT_PKL,
    "selfjudge": SELFJUDGE_PKL,
    "gold": GOLD_PKL,
    "rlhf": RLHF_PKL,
}

# ── Thinking 후처리 ──────────────────────────────────────────────────────

# 1) 닫힌 <think>...</think> 블록 제거 (정상 thinking 모델: 답이 </think> 뒤)
_RE_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL)

# 2) 안 닫힌 <think> → 끝까지 제거
#    (qwen35: 답을 먼저 내고 trailing으로 <think>가 붙고 안 닫힘)
_RE_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL)

# 3) 태그 없는 영어 CoT가 trailing으로 붙는 경우 → 그 지점부터 끝까지 제거
#    (qwen35: "...post-op\n\nThinking Process:\n1. Analyze..." 형태)
_RE_THINK_TRAILING = re.compile(
    r"\n*\s*(?:Thinking Process|Analyze the Request)\b.*$",
    re.DOTALL | re.IGNORECASE,
)

# 4) "Thinking Process:" 스타일 CoT가 맨 앞에 오는 경우 → 한국어 본문 직전까지 제거
_RE_THINK_PREAMBLE = re.compile(
    r"^\s*(?:Thinking Process|Analyze the Request|Analysis|Step \d|<think>).*?"
    r"(?=(?:##|환아|환자|\*\*환자|소아|▶|\d{2}:\d{2}|특이사항))",
    re.DOTALL | re.IGNORECASE,
)

# 5) 학습 데이터 오염 패턴
_RE_JUNK = re.compile(
    r"Name:\s*\d+,\s*dtype:\s*\w+|"
    r"^(?:assistant|user)\s*$|"
    r"위 데이터를 바탕으로[^。\n]*작성하세요\.?",
    re.MULTILINE,
)


def clean_output(text: str) -> str:
    """
    Thinking 블록 및 오염 패턴을 제거하고 실제 한국어 인계요약만 반환.
    원본은 generated_raw에 보존, 이 함수 결과는 generated 필드에 저장됨.

    순서 중요:
      1) 닫힌 <think>..</think> 먼저 제거 (그래야 그 뒤 본문 보존)
      2) 남은 안 닫힌 <think> → 끝까지 제거
      3) 태그 없는 trailing CoT 제거
    """
    text = _RE_THINK_CLOSED.sub("", text)
    text = _RE_THINK_OPEN.sub("", text)
    text = _RE_THINK_TRAILING.sub("", text)
    text = _RE_JUNK.sub("", text)
    text = _RE_THINK_PREAMBLE.sub("", text, count=1)

    cleaned = text.strip()
    if len(cleaned) < 5:
        cleaned = "[생성 실패: 출력 없음]"
    return cleaned


def is_thinking_model(model_path: str, tokenizer) -> bool:
    """Qwen3 계열 thinking 모델 여부 판별."""
    path_lower = model_path.lower()
    if any(k in path_lower for k in ("qwen3", "qwen35", "hari")):
        return True
    # tokenizer chat_template에 <think> 토큰이 있으면 thinking 모델
    tmpl = getattr(tokenizer, "chat_template", "") or ""
    return "<think>" in tmpl


# ── 유틸 ────────────────────────────────────────────────────────────────


def load_model(model_path: str, base_model_id: str = None):
    """모델 로드. PEFT 체크포인트면 base + adapter 방식."""
    adapter_config = Path(model_path) / "adapter_config.json"
    is_peft = adapter_config.exists()

    if is_peft and base_model_id:
        print(f"  PEFT 모델 로드: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _max_mem = {i: "40GiB" for i in range(torch.cuda.device_count())}
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=_max_mem,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print(f"  전체 모델 로드: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _max_mem = {i: "40GiB" for i in range(torch.cuda.device_count())}
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=_max_mem,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def apply_chat_template_safe(tokenizer, msgs: list, enable_thinking: bool) -> str:
    """
    apply_chat_template 호출.
    원본 노트북과 동일: 기본값은 enable_thinking=False.
    --thinking 플래그를 줬을 때만 True로 활성화.
    구버전 tokenizer는 enable_thinking 미지원이므로 try/except 처리.
    """
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=True, **kwargs)
        except TypeError:
            pass  # enable_thinking 미지원 → fallback
    else:
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(msgs, **kwargs)


# ── vLLM 경로 (tensor parallel — 31B 등 대형/다GPU에서 수십배 빠름) ────────


def _merge_lora_for_vllm(model_path: str, base_model: str) -> str:
    """PEFT 어댑터면 base와 merge해서 임시 dir에 저장 후 그 경로 반환 (캐시).
    vLLM은 merged full model을 로드. CPU에서 merge해 GPU는 vLLM이 독점."""
    adapter = Path(model_path) / "adapter_config.json"
    if not adapter.exists():
        return model_path  # 이미 full model
    if not base_model:
        raise ValueError("PEFT 체크포인트인데 --base_model 미지정")

    merged = Path(model_path) / "_merged_for_vllm"
    if (merged / "config.json").exists():
        print(f"  merged 캐시 사용: {merged}")
        return str(merged)

    print(f"  LoRA merge 중 (CPU)... base={base_model}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    m = PeftModel.from_pretrained(base, model_path)
    m = m.merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(merged))
    tok.save_pretrained(str(merged))
    del m, base
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  merge 완료: {merged}")
    return str(merged)


def run_inference_vllm(args) -> bool:
    """vLLM tensor-parallel 배치 추론. 성공 True, 미설치/실패 False(→HF fallback)."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  [vLLM] 미설치 → HF 경로로 fallback")
        return False

    split_pkl = SPLIT_MAP[args.split]
    tag = args.out_tag if args.out_tag else Path(args.model_path).name
    out_dir = INFER_OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.split}_results.jsonl"

    n_gpu = torch.cuda.device_count()
    df = pd.read_pickle(split_pkl)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    print(f"\n[Inference · vLLM]")
    print(f"  모델:   {args.model_path}")
    print(f"  Split:  {args.split}  ({split_pkl.name})  데이터 {len(df)}건")
    print(f"  GPU:    {n_gpu}개 (tensor_parallel)")

    try:
        model_dir = _merge_lora_for_vllm(args.model_path, args.base_model)
    except Exception as e:
        print(f"  [vLLM] merge 실패: {e} → HF fallback")
        return False

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_thinking_arch = is_thinking_model(args.base_model or args.model_path, tokenizer)
    enable_thinking = args.thinking if args.thinking is not None else INFER_ENABLE_THINKING
    cfg = INFER_CONFIG_THINKING if (is_thinking_arch and enable_thinking) else INFER_CONFIG

    # 프롬프트 일괄 생성
    prompts, metas = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vital_map.get(sid, "")
        user = build_user_prompt(emr, vital)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompts.append(apply_chat_template_safe(tokenizer, msgs, enable_thinking))
        metas.append((i, sid))

    try:
        llm = LLM(
            model=model_dir,
            tensor_parallel_size=n_gpu,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  [vLLM] 모델 로드 실패({type(e).__name__}: {e}) → HF fallback")
        return False

    sp_kwargs = dict(max_tokens=cfg["max_new_tokens"])
    if cfg.get("do_sample"):
        sp_kwargs["temperature"] = cfg.get("temperature", 0.7)
        if cfg.get("top_p") is not None:
            sp_kwargs["top_p"] = cfg["top_p"]
    else:
        sp_kwargs["temperature"] = 0.0
    sampling = SamplingParams(**sp_kwargs)

    print(f"  배치 생성 시작 ({len(prompts)}건)...")
    outputs = llm.generate(prompts, sampling)

    with open(out_file, "w", encoding="utf-8") as fout:
        for (i, sid), out in zip(metas, outputs):
            raw = out.outputs[0].text.strip()
            cleaned = clean_output(raw)
            rec = {"idx": i, "sid": sid, "generated_raw": raw, "generated": cleaned}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[완료·vLLM] {len(df)}건 저장: {out_file}")
    return True


# ── 메인 ────────────────────────────────────────────────────────────────


def run_inference(args):
    model_path = args.model_path
    split_pkl = SPLIT_MAP[args.split]

    tag = args.out_tag if args.out_tag else Path(model_path).name
    out_dir = INFER_OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.split}_results.jsonl"

    n_gpu = torch.cuda.device_count()
    print(f"\n[Inference]")
    print(f"  모델:   {model_path}")
    print(f"  Split:  {args.split}  ({split_pkl.name})")
    print(f"  출력:   {out_file}")
    print(f"  GPU:    {n_gpu}개")

    # 데이터 로드
    df = pd.read_pickle(split_pkl)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  데이터: {len(df)}건")

    # 모델 로드
    model, tokenizer = load_model(model_path, args.base_model)

    # Thinking 모드 결정:
    # 1) --thinking 플래그가 있으면 무조건 ON
    # 2) 없으면 config의 INFER_ENABLE_THINKING 기본값(False) 사용
    # ※ 원본 학습은 모두 enable_thinking=False → 기본값은 False 권장
    is_thinking_arch = is_thinking_model(args.base_model or model_path, tokenizer)
    enable_thinking = (
        args.thinking if args.thinking is not None else INFER_ENABLE_THINKING
    )

    if is_thinking_arch and enable_thinking:
        cfg = INFER_CONFIG_THINKING
        print(f"  Thinking 모드: ON  (max_new_tokens={cfg['max_new_tokens']})")
        print(
            "  ※ 주의: 학습이 enable_thinking=False로 진행됐다면 분포 불일치 발생 가능"
        )
    elif is_thinking_arch and not enable_thinking:
        cfg = INFER_CONFIG
        print(
            f"  Thinking 모드: OFF  (max_new_tokens={cfg['max_new_tokens']}, 원본 학습과 일치)"
        )
    else:
        cfg = INFER_CONFIG
        print(f"  max_new_tokens: {cfg['max_new_tokens']}")

    with open(out_file, "w", encoding="utf-8") as fout:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            emr = build_emr_text(row)
            try:
                v = row["수술 ID"]
                sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
            except Exception:
                sid = -1
            vital = vital_map.get(sid, "")
            user = build_user_prompt(emr, vital)

            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]

            prompt = apply_chat_template_safe(tokenizer, msgs, enable_thinking)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=cfg["max_new_tokens"],
                    do_sample=cfg.get("do_sample", False),
                    pad_token_id=tokenizer.eos_token_id,
                )
                if cfg.get("temperature") is not None:
                    gen_kwargs["temperature"] = cfg["temperature"]
                if cfg.get("top_p") is not None and cfg.get("do_sample"):
                    gen_kwargs["top_p"] = cfg["top_p"]
                output_ids = model.generate(**inputs, **gen_kwargs)

            raw = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            # thinking 블록 제거 (thinking ON/OFF 관계없이 항상 적용 — 방어적 처리)
            cleaned = clean_output(raw)

            rec = {
                "idx": i,
                "sid": sid,
                "generated_raw": raw,  # thinking 포함 원본 (분석용)
                "generated": cleaned,  # 정제본 (judge용)
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[완료] {len(df)}건 저장: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="모델 경로 (full model or PEFT adapter 폴더)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="PEFT 체크포인트일 때 베이스 모델 경로",
    )
    parser.add_argument(
        "--split",
        choices=list(SPLIT_MAP.keys()),
        default="gold",
        help="평가 데이터 split",
    )
    parser.add_argument(
        "--out_tag",
        type=str,
        default=None,
        help="출력 폴더 태그 (예: llama_raw). 없으면 모델명으로 자동 결정",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=None,
        help=(
            "Qwen3 thinking 모드 활성화. "
            "지정 시 max_new_tokens=8192(INFER_CONFIG_THINKING) 자동 적용. "
            "※ 원본 학습은 enable_thinking=False이므로 기본값은 OFF."
        ),
    )
    parser.add_argument(
        "--gpus", type=str, default=None, help="사용할 GPU 번호. 예: '0' 또는 '0,1'"
    )
    parser.add_argument(
        "--engine", choices=["auto", "vllm", "hf"], default="auto",
        help="추론 엔진. auto=vLLM 가능하면 사용(전 GPU tensor parallel), 실패 시 HF. "
             "hf=기존 transformers(느림). vllm=vLLM 강제.",
    )
    _args = parser.parse_args()

    # vLLM 우선 (auto/vllm), 실패하면 HF로 fallback
    if _args.engine in ("auto", "vllm"):
        ok = run_inference_vllm(_args)
        if ok:
            sys.exit(0)
        if _args.engine == "vllm":
            print("[오류] --engine vllm 강제인데 vLLM 실행 실패")
            sys.exit(1)
        print("[fallback] HF transformers 경로로 실행")
    run_inference(_args)
