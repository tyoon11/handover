"""
04_inference.py — 배치 Inference

실행 예시:
  python 04_inference.py --model_path outputs/sft/llama_3ep/final --split eval --gpus 0
  python 04_inference.py --model_path outputs/rlaif/llama_dpo/final --split sft --gpus 2,3
"""

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
    INFER_CONFIG,
    EMR_PREOP_SUM_COL,
    EMR_PREMED_COL,
)

SPLIT_MAP = {
    "sft": SFT_PKL,
    "selfjudge": SELFJUDGE_PKL,
    "gold": GOLD_PKL,
    "rlhf": RLHF_PKL,
}

# ── Thinking 후처리 ──────────────────────────────────────────────────────

# 1) 정식 <think>...</think> 블록 제거
_RE_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)

# 2) "Thinking Process:" 스타일의 영어 CoT 제거
#    한국어 내용(##, 환아/환자, 시간 타임스탬프)이 시작되기 직전까지 제거
_RE_THINK_PREAMBLE = re.compile(
    r"^\s*(?:Thinking Process|Analyze the Request|Analysis|Step \d|<think>).*?"
    r"(?=(?:##|환아|환자|\*\*환자|소아|▶|\d{2}:\d{2}|특이사항))",
    re.DOTALL | re.IGNORECASE,
)

# 3) 학습 데이터 오염 패턴
_RE_JUNK = re.compile(
    r"Name:\s*\d+,\s*dtype:\s*\w+|"
    r"^(?:assistant|user)\s*$|"
    r"위 데이터를 바탕으로[^。\n]*작성하세요\.?",
    re.MULTILINE,
)

# 4) Qwen3 thinking 후 출력의 앞부분 잘린 패턴
#    (예: "* 환아는 CASK..."처럼 drafting 도중 잘린 경우)
_RE_INCOMPLETE_BULLET = re.compile(r"^\s*\*\s+[가-힣]", re.MULTILINE)


def clean_output(text: str) -> str:
    """
    Thinking 블록 및 오염 패턴을 제거하고 실제 한국어 인계요약만 반환.
    원본은 generated_raw에 보존, 이 함수 결과는 generated 필드에 저장됨.
    """
    text = _RE_THINK_TAG.sub("", text)
    text = _RE_JUNK.sub("", text)
    text = _RE_THINK_PREAMBLE.sub("", text, count=1)

    # 남은 내용이 없거나 의미없는 단편이면 명시
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


def load_model(model_path: str, base_model_id: str = None):
    """모델 로드. PEFT 체크포인트면 base + adapter 방식."""
    adapter_config = Path(model_path) / "adapter_config.json"
    is_peft = adapter_config.exists()

    if is_peft and base_model_id:
        print(f"  PEFT 모델 로드: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print(f"  전체 모델 로드: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def apply_chat_template_safe(tokenizer, msgs: list, thinking: bool) -> str:
    """
    apply_chat_template 호출.
    Qwen3 계열이면 enable_thinking=True로 thinking 활성화.
    구버전 tokenizer는 enable_thinking 미지원이므로 try/except 처리.
    """
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if thinking:
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=True, **kwargs)
        except TypeError:
            pass  # enable_thinking 미지원 tokenizer → 일반 방식으로 fallback
    return tokenizer.apply_chat_template(msgs, **kwargs)


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

    # Thinking 모델 여부
    thinking = is_thinking_model(args.base_model or model_path, tokenizer)
    print(f"  Thinking 모드: {'ON' if thinking else 'OFF'}")

    cfg = INFER_CONFIG
    print(f"  max_new_tokens: {cfg['max_new_tokens']}")

    with open(out_file, "w", encoding="utf-8") as fout:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            emr = _emr_text(row)
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

            prompt = apply_chat_template_safe(tokenizer, msgs, thinking)

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

            # thinking 블록 제거 → judge용 정제 텍스트
            cleaned = clean_output(raw) if thinking else raw

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
        "--gpus", type=str, default=None, help="사용할 GPU 번호. 예: '0' 또는 '0,1'"
    )
    run_inference(parser.parse_args())
