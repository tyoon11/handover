"""
05_evaluate.py — LLM-as-Judge + SCALE 평가

실행 예시:
  python 05_evaluate.py --result_file outputs/inference/llama_3ep/gold_results.jsonl --gpus 0
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

import json
import pickle
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    GOLD_PKL,
    VITAL_MAP_PKL,
    EVAL_OUT,
    EVAL_JUDGE_MODEL,
    SYSTEM_PROMPT,
    build_user_prompt,
    EMR_PREOP_SUM_COL,
    EMR_PREMED_COL,
)

# ── Thinking 후처리 (04_inference와 동일 — judge 입력 이중 방어용) ───────

_RE_THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)
_RE_THINK_PREAMBLE = re.compile(
    r"^\s*(?:Thinking Process|Analyze the Request|Analysis|Step \d|<think>).*?"
    r"(?=(?:##|환아|환자|\*\*환자|소아|▶|\d{2}:\d{2}|특이사항))",
    re.DOTALL | re.IGNORECASE,
)
_RE_JUNK = re.compile(
    r"Name:\s*\d+,\s*dtype:\s*\w+|"
    r"^(?:assistant|user)\s*$|"
    r"위 데이터를 바탕으로[^。\n]*작성하세요\.?",
    re.MULTILINE,
)


def clean_output(text: str) -> str:
    """judge 입력 전 최종 정제. 04_inference에서 이미 정제됐으면 no-op에 가까움."""
    text = _RE_THINK_TAG.sub("", text)
    text = _RE_JUNK.sub("", text)
    text = _RE_THINK_PREAMBLE.sub("", text, count=1)
    cleaned = text.strip()
    return cleaned if len(cleaned) >= 5 else "특이사항 없음"


# ── 평가 루브릭 ───────────────────────────────────────────────────────────

BREVITY_RUBRIC = """Evaluate the handoff based on 'Time Efficiency' for a critical situation.
CRITICAL RULE: listing normal/stable vital signs or routine findings is 'CLINICAL NOISE' and must be penalized.
If the patient is stable, 'Unremarkable' or 'No issues' is the ONLY 5-point answer. Verbosity is a failure.
Score 1-5."""

CRITICAL_RUBRIC = """Evaluate whether the handoff includes ONLY clinically actionable abnormalities or correctly states no actionable issues exist.
Any mention of normal findings, stability, reassurance, or routine postoperative care is noise and must reduce the score.
Short responses are fully acceptable when accurate. Score 1-5."""


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


def load_judge_model(model_id: str):
    print(f"Judge 모델 로드: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    return mdl, tok


def judge_score(
    model, tokenizer, instruction: str, response: str, rubric: str
) -> float:
    """Prometheus 스타일 scoring → 1~5점."""
    prompt = (
        f"###Task Description:\n"
        f"Instruction: {instruction}\n"
        f"Response: {response}\n"
        f"Score Rubric: {rubric}\n"
        f"###Feedback:"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    m = re.search(r"\[RESULT\]\s*([1-5])", text)
    if m:
        return float(m.group(1))
    nums = re.findall(r"\b([1-5])\b", text)
    return float(nums[-1]) if nums else 3.0


# ── 메인 ────────────────────────────────────────────────────────────────


def evaluate(args):
    result_file = Path(args.result_file)
    tag = args.out_tag if args.out_tag else result_file.parent.name
    out_dir = EVAL_OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / result_file.name.replace(".jsonl", "_scores.jsonl")

    print(f"\n[Evaluate]")
    print(f"  결과 파일: {result_file}")
    print(f"  출력:      {out_file}")

    # 결과 로드
    with open(result_file, encoding="utf-8") as f:
        results = [json.loads(l) for l in f]
    print(f"  샘플 수:  {len(results)}건")

    # generated 필드 없는 구버전 jsonl 호환
    # 구버전: "generated" 필드가 raw 출력 (thinking 포함 가능)
    # 신버전: "generated" = 정제본, "generated_raw" = 원본
    has_raw_field = any("generated_raw" in r for r in results[:3])
    print(
        f"  포맷: {'신버전 (generated_raw 있음)' if has_raw_field else '구버전 (generated만 있음)'}"
    )

    # Gold 데이터 로드 (EMR 텍스트 재구성용)
    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    # Judge 모델
    judge_model, judge_tok = load_judge_model(args.judge_model or EVAL_JUDGE_MODEL)

    scored = []
    skipped = 0

    with open(out_file, "w", encoding="utf-8") as fout:
        for rec in tqdm(results):
            idx = rec["idx"]
            sid = rec.get("sid", -1)

            # generated 필드 추출 + 이중 정제
            # 신버전은 이미 정제됐지만 혹시 모를 잔여 오염도 제거
            gen_raw = rec.get("generated", rec.get("response", ""))
            gen = clean_output(gen_raw)

            # 생성 실패 케이스 스킵
            if gen in ("[생성 실패: 출력 없음]", "특이사항 없음") and len(gen_raw) < 5:
                skipped += 1
                out_rec = {
                    **rec,
                    "brevity_score": 1.0,
                    "critical_score": 1.0,
                    "sum_score": 2.0,
                    "judge_note": "생성 실패 — 기본값 1점 부여",
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                scored.append(out_rec)
                continue

            # EMR 텍스트 재구성
            row = gold_df.iloc[idx] if idx < len(gold_df) else None
            emr = _emr_text(row) if row is not None else ""
            vital = vital_map.get(sid, "")
            instruction = build_user_prompt(emr, vital)

            b_score = judge_score(
                judge_model, judge_tok, instruction, gen, BREVITY_RUBRIC
            )
            c_score = judge_score(
                judge_model, judge_tok, instruction, gen, CRITICAL_RUBRIC
            )

            out_rec = {
                **rec,
                "generated": gen,  # 정제된 버전으로 덮어쓰기
                "brevity_score": b_score,
                "critical_score": c_score,
                "sum_score": b_score + c_score,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            scored.append(out_rec)

    # 최종 통계
    scores = pd.DataFrame(scored)
    print(f"\n[결과 요약]")
    print(f"  Brevity:  {scores['brevity_score'].mean():.3f}")
    print(f"  Critical: {scores['critical_score'].mean():.3f}")
    print(f"  SUM:      {scores['sum_score'].mean():.3f}")
    if skipped:
        print(f"  생성 실패: {skipped}건 (점수 1점 처리)")
    print(f"\n  저장: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge 평가")
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="04_inference.py 출력 jsonl 파일 경로",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Judge 모델 경로 (기본값: config.EVAL_JUDGE_MODEL)",
    )
    parser.add_argument(
        "--out_tag", type=str, default=None, help="출력 폴더 태그 (예: llama_raw)"
    )
    parser.add_argument(
        "--gpus", type=str, default=None, help="사용할 GPU 번호. 예: '0' 또는 '0,1'"
    )
    evaluate(parser.parse_args())
