"""
01_synthetic_gen.py — 합성데이터 생성 → LLM-as-Judge 채점 → chosen/rejected 선정

★ SYNTH_PKL(jsft_score_251213.pkl)이 이미 존재하면 이 스크립트를 건너뛸 수 있습니다.
   vitals는 02_sft_train.py에서 build_user_prompt(emr, vital)로 학습 시 주입됩니다.

실행:
  python 01_synthetic_gen.py                      # llama + qwen으로 생성
  python 01_synthetic_gen.py --models llama       # llama만 사용
  python 01_synthetic_gen.py --debug              # 소량(5건)만 실행
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    EMR_HANDOFF_COL,
    EMR_SID_COL,
    EVAL_JUDGE_MODEL,
    GOLD_REF_PKL,
    SFT_MODELS,
    SFT_PKL,
    SYNTH_PKL,
    SYSTEM_PROMPT,
    VITAL_MAP_PKL,
    build_emr_text,
    build_user_prompt,
)

RANDOM_SEED = 42
MAX_INPUT_LEN = 2048
MAX_OUTPUT_LEN = 512

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Judge 프롬프트 (원본 data_synthetic_scoring.ipynb와 동일 구조) ─────────
_SCORE_PROMPT = """\
###Task Description:
An instruction, a response to evaluate, a reference answer (score 5), and a score rubric are given.
1. Write detailed feedback strictly based on the score rubric.
2. After feedback, write a score (integer 1-5).
3. Output format: "(feedback) [RESULT] (score)"
4. No other opening or closing.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubric:
{rubric}

###Feedback:"""

_BREVITY_RUBRIC = (
    "Evaluate for brevity and exception-based reporting. "
    "Score 5: ONLY abnormal findings mentioned, nothing normal. "
    "Score 1: verbose, lists normal/stable findings."
)
_CRITICAL_RUBRIC = (
    "Evaluate for clinical actionability. "
    "Score 5: only actionable abnormalities or minimal 'None'. "
    "Score 1: normal findings, reassurance phrases, routine context."
)


# ── 유틸 ─────────────────────────────────────────────────────────────────


def _get_sid(row) -> int:
    """MultiIndex Series에서 수술ID 안전 추출."""
    try:
        v = row[EMR_SID_COL]
        return int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
    except Exception:
        return -1


def _get_handoff(gold_df: pd.DataFrame) -> str:
    """Gold 데이터에서 human handoff 텍스트 추출 (few-shot reference용)."""
    # flat 컬럼 우선 탐색
    for col in ["Human_handoff", "human_handoff", "handoff"]:
        if col in gold_df.columns:
            for v in gold_df[col]:
                s = str(v).strip()
                if s and s not in ("nan", "None", ""):
                    return s
    # MultiIndex 컬럼
    try:
        for v in gold_df[EMR_HANDOFF_COL]:
            s = str(v).strip()
            if s and s not in ("nan", "None", ""):
                return s
    except Exception:
        pass
    return "특이사항 없음"


def _apply_chat(tokenizer, system: str, user: str, is_qwen: bool) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if is_qwen:
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(msgs, **kwargs)


def _generate(model, tokenizer, text: str) -> str:
    ids = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN
    ).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=MAX_OUTPUT_LEN,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def _score(judge_model, judge_tok, instruction: str, response: str, reference: str, rubric: str) -> float:
    prompt = _SCORE_PROMPT.format(
        instruction=instruction, response=response, reference=reference, rubric=rubric
    )
    ids = judge_tok(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).input_ids.to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(
            ids,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=judge_tok.eos_token_id,
        )
    text = judge_tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    m = re.search(r"\[RESULT\]\s*([1-5])", text)
    try:
        return float(m.group(1))
    except Exception:
        nums = re.findall(r"\b([1-5])\b", text)
        return float(nums[-1]) if nums else 3.0


def _build_fewshot_user(emr_text: str, vital_summary: str, gold_df: pd.DataFrame, n_shot: int) -> str:
    """few-shot 예시 포함 user 프롬프트 생성."""
    examples = gold_df.sample(n=min(n_shot, len(gold_df)), random_state=RANDOM_SEED)
    shot_text = ""
    for i, (_, ex) in enumerate(examples.iterrows(), 1):
        ex_emr = build_emr_text(ex)
        try:
            v = ex[EMR_HANDOFF_COL]
            ex_ho = str(v).strip() if v and str(v) not in ("nan", "None", "") else "특이사항 없음"
        except Exception:
            ex_ho = "특이사항 없음"
        shot_text += f"\nExample {i}\n**EMR**\n{ex_emr}\n**PACU/ICU Handoff**\n{ex_ho}\n--"

    base = build_user_prompt(emr_text, vital_summary)
    return (
        "Using the EMR below, generate an ultra-brief PACU/ICU handoff.\n"
        '- If NO clinically meaningful issues → output exactly "특이사항 없음".\n'
        "- If issues exist → 1-5 very short sentences in formal Korean.\n\n"
        f"### EXAMPLES (follow style, do NOT copy content)\n{shot_text}\n\n"
        f"### EMR\n{base.split('### EMR')[1].split('### OUTPUT')[0].strip()}\n\n"
        "### OUTPUT"
    )


# ── 메인 ─────────────────────────────────────────────────────────────────


def main(args):
    print("[1] 데이터 로드...")
    sft_df = pd.read_pickle(SFT_PKL)
    gold_df = pd.read_pickle(GOLD_REF_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    print(f"  SFT: {len(sft_df)}건  |  Gold: {len(gold_df)}건  |  Vital: {len(vital_map)}건")

    if args.debug:
        sft_df = sft_df.head(5)
        print("  [DEBUG] 5건만 실행")

    ref_handoff = _get_handoff(gold_df)
    print(f"  Few-shot reference: '{ref_handoff[:50]}...'")

    print("[2] Judge 모델 로드...")
    j_tok = AutoTokenizer.from_pretrained(EVAL_JUDGE_MODEL, trust_remote_code=True)
    if j_tok.pad_token is None:
        j_tok.pad_token = j_tok.eos_token
    j_model = AutoModelForCausalLM.from_pretrained(
        EVAL_JUDGE_MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    j_model.eval()

    # idx → {model_key: [response_list]} 형태로 수집
    # row당 2개 이상 응답을 모아서 best/worst 선택
    all_responses: dict[int, list[dict]] = {}  # idx -> [{"response": str, "score": float}]

    for model_key in args.models:
        model_path = str(SFT_MODELS[model_key])
        is_qwen = "qwen" in model_key.lower()
        print(f"\n[3] 생성 모델: {model_key}  ({model_path})")

        g_tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if g_tok.pad_token is None:
            g_tok.pad_token = g_tok.eos_token
        g_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        g_model.eval()

        for n_shot in [2, 4]:
            print(f"  {n_shot}-shot 생성 중...")
            for idx, row in tqdm(sft_df.iterrows(), total=len(sft_df)):
                emr = build_emr_text(row)
                sid = _get_sid(row)
                vital = vital_map.get(sid, "")
                user = _build_fewshot_user(emr, vital, gold_df, n_shot)
                try:
                    prompt = _apply_chat(g_tok, SYSTEM_PROMPT, user, is_qwen)
                    response = _generate(g_model, g_tok, prompt)
                except Exception as e:
                    print(f"    생성 오류 idx={idx}: {e}")
                    response = "특이사항 없음"

                instruction = build_user_prompt(emr, vital)
                s_b = _score(j_model, j_tok, instruction, response, ref_handoff, _BREVITY_RUBRIC)
                s_c = _score(j_model, j_tok, instruction, response, ref_handoff, _CRITICAL_RUBRIC)

                if idx not in all_responses:
                    all_responses[idx] = []
                all_responses[idx].append({"response": response, "score": s_b + s_c})

        del g_model
        torch.cuda.empty_cache()

    print("\n[4] chosen / rejected 선정...")
    chosen_rows = []
    for idx, row in sft_df.iterrows():
        candidates = all_responses.get(idx, [])
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda x: x["score"], reverse=True)
        chosen = candidates[0]["response"]
        # rejected: 최저점 (복수일 때 랜덤)
        worst_score = candidates[-1]["score"]
        worst_pool = [c for c in candidates if c["score"] == worst_score]
        rejected = random.choice(worst_pool)["response"]

        new_row = row.copy()
        new_row["chosen"] = chosen
        new_row["rejected"] = rejected
        chosen_rows.append(new_row)

    synth_df = pd.DataFrame(chosen_rows)
    SYNTH_PKL.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_pickle(SYNTH_PKL)
    print(f"  완료: {len(synth_df)}건 저장 → {SYNTH_PKL}")
    print("\n[완료] 01_synthetic_gen.py 실행 완료.")
    print("  다음 단계: python 02_sft_train.py --base llama --epochs 3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="합성데이터 생성 + LLM-as-Judge 채점")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(SFT_MODELS.keys()),
        default=["llama", "qwen"],
        help="합성 응답 생성에 사용할 모델 (default: llama qwen)",
    )
    parser.add_argument("--debug", action="store_true", help="소량(5건)만 실행")
    main(parser.parse_args())
