"""
01_synthetic_gen.py
합성데이터 생성 → LLM-as-Judge 채점 → chosen/rejected 선정

실행: python 01_synthetic_gen.py --model llama  (또는 qwen)
"""

import argparse, pickle, random, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from config import (
    SPLIT_PKL,
    VITAL_MAP,
    SYNTH_PKL,
    SID_COL,
    AGE_COL,
    EMR_COLS,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    EVAL_RUBRICS,
    MAX_INPUT_LEN,
    MAX_OUTPUT_LEN,
    RANDOM_SEED,
    SYNTH_MODELS,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# EMR 포맷팅
# ─────────────────────────────────────────────


def get_col(row, key):
    cols = EMR_COLS[key]
    try:
        v = row
        for c in cols:
            v = v[c]
        return str(v).strip() if pd.notna(v) else ""
    except Exception:
        return ""


def emr_input(row, vital_map: dict) -> str:
    sid = row[SID_COL]
    pre_summary = get_col(row, "pre_summary")
    premed = get_col(row, "premedication")
    totals = get_col(row, "totals")
    record = get_col(row, "record")
    vital_text = vital_map.get(sid, "")

    parts = [
        "- Pre-anesthetic Patient Conditions Summary",
        pre_summary,
        "\n- Preoperative Preparations and Premedication",
        premed,
        "\n- Anesthetic TOTALS",
        totals,
        "\n- Anesthetic Record",
        record,
    ]
    if vital_text:
        parts += ["\n" + vital_text]
    return "\n".join(p for p in parts if p.strip())


# ─────────────────────────────────────────────
# 합성데이터 생성 (few-shot)
# ─────────────────────────────────────────────


def make_prompt_fewshot(emr_text: str, gold_df, n_shot: int, tokenizer) -> list[dict]:
    """n_shot개 예시를 K-means 대신 랜덤으로 선택 (간소화)"""
    examples = gold_df.sample(n=n_shot, random_state=RANDOM_SEED)
    shot_text = ""
    for i, (_, ex) in enumerate(examples.iterrows(), 1):
        shot_text += (
            f"\nExample {i}\n**EMR**\n{ex['emr_text']}\n"
            f"**PACU/ICU Handoff**\n{ex['Human_handoff']}\n--"
        )

    user = (
        "Using the EMR below, generate an ultra-brief PACU/ICU handoff.\n"
        '- If NO clinically meaningful issues → output exactly "특이사항 없음".\n'
        "- If issues exist → 1-5 very short sentences in formal Korean.\n\n"
        f"### EXAMPLES (follow style, do NOT copy content)\n{shot_text}\n\n"
        f"### EMR\n{emr_text}\n\n### OUTPUT"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_one(model, tokenizer, messages, device, max_new=MAX_OUTPUT_LEN) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN
    ).input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# LLM-as-Judge 채점
# ─────────────────────────────────────────────

SCORE_PROMPT = """###Task Description:
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


def score_response(
    judge_model, judge_tok, emr_text, response, reference, rubric_key, device
):
    rubric = EVAL_RUBRICS[rubric_key]
    rubric_str = (
        rubric["criteria"]
        + "\n"
        + "\n".join(f"Score {k}: {v}" for k, v in rubric["scores"].items())
    )
    prompt = SCORE_PROMPT.format(
        instruction=USER_PROMPT_TEMPLATE.format(emr_text=emr_text),
        response=response,
        reference=reference,
        rubric=rubric_str,
    )
    ids = judge_tok(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    ).input_ids.to(device)
    logits_list = []
    with torch.no_grad():
        out = judge_model.generate(
            ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=judge_tok.eos_token_id,
        )
    # 점수 토큰(1~5) 확률 가중합
    score_ids = {
        str(i): judge_tok.encode(str(i), add_special_tokens=False)[0]
        for i in range(1, 6)
    }
    text = judge_tok.decode(out.sequences[0][ids.shape[1] :], skip_special_tokens=True)
    match = re.search(r"\[RESULT\]\s*(\d)", text)
    try:
        base_score = int(match.group(1))
    except Exception:
        base_score = 3

    # 로짓 기반 소수점 점수 (tie-break용)
    first_scores = out.scores[0][0]
    probs = torch.softmax(first_scores, dim=-1)
    soft = sum(
        i * probs[tid].item() for i, tid in [(int(k), v) for k, v in score_ids.items()]
    )
    return soft if soft > 0 else float(base_score)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1] 데이터 로드...")
    with open(SPLIT_PKL, "rb") as f:
        splits = pickle.load(f)
    with open(VITAL_MAP, "rb") as f:
        vital_map = pickle.load(f)
    sft_df = splits["sft"].copy()
    gold_df = splits["gold"].copy()

    # gold에 emr_text 컬럼 추가
    gold_df["emr_text"] = [emr_input(row, vital_map) for _, row in gold_df.iterrows()]

    print("[2] Judge 모델 로드 (prometheus-8x7b-v2.0)...")
    JUDGE_MODEL = "prometheus-eval/prometheus-8x7b-v2.0"
    j_tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    j_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    j_model.eval()

    all_records = []

    for gen_model_id in SYNTH_MODELS:
        print(f"\n[3] 합성데이터 생성: {gen_model_id}")
        g_tok = AutoTokenizer.from_pretrained(gen_model_id)
        g_model = AutoModelForCausalLM.from_pretrained(
            gen_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        g_model.eval()

        for n_shot in [2, 4]:
            print(f"  {n_shot}-shot 생성 중...")
            for idx, row in tqdm(sft_df.iterrows(), total=len(sft_df)):
                emr_text = emr_input(row, vital_map)
                messages = make_prompt_fewshot(emr_text, gold_df, n_shot, g_tok)
                try:
                    response = generate_one(g_model, g_tok, messages, device)
                except Exception as e:
                    print(f"    생성 오류 idx={idx}: {e}")
                    response = "특이사항 없음"

                # 채점
                ref = gold_df["Human_handoff"].iloc[0]  # 참고용 reference
                s_brev = score_response(
                    j_model, j_tok, emr_text, response, ref, "brevity", device
                )
                s_crit = score_response(
                    j_model, j_tok, emr_text, response, ref, "critical_focus", device
                )

                all_records.append(
                    {
                        SID_COL: row[SID_COL],
                        "emr_text": emr_text,
                        "response": response,
                        "gen_model": gen_model_id,
                        "n_shot": n_shot,
                        "score_brev": s_brev,
                        "score_crit": s_crit,
                        "score_sum": s_brev + s_crit,
                    }
                )

        del g_model
        torch.cuda.empty_cache()

    print("\n[4] chosen / rejected 선정...")
    records_df = pd.DataFrame(all_records)
    chosen_records = []

    for sid, grp in records_df.groupby(SID_COL):
        best_idx = grp["score_sum"].idxmax()
        best_row = grp.loc[best_idx]
        rest = grp.drop(index=best_idx)
        rej_row = rest.sample(1, random_state=RANDOM_SEED).iloc[0]

        emr_text = best_row["emr_text"]
        chosen_records.append(
            {
                SID_COL: sid,
                "emr_text": emr_text,
                "chosen": best_row["response"],
                "rejected": rej_row["response"],
            }
        )

    synth_df = pd.DataFrame(chosen_records)
    with open(SYNTH_PKL, "wb") as f:
        pickle.dump(synth_df, f)
    print(f"  완료: {len(synth_df)}건 저장 → {SYNTH_PKL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="소량만 실행")
    main(parser.parse_args())
