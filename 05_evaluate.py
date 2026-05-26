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
    build_emr_text,
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


# ── Judge 시스템 프롬프트 (원본 system_prompt7 동일) ──────────────────────
# evaluation.ipynb의 system_prompt7에서 가져옴
JUDGE_SYSTEM_PROMPT = (
    "You grade PACU/ICU handoffs using strict exception-based rules and must follow "
    "the rubric exactly as written at all times. You must not reward length, detail, "
    "completeness, or fluency unless the rubric explicitly requires it. Short or minimal "
    "responses such as 'None' or 'No issues' must be given full credit when they satisfy "
    "the rubric. Any mention of normal findings, stability, routine postoperative care, or "
    "reassurance must be treated as noise and penalized. If any conflict arises between "
    "your general judgment and the rubric rules, you must always prioritize the rubric."
)

# Prometheus ABSOLUTE_PROMPT_WO_REF 포맷 (원본과 동일)
_ABSOLUTE_PROMPT = """\
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1) Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2) After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3) The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4) Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubric:
{rubric}

###Feedback:"""

# ── 평가 루브릭 (원본 evaluation.ipynb BEST 버전과 동일) ──────────────────
BREVITY_RUBRIC = """\
[Criteria] Evaluate the handoff based on 'Time Efficiency' for a critical situation. The highest score must be given to the shortest possible text that conveys the patient's stability.

CRITICAL RULE: listing normal/stable vital signs or routine findings (e.g., 'BP is stable', 'No fever') is considered 'CLINICAL NOISE' and must be penalized. If the patient is stable, a simple 'Unremarkable' or 'No issues' is the ONLY 5-point answer. Verbosity is a failure.

[Score 1]: Inefficient. The summary lists normal parameters, routine lab results, or standard procedures that do not require action. It wastes the receiver's time with 'stable' data.
[Score 2]: Verbose. It uses full sentences or conversational fillers to describe a condition that could be summarized in keywords. It includes non-essential positive findings.
[Score 3]: Acceptable. It is relatively short but still mentions 1-2 things that are 'normal' or 'routine' which could have been omitted for higher speed.
[Score 4]: Concise. Uses telegraphic style (fragments). Focuses mostly on active issues. Only very minor unnecessary words.
[Score 5]: Maximum Efficiency. It strictly follows 'Reporting by Exception'. It mentions NOTHING unless it is abnormal or actionable. If the patient is stable, it uses minimal words like 'Stable' or 'None'. No wasted syllables."""

CRITICAL_RUBRIC = """\
[Criteria] Evaluate whether the handoff includes ONLY clinically actionable abnormalities or correctly states that no actionable issues exist. Any mention of normal findings, stability, reassurance, or routine postoperative care is considered noise and must reduce the score. Short responses are fully acceptable when accurate, and length must not influence scoring.

[Score 1]: The handoff lists normal findings, stable vitals, reassurance statements, or routine postoperative plans, or it fails to identify whether any actionable issues exist. Any normal or routine information automatically places the response in this category.
[Score 2]: The handoff mentions the correct critical issue but mixes it with mild noise such as stable findings or minor unnecessary context, weakening the prioritization.
[Score 3]: The handoff identifies the main abnormal issue or states the absence of abnormalities but still includes mild dilution, extra wording, or unnecessary framing.
[Score 4]: The handoff clearly states the clinically relevant abnormality or confidently states that no abnormality exists, while maintaining minimal noise and strong focus.
[Score 5]: The handoff contains ONLY actionable abnormal information or, when there are no abnormal findings, uses a minimal phrase such as 'None' or 'No issues' without any normal findings, reassurance phrases, or unrelated context. Short statements must receive full credit."""


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
    """Prometheus 스타일 scoring → 1~5점. 원본 ABSOLUTE_PROMPT_WO_REF + system_prompt7 사용."""
    content = _ABSOLUTE_PROMPT.format(
        instruction=instruction,
        response=response,
        rubric=rubric,
    )
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    # Prometheus는 mistral 기반 → apply_chat_template 없이 직접 포맷
    # HF 모델로 로드한 경우 tokenizer의 chat_template 사용
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        # fallback: [INST] ... [/INST] mistral 포맷
        prompt = f"[INST] {JUDGE_SYSTEM_PROMPT}\n\n{content} [/INST]"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=3072
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
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
            emr = build_emr_text(row) if row is not None else ""
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

    # ── SCALE 평가 (--scale 플래그 지정 시) ──────────────────────────────
    if args.scale:
        run_scale_eval(scored, out_dir, result_file)


def run_scale_eval(scored: list, out_dir: Path, result_file: Path):
    """
    SCALE (Flan-T5 기반 factual consistency) 평가.
    원본 evaluation.ipynb의 scale_score 블록과 동일 로직.
    large + xl 두 모델 모두 실행, chunk max → final score.
    """
    try:
        from scale_score.scorer import SCALEScorer
    except ImportError:
        print("[SCALE] scale_score 미설치 — pip install scale_score 후 재시도")
        return

    print("\n[SCALE 평가 시작]")
    from config import EVAL_MODELS

    device_large = "cuda:0"
    device_xl    = "cuda:1"

    scorer_large = SCALEScorer(size="large", device=device_large)
    scorer_xl    = SCALEScorer(size="xl",    device=device_xl)

    CHUNK_SIZE = 100
    MAX_LEN    = 512

    premises = [r.get("emr_context", "") for r in scored]
    hypotheses = [r.get("generated", "") for r in scored]

    def make_hypo_chunks(hypo_list, tokenizer):
        chunked, counts = [], []
        for h in hypo_list:
            ids = tokenizer.encode(h, add_special_tokens=False)
            chunks = []
            for i in range(0, min(len(ids), MAX_LEN), CHUNK_SIZE):
                sub = tokenizer.decode(ids[i:i + CHUNK_SIZE], skip_special_tokens=True)
                if sub.strip():
                    chunks.append(sub)
            chunked.append(chunks or [h])
            counts.append(len(chunked[-1]))
        return chunked, counts

    hypo_large, cnt_large = make_hypo_chunks(hypotheses, scorer_large.tokenizer)
    hypo_xl,    cnt_xl    = make_hypo_chunks(hypotheses, scorer_xl.tokenizer)

    raw_large = scorer_large.score(premises, hypo_large)
    raw_xl    = scorer_xl.score(premises, hypo_xl)

    def aggregate(raw, counts):
        result, idx = [], 0
        for c in counts:
            result.append(max(raw[idx:idx + c]))
            idx += c
        return result

    scale_large = aggregate(raw_large, cnt_large)
    scale_xl    = aggregate(raw_xl,    cnt_xl)

    # 결과 병합 저장
    scale_file = out_dir / result_file.name.replace(".jsonl", "_scale.jsonl")
    with open(scale_file, "w", encoding="utf-8") as f:
        for rec, sl, sx in zip(scored, scale_large, scale_xl):
            f.write(
                json.dumps(
                    {**rec, "scale_large": sl, "scale_xl": sx},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"  SCALE large mean: {sum(scale_large)/len(scale_large):.4f}")
    print(f"  SCALE xl    mean: {sum(scale_xl)/len(scale_xl):.4f}")
    print(f"  저장: {scale_file}")


def compare_models(file_a: str, file_b: str):
    """
    두 모델 결과 jsonl 파일 간 통계 검정.
    원본 evaluation.ipynb의 paired t-test / Wilcoxon / Sign test 블록 재현.
    """
    import numpy as np
    from scipy import stats

    def load_scores(path):
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        return pd.DataFrame(recs)

    df_a = load_scores(file_a)
    df_b = load_scores(file_b)

    print("\n" + "=" * 70)
    print(f"Model A: {file_a}")
    print(f"Model B: {file_b}")
    print("=" * 70)
    print(
        "t-test / Wilcoxon 모두 p<0.05 → 두 값의 차이 유의미\n"
        "  · mean Δ 양수  → Model A가 더 좋음\n"
        "  · dz: |0.2| 작음 / |0.5| 중간 / |0.8| 큼\n"
        "  · rank-biserial: +1 A 항상 > B / 0 차이 없음 / -1 반대\n"
    )

    cols = [c for c in ["brevity_score", "critical_score", "sum_score",
                         "scale_large", "scale_xl", "text_length"] if c in df_a.columns]

    for col in cols:
        A = np.array(df_a[col])
        B = np.array(df_b[col])
        d = A - B
        n = len(d)

        t_stat, p_t = stats.ttest_rel(A, B)
        dz = d.mean() / d.std(ddof=1)
        se = d.std(ddof=1) / np.sqrt(n)
        ci = (d.mean() - 1.984 * se, d.mean() + 1.984 * se)

        w_stat, p_w = stats.wilcoxon(A, B, zero_method="wilcox")
        wins = int((d > 0).sum()); losses = int((d < 0).sum())
        rb = (wins - losses) / (wins + losses) if (wins + losses) > 0 else float("nan")

        p_sign = stats.binomtest(wins, wins + losses, p=0.5).pvalue

        sig_t  = " !!!" if p_t    < 0.05 else ""
        sig_w  = " !!!" if p_w    < 0.05 else ""
        sig_s  = " !!!" if p_sign < 0.05 else ""

        print(f"\n── {col} ──")
        print(f"  Paired t  : t={t_stat:.3f}, p={p_t:.4g}, Δ={d.mean():.4f}, CI={ci}, dz={dz:.3f}{sig_t}")
        print(f"  Wilcoxon  : W={w_stat}, p={p_w:.4g}, rank-biserial={rb:.3f}{sig_w}")
        print(f"  Sign test : wins={wins}, losses={losses}, p={p_sign:.4g}{sig_s}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge 평가")
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
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
        "--scale",
        action="store_true",
        help="SCALE (Flan-T5 factual consistency) 평가 추가 실행",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        default=None,
        help="두 모델 결과 jsonl 간 통계 검정 (t-test / Wilcoxon / Sign test)",
    )
    parser.add_argument(
        "--gpus", type=str, default=None, help="사용할 GPU 번호. 예: '0' 또는 '0,1'"
    )
    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare[0], args.compare[1])
    elif args.result_file:
        evaluate(args)
    else:
        parser.error("--result_file 또는 --compare 중 하나를 지정하세요.")
