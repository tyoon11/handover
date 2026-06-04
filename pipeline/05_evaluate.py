"""
05_evaluate.py — LLM-as-Judge + SCALE 평가

실행 예시:
  python 05_evaluate.py --result_file outputs/inference/llama_3ep/gold_results.jsonl --gpus 0
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
    import tempfile
    print(f"Judge 모델 로드: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _max_mem = {i: "40GiB" for i in range(torch.cuda.device_count())}
    # MoE 모델(Mixtral 계열)은 weight 재저장 시 offload_folder 필요
    _offload_dir = tempfile.mkdtemp(prefix="hf_offload_judge_")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=_max_mem,
        low_cpu_mem_usage=True,
        offload_folder=_offload_dir,
        trust_remote_code=True,
    )
    mdl.eval()
    return mdl, tok


def judge_score(
    model, tokenizer, instruction: str, response: str, rubric: str
) -> float:
    """Absolute scoring → 1~5점. 텍스트 [RESULT] N 파싱 방식 (단독 평가 시)."""
    content = _ABSOLUTE_PROMPT.format(
        instruction=instruction,
        response=response,
        rubric=rubric,
    )
    # Mistral/Prometheus chat_template은 system role을 지원하지 않아 user에 합침
    msgs = [{"role": "user", "content": f"{JUDGE_SYSTEM_PROMPT}\n\n{content}"}]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
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


_AB_TOKEN_IDS: dict = {}


def _get_ab_token_ids(tokenizer) -> tuple:
    """' A', ' B' 토큰 ID 캐싱 (모델별로 다름)."""
    key = id(tokenizer)
    if key not in _AB_TOKEN_IDS:
        _AB_TOKEN_IDS[key] = (
            tokenizer.encode(" A", add_special_tokens=False)[-1],
            tokenizer.encode(" B", add_special_tokens=False)[-1],
        )
    return _AB_TOKEN_IDS[key]


def judge_score_ab(
    model, tokenizer, instruction: str, resp_a: str, resp_b: str, rubric: str
) -> float:
    """Pairwise A/B logprob scoring → P(A) / (P(A)+P(B)).

    원본 evaluation.ipynb 방식: 두 응답을 비교하여 다음 토큰의 logprob으로 소프트 스코어 반환.
    위치 편향 제거를 위해 A/B 순서를 반전한 두 번의 평가 평균을 호출 측에서 처리.
    """
    tok_a, tok_b = _get_ab_token_ids(tokenizer)

    content = (
        f"###Score Rubric:\n{rubric}\n\n"
        f"###Instruction:\n{instruction}\n\n"
        f"###Response A:\n{resp_a}\n\n"
        f"###Response B:\n{resp_b}\n\n"
        "###Question: Which response better follows the rubric?\n\n"
        "###Answer: Response"
    )
    # Mistral/Prometheus chat_template은 system role을 지원하지 않아 user에 합침
    msgs = [{"role": "user", "content": f"{JUDGE_SYSTEM_PROMPT}\n\n{content}"}]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"[INST] {JUDGE_SYSTEM_PROMPT}\n\n{content} [/INST]"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=3500
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    log_probs = torch.log_softmax(logits, dim=-1)
    p_a = log_probs[tok_a].exp().item()
    p_b = log_probs[tok_b].exp().item()
    denom = p_a + p_b
    return p_a / denom if denom > 1e-10 else 0.5


# ── 메인 ────────────────────────────────────────────────────────────────


def evaluate_pairwise(args):
    """A/B logprob 쌍 비교 평가 (원본 evaluation.ipynb 방식).

    result_file (모델 A)과 result_file_b (모델 B)를 케이스별로 매칭하여
    Brevity + Critical 각 루브릭에서 P(A) / (P(A)+P(B)) 소프트 스코어를 산출.
    위치 편향 제거를 위해 A→B 순과 B→A 순을 모두 평가한 후 평균.
    """
    file_a = Path(args.result_file)
    file_b = Path(args.result_file_b)

    tag = (
        args.out_tag
        if args.out_tag
        else f"{file_a.parent.name}_vs_{file_b.parent.name}"
    )
    out_dir = EVAL_OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_a = out_dir / file_a.name.replace(".jsonl", "_A_scores.jsonl")
    out_b = out_dir / file_b.name.replace(".jsonl", "_B_scores.jsonl")

    print(f"\n[Pairwise Evaluate]")
    print(f"  Model A: {file_a}")
    print(f"  Model B: {file_b}")

    with open(file_a, encoding="utf-8") as f:
        recs_a = [json.loads(l) for l in f]
    with open(file_b, encoding="utf-8") as f:
        recs_b = [json.loads(l) for l in f]

    assert len(recs_a) == len(recs_b), "두 파일의 케이스 수가 다릅니다."

    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    judge_model, judge_tok = load_judge_model(args.judge_model or EVAL_JUDGE_MODEL)

    scored_a, scored_b = [], []
    with open(out_a, "w", encoding="utf-8") as fa, open(
        out_b, "w", encoding="utf-8"
    ) as fb:
        for rec_a, rec_b in tqdm(zip(recs_a, recs_b), total=len(recs_a)):
            idx = rec_a["idx"]
            sid = rec_a.get("sid", -1)
            gen_a = clean_output(rec_a.get("generated", ""))
            gen_b = clean_output(rec_b.get("generated", ""))

            row = gold_df.iloc[idx] if idx < len(gold_df) else None
            emr = build_emr_text(row) if row is not None else ""
            vital = vital_map.get(sid, "")
            instruction = build_user_prompt(emr, vital)

            scores_a_brevity, scores_b_brevity = [], []
            scores_a_critical, scores_b_critical = [], []
            for rubric, sa_list, sb_list in [
                (BREVITY_RUBRIC, scores_a_brevity, scores_b_brevity),
                (CRITICAL_RUBRIC, scores_a_critical, scores_b_critical),
            ]:
                # 순서 1: A vs B → P(A wins)
                p_a = judge_score_ab(
                    judge_model, judge_tok, instruction, gen_a, gen_b, rubric
                )
                # 순서 2: B vs A → P(B wins) = 1 - P(A wins in reverse)
                p_b_rev = judge_score_ab(
                    judge_model, judge_tok, instruction, gen_b, gen_a, rubric
                )
                # 위치 편향 제거: P(A) = 평균(순서1 P(A), 1 - 순서2 P(B as A))
                sa = (p_a + (1.0 - p_b_rev)) / 2.0
                sb = 1.0 - sa
                sa_list.append(sa)
                sb_list.append(sb)

            # 1~5 스케일 변환: soft_score * 4 + 1
            def to_scale(p):
                return p * 4.0 + 1.0

            ra = {
                **rec_a,
                "brevity_score": to_scale(scores_a_brevity[0]),
                "critical_score": to_scale(scores_a_critical[0]),
                "sum_score": to_scale(scores_a_brevity[0])
                + to_scale(scores_a_critical[0]),
                "brevity_soft": scores_a_brevity[0],
                "critical_soft": scores_a_critical[0],
            }
            rb = {
                **rec_b,
                "brevity_score": to_scale(scores_b_brevity[0]),
                "critical_score": to_scale(scores_b_critical[0]),
                "sum_score": to_scale(scores_b_brevity[0])
                + to_scale(scores_b_critical[0]),
                "brevity_soft": scores_b_brevity[0],
                "critical_soft": scores_b_critical[0],
            }
            fa.write(json.dumps(ra, ensure_ascii=False) + "\n")
            fb.write(json.dumps(rb, ensure_ascii=False) + "\n")
            scored_a.append(ra)
            scored_b.append(rb)

    for label, scored, out_file in [("A", scored_a, out_a), ("B", scored_b, out_b)]:
        df = pd.DataFrame(scored)
        print(
            f"\n[Model {label}] Brevity: {df['brevity_score'].mean():.3f}  "
            f"Critical: {df['critical_score'].mean():.3f}  "
            f"SUM: {df['sum_score'].mean():.3f}"
        )
        print(f"  저장: {out_file}")


def _load_eval_deps(judge_model_id: str):
    """Judge 모델, gold_df, vital_map을 1회만 로드 (batch 평가용)."""
    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    judge_model, judge_tok = load_judge_model(judge_model_id)
    return judge_model, judge_tok, gold_df, vital_map


def _evaluate_single(result_file, out_tag, judge_model, judge_tok, gold_df, vital_map, args):
    """파일 1개 평가 (judge/gold/vital은 미리 로드된 것 재사용)."""
    result_file = Path(result_file)
    tag = out_tag if out_tag else result_file.parent.name
    out_dir = EVAL_OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / result_file.name.replace(".jsonl", "_scores.jsonl")

    print(f"\n[Evaluate] {result_file} → {out_file}")

    with open(result_file, encoding="utf-8") as f:
        results = [json.loads(l) for l in f]
    print(f"  샘플 수: {len(results)}건")

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


def evaluate(args):
    """단일 파일 평가 (구 인터페이스 — judge 모델 매번 로드)."""
    if args.result_file_b:
        evaluate_pairwise(args)
        return
    judge, tok, gold_df, vmap = _load_eval_deps(args.judge_model or EVAL_JUDGE_MODEL)
    _evaluate_single(args.result_file, args.out_tag, judge, tok, gold_df, vmap, args)


def evaluate_batch(args):
    """다중 파일 평가 — judge 1회 로드 + (옵션) SCALE 1회 로드."""
    judge, tok, gold_df, vmap = _load_eval_deps(args.judge_model or EVAL_JUDGE_MODEL)

    # Phase A: judge 평가 (SCALE은 잠시 끄고 두 번째 패스에서 일괄)
    do_scale = args.scale
    args.scale = False  # _evaluate_single 내부 SCALE 호출 비활성화
    print(f"\n[Batch Evaluate] Phase A: judge {len(args.result_files)}개 파일")
    for i, rf in enumerate(args.result_files):
        print(f"\n──── [{i+1}/{len(args.result_files)}] judge ────")
        out_tag = Path(rf).parent.name
        _evaluate_single(rf, out_tag, judge, tok, gold_df, vmap, args)
    args.scale = do_scale  # 복원

    # Phase B: SCALE 1회 로드 후 일괄 평가
    if do_scale:
        import gc
        del judge  # judge 모델 메모리 해제 (SCALE 위해)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[Batch Evaluate] Phase B: SCALE {len(args.result_files)}개 파일")
        scorer_large, scorer_xl = _load_scale_scorers()
        if scorer_large is None:
            return
        for i, rf in enumerate(args.result_files):
            out_tag = Path(rf).parent.name
            score_file = EVAL_OUT / out_tag / Path(rf).name.replace(".jsonl", "_scores.jsonl")
            if not score_file.exists():
                print(f"  [{i+1}/{len(args.result_files)}] [SKIP] {score_file.name} 없음")
                continue
            scored = [json.loads(l) for l in open(score_file, encoding="utf-8") if l.strip()]
            print(f"\n──── [{i+1}/{len(args.result_files)}] scale: {out_tag} ────")
            _run_scale_eval(scored, score_file.parent, score_file,
                            scorer_large, scorer_xl, gold_df, vmap)


def _load_scale_scorers():
    """SCALE (Flan-T5) scorers 1회 로드 — large + xl.
    scale_score 0.x는 get_flan_T5_model이 tokenizer를 항상 HF Hub에서 받음(버그).
    monkey-patch로 우회해서 로컬 경로 강제."""
    # SSL 차단 환경: HF Hub 접근 자체를 막아서 다른 경로로 다운로드 시도해도 즉시 실패
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from scale_score import scorer as _scale_scorer
        from scale_score import utils as _scale_utils
        SCALEScorer = _scale_scorer.SCALEScorer
    except ImportError:
        print("[SCALE] scale_score 미설치 — pip install scale_score 후 재시도")
        return None, None

    try:
        from config import EVAL_MODELS
        _p_large = EVAL_MODELS.get("flan-large")
        _p_xl = EVAL_MODELS.get("flan-xl")
        path_large = str(_p_large) if _p_large and Path(_p_large).exists() else None
        path_xl = str(_p_xl) if _p_xl and Path(_p_xl).exists() else None
    except ImportError:
        path_large = path_xl = None

    # scale_score 라이브러리 버그 우회
    # 1) get_flan_T5_model이 tokenizer를 항상 HF Hub에서 받음
    # 2) device 인자 받지만 model.to(device) 호출 안 함 → CPU에 머무름
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        device_large, device_xl = "cuda:0", "cuda:1"
    else:
        device_large = device_xl = "cuda:0" if n_gpu else "cpu"

    def _make_patched(device):
        """closure로 device 캡처 → tokenizer + model을 로컬에서 로드 + 명시적으로 to(device)"""
        def _patched(size, model_path):
            src = model_path or f"google/flan-t5-{size}"
            print(f"  [patched] load Flan-T5-{size} from: {src} → {device}")
            tokenizer = T5Tokenizer.from_pretrained(src, local_files_only=True)
            model = T5ForConditionalGeneration.from_pretrained(src, local_files_only=True)
            model = model.to(device)
            model.eval()
            return model, tokenizer
        return _patched

    _orig_utils_get = _scale_utils.get_flan_T5_model
    _orig_scorer_get = getattr(_scale_scorer, "get_flan_T5_model", None)

    def _apply_patch(patch_fn):
        _scale_utils.get_flan_T5_model = patch_fn
        if _orig_scorer_get is not None:
            _scale_scorer.get_flan_T5_model = patch_fn

    print(f"\n[SCALE] scorer 로드 (large: {device_large}, xl: {device_xl})")
    print(f"  flan-large path: {path_large or 'HF Hub'}")
    print(f"  flan-xl    path: {path_xl or 'HF Hub'}")

    try:
        _apply_patch(_make_patched(device_large))
        scorer_large = SCALEScorer(size="large", device=device_large, model_path=path_large)
        _apply_patch(_make_patched(device_xl))
        scorer_xl = SCALEScorer(size="xl", device=device_xl, model_path=path_xl)
    finally:
        _scale_utils.get_flan_T5_model = _orig_utils_get
        if _orig_scorer_get is not None:
            _scale_scorer.get_flan_T5_model = _orig_scorer_get

    # 추가 방어: SCALEScorer가 self.device 속성 못 쓰는 경우 강제 동기화
    for scorer, dev in [(scorer_large, device_large), (scorer_xl, device_xl)]:
        if hasattr(scorer, "device"):
            scorer.device = dev
        if hasattr(scorer, "model") and hasattr(scorer.model, "to"):
            scorer.model.to(dev)

    return scorer_large, scorer_xl


def _run_scale_eval(scored: list, out_dir: Path, result_file: Path,
                    scorer_large, scorer_xl, gold_df=None, vital_map=None):
    """SCALE 평가 본체. scorer는 외부에서 1회 로드된 것을 받아 사용."""
    if scorer_large is None or scorer_xl is None:
        return

    CHUNK_SIZE = 100
    MAX_LEN = 512

    # premise(EMR) 재구성 — scored에 emr_context 없으므로 idx로 gold_df에서 가져옴
    if gold_df is None:
        gold_df = pd.read_pickle(GOLD_PKL)
    if vital_map is None:
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)

    premises, hypotheses = [], []
    for r in scored:
        idx = r.get("idx", 0)
        sid = r.get("sid", -1)
        row = gold_df.iloc[idx] if idx < len(gold_df) else None
        emr = build_emr_text(row) if row is not None else ""
        vital = vital_map.get(sid, "")
        premises.append(build_user_prompt(emr, vital))
        hypotheses.append(r.get("generated", ""))

    def make_hypo_chunks(hypo_list, tokenizer):
        chunked, counts = [], []
        for h in hypo_list:
            ids = tokenizer.encode(h, add_special_tokens=False)
            chunks = []
            for i in range(0, min(len(ids), MAX_LEN), CHUNK_SIZE):
                sub = tokenizer.decode(ids[i:i+CHUNK_SIZE], skip_special_tokens=True)
                if sub.strip():
                    chunks.append(sub)
            chunked.append(chunks or [h])
            counts.append(len(chunked[-1]))
        return chunked, counts

    hypo_large, cnt_large = make_hypo_chunks(hypotheses, scorer_large.tokenizer)
    hypo_xl, cnt_xl = make_hypo_chunks(hypotheses, scorer_xl.tokenizer)

    raw_large = scorer_large.score(premises, hypo_large)
    raw_xl = scorer_xl.score(premises, hypo_xl)

    def aggregate(raw, counts):
        result, i = [], 0
        for c in counts:
            result.append(max(raw[i:i+c]))
            i += c
        return result

    scale_large = aggregate(raw_large, cnt_large)
    scale_xl = aggregate(raw_xl, cnt_xl)

    scale_file = out_dir / result_file.name.replace(".jsonl", "_scale.jsonl")
    with open(scale_file, "w", encoding="utf-8") as f:
        for rec, sl, sx in zip(scored, scale_large, scale_xl):
            f.write(json.dumps({**rec, "scale_large": sl, "scale_xl": sx},
                               ensure_ascii=False) + "\n")
    print(f"  SCALE large={sum(scale_large)/len(scale_large):.4f}  "
          f"xl={sum(scale_xl)/len(scale_xl):.4f}  → {scale_file.name}")


def run_scale_eval(scored, out_dir, result_file):
    """기존 단일 파일 인터페이스 (backward compat)."""
    scorer_large, scorer_xl = _load_scale_scorers()
    _run_scale_eval(scored, out_dir, result_file, scorer_large, scorer_xl)


def evaluate_scale_only(args):
    """기존 judge 점수 jsonl에 SCALE만 추가 (SCALE 모델 1회 로드)."""
    score_files = args.score_files
    print(f"\n[SCALE-only] {len(score_files)}개 점수 파일에 SCALE 추가")

    scorer_large, scorer_xl = _load_scale_scorers()
    if scorer_large is None:
        return

    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    for i, sf in enumerate(score_files):
        sf = Path(sf)
        if not sf.exists():
            print(f"  [{i+1}/{len(score_files)}] 없음: {sf}")
            continue
        scored = [json.loads(l) for l in open(sf, encoding="utf-8") if l.strip()]
        if not scored:
            print(f"  [{i+1}/{len(score_files)}] 비어있음: {sf}")
            continue
        print(f"\n──── [{i+1}/{len(score_files)}] {sf.parent.name} ({len(scored)}건) ────")
        _run_scale_eval(scored, sf.parent, sf, scorer_large, scorer_xl, gold_df, vital_map)


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

    cols = [
        c
        for c in [
            "brevity_score",
            "critical_score",
            "sum_score",
            "scale_large",
            "scale_xl",
            "text_length",
        ]
        if c in df_a.columns
    ]

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
        wins = int((d > 0).sum())
        losses = int((d < 0).sum())
        rb = (wins - losses) / (wins + losses) if (wins + losses) > 0 else float("nan")

        p_sign = stats.binomtest(wins, wins + losses, p=0.5).pvalue

        sig_t = " !!!" if p_t < 0.05 else ""
        sig_w = " !!!" if p_w < 0.05 else ""
        sig_s = " !!!" if p_sign < 0.05 else ""

        print(f"\n── {col} ──")
        print(
            f"  Paired t  : t={t_stat:.3f}, p={p_t:.4g}, Δ={d.mean():.4f}, CI={ci}, dz={dz:.3f}{sig_t}"
        )
        print(f"  Wilcoxon  : W={w_stat}, p={p_w:.4g}, rank-biserial={rb:.3f}{sig_w}")
        print(f"  Sign test : wins={wins}, losses={losses}, p={p_sign:.4g}{sig_s}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge 평가")
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
        help="04_inference.py 출력 jsonl 파일 경로 (모델 A)",
    )
    parser.add_argument(
        "--result_files",
        nargs="+",
        default=None,
        help="여러 추론 결과를 judge 1회 로드로 일괄 평가. out_tag는 부모폴더명에서 자동 추출.",
    )
    parser.add_argument(
        "--result_file_b",
        type=str,
        default=None,
        help="쌍 비교 시 모델 B의 jsonl 경로. 지정 시 A/B logprob 방식으로 평가.",
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
        "--scale_only",
        action="store_true",
        help="기존 judge 점수 jsonl에 SCALE만 추가 (judge 재실행 안함). --score_files 지정 필요.",
    )
    parser.add_argument(
        "--score_files",
        nargs="+",
        default=None,
        help="--scale_only 모드용 점수 jsonl 파일들 (eval/<tag>/gold_results_scores.jsonl).",
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
    elif args.scale_only:
        if not args.score_files:
            parser.error("--scale_only 사용 시 --score_files 필수")
        evaluate_scale_only(args)
    elif args.result_files:
        evaluate_batch(args)
    elif args.result_file:
        evaluate(args)
    else:
        parser.error("--result_file / --result_files / --compare / --scale_only 중 하나를 지정하세요.")
