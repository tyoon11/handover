"""
make_fewshot_bank.py — held-out few-shot 예시 뱅크 생성 (T1)

v1의 치명 결함(few-shot을 평가셋 22케이스에서 추출 → 유출)을 대체한다:
  - 예시 케이스 = RLHF held-out 150 중 few-shot pool 40건 (gold와 분리 assert됨)
  - 예시 인계문 = 강한 로컬 모델(기본 gemma4_31b)이 zero-shot 작성한 초안.
    gold가 아니라 '스타일 앵커'다. JSON으로 저장되므로 전문의가 직접 수정 가능.
  - 생성 붕괴(clean_v3 != ok) 예시는 뱅크에서 제외 (쓰레기 스타일 전파 방지, T3)

실행:
  python -m pipeline_v3.make_fewshot_bank --gpus 0,1,2,3
출력:
  {DATA_DIR}/preprocessed/fewshot_bank_v3.json
  [{"sid": ..., "emr": ..., "vital": ..., "handoff": ..., "source": "model_draft"}, ...]
"""

import argparse
import json
import os
from pathlib import Path


def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    g = p.parse_known_args()[0].gpus
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = g
        print(f"[GPU] CUDA_VISIBLE_DEVICES={g}")


_early_gpus()

from .config_v3 import DATA_DIR, model_path                          # noqa: E402
from .data_splits import load_splits                                 # noqa: E402
from .eval_v3.cleaning import clean_v3                                # noqa: E402
from .prompt_utils import build_emr_text, get_sid                     # noqa: E402

FEWSHOT_BANK_JSON = DATA_DIR / "preprocessed" / "fewshot_bank_v3.json"


def load_fewshot_bank(path=FEWSHOT_BANK_JSON) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--writer", type=str, default="gemma4_31b",
                    help="예시 인계문을 작성할 로컬 모델 키")
    ap.add_argument("--backend", type=str, default="auto")
    args = ap.parse_args()

    splits = load_splits(need=("gold", "fewshot"))
    fewshot_df = splits["fewshot"]
    vital_map = splits["vital_map"]

    from .eval_v3.engine import JudgeEngine
    engine = JudgeEngine(model_path(args.writer), backend=args.backend,
                         gen_cfg=dict(max_new_tokens=256, temperature=0.0))

    from .prompt_utils import SYSTEM_PROMPT, build_user_prompt
    cases, prompts = [], []
    for i in range(len(fewshot_df)):
        row = fewshot_df.iloc[i]
        sid = get_sid(row)
        emr = build_emr_text(row)
        vital = vital_map.get(sid, "")
        # engine.chat은 user 문자열을 받으므로 user 프롬프트만 예산 내로 구성
        user = build_user_prompt(emr, vital)
        cases.append(dict(sid=sid, emr=emr, vital=vital))
        prompts.append(user)

    outs = engine.chat(prompts, system=SYSTEM_PROMPT)
    engine.unload()

    bank, dropped = [], 0
    for case, out in zip(cases, outs):
        text, status, _flags = clean_v3(out)
        if status != "ok":
            dropped += 1
            continue
        bank.append(dict(sid=case["sid"], emr=case["emr"], vital=case["vital"],
                         handoff=text, source=f"model_draft:{args.writer}"))

    FEWSHOT_BANK_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = FEWSHOT_BANK_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(FEWSHOT_BANK_JSON)
    print(f"[fewshot_bank] {len(bank)}건 저장 (붕괴 제외 {dropped}건) → {FEWSHOT_BANK_JSON}")
    print("  ※ 예시 인계문은 모델 초안('스타일 앵커')이다 — 전문의 수정 가능, gold 아님.")


if __name__ == "__main__":
    main()
