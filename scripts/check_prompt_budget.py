#!/usr/bin/env python3
"""
check_prompt_budget.py — 프롬프트 토큰 예산 사전 점검

gen_pairs 는 few-shot 까지 붙여야 예산 초과가 드러나므로, 몇 시간 돌린 뒤
PromptTruncationError 로 죽는 일이 생긴다. 실제 tokenizer 로 미리 재본다.

  python scripts/check_prompt_budget.py --models llama qwen35 gemma4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_v3.config_v3 import (      # noqa: E402
    MAX_PROMPT_TOKENS, PAIRGEN, model_path,
)
from pipeline_v3.gen_pairs import fewshot_block_for_row      # noqa: E402
from pipeline_v3.make_fewshot_bank import load_fewshot_bank  # noqa: E402
from pipeline_v3.prompt_utils import (                       # noqa: E402
    SYSTEM_PROMPT, apply_chat_template, build_user_prompt, n_tokens,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama", "qwen35", "gemma4"])
    args = ap.parse_args()

    bank = load_fewshot_bank()
    if not bank:
        sys.exit("fewshot bank 없음 — 먼저 make_fewshot_bank 실행")
    print(f"fewshot bank {len(bank)}건 · n_fewshot={PAIRGEN['n_fewshot']} · "
          f"예시 EMR {PAIRGEN['fewshot_emr_chars']}자 · 예산 {MAX_PROMPT_TOKENS}\n")

    from transformers import AutoTokenizer
    bad = False
    for key in args.models:
        tok = AutoTokenizer.from_pretrained(str(model_path(key)), trust_remote_code=True)
        # EMR 을 완전히 비운 최악의 경우 = 지시문+few-shot 만의 비용
        worst = 0
        for row_idx in range(0, 200, 7):
            fs = fewshot_block_for_row(bank, row_idx, PAIRGEN["n_fewshot"])
            user = fs + "\n\n" + build_user_prompt("", "")
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
            worst = max(worst, n_tokens(tok, apply_chat_template(tok, msgs)))
        room = MAX_PROMPT_TOKENS - worst
        mark = "OK " if room > 300 else ("빠듯" if room > 0 else "초과")
        if room <= 0:
            bad = True
        print(f"  [{mark}] {key:8s} 고정비용 {worst:5d} tok · EMR 여유 {room:5d} tok")

    print()
    if bad:
        print("→ 초과. PAIRGEN['fewshot_emr_chars'] 또는 n_fewshot 을 줄이거나 "
              "SYSTEM_PROMPT 를 압축하세요.")
        sys.exit(1)
    print("→ 통과. gen_pairs 진행 가능.")


if __name__ == "__main__":
    main()
