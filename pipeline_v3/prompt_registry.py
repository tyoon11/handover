"""
prompt_registry.py — 프롬프트 전체의 '한 벌' 목록 + 내용 지문 (v3.2 §4 동결 게이트)

왜 필요한가:
  프롬프트를 바꾸면 (a) gold_checklist 추출 결과, (b) judge calibration, (c) 생성 산출물이
  전부 무효가 된다. v3.1까지는 이 사실이 사람 기억에만 있어서, 프롬프트를 손본 뒤 옛 캐시를
  그대로 재사용하는 '조용히 틀린 결과'가 반복됐다. 지문을 산출물에 박아두고, 지문이 다르면
  재생성을 강제한다.

사용:
  python -m pipeline_v3.prompt_registry            # 지문 + 각 프롬프트 길이 출력
  python -m pipeline_v3.prompt_registry --dump DIR # 사람 검토용으로 전문 저장
  from .prompt_registry import fingerprint         # 산출물 메타에 기록
"""

import argparse
import hashlib
import json
from pathlib import Path

from .eval_v3 import checklist as CK
from .eval_v3 import metrics as M
from .prompt_utils import PROMPT_SPEC_VERSION, SYSTEM_PROMPT, build_user_prompt
from .prompts_pairgen import COVERAGE_RUBRIC, FIDELITY_BREVITY_RUBRIC, JUDGE_TMPL
from .required_categories import prompt_block

_PLACEHOLDER = "{EMR}"


def all_prompts() -> dict:
    """이름 → 프롬프트 전문. 여기 없는 프롬프트는 존재해선 안 된다 (한 벌 원칙)."""
    return {
        # 생성 (학습·추론·후보생성 공통)
        "gen.system": SYSTEM_PROMPT,
        "gen.user": build_user_prompt(_PLACEHOLDER, "{VITAL}"),
        # 선호쌍 judge
        "pairgen.judge_tmpl": JUDGE_TMPL,
        "pairgen.rubric_coverage": COVERAGE_RUBRIC,
        "pairgen.rubric_fidelity_brevity": FIDELITY_BREVITY_RUBRIC,
        # gold checklist 추출
        "checklist.system": CK._EXTRACT_SYSTEM,
        "checklist.tmpl": CK._EXTRACT_TMPL,
        # 평가 3축
        "eval.coverage_system": M._COVERAGE_SYSTEM,
        "eval.coverage_tmpl": M._COVERAGE_TMPL,
        "eval.faith_system": M._FAITH_SYSTEM,
        "eval.faith_tmpl": M._FAITH_TMPL,
        "eval.brevity_system": M._BREVITY_SYSTEM,
        "eval.brevity_tmpl": M._BREVITY_TMPL,
        # 필수 항목군 블록 (프롬프트에 삽입되는 실질 지시문)
        "categories.en": prompt_block(lang="en"),
    }


def fingerprint() -> dict:
    """{spec_version, sha1, per_prompt_sha1} — 산출물 메타·provenance에 기록한다."""
    prompts = all_prompts()
    per = {k: hashlib.sha1(v.encode("utf-8")).hexdigest()[:12]
           for k, v in sorted(prompts.items())}
    joined = "\n\x00".join(f"{k}\x01{prompts[k]}" for k in sorted(prompts))
    return dict(spec_version=PROMPT_SPEC_VERSION,
                sha1=hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16],
                per_prompt_sha1=per)


def assert_same(recorded: dict, stage: str = ""):
    """산출물에 박힌 지문과 현재 코드가 다르면 즉시 실패 (옛 캐시 재사용 차단)."""
    cur = fingerprint()
    if not recorded:
        return
    if recorded.get("sha1") != cur["sha1"]:
        changed = [k for k, v in cur["per_prompt_sha1"].items()
                   if recorded.get("per_prompt_sha1", {}).get(k) != v]
        raise RuntimeError(
            f"[프롬프트 불일치{(' · ' + stage) if stage else ''}] "
            f"산출물 {recorded.get('spec_version')}/{recorded.get('sha1')} vs "
            f"현재 {cur['spec_version']}/{cur['sha1']} — 바뀐 프롬프트: {changed or '(목록불가)'}. "
            "gold_checklist·calibration·생성 산출물을 재생성해야 한다 "
            "(scripts/invalidate_v3.py)."
        )


def main():
    ap = argparse.ArgumentParser(description="프롬프트 지문 확인 / 전문 덤프")
    ap.add_argument("--dump", type=str, default=None, help="프롬프트 전문을 저장할 디렉토리")
    args = ap.parse_args()

    fp = fingerprint()
    print(f"spec_version = {fp['spec_version']}")
    print(f"sha1         = {fp['sha1']}\n")
    prompts = all_prompts()
    for k in sorted(prompts):
        n_ko = sum(1 for ch in prompts[k] if '가' <= ch <= '힣')
        print(f"  {k:34s} {len(prompts[k]):6d}자  한글 {n_ko:4d}자  {fp['per_prompt_sha1'][k]}")

    if args.dump:
        d = Path(args.dump)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in prompts.items():
            (d / f"{k}.txt").write_text(v, encoding="utf-8")
        (d / "fingerprint.json").write_text(json.dumps(fp, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        print(f"\n덤프: {d}")


if __name__ == "__main__":
    main()
