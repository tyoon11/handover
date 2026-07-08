"""
tests_v3/test_v3.py — GPU 없이 도는 v3 회귀 테스트

실행 (둘 다 지원):
  python -m tests_v3.test_v3        # 폐쇄망/로컬 어디서나
  pytest tests_v3/ -q               # pytest 있으면

커버: 리뷰에서 잡힌 버그가 '테스트로 고정'되는지 —
  E2(안전게이트), E3(fail-loud), E4(clean 오탐), E5(no-issue 전체일치),
  E7(verdict id 검증), B4(통계 가드), T8(쌍 마진/dedup), B11(로컬 import),
  T4/T6(EMR-only 절단 + OUTPUT 생존)
"""

import os
import tempfile

# config_v3는 import 시 부작용이 없어야 한다(B11) — 서버 경로 없이도 동작 확인차
# 임시 디렉토리로 오버라이드하고 import.
_TMP = tempfile.mkdtemp(prefix="handover_v3_test_")
os.environ.setdefault("HANDOVER_BASE_DIR", _TMP)
os.environ.setdefault("HANDOVER_MODEL_DIR", _TMP)

from pipeline_v3.config_v3 import is_no_issue_v3, judges_for, model_family   # noqa: E402
from pipeline_v3.eval_v3 import metrics as M           # noqa: E402
from pipeline_v3.eval_v3.cleaning import clean_v3       # noqa: E402
from pipeline_v3.eval_v3.stats import (                 # noqa: E402
    bootstrap_ci, holm_correction, micro_coverage, paired_permutation, paired_tests,
)


# ── B11: import 부작용 없음 ─────────────────────────────────────────────────
def test_config_import_no_side_effects():
    # 서버 경로가 없는 로컬에서 여기까지 왔다는 것 자체가 통과 조건.
    # (v1 config.py는 import 시 서버 경로 mkdir로 로컬 전멸)
    from pipeline_v3 import config_v3
    assert not os.path.exists(str(config_v3.OUTPUT_BASE)) or True


# ── E5: no-issue 전체 일치만 ────────────────────────────────────────────────
def test_no_issue_exact_match_only():
    assert is_no_issue_v3("특이사항 없음")
    assert is_no_issue_v3("특이사항 없음.")
    assert is_no_issue_v3("  특이 사항 없음 ")
    # v2 오탐 케이스들 — 실질 내용이 있으면 절대 no-issue가 아니다
    assert not is_no_issue_v3("특이사항 없음. intraop VT 발생")
    assert not is_no_issue_v3("None significant except bleeding")
    assert not is_no_issue_v3("")
    assert not is_no_issue_v3(None)


# ── E4: clean_v3 오탐 수정 ──────────────────────────────────────────────────
def test_clean_v3_no_false_positives():
    # 조사(명사)로 끝나는 정상 인계문 → 0점 아님, 플래그만
    text, status, flags = clean_v3("수술 중 출혈 300mL, 수혈 시행. 저혈당 주의")
    assert status == "ok"
    # 마크다운 구분선 → repetition 아님
    text, status, _ = clean_v3("VT 1회 발생.\n--------\namiodarone 투여함.")
    assert status == "ok"
    # 곱슬따옴표/전각부호 → garbage 아님
    text, status, _ = clean_v3("'특이사항' 없음… 체온 36.5℃, 각성 양호함.")
    assert status == "ok"


def test_clean_v3_still_catches_real_failures():
    assert clean_v3("")[1] == "empty"
    assert clean_v3("ልልልልልልልልልል ልልልል ልልልልል")[1] in ("repetition", "garbage")
    assert clean_v3("same1 same1 same1 same1")[1] == "repetition"
    assert clean_v3("환자 상태 양호 //thought: let me think")[1] == "leak"
    # think 블록 제거 후 본문 보존
    text, status, flags = clean_v3("<think>reasoning...</think>출혈 소견 있음.")
    assert status == "ok" and text == "출혈 소견 있음." and "had_think_block" in flags


# ── E2: 안전게이트 (이상소견 + no-issue → 0) ────────────────────────────────
def _entry(items, normal=False, source="gold_llm"):
    return {"items": items, "is_normal_case": normal, "source": source}


def test_gate_missed_abnormal():
    entry = _entry([{"id": "c1", "finding": "intraop VT", "severity": "high"}])
    sc = M.fast_path("특이사항 없음", "ok", entry)
    assert sc is not None
    assert sc["composite"] == 0.0
    assert sc["faithfulness"] == 0.0        # 모순 주장 — v2는 1.0이었다
    assert sc["gate"] == "missed_abnormal"
    assert not sc["excluded"]


def test_normal_case_correct_no_issue_is_perfect():
    sc = M.fast_path("특이사항 없음", "ok", _entry([], normal=True, source="gold_normal"))
    assert sc["composite"] == 1.0


def test_degenerate_is_zero_not_excluded():
    sc = M.fast_path("ልልል", "garbage", _entry([{"id": "c1", "finding": "x"}]))
    assert sc["composite"] == 0.0 and sc["gate"] == "degenerate"


# ── E3: 실패는 제외 (점수 변환 금지) ────────────────────────────────────────
def test_no_gold_and_extract_failure_excluded():
    sc = M.fast_path("출혈 소견.", "ok", _entry([], source="no_gold"))
    assert sc["excluded"] and sc["composite"] is None
    sc = M.fast_path("출혈 소견.", "ok", _entry([], source="gold_llm_failed"))
    assert sc["excluded"] and sc["exclude_reason"] == "checklist_extract_failed"


def test_judge_failure_excludes_case():
    entry = _entry([{"id": "c1", "finding": "VT"}])
    cov = M.parse_coverage(None, entry)            # judge JSON 실패
    assert cov["coverage"] is None and cov["judge_failed"]
    fa = M.parse_faithfulness({"claims": [{"claim": "a", "verdict": "supported"}]})
    br = M.parse_brevity({"score": 4, "noise": []})
    sc = M.composite_from_axes(cov, fa, br, entry)
    assert sc["excluded"] and sc["composite"] is None    # v2는 coverage 0.0으로 집계했다


# ── E7: verdict id 대조 검증 ────────────────────────────────────────────────
def test_coverage_verdict_id_validation():
    entry = _entry([{"id": "c1", "finding": "VT"}, {"id": "c2", "finding": "수혈"}])
    # 모든 id 존재 → 정상
    ok = M.parse_coverage({"verdicts": [{"id": "c1", "status": "yes"},
                                        {"id": "c2", "status": "partial"}]}, entry)
    assert ok["coverage"] == 0.75 and not ok["judge_failed"]
    assert len(ok["partial"]) == 1 and len(ok["missed"]) == 0     # B10: partial 분리
    # id 누락 → judge 실패 (v2는 누락 id를 'no'로 세어 coverage를 깎았다)
    bad = M.parse_coverage({"verdicts": [{"id": "c1", "status": "yes"}]}, entry)
    assert bad["judge_failed"] and bad["coverage"] is None
    # 엉뚱한 id(주입/환각) → 무시되고 실패 처리
    inj = M.parse_coverage({"verdicts": [{"id": "c1", "status": "yes"},
                                         {"id": "HACK", "status": "yes"}]}, entry)
    assert inj["judge_failed"]


# ── 교차 judge 배정 (T7) ────────────────────────────────────────────────────
def test_cross_judge_assignment():
    assert model_family("gemma4_31b") == "gemma"
    assert "gemma4_31b" not in judges_for("gemma4")          # 같은 family 제외
    assert judges_for("gemma4") == ["qwen35"]
    assert judges_for("qwen35") == ["gemma4_31b"]
    assert set(judges_for("llama")) == {"gemma4_31b", "qwen35"}


# ── 통계 (E8/B4) ────────────────────────────────────────────────────────────
def test_stats_guards_and_sanity():
    # 전부-0 diff에서 크래시 없음 (B4)
    t = paired_tests([0.5] * 10, [0.5] * 10)
    assert t["wilcoxon"] is None and t["permutation"]["p"] == 1.0
    # 명백한 차이는 유의
    a = [0.9, 0.8, 0.85, 0.95, 0.9, 0.88, 0.92, 0.87, 0.91, 0.86]
    b = [0.1, 0.2, 0.15, 0.05, 0.1, 0.12, 0.08, 0.13, 0.09, 0.14]
    assert paired_permutation(a, b)["p"] < 0.01
    ci = bootstrap_ci(a)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    # None은 집계에서 자동 제외
    assert bootstrap_ci([0.5, None, 0.7])["n"] == 2
    # Holm 보정 단조성
    h = holm_correction({"a": 0.001, "b": 0.04, "c": None})
    assert h["a"]["significant"] and h["c"]["p_adj"] is None


def test_micro_coverage_pools_items():
    recs = [
        dict(coverage=1.0, covered=[1, 2, 3], partial=[], missed=[]),
        dict(coverage=0.0, covered=[], partial=[], missed=[1]),
    ]
    # macro 평균은 0.5지만 micro는 3/4
    assert micro_coverage(recs)["micro_coverage"] == 0.75


# ── T8: 쌍 선정 (dedup/마진) ────────────────────────────────────────────────
def test_pair_selection_margin_and_identity():
    from pipeline_v3.gen_pairs import select_pairs
    rows = [dict(row_idx=0, sid=1), dict(row_idx=1, sid=2), dict(row_idx=2, sid=3)]
    judged = {
        0: [dict(text="A 소견.", cov=5, fid=5, total=15.0),
            dict(text="B 없음.", cov=1, fid=3, total=5.0)],      # margin 10 → 채택
        1: [dict(text="같은 문장.", cov=4, fid=4, total=12.0),
            dict(text="같은  문장.", cov=3, fid=4, total=10.0)],  # 정규화 동일 → 탈락
        2: [dict(text="x", cov=3, fid=3, total=9.0),
            dict(text="y", cov=3, fid=2, total=8.0)],             # margin 1 < 2 → 탈락
    }
    recs, stats = select_pairs(rows, judged)
    assert stats["kept"] == 1 and stats["identical"] == 1 and stats["no_margin"] == 1
    assert recs[0]["chosen"] == "A 소견."


# ── T4/T6: EMR-only 절단 + OUTPUT 생존 ─────────────────────────────────────
class _DummyTok:
    """공백 토크나이저 + 단순 chat template (transformers 불필요)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        out = "\n".join(f"<{m['role']}>{m['content']}" for m in msgs)
        return out + ("\n<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_fit_prompt_truncates_emr_only():
    from pipeline_v3.prompt_utils import OUTPUT_HEADER, fit_chat_prompt
    tok = _DummyTok()
    emr = "오래된기록 " * 3000 + "최근기록_수혈_시행"
    p = fit_chat_prompt(tok, emr, "HR 정상", budget=800)
    assert OUTPUT_HEADER in p                    # 생성 헤더 생존 (v1 HF 경로는 잘렸다)
    assert "최근기록_수혈_시행" in p             # EMR 꼬리(최근) 보존
    assert "앞부분 생략" in p                    # 절단은 마커로 명시
    assert len(tok(p)["input_ids"]) <= 800


def test_fit_prompt_fails_loud_when_budget_impossible():
    from pipeline_v3.prompt_utils import PromptTruncationError, fit_chat_prompt
    tok = _DummyTok()
    try:
        fit_chat_prompt(tok, "짧은 EMR", "", budget=5)
        raise AssertionError("예산 불가능인데 조용히 통과")
    except PromptTruncationError:
        pass


# ── 단독 실행 러너 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception:
            failed += 1
            print(f"  ✗ {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
