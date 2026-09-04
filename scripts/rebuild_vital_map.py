#!/usr/bin/env python3
"""
rebuild_vital_map.py — vital_summary_map.pkl 재생성 (v3 경로 기준)

임계값을 고친 뒤 반드시 돌려야 한다. v1 `pipeline/00_preprocess.py` 의
`build_and_save_vital_map()` 은 **파일이 있으면 그냥 로드**하므로 새 임계값이
반영되지 않는다. v1은 legacy 봉인 상태라 여기서 v3 경로로 직접 만든다.

  python scripts/rebuild_vital_map.py --sample 3     # 샘플만 찍어보고 저장 안 함
  python scripts/rebuild_vital_map.py --apply
  python scripts/rebuild_vital_map.py --apply --force   # 기존 파일 덮어쓰기
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd      # noqa: E402

from pipeline_v3.config_v3 import (                      # noqa: E402
    EMR_PKL, VITAL_AUDIT_PKL, VITAL_MAP_PKL, VITAL_PKL,
)
from utils.vital_summarizer import (                      # noqa: E402
    DROPPED_ARTIFACTS, HDR_HANDOFF, HDR_REPORTABLE, NO_REPORTABLE, RULE_DIAG,
    build_vital_map, selftest,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="pkl 저장")
    ap.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    ap.add_argument("--sample", type=int, default=3,
                    help="저장 전 미리보기 케이스 수")
    ap.add_argument("--skip_selftest", action="store_true",
                    help="합성데이터 규칙 검증을 생략 (권장하지 않음)")
    args = ap.parse_args()

    # 규칙(R1~R4·baseline 게이트)이 의도대로 동작하는지 먼저 확인한다.
    # 2900건을 돌린 뒤 틀린 걸 발견하는 것보다 30초가 싸다.
    if not args.skip_selftest:
        print("=" * 70)
        print("[selftest] 합성데이터로 판정 규칙 검증")
        if not selftest(verbose=False):
            sys.exit("판정 규칙 selftest 실패 — 재생성을 중단한다")
        RULE_DIAG.clear()
        DROPPED_ARTIFACTS.clear()
        print("=" * 70)

    if VITAL_MAP_PKL.exists() and args.apply and not args.force:
        sys.exit(f"이미 존재: {VITAL_MAP_PKL}\n"
                 f"  새 임계값을 반영하려면 --force 를 주거나 "
                 f"scripts/invalidate_v3.py --apply --scope vital 로 먼저 지우세요.")

    for p in (VITAL_PKL, EMR_PKL):
        if not p.exists():
            sys.exit(f"입력 없음: {p}")

    print(f"[load] vital: {VITAL_PKL}")
    vital_df = pd.read_pickle(VITAL_PKL)
    print(f"       shape={vital_df.shape}")
    print(f"[load] emr  : {EMR_PKL}")
    emr_df = pd.read_pickle(EMR_PKL)
    print(f"       shape={emr_df.shape}")

    vital_map, audit_map = build_vital_map(vital_df, emr_df, with_audit=True)

    if args.sample:
        print(f"\n{'=' * 70}\n미리보기 {args.sample}건\n{'=' * 70}")
        for sid in list(vital_map)[:args.sample]:
            print(f"\n--- sid={sid} ---\n{vital_map[sid]}")

    # ── 자가진단 (조용한 누락 방지) ───────────────────────────────────────
    n = len(vital_map)
    n_rep = sum(1 for v in vital_map.values()
                if HDR_REPORTABLE in v and NO_REPORTABLE not in v)
    n_none = sum(1 for v in vital_map.values() if NO_REPORTABLE in v)
    n_ho = sum(1 for v in vital_map.values() if HDR_HANDOFF in v)
    n_dur = sum(1 for v in vital_map.values() if "최장" in v)
    n_iv = sum(1 for v in vital_map.values() if "개입:" in v)
    n_planned = sum(1 for v in vital_map.values() if "계획된 저체온" in v)
    ev_tot = RULE_DIAG.get("events_total", 0)
    ev_rep = RULE_DIAG.get("events_reportable", 0)

    print(f"\n[check] REPORTABLE 있음 {n_rep}/{n} · 없음 {n_none}/{n} · "
          f"AT HANDOFF 블록 {n_ho}/{n}")
    print(f"[check] 지속시간 표기 {n_dur}/{n} · 개입 연동(R1) 표기 {n_iv}/{n} · "
          f"계획된 저체온 {n_planned}/{n}")
    if ev_tot:
        print(f"[check] 이벤트 {ev_tot}개 중 REPORTABLE {ev_rep}개 "
              f"({ev_rep / ev_tot:.0%}) · 규칙별 "
              + " ".join(f"{k.replace('reason_', '')}={RULE_DIAG[k]}"
                         for k in sorted(RULE_DIAG) if k.startswith("reason_")))
    if DROPPED_ARTIFACTS:
        print(f"[check] 측정오류로 배제한 표본: {DROPPED_ARTIFACTS}")
        print("        (DHCA 저체온·청색성 SpO2 같은 '극단이지만 실제' 값은 보존됨)")
    else:
        print("[check] 측정오류 배제 0건")
    for k in ("ebl_no_weight", "uo_no_weight_or_time", "cases_no_vital",
              "cases_without_intervention"):
        if RULE_DIAG.get(k):
            print(f"[check] {k}: {RULE_DIAG[k]}건")

    if not n_rep:
        print("  ⚠ REPORTABLE 이벤트가 한 건도 없다 — 규칙이 과하게 막고 있는지 확인")
    if not n_iv:
        print("  ⚠ 개입 연동(R1) 표기가 0건 — 마취기록 매핑/키워드를 확인 "
              "(utils/anesthetic_record.py)")
    if ev_tot and ev_rep / ev_tot > 0.7:
        print(f"  ⚠ 이벤트의 {ev_rep / ev_tot:.0%}가 REPORTABLE — v3.1 수준의 과대등재일 수 있다")

    if not args.apply:
        print("\n[dry-run] 저장 안 함. 저장하려면 --apply")
        return
    VITAL_MAP_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(VITAL_MAP_PKL, "wb") as f:
        pickle.dump(vital_map, f)
    print(f"\n[save] {VITAL_MAP_PKL} ({len(vital_map)}건)")
    with open(VITAL_AUDIT_PKL, "wb") as f:
        pickle.dump(audit_map, f)
    print(f"[save] {VITAL_AUDIT_PKL} (이벤트 감사본 — "
          f"scripts/vital_flag_audit.py 가 읽는다)")


if __name__ == "__main__":
    main()
