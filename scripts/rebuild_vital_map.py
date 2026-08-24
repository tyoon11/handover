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

from pipeline_v3.config_v3 import EMR_PKL, VITAL_MAP_PKL, VITAL_PKL   # noqa: E402
from utils.vital_summarizer import (                      # noqa: E402
    DROPPED_ARTIFACTS, build_vital_map, summarize_vitals,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="pkl 저장")
    ap.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    ap.add_argument("--sample", type=int, default=3,
                    help="저장 전 미리보기 케이스 수")
    args = ap.parse_args()

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

    vital_map = build_vital_map(vital_df, emr_df)

    if args.sample:
        print(f"\n{'=' * 70}\n미리보기 {args.sample}건\n{'=' * 70}")
        for sid in list(vital_map)[:args.sample]:
            print(f"\n--- sid={sid} ---\n{vital_map[sid]}")

    # 새 임계값이 실제로 반영됐는지 지표 — ⚑(유의) 이벤트가 하나도 없으면 의심
    n_crit = sum(1 for v in vital_map.values() if "⚑" in v)
    n_dur = sum(1 for v in vital_map.values() if "분," in v or "시간," in v)
    print(f"\n[check] ⚑ 유의 이벤트 포함 {n_crit}/{len(vital_map)}건 · "
          f"지속시간 표기 포함 {n_dur}/{len(vital_map)}건")
    if DROPPED_ARTIFACTS:
        print(f"[check] 측정오류로 배제한 표본: {DROPPED_ARTIFACTS}")
        print("        (DHCA 저체온·청색성 SpO2 같은 '극단이지만 실제' 값은 보존됨)")
    else:
        print("[check] 측정오류 배제 0건")
    if not n_crit:
        print("  ⚠ ⚑ 이벤트가 0건 — 임계값 반영 여부를 확인하세요")

    if not args.apply:
        print("\n[dry-run] 저장 안 함. 저장하려면 --apply")
        return
    VITAL_MAP_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(VITAL_MAP_PKL, "wb") as f:
        pickle.dump(vital_map, f)
    print(f"\n[save] {VITAL_MAP_PKL} ({len(vital_map)}건)")


if __name__ == "__main__":
    main()
