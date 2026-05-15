"""
00_preprocess.py — 전처리 스크립트

역할:
  1. 기존 split pkl 확인 (jsft/selfjudge/rlhf/gold는 이미 존재)
  2. Vital summary map 생성 및 저장
  3. EMR ↔ Vital 수술ID 매칭 검증

실행: python 00_preprocess.py
"""

import pickle
import sys
import pandas as pd

from config import (
    EMR_PKL, VITAL_PKL,
    GOLD_PKL, GOLD_REF_PKL,
    SFT_PKL, SELFJUDGE_PKL, RLHF_PKL, SYNTH_PKL,
    VITAL_MAP_PKL,
    VIT_SID_COL,
)
from vital_summarizer import build_vital_map


def check_pkl_exists():
    """이미 존재하는 pkl 파일 확인."""
    pkls = {
        "EMR (raw)":       EMR_PKL,
        "Vital":           VITAL_PKL,
        "Gold (228col)":   GOLD_PKL,
        "Gold (8col ref)": GOLD_REF_PKL,
        "SFT":             SFT_PKL,
        "SelfJudge":       SELFJUDGE_PKL,
        "RLHF":            RLHF_PKL,
        "Synth+Score":     SYNTH_PKL,
    }
    print("=" * 60)
    print("[파일 존재 확인]")
    all_ok = True
    for name, path in pkls.items():
        exists = path.exists()
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status}  {name}: {path.name}")
        if not exists:
            all_ok = False
    print()
    return all_ok


def verify_sid_match(emr_df: pd.DataFrame, vital_df: pd.DataFrame):
    """EMR '수술 ID'와 Vital '수술ID' 매칭 현황 출력."""
    emr_sids   = set(emr_df["수술 ID"].astype(int))
    vital_sids = set(vital_df[VIT_SID_COL].astype(int))

    both    = emr_sids & vital_sids
    emr_only  = emr_sids  - vital_sids
    vital_only = vital_sids - emr_sids

    print("[수술ID 매칭 현황]")
    print(f"  EMR 전체:          {len(emr_sids):,}건")
    print(f"  Vital 전체:        {len(vital_sids):,}건")
    print(f"  양쪽 모두 존재:    {len(both):,}건  ← vital summary 생성 대상")
    print(f"  EMR에만 존재:      {len(emr_only):,}건  (vital 없음, summary 빈칸)")
    print(f"  Vital에만 존재:    {len(vital_only):,}건  (EMR 없음, 무시)")
    print()
    return both


def build_and_save_vital_map(vital_df: pd.DataFrame, emr_df: pd.DataFrame):
    """Vital summary map 생성 → VITAL_MAP_PKL 저장."""
    if VITAL_MAP_PKL.exists():
        print(f"[SKIP] Vital map 이미 존재: {VITAL_MAP_PKL.name}")
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)
        print(f"  로드: {len(vital_map)}건")
        return vital_map

    print("[Vital summary map 생성 중...]")
    vital_map = build_vital_map(vital_df, emr_df)

    VITAL_MAP_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(VITAL_MAP_PKL, "wb") as f:
        pickle.dump(vital_map, f)
    print(f"  저장 완료: {VITAL_MAP_PKL}")
    return vital_map


def sample_check(vital_map: dict, emr_df: pd.DataFrame, n: int = 3):
    """샘플 수술 vital summary 출력."""
    sids = list(vital_map.keys())[:n]
    print(f"\n[샘플 Vital Summary ({n}건)]")
    for sid in sids:
        # EMR에서 수술명 찾기
        mask = emr_df["수술 ID"].astype(int) == sid
        if mask.any():
            try:
                opname = emr_df.loc[mask, ("수술", "수술명", "")].values[0]
                age    = emr_df.loc[mask, ("수술", "수술당시나이", "")].values[0]
                dept   = emr_df.loc[mask, ("수술", "수술진료과", "")].values[0] \
                         if ("수술", "수술진료과", "") in emr_df.columns else "N/A"
            except Exception:
                opname = age = dept = "N/A"
        else:
            opname = age = dept = "(EMR 없음)"

        print(f"\n  수술ID: {sid}  |  {dept}  |  {opname}  |  {age}세")
        print("  " + "\n  ".join(vital_map[sid].split("\n")))


def main():
    # 1. 파일 존재 확인
    all_ok = check_pkl_exists()
    if not all_ok:
        print("[WARNING] 일부 필수 파일 없음. 경로를 확인하세요.")

    # 2. EMR / Vital 로드
    print("EMR pkl 로드 중...")
    emr_df = pd.read_pickle(EMR_PKL)
    print(f"  shape: {emr_df.shape}")

    print("Vital pkl 로드 중...")
    vital_df = pd.read_pickle(VITAL_PKL)
    print(f"  shape: {vital_df.shape}")
    print()

    # 3. 수술ID 매칭 검증
    verify_sid_match(emr_df, vital_df)

    # 4. Vital summary map 생성/저장
    vital_map = build_and_save_vital_map(vital_df, emr_df)

    # 5. 샘플 출력
    sample_check(vital_map, emr_df)

    # 6. SFT/RLAIF 데이터 요약 출력
    print("\n[기존 Split 데이터 요약]")
    for name, path in [
        ("SFT",      SFT_PKL),
        ("SelfJudge",SELFJUDGE_PKL),
        ("RLHF",     RLHF_PKL),
        ("Gold",     GOLD_PKL),
        ("Synth",    SYNTH_PKL),
    ]:
        if path.exists():
            df = pd.read_pickle(path)
            print(f"  {name}: {df.shape}")
        else:
            print(f"  {name}: MISSING")

    print("\n[완료] 00_preprocess.py 실행 완료.")
    print(f"  Vital map: {VITAL_MAP_PKL}")
    print("  다음 단계: python 02_sft_train.py")


if __name__ == "__main__":
    main()
