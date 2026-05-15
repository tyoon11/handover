"""
vital_summarizer.py — Rule-based Vital Sign Summarizer
진료서식구성원소명 파싱 ("HR <- HR <- 마취기록" → "HR") 포함
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from config import (
    VIT_SID_COL, VIT_TIME_COL, VIT_ITEM_COL, VIT_VAL_COL,
    VITAL_ITEMS, BP_ITEMS, IBP_ITEMS,
)

warnings.filterwarnings("ignore")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 소아 연령 그룹별 Threshold
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _age_group(age: float) -> str:
    if age < 1:   return "infant"
    elif age < 3: return "toddler"
    elif age < 6: return "preschool"
    elif age < 12: return "school"
    else:          return "adolescent"


HR_THRESH = {
    "infant":     (100, 160),
    "toddler":    (90,  150),
    "preschool":  (80,  140),
    "school":     (70,  130),
    "adolescent": (60,  110),
}

SBP_HIGH = {
    "infant":     100,
    "toddler":    104,
    "preschool":  108,
    "school":     116,
    "adolescent": 130,
}

DBP_THRESH = {
    "infant":     (30, 65),
    "toddler":    (35, 70),
    "preschool":  (38, 72),
    "school":     (40, 76),
    "adolescent": (45, 82),
}

def sbp_low(age):  return 70.0 if age < 1 else (70 + 2*age if age <= 10 else 90.0)
def mbp_low(age):  return 1.5 * age + 40.0
def qtc_thresh(age): return 450.0 if age < 8 else 460.0

SPO2_WARN = 95.0
SPO2_CRIT = 90.0
TEMP_LOW  = 35.5
TEMP_HIGH = 38.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파싱 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_item_name(raw: str) -> str:
    """'HR <- HR <- 마취기록' → 'HR'"""
    return raw.split(" <- ")[0].strip()


def _to_numeric(series: pd.Series) -> pd.Series:
    """object → numeric, 0값·비숫자 제거 (센서 미연결 / 오입력)."""
    s = pd.to_numeric(series, errors="coerce")
    return s.replace(0, np.nan)


def _load_sub(vital_df: pd.DataFrame, sid: int) -> pd.DataFrame:
    """수술 ID 필터 + 항목명 파싱 + 시간 파싱."""
    sub = vital_df[vital_df[VIT_SID_COL] == sid].copy()
    if sub.empty:
        return sub
    sub["_item"] = sub[VIT_ITEM_COL].apply(_parse_item_name)
    sub[VIT_TIME_COL] = pd.to_datetime(sub[VIT_TIME_COL], errors="coerce")
    return sub


def _series(sub: pd.DataFrame, item: str) -> pd.Series:
    """항목 이름 → 정제된 numeric 시계열."""
    rows = sub[sub["_item"] == item].sort_values(VIT_TIME_COL).copy()
    rows[VIT_VAL_COL] = _to_numeric(rows[VIT_VAL_COL])
    rows = rows.dropna(subset=[VIT_VAL_COL])
    return rows.set_index(VIT_TIME_COL)[VIT_VAL_COL]


def _stat(s: pd.Series, unit: str = "") -> str:
    if s.empty: return "데이터 없음"
    u = f" {unit}" if unit else ""
    return f"중앙값 {s.median():.0f}{u} (범위 {s.min():.0f}–{s.max():.0f}{u})"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 항목별 요약 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _hr(s, age):
    if s.empty: return ""
    g = _age_group(age)
    brady, tachy = HR_THRESH[g]
    evs = []
    nb = (s < brady).sum(); nt = (s > tachy).sum()
    if nb: evs.append(f"서맥(<{brady}) {nb}회")
    if nt: evs.append(f"빈맥(>{tachy}) {nt}회")
    ev = f", 이벤트: {', '.join(evs)}" if evs else ", 이상 없음"
    return f"HR: {_stat(s, 'bpm')}{ev}"


def _bp(item, s, age):
    if s.empty: return ""
    g = _age_group(age)
    evs = []
    if item in ("SBP", "ISBP1"):
        lo, hi = sbp_low(age), SBP_HIGH[g]
        nl = (s < lo).sum(); nh = (s > hi).sum()
        if nl: evs.append(f"저혈압(<{lo:.0f}) {nl}회")
        if nh: evs.append(f"고혈압(>{hi}) {nh}회")
    elif item in ("MBP", "IMBP1"):
        lo = mbp_low(age)
        nl = (s < lo).sum()
        if nl: evs.append(f"저혈압(<{lo:.0f}) {nl}회")
    elif item in ("DBP", "IDBP1"):
        lo, hi = DBP_THRESH[g]
        nl = (s < lo).sum(); nh = (s > hi).sum()
        if nl: evs.append(f"저(<{lo}) {nl}회")
        if nh: evs.append(f"고(>{hi}) {nh}회")
    ev = f", 이벤트: {', '.join(evs)}" if evs else ", 이상 없음"
    return f"{item}: {_stat(s, 'mmHg')}{ev}"


def _spo2(s):
    if s.empty: return ""
    evs = []
    nc = (s < SPO2_CRIT).sum(); nw = (s < SPO2_WARN).sum()
    if nc: evs.append(f"SpO2<90% {nc}회")
    elif nw: evs.append(f"SpO2<95% {nw}회")
    ev = f", 이벤트: {', '.join(evs)}" if evs else ", 이상 없음"
    return f"SpO2: {_stat(s, '%')}{ev}"


def _temp(s):
    if s.empty: return ""
    evs = []
    nl = (s < TEMP_LOW).sum(); nh = (s > TEMP_HIGH).sum()
    if nl: evs.append(f"저체온(<{TEMP_LOW}) {nl}회")
    if nh: evs.append(f"발열(>{TEMP_HIGH}) {nh}회")
    ev = f", 이벤트: {', '.join(evs)}" if evs else ", 이상 없음"
    return f"체온(T1): {_stat(s, '°C')}{ev}"


def _qtc(s, age):
    if s.empty: return ""
    thresh = qtc_thresh(age)
    nl = (s > thresh).sum()
    if nl:
        return f"QTc: 연장({nl}회 >{thresh:.0f}ms, 최대 {s.max():.0f}ms)"
    return f"QTc: 정상 범위 (최대 {s.max():.0f}ms)"


def _ebl(sub):
    rows = sub[sub["_item"] == "EBL"].copy()
    if rows.empty: return ""
    v = pd.to_numeric(rows[VIT_VAL_COL], errors="coerce").dropna()
    if v.empty: return ""
    return f"EBL: {v.max():.0f} mL"


def _uo(sub, weight_kg=None):
    rows = sub[sub["_item"] == "UO"].copy()
    if rows.empty: return ""
    raw = rows[VIT_VAL_COL].astype(str).str.strip()
    if raw.str.upper().str.contains(r"NF|NO.?FOLEY", regex=True).any():
        return "UO: No Foley"
    v = pd.to_numeric(rows[VIT_VAL_COL], errors="coerce").dropna()
    if v.empty: return ""
    total = v.max()
    parts = [f"UO: {total:.0f} mL"]
    if weight_kg and weight_kg > 0:
        n_min = len(rows)
        if n_min > 0:
            rate = total / (weight_kg * n_min / 60)
            parts.append(f"({rate:.1f} mL/kg/hr)")
    return " ".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 공개 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def summarize_vitals(
    vital_df: pd.DataFrame,
    sid: int,
    age_years: float,
    weight_kg: Optional[float] = None,
) -> str:
    """
    단일 수술에 대한 바이탈 요약 문자열 생성.

    Parameters
    ----------
    vital_df   : 전체 vital DataFrame
    sid        : 수술ID (int) — VIT_SID_COL 기준
    age_years  : 수술당시나이 (세)
    weight_kg  : 체중 (UO 속도 계산용, optional)
    """
    sub = _load_sub(vital_df, sid)
    if sub.empty:
        return "(바이탈 데이터 없음)"

    lines = []

    lines.append(_hr(_series(sub, "HR"), age_years))

    for item in BP_ITEMS:
        lines.append(_bp(item, _series(sub, item), age_years))

    # 침습혈압: 데이터 있는 경우만
    has_ibp = any(not _series(sub, i).empty for i in IBP_ITEMS)
    if has_ibp:
        for item in IBP_ITEMS:
            lines.append(_bp(item, _series(sub, item), age_years))

    lines.append(_spo2(_series(sub, "SpO2")))

    ppeak = _series(sub, "Ppeak")
    if not ppeak.empty:
        lines.append(f"Ppeak: {_stat(ppeak, 'cmH₂O')}")

    lines.append(_temp(_series(sub, "T1")))
    lines.append(_qtc(_series(sub, "QTc"), age_years))
    lines.append(_ebl(sub))
    lines.append(_uo(sub, weight_kg))

    return "\n".join(l for l in lines if l)


def build_vital_map(vital_df: pd.DataFrame, emr_df: pd.DataFrame) -> dict:
    """
    전체 수술 ID에 대한 {수술ID(int): vital_summary_str} dict 생성.

    EMR df의 '수술 ID' (공백 있음, level0 키) 기준으로 나이 매핑.
    """
    # 나이 매핑: EMR 수술ID → 수술당시나이
    try:
        sid_series  = emr_df["수술 ID"]
        age_series  = emr_df[("수술", "수술당시나이", "")]
        age_map = dict(zip(sid_series.astype(int), age_series.astype(float)))
    except Exception as e:
        print(f"[build_vital_map] 나이 매핑 실패: {e}, 기본값 5세 사용")
        age_map = {}

    vital_sids = vital_df[VIT_SID_COL].unique()
    vital_map  = {}

    for i, sid in enumerate(vital_sids):
        sid_int = int(sid)
        age     = age_map.get(sid_int, 5.0)
        vital_map[sid_int] = summarize_vitals(vital_df, sid_int, age)
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(vital_sids)}] 처리 중...")

    print(f"[build_vital_map] 완료: {len(vital_map)}건")
    return vital_map


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단독 실행 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    from config import VITAL_PKL
    print("Vital pkl 로드 중...")
    vital_df = pd.read_pickle(VITAL_PKL)
    print(f"  shape: {vital_df.shape}")

    # 샘플 수술ID 3개 테스트
    for sid in vital_df[VIT_SID_COL].unique()[:3]:
        print(f"\n{'='*50}")
        print(f"수술ID: {sid}")
        print(summarize_vitals(vital_df, int(sid), age_years=5.0))
