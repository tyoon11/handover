"""
vital_summarizer.py — Rule-based Vital Sign Summarizer
진료서식구성원소명 파싱 ("HR <- HR <- 마취기록" → "HR") 포함

임계값은 전부 [utils/vital_thresholds.py](vital_thresholds.py) 한 곳에서 온다
(Smith's Anesthesia 2021 / Miller's Anesthesia 2024 근거). 규칙 문서는
[VITAL_SUMMARY_RULES.md](../VITAL_SUMMARY_RULES.md).

이벤트는 2-tier로 표기한다:
  - "정상범위 이탈" — 연령별 참조범위 밖 (통계적 이상)
  - "⚑" 접두 — 소생·개입 기준 초과 (임상적으로 유의, 인계문에 반드시 포함)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from config import (
    VIT_SID_COL,
    VIT_TIME_COL,
    VIT_ITEM_COL,
    VIT_VAL_COL,
    VITAL_ITEMS,
    BP_ITEMS,
    IBP_ITEMS,
)

from utils.vital_thresholds import (      # noqa: E402
    AGE_GROUP_LABELS,
    implausible_mask,
    DBP_HAS_LOWER_THRESHOLD,
    EBL_MASSIVE_PCT,
    EBL_SIGNIFICANT_PCT,
    SPO2_CRIT,
    SPO2_TARGET_LOW,
    TEMP_FEVER,
    TEMP_HYPOTHERMIA,
    TEMP_SAFE_HIGH,
    TEMP_SAFE_LOW,
    UO_OLIGURIA,
    QTC_PROLONGED,
    age_group,
    dbp_hypertension,
    ebl_percent_ebv,
    hr_critical_range,
    hr_normal_range,
    map_hypotension,
    qtc_upper_normal,
    sbp_hypertension,
    sbp_hypotension,
)

warnings.filterwarnings("ignore")

CRIT = "⚑"      # 임상적으로 유의한 이벤트 표시


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


DROPPED_ARTIFACTS = {}      # {item: n} — 마지막 summarize 호출의 배제 건수 (진단용)


def _series(sub: pd.DataFrame, item: str) -> pd.Series:
    """항목 이름 → 정제된 numeric 시계열 (측정 오류 배제 포함).

    센서 이탈로만 설명되는 값을 버린다. 임상적으로 극단적이지만 가능한 값
    (DHCA 저체온 18–20°C, 청색성 심질환 SpO2 40–70%)은 **보존**한다 —
    그게 바로 인계해야 할 소견이다. 배제 건수는 DROPPED_ARTIFACTS에 남긴다.
    """
    rows = sub[sub["_item"] == item].sort_values(VIT_TIME_COL).copy()
    rows[VIT_VAL_COL] = _to_numeric(rows[VIT_VAL_COL])
    rows = rows.dropna(subset=[VIT_VAL_COL])
    if not rows.empty:
        bad = implausible_mask(item, rows[VIT_VAL_COL])
        n_bad = int(bad.sum())
        if n_bad:
            DROPPED_ARTIFACTS[item] = DROPPED_ARTIFACTS.get(item, 0) + n_bad
            rows = rows[~bad]
    return rows.set_index(VIT_TIME_COL)[VIT_VAL_COL]


def _stat(s: pd.Series, unit: str = "", digits: int = 0) -> str:
    """digits: 체온처럼 0.1 단위가 임계값 판정을 가르는 항목만 1을 쓴다."""
    if s.empty:
        return "데이터 없음"
    u = f" {unit}" if unit else ""
    f = f".{digits}f"
    return (f"중앙값 {s.median():{f}}{u} "
            f"(범위 {s.min():{f}}–{s.max():{f}}{u})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 항목별 요약 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _events(evs) -> str:
    """이벤트 목록 → 표기 문자열. 유의(⚑) 이벤트를 앞으로 정렬."""
    if not evs:
        return ", 이상 없음"
    evs = sorted(evs, key=lambda e: not e.startswith(CRIT))
    return f", 이벤트: {' | '.join(evs)}"


DEFAULT_STEP_MIN = 5.0      # 마취기록 표본 간격 기본값 (간격 추정 불가 시)


def _step_min(s: pd.Series) -> float:
    """표본 간격(분) 중앙값. 지속시간 환산의 단위."""
    idx = pd.DatetimeIndex(s.index).dropna()
    if len(idx) < 2:
        return DEFAULT_STEP_MIN
    d = pd.Series(idx).diff().dropna().dt.total_seconds() / 60.0
    d = d[d > 0]
    return float(d.median()) if len(d) else DEFAULT_STEP_MIN


def _fmt_min(m: float) -> str:
    """분 → 사람이 읽는 문자열."""
    if m < 1:
        return "<1분"
    if m < 60:
        return f"{m:.0f}분"
    return f"{m / 60:.1f}시간"


def _ev(label: str, s: pd.Series, mask, thresh: float, *, low: bool,
        crit: bool = False, digits: int = 0, unit: str = "") -> str:
    """임계값 위반 이벤트를 '횟수 / 지속시간 / 기준 대비 편차'로 표기.

    지속시간은 위반 표본 1건 = 표본 간격 1구간으로 환산한 **추정치**다
    ([M] Ch.4가 술중저혈압을 'time or fraction of the case below threshold'로
    정량화하라고 권하는 방식). 연속 구간(episode)이 2개 이상이면 최장 구간도 적는다.

    출력 예: `⚑저혈압(<70): 3회 15분(최장 10분), 최저 52 (기준-18)`
    """
    m = pd.Series(mask, index=s.index).fillna(False).astype(bool)
    n = int(m.sum())
    if not n:
        return ""

    f = f".{digits}f"
    step = _step_min(s)
    # 연속 위반 구간(episode)별 지속시간
    grp = (m != m.shift()).cumsum()[m]
    durs = []
    for _, run in m[m].groupby(grp):
        t = pd.DatetimeIndex(run.index).dropna()
        span = ((t.max() - t.min()).total_seconds() / 60.0) if len(t) >= 2 else 0.0
        durs.append(span + step)
    total = sum(durs)

    vals = s[m]
    if low:
        ext = vals.min()
        delta = thresh - ext
        ext_lab = f"최저 {ext:{f}}{unit} (기준-{delta:{f}})"
    else:
        ext = vals.max()
        delta = ext - thresh
        ext_lab = f"최고 {ext:{f}}{unit} (기준+{delta:{f}})"

    dur = _fmt_min(total)
    if len(durs) > 1:
        dur += f"(최장 {_fmt_min(max(durs))})"
    mark = CRIT if crit else ""
    return f"{mark}{label}: {n}회 {dur}, {ext_lab}"


def _hr(s, age):
    """[S] Table 18.1 정상범위 / Table 57.3(PALS) 유의 기준 — 2-tier."""
    if s.empty:
        return ""
    lo, hi = hr_normal_range(age)
    brady, tachy = hr_critical_range(age)
    evs = [
        _ev(f"서맥(<{brady:.0f})", s, s < brady, brady, low=True, crit=True),
        _ev(f"빈맥(>{tachy:.0f})", s, s > tachy, tachy, low=False, crit=True),
        # 정상범위 이탈은 유의 기준을 넘지 않은 것만 (이중계수 방지)
        _ev(f"정상하한 미만(<{lo:.0f})", s, (s < lo) & (s >= brady), lo, low=True),
        _ev(f"정상상한 초과(>{hi:.0f})", s, (s > hi) & (s <= tachy), hi, low=False),
    ]
    return f"HR: {_stat(s, 'bpm')}{_events([e for e in evs if e])}"


def _bp(item, s, age):
    """SBP/MBP 저혈압은 PALS·성인 술중저혈압 기준(유의), 고혈압은 95th %ile."""
    if s.empty:
        return ""
    evs = []
    if item in ("SBP", "ISBP1"):
        lo, hi = sbp_hypotension(age), sbp_hypertension(age)
        evs = [_ev(f"저혈압(<{lo:.0f})", s, s < lo, lo, low=True, crit=True),
               _ev(f"고혈압(>{hi:.0f})", s, s > hi, hi, low=False)]
    elif item in ("MBP", "IMBP1"):
        lo = map_hypotension(age)
        evs = [_ev(f"저혈압(<{lo:.0f})", s, s < lo, lo, low=True, crit=True)]
    elif item in ("DBP", "IDBP1"):
        # 하한 기준은 두 교과서에 없어 판정하지 않는다 (vital_thresholds 참조)
        hi = dbp_hypertension(age)
        evs = [_ev(f"고(>{hi:.0f})", s, s > hi, hi, low=False)]
        if not DBP_HAS_LOWER_THRESHOLD and not any(evs):
            return f"{item}: {_stat(s, 'mmHg')}, 상한 이상 없음"
    return f"{item}: {_stat(s, 'mmHg')}{_events([e for e in evs if e])}"


def _spo2(s):
    """[S] Ch.57 목표 94–99%. <90% = 탈포화(유의), 90–93% = 목표 이탈."""
    if s.empty:
        return ""
    evs = [
        _ev(f"SpO2<{SPO2_CRIT:.0f}%", s, s < SPO2_CRIT, SPO2_CRIT,
            low=True, crit=True, unit="%"),
        _ev(f"목표미달({SPO2_CRIT:.0f}–{SPO2_TARGET_LOW - 1:.0f}%)", s,
            (s < SPO2_TARGET_LOW) & (s >= SPO2_CRIT), SPO2_TARGET_LOW,
            low=True, unit="%"),
    ]
    return f"SpO2: {_stat(s, '%')}{_events([e for e in evs if e])}"


def _temp(s):
    """[S] Ch.7 저체온 <36.0 / Ch.21 안전범위 35.5–37.5."""
    if s.empty:
        return ""
    evs = [
        _ev(f"저체온(<{TEMP_SAFE_LOW})", s, s < TEMP_SAFE_LOW, TEMP_SAFE_LOW,
            low=True, crit=True, digits=1),
        _ev(f"저체온(<{TEMP_HYPOTHERMIA})", s,
            (s < TEMP_HYPOTHERMIA) & (s >= TEMP_SAFE_LOW), TEMP_HYPOTHERMIA,
            low=True, digits=1),
        _ev(f"발열(>{TEMP_FEVER})", s, s > TEMP_FEVER, TEMP_FEVER,
            low=False, crit=True, digits=1),
        _ev(f"안전범위 초과(>{TEMP_SAFE_HIGH})", s,
            (s > TEMP_SAFE_HIGH) & (s <= TEMP_FEVER), TEMP_SAFE_HIGH,
            low=False, digits=1),
    ]
    return f"체온(T1): {_stat(s, '°C', digits=1)}{_events([e for e in evs if e])}"


def _qtc(s, age):
    """[S] Ch.5 정상상한(신생아 470 / 소아 440) vs [M] 연장 기준 480 — 2-tier.

    v1은 450/460 단일 기준이라 임상적으로 무의미한 QTc '연장'이 대량 생성되어
    gold checklist를 오염시켰다 (PROMPTS_V3.md L235).
    """
    if s.empty:
        return ""
    upper = qtc_upper_normal(age)
    crit = _ev(f"연장(>{QTC_PROLONGED:.0f}ms)", s, s > QTC_PROLONGED,
               QTC_PROLONGED, low=False, crit=True, unit="ms")
    if crit:
        return f"QTc: {crit}"
    up = _ev(f"정상상한 초과(>{upper:.0f}ms)", s, s > upper, upper,
             low=False, unit="ms")
    if up:
        return f"QTc: {up} — 연장기준 {QTC_PROLONGED:.0f}ms 미만"
    return f"QTc: 정상 범위 (최대 {s.max():.0f}ms)"


def _ebl(sub, age, weight_kg=None):
    """실혈량 — 절대 mL + 순환혈액량(EBV) 대비 %. [S] Table 21.6 / Ch.18."""
    rows = sub[sub["_item"] == "EBL"].copy()
    if rows.empty:
        return ""
    v = pd.to_numeric(rows[VIT_VAL_COL], errors="coerce").dropna()
    if v.empty:
        return ""
    total = v.max()
    parts = [f"EBL: {total:.0f} mL"]
    pct = ebl_percent_ebv(total, age, weight_kg)
    if pct is not None:
        parts.append(f"(EBV의 {pct:.0f}%)")
        if pct > EBL_MASSIVE_PCT:
            parts.append(f"{CRIT}대량 실혈(>{EBL_MASSIVE_PCT:.0f}% EBV)")
        elif pct > EBL_SIGNIFICANT_PCT:
            parts.append(f"{CRIT}유의 실혈(>{EBL_SIGNIFICANT_PCT:.0f}% EBV)")
    return " ".join(parts)


def _hours_span(rows) -> Optional[float]:
    """기록 구간의 실제 경과 시간(hr). 타임스탬프가 부족하면 None."""
    ts = pd.to_datetime(rows[VIT_TIME_COL], errors="coerce").dropna()
    if len(ts) < 2:
        return None
    hrs = (ts.max() - ts.min()).total_seconds() / 3600.0
    return hrs if hrs > 0 else None


def _uo(sub, weight_kg=None):
    """소변량 — 누적 mL + mL/kg/hr. 핍뇨 기준 <0.5 mL/kg/hr ([M] Ch.24).

    속도는 '기록 건수 = 분'이라는 v1 가정(B8) 대신 실제 타임스탬프 경과
    시간으로 계산한다. UO 행이 1건뿐이면 케이스 전체 기록 구간으로 대체한다.
    """
    rows = sub[sub["_item"] == "UO"].copy()
    if rows.empty:
        return ""
    raw = rows[VIT_VAL_COL].astype(str).str.strip()
    if raw.str.upper().str.contains(r"NF|NO.?FOLEY", regex=True).any():
        return "UO: No Foley"
    v = pd.to_numeric(rows[VIT_VAL_COL], errors="coerce").dropna()
    if v.empty:
        return ""
    total = v.max()
    parts = [f"UO: {total:.0f} mL"]
    hrs = _hours_span(rows) or _hours_span(sub)
    if weight_kg and weight_kg > 0 and hrs:
        rate = total / (weight_kg * hrs)
        parts.append(f"({rate:.1f} mL/kg/hr)")
        if rate < UO_OLIGURIA:
            parts.append(f"{CRIT}핍뇨(<{UO_OLIGURIA} mL/kg/hr)")
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
    weight_kg  : 체중 (UO 속도·EBL %EBV 계산용, optional)
    """
    sub = _load_sub(vital_df, sid)
    if sub.empty:
        return "(바이탈 데이터 없음)"

    lines = [f"(연령군: {AGE_GROUP_LABELS[age_group(age_years)]})"]

    lines.append(_hr(_series(sub, "HR"), age_years))

    for item in BP_ITEMS:
        lines.append(_bp(item, _series(sub, item), age_years))

    # 침습혈압: 데이터 있는 경우만
    has_ibp = any(not _series(sub, i).empty for i in IBP_ITEMS)
    if has_ibp:
        for item in IBP_ITEMS:
            lines.append(_bp(item, _series(sub, item), age_years))

    lines.append(_spo2(_series(sub, "SpO2")))

    # Ppeak: 소아 일반마취용 임계값이 교과서에 없어 통계만 (판정하지 않음)
    ppeak = _series(sub, "Ppeak")
    if not ppeak.empty:
        lines.append(f"Ppeak: {_stat(ppeak, 'cmH₂O')}")

    lines.append(_temp(_series(sub, "T1")))
    lines.append(_qtc(_series(sub, "QTc"), age_years))
    lines.append(_ebl(sub, age_years, weight_kg))
    lines.append(_uo(sub, weight_kg))

    return "\n".join(l for l in lines if l)


def _find_weight_map(emr_df: pd.DataFrame, sid_series) -> dict:
    """EMR에서 체중 컬럼을 찾아 {수술ID: kg}. 못 찾으면 빈 dict.

    컬럼 이름이 데이터셋마다 달라 후보를 하드코딩하지 않고, MultiIndex 어느
    레벨이든 '체중'/'weight'가 들어간 컬럼을 찾아 쓴다. kg 범위(0.4~150)를
    벗어나는 값은 단위 오류로 보고 버린다 (신생아 0.5kg ~ 성인 체격 청소년).
    """
    cand = []
    for col in emr_df.columns:
        parts = col if isinstance(col, tuple) else (col,)
        joined = " ".join(str(p) for p in parts).lower()
        if "체중" in joined or "weight" in joined or "몸무게" in joined:
            cand.append(col)
    if not cand:
        return {}
    for col in cand:
        try:
            w = pd.to_numeric(emr_df[col], errors="coerce")
            w = w.where((w >= 0.4) & (w <= 150))
            m = {int(s): float(v) for s, v in zip(sid_series, w)
                 if pd.notna(s) and pd.notna(v)}
            if m:
                print(f"[build_vital_map] 체중 컬럼 사용: {col} ({len(m)}건)")
                return m
        except Exception:
            continue
    return {}


def build_vital_map(vital_df: pd.DataFrame, emr_df: pd.DataFrame) -> dict:
    """
    전체 수술 ID에 대한 {수술ID(int): vital_summary_str} dict 생성.

    EMR df의 '수술 ID' (공백 있음, level0 키) 기준으로 나이·체중을 매핑한다.
    체중은 EBL의 EBV 대비 %와 UO의 mL/kg/hr(핍뇨 판정)에 필요하다 — 없으면
    두 판정이 조용히 빠지므로 매핑 실패를 명시적으로 경고한다.
    """
    age_map, weight_map = {}, {}
    try:
        sid_series = emr_df["수술 ID"].astype("Int64")
        age_series = emr_df[("수술", "수술당시나이", "")]
        age_map = {int(s): float(a) for s, a in zip(sid_series, age_series)
                   if pd.notna(s) and pd.notna(a)}
        weight_map = _find_weight_map(emr_df, sid_series)
    except Exception as e:
        print(f"[build_vital_map] 나이 매핑 실패: {e}, 기본값 5세 사용")

    if not weight_map:
        print("[build_vital_map] ⚠ 체중 컬럼을 찾지 못했다 — "
              "EBL의 EBV 대비 %와 UO 핍뇨 판정이 생략된다.")

    vital_sids = vital_df[VIT_SID_COL].unique()
    vital_map = {}
    n_age, n_w = 0, 0

    for i, sid in enumerate(vital_sids):
        sid_int = int(sid)
        age = age_map.get(sid_int)
        w = weight_map.get(sid_int)
        n_age += age is not None
        n_w += w is not None
        vital_map[sid_int] = summarize_vitals(
            vital_df, sid_int, 5.0 if age is None else age, w)
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(vital_sids)}] 처리 중...")

    n = len(vital_map)
    print(f"[build_vital_map] 완료: {n}건 "
          f"(나이 매핑 {n_age}/{n}, 체중 매핑 {n_w}/{n})")
    if n and n_age < n:
        print(f"  ⚠ 나이 미매핑 {n - n_age}건은 기본값 5세로 판정됨 — "
              "연령별 임계값이 틀어진다")
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
