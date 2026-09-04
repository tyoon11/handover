"""
vital_summarizer.py — 바이탈 시계열 → 인계용 요약 (v3.2 재설계)

v3.1의 문제와 v3.2의 처방은 docs/PIPELINE_V3.2.md §7b 에 있다. 요지:

  v3.1  임계값을 넘은 표본이 1개라도 있으면 이벤트로 등재하고, 이벤트가 없는
        항목까지 중앙값·범위를 출력했다. NIBP 한 번 55mmHg가 `[유의]저혈압`이 되고
        프롬프트가 "[유의]는 반드시 포함"이라 모델이 반드시 썼다 → 바이탈 편향.

  v3.2  **인계 대상은 세 가지 중 하나여야 한다** — ① 종료 시점까지 영향이 남은 것,
        ② 개입이 필요했던 것, ③ 지속·심도가 임계 이상인 것. 나머지는 MINOR로 내려
        프롬프트에 넣지 않는다. 판정 규칙 R1~R4 는 vital_thresholds.py 참조.

출력 구조 (프롬프트에 들어가는 것은 앞의 두 블록뿐)

    ### VITAL — REPORTABLE
    - 저혈압(<70): 최장 15분 / 총 20분, 최저 55 mmHg (기준-15, baseline 78 대비 -29%) — 개입: 승압제 13:20
    ### VITAL — AT HANDOFF (last 15 min)
    - HR 150 · SBP 82 · SpO2 99% · T1 36.8 — 종료 15분 내 처치: 승압제 15:20

  MINOR 블록은 문자열에 넣지 않고 **감사 데이터**(`return_audit=True`)로만 낸다.

임계값·판정규칙은 utils/vital_thresholds.py 한 곳에서 온다
(Smith's Anesthesia 2021 / Miller's Anesthesia 2024). 규칙 문서는 VITAL_SUMMARY_RULES.md.
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
    BP_ITEMS,
    IBP_ITEMS,
)

from utils.anesthetic_record import (        # noqa: E402
    extract_interventions, match_interventions, render_interventions,
)
from utils.vital_thresholds import (      # noqa: E402
    AGE_GROUP_LABELS,
    BASELINE_DROP_REQUIRED,
    MIN_DEVIATION,
    VITAL_REPORT_RULES,
    implausible_mask,
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
    hard_lines,
    has_planned_hypothermia,
    hr_critical_range,
    hr_normal_range,
    map_hypotension,
    qtc_upper_normal,
    sbp_hypertension,
    sbp_hypotension,
)

warnings.filterwarnings("ignore")

CRIT = "[유의]"   # v3.1 포맷 마커 — 구 산출물 파싱(감사 스크립트) 호환용으로만 남긴다

HDR_REPORTABLE = "### VITAL — REPORTABLE"
HDR_HANDOFF = "### VITAL — AT HANDOFF (last 15 min)"
NO_REPORTABLE = "- (인계 대상 바이탈 이상 없음)"

# 진단 카운터 (rebuild_vital_map 이 출력한다 — 조용한 누락 방지)
DROPPED_ARTIFACTS = {}
RULE_DIAG = {}


def _diag(key, n=1):
    RULE_DIAG[key] = RULE_DIAG.get(key, 0) + n


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파싱 유틸 (v3.1과 동일 — 검증된 부분)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _parse_item_name(raw: str) -> str:
    """'HR <- HR <- 마취기록' → 'HR'"""
    return raw.split(" <- ")[0].strip()


def _to_numeric(series: pd.Series) -> pd.Series:
    """object → numeric, 0값·비숫자 제거 (센서 미연결 / 오입력)."""
    s = pd.to_numeric(series, errors="coerce")
    return s.replace(0, np.nan)


def _load_sub(vital_df: pd.DataFrame, sid: int) -> pd.DataFrame:
    sub = vital_df[vital_df[VIT_SID_COL] == sid].copy()
    if sub.empty:
        return sub
    sub["_item"] = sub[VIT_ITEM_COL].apply(_parse_item_name)
    sub[VIT_TIME_COL] = pd.to_datetime(sub[VIT_TIME_COL], errors="coerce")
    return sub


def _series(sub: pd.DataFrame, item: str) -> pd.Series:
    """항목 이름 → 정제된 numeric 시계열 (측정 오류 배제 포함).

    센서 이탈로만 설명되는 값을 버린다. 임상적으로 극단이지만 가능한 값
    (DHCA 저체온 18–20°C, 청색성 심질환 SpO2 40–70%)은 **보존**한다.
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


DEFAULT_STEP_MIN = 5.0      # 마취기록 표본 간격 기본값 (간격 추정 불가 시)


def _step_min(s: pd.Series) -> float:
    idx = pd.DatetimeIndex(s.index).dropna()
    if len(idx) < 2:
        return DEFAULT_STEP_MIN
    d = pd.Series(idx).diff().dropna().dt.total_seconds() / 60.0
    d = d[d > 0]
    return float(d.median()) if len(d) else DEFAULT_STEP_MIN


def _fmt_min(m: float) -> str:
    if m < 1:
        return "<1분"
    if m < 60:
        return f"{m:.0f}분"
    return f"{m / 60:.1f}시간"


def _minute_of_day(ts) -> Optional[int]:
    try:
        t = pd.Timestamp(ts)
        return int(t.hour) * 60 + int(t.minute)
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 이벤트 구조화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _baseline(s: pd.Series) -> Optional[float]:
    """기록 시작 후 baseline_window_min 창의 중앙값.

    유도 전후 구간이다(모니터링은 유도 전에 시작된다). 정확한 '유도 시각'은
    바이탈 시트에 없으므로 시작 창을 대리값으로 쓴다 — 문서에 명시된 근사다.
    """
    if s.empty:
        return None
    idx = pd.DatetimeIndex(s.index).dropna()
    if len(idx) < VITAL_REPORT_RULES["baseline_min_samples"]:
        return None
    t0 = idx.min()
    win = s[(pd.DatetimeIndex(s.index) >= t0) &
            (pd.DatetimeIndex(s.index) <= t0 + pd.Timedelta(
                minutes=VITAL_REPORT_RULES["baseline_window_min"]))]
    if len(win) < VITAL_REPORT_RULES["baseline_min_samples"]:
        return None
    return float(win.median())


def _handoff_start(sub: pd.DataFrame):
    """케이스 마지막 표본 - handoff_window_min."""
    ts = pd.to_datetime(sub[VIT_TIME_COL], errors="coerce").dropna()
    if ts.empty:
        return None
    return ts.max() - pd.Timedelta(minutes=VITAL_REPORT_RULES["handoff_window_min"])


def _make_event(item, kind, label, s, mask, thresh, *, low, unit="", digits=0,
                tier2=False, baseline=None, handoff_start=None):
    """임계값 위반 → 구조화 이벤트 dict. 위반 없으면 None."""
    m = pd.Series(mask, index=s.index).fillna(False).astype(bool)
    n = int(m.sum())
    if not n:
        return None

    step = _step_min(s)
    grp = (m != m.shift()).cumsum()[m]
    episodes, minutes = [], []
    for _, run in m[m].groupby(grp):
        t = pd.DatetimeIndex(run.index).dropna()
        span = ((t.max() - t.min()).total_seconds() / 60.0) if len(t) >= 2 else 0.0
        episodes.append(dict(start=t.min() if len(t) else None,
                             end=t.max() if len(t) else None,
                             minutes=span + step, n=int(len(run))))
        for x in (t.min(), t.max()):
            mo = _minute_of_day(x)
            if mo is not None:
                minutes.append(mo)

    total = sum(e["minutes"] for e in episodes)
    longest = max((e["minutes"] for e in episodes), default=0.0)
    max_run = max((e["n"] for e in episodes), default=0)

    vals = s[m]
    ext = float(vals.min() if low else vals.max())
    delta = float(thresh - ext) if low else float(ext - thresh)

    # R3 "종료 시점 지속" — 마지막 창에 **스쳐간 1표본**은 지속이 아니다.
    #   episode가 2표본 이상이거나, 케이스 마지막 표본 자체가 위반일 때만 인정한다.
    #   (이 조건이 없으면 종료 15분 안의 아무 blip이나 R3로 승격된다 — selftest 케이스1)
    persists = False
    idx_all = pd.DatetimeIndex(s.index).dropna()
    t_series_last = idx_all.max() if len(idx_all) else None
    if handoff_start is not None:
        for e in episodes:
            if e["end"] is None or e["end"] < handoff_start:
                continue
            includes_last = (t_series_last is not None
                             and e["start"] is not None
                             and e["start"] <= t_series_last <= e["end"])
            if e["n"] >= 2 or includes_last:
                persists = True
                break

    drop_ok, drop_txt = _baseline_drop_ok(item, ext, baseline, low)

    return dict(item=item, kind=kind, label=label, threshold=float(thresh),
                low=bool(low), unit=unit, digits=int(digits), tier2=bool(tier2),
                n=n, total_min=round(total, 1), longest_min=round(longest, 1),
                max_run=max_run, ext=ext, delta=round(delta, 2),
                baseline=baseline, baseline_drop_ok=drop_ok, baseline_txt=drop_txt,
                persists_at_handoff=persists,
                episode_minutes=minutes, n_episodes=len(episodes),
                interventions=[], reportable=False, reasons=[])


_BASELINE_ABS_UNIT = {"SpO2": "%p", "T1": "°C"}


def _baseline_drop_ok(item, ext, baseline, low):
    """baseline 대비 하락 조건 (BASELINE_DROP_REQUIRED). (통과여부, 표기문자열).

    - 조건이 없는 항목이거나 baseline이 없으면 **통과**로 본다(억제하지 않는다).
    - '상승' 이벤트(low=False)에는 적용하지 않는다.
    """
    spec = BASELINE_DROP_REQUIRED.get(item)
    if spec is None or baseline is None or not low:
        return True, ""
    mode, amount = spec
    if mode == "rel":
        need = baseline * (1.0 - amount)
        pct = (baseline - ext) / baseline * 100.0 if baseline else 0.0
        return ext <= need, f"baseline {baseline:.0f} 대비 -{pct:.0f}%"
    need = baseline - amount
    unit = _BASELINE_ABS_UNIT.get(item, "")
    return ext <= need, f"baseline {baseline:g} 대비 -{baseline - ext:.1f}{unit}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 판정 (R1~R4) — docs/PIPELINE_V3.2.md §7b.4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def classify_event(ev, age, interventions, use_intervention_link=True):
    """이벤트를 REPORTABLE/MINOR로 판정. ev를 제자리 수정하고 reportable 반환."""
    rules = VITAL_REPORT_RULES
    reasons = []

    # R1 개입 동반
    if use_intervention_link and interventions:
        hits = match_interventions(ev["kind"], ev["episode_minutes"], interventions,
                                   rules["intervention_window_min"])
        if hits:
            ev["interventions"] = hits
            reasons.append("R1")

    # baseline 게이트 — **R2·R3·R4 공통**. R1(개입)에는 적용하지 않는다.
    #   "그 환자의 기저 상태는 수술 중 발생한 사건이 아니다. 다만 처치가 있었다면 예외."
    #   청색성 심질환의 기저 SpO2 75%는 절대기준(<90)과 위험선(<80)을 상시 위반하지만
    #   이벤트가 아니다 — 그 상태는 AT HANDOFF 블록이 전달한다.
    #   baseline이 없거나 해당 항목이 아니면 게이트는 항상 통과다(억제하지 않는다).
    gate = ev["baseline_drop_ok"]

    # R4 절대 위험선 (1표본이라도)
    hl = hard_lines(ev["item"], age)
    if gate and hl:
        low_lt, high_gt = hl
        if ev["low"] and low_lt is not None and ev["ext"] < low_lt:
            reasons.append("R4")
        if (not ev["low"]) and high_gt is not None and ev["ext"] > high_gt:
            reasons.append("R4")

    # R3 종료 시점 지속
    if gate and ev["persists_at_handoff"]:
        reasons.append("R3")

    # R2 지속×심도. tier2(정상범위 이탈)는 R2로 올리지 않는다 —
    #    개입이 있었거나(R1) 인계 시점에 남아 있을 때(R3)만 인계 대상이다.
    if gate and not ev["tier2"]:
        long_enough = (ev["longest_min"] >= rules["min_longest_min"]
                       or ev["max_run"] >= rules["min_samples"])
        deep_enough = ev["delta"] >= MIN_DEVIATION.get(ev["item"], 0.0)
        if long_enough and deep_enough:
            reasons.append("R2")

    ev["reasons"] = sorted(set(reasons))
    ev["reportable"] = bool(ev["reasons"])
    _diag("events_total")
    _diag("events_reportable" if ev["reportable"] else "events_minor")
    for r in ev["reasons"]:
        _diag(f"reason_{r}")
    return ev["reportable"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 렌더링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_event(ev) -> str:
    f = f".{ev['digits']}f"
    dur = f"최장 {_fmt_min(ev['longest_min'])}"
    if ev["n_episodes"] > 1:
        dur += f" / 총 {_fmt_min(ev['total_min'])}({ev['n_episodes']}회)"
    ext_lab = ("최저" if ev["low"] else "최고")
    parts = [f"{ev['label']}: {dur}, {ext_lab} {ev['ext']:{f}}{ev['unit']}"]
    inner = [f"기준{'-' if ev['low'] else '+'}{ev['delta']:{f}}"]
    if ev["baseline_txt"]:
        inner.append(ev["baseline_txt"])
    parts[0] += f" ({', '.join(inner)})"
    if ev["interventions"]:
        parts.append(f" — 개입: {render_interventions(ev['interventions'])}")
    return "- " + "".join(parts)


def render_minor(ev) -> str:
    f = f".{ev['digits']}f"
    return (f"{ev['label']} {ev['n']}회/{_fmt_min(ev['longest_min'])} "
            f"({'최저' if ev['low'] else '최고'} {ev['ext']:{f}}{ev['unit']})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 항목별 이벤트 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _hr_events(s, age, **kw):
    if s.empty:
        return []
    lo, hi = hr_normal_range(age)
    brady, tachy = hr_critical_range(age)
    out = [
        _make_event("HR", "bradycardia", f"서맥(<{brady:.0f})", s, s < brady, brady,
                    low=True, unit="bpm", **kw),
        _make_event("HR", "tachycardia", f"빈맥(>{tachy:.0f})", s, s > tachy, tachy,
                    low=False, unit="bpm", **kw),
        _make_event("HR", "bradycardia", f"정상하한 미만(<{lo:.0f})", s,
                    (s < lo) & (s >= brady), lo, low=True, unit="bpm",
                    tier2=True, **kw),
        _make_event("HR", "tachycardia", f"정상상한 초과(>{hi:.0f})", s,
                    (s > hi) & (s <= tachy), hi, low=False, unit="bpm",
                    tier2=True, **kw),
    ]
    return [e for e in out if e]


def _bp_events(item, s, age, **kw):
    if s.empty:
        return []
    out = []
    if item in ("SBP", "ISBP1"):
        lo, hi = sbp_hypotension(age), sbp_hypertension(age)
        out += [
            _make_event(item, "hypotension", f"저혈압(<{lo:.0f})", s, s < lo, lo,
                        low=True, unit="mmHg", **kw),
            _make_event(item, "hypertension", f"고혈압(>{hi:.0f})", s, s > hi, hi,
                        low=False, unit="mmHg", tier2=True, **kw),
        ]
    elif item in ("MBP", "IMBP1"):
        lo = map_hypotension(age)
        out += [_make_event(item, "hypotension", f"저혈압(<{lo:.0f})", s, s < lo, lo,
                            low=True, unit="mmHg", **kw)]
    elif item in ("DBP", "IDBP1"):
        # 하한 기준은 두 교과서에 없어 판정하지 않는다 (vital_thresholds 참조)
        hi = dbp_hypertension(age)
        out += [_make_event(item, "hypertension", f"이완기 고혈압(>{hi:.0f})", s,
                            s > hi, hi, low=False, unit="mmHg", tier2=True, **kw)]
    return [e for e in out if e]


def _spo2_events(s, **kw):
    if s.empty:
        return []
    out = [
        _make_event("SpO2", "desaturation", f"탈포화(<{SPO2_CRIT:.0f}%)", s,
                    s < SPO2_CRIT, SPO2_CRIT, low=True, unit="%", **kw),
        _make_event("SpO2", "desaturation",
                    f"목표미달({SPO2_CRIT:.0f}–{SPO2_TARGET_LOW - 1:.0f}%)", s,
                    (s < SPO2_TARGET_LOW) & (s >= SPO2_CRIT), SPO2_TARGET_LOW,
                    low=True, unit="%", tier2=True, **kw),
    ]
    return [e for e in out if e]


def _temp_events(s, **kw):
    if s.empty:
        return []
    out = [
        _make_event("T1", "hypothermia", f"저체온(<{TEMP_SAFE_LOW})", s,
                    s < TEMP_SAFE_LOW, TEMP_SAFE_LOW, low=True, unit="°C",
                    digits=1, **kw),
        _make_event("T1", "hypothermia", f"저체온(<{TEMP_HYPOTHERMIA})", s,
                    (s < TEMP_HYPOTHERMIA) & (s >= TEMP_SAFE_LOW), TEMP_HYPOTHERMIA,
                    low=True, unit="°C", digits=1, tier2=True, **kw),
        _make_event("T1", "fever", f"발열(>{TEMP_FEVER})", s, s > TEMP_FEVER,
                    TEMP_FEVER, low=False, unit="°C", digits=1, **kw),
        _make_event("T1", "fever", f"안전범위 초과(>{TEMP_SAFE_HIGH})", s,
                    (s > TEMP_SAFE_HIGH) & (s <= TEMP_FEVER), TEMP_SAFE_HIGH,
                    low=False, unit="°C", digits=1, tier2=True, **kw),
    ]
    return [e for e in out if e]


def _qtc_events(s, age, **kw):
    """QTc는 유의 기준(>480ms)만 본다.

    자동 QTc는 Bazett 보정이 빈맥에서 과대평가되고 노이즈에 민감하다.
    v3.1의 tier2(정상상한 초과)는 임상적으로 무의미한 '연장'을 대량 생성해
    checklist를 오염시켰다 → **tier2 폐기** (§7b.4).
    """
    if s.empty:
        return []
    e = _make_event("QTc", "qtc", f"QTc 연장(>{QTC_PROLONGED:.0f}ms)", s,
                    s > QTC_PROLONGED, QTC_PROLONGED, low=False, unit="ms", **kw)
    return [e] if e else []


# ── 총량 항목 (episode 아님 — 종료 시점 상태이므로 flag되면 REPORTABLE) ──────
def _ebl_line(sub, age, weight_kg=None):
    rows = sub[sub["_item"] == "EBL"]
    if rows.empty:
        return None, None
    v = pd.to_numeric(rows[VIT_VAL_COL], errors="coerce").dropna()
    if v.empty:
        return None, None
    total = float(v.max())
    pct = ebl_percent_ebv(total, age, weight_kg)
    txt = f"실혈 {total:.0f} mL"
    flagged = False
    if pct is not None:
        txt += f" (EBV의 {pct:.0f}%)"
        if pct > EBL_MASSIVE_PCT:
            txt += f" — 대량 실혈(>{EBL_MASSIVE_PCT:.0f}% EBV)"
            flagged = True
        elif pct > EBL_SIGNIFICANT_PCT:
            txt += f" — 유의 실혈(>{EBL_SIGNIFICANT_PCT:.0f}% EBV)"
            flagged = True
    else:
        _diag("ebl_no_weight")
    return txt, flagged


def _hours_span(rows) -> Optional[float]:
    ts = pd.to_datetime(rows[VIT_TIME_COL], errors="coerce").dropna()
    if len(ts) < 2:
        return None
    hrs = (ts.max() - ts.min()).total_seconds() / 3600.0
    return hrs if hrs > 0 else None


def _uo_line(sub, weight_kg=None):
    rows = sub[sub["_item"] == "UO"]
    if rows.empty:
        return None, None
    raw = rows[VIT_VAL_COL].astype(str).str.strip()
    if raw.str.upper().str.contains(r"NF|NO.?FOLEY", regex=True).any():
        return "소변량: No Foley", False
    v = pd.to_numeric(rows[VIT_VAL_COL], errors="coerce").dropna()
    if v.empty:
        return None, None
    total = float(v.max())
    txt = f"소변량 {total:.0f} mL"
    flagged = False
    hrs = _hours_span(rows) or _hours_span(sub)
    if weight_kg and weight_kg > 0 and hrs:
        rate = total / (weight_kg * hrs)
        txt += f" ({rate:.1f} mL/kg/hr)"
        if rate < UO_OLIGURIA:
            txt += f" — 핍뇨(<{UO_OLIGURIA} mL/kg/hr)"
            flagged = True
    else:
        _diag("uo_no_weight_or_time")
    return txt, flagged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AT HANDOFF 블록
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_HANDOFF_SPEC = [("HR", "HR", "", 0), ("SBP", "SBP", "", 0), ("ISBP1", "ISBP", "", 0),
                 ("MBP", "MBP", "", 0), ("SpO2", "SpO2", "%", 0),
                 ("T1", "T1", "°C", 1)]


def _handoff_block(ser, handoff_start, interventions):
    """마지막 창의 대표값(중앙값) + 그 창에 있었던 처치. ser = item→Series 접근자."""
    if handoff_start is None:
        return []
    parts = []
    for item, label, unit, digits in _HANDOFF_SPEC:
        s = ser(item)
        if s.empty:
            continue
        win = s[pd.DatetimeIndex(s.index) >= handoff_start]
        if win.empty:
            continue
        parts.append(f"{label} {win.median():.{digits}f}{unit}")
    if not parts:
        return []
    line = "- " + " · ".join(parts)
    hs = _minute_of_day(handoff_start)
    if interventions and hs is not None:
        late = [(cat, f"{mo // 60:02d}:{mo % 60:02d}")
                for (mo, cat, _snip) in interventions if mo >= hs]
        if late:
            line += f" — 종료 15분 내 처치: {render_interventions(late)}"
    return [line]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 공개 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def summarize_vitals(vital_df: pd.DataFrame, sid: int, age_years: float,
                     weight_kg: Optional[float] = None, record_text: str = "",
                     return_audit: bool = False, use_intervention_link: bool = True):
    """단일 수술의 바이탈 요약.

    반환: 프롬프트용 문자열 (return_audit=True 면 (문자열, 감사 dict))

    record_text : 마취기록 원문 — R1(개입 연동)과 계획된 저체온 판정에 쓴다.
                  비우면 R1이 꺼지고(누락 카운터 증가) 절대·지속 규칙만 적용된다.
    """
    sub = _load_sub(vital_df, sid)
    if sub.empty:
        _diag("cases_no_vital")
        txt = "(바이탈 데이터 없음)"
        return (txt, dict(sid=sid, reportable=[], minor=[], note="no_vital")) \
            if return_audit else txt

    interventions = extract_interventions(record_text) if record_text else []
    _diag("cases_with_intervention" if interventions
          else "cases_without_intervention")
    planned_hypo = has_planned_hypothermia(record_text)
    if planned_hypo:
        _diag("cases_planned_hypothermia")

    handoff_start = _handoff_start(sub)

    # 항목별 시계열 캐시 — 같은 항목을 여러 번 뽑으면 DROPPED_ARTIFACTS 가 이중계수된다
    _cache = {}

    def ser(item):
        if item not in _cache:
            _cache[item] = _series(sub, item)
        return _cache[item]

    def evkw(item):
        return dict(baseline=_baseline(ser(item)), handoff_start=handoff_start)

    events = []
    events += _hr_events(ser("HR"), age_years, **evkw("HR"))
    for item in BP_ITEMS:
        events += _bp_events(item, ser(item), age_years, **evkw(item))
    if any(not ser(i).empty for i in IBP_ITEMS):
        for item in IBP_ITEMS:
            events += _bp_events(item, ser(item), age_years, **evkw(item))
    events += _spo2_events(ser("SpO2"), **evkw("SpO2"))
    events += _qtc_events(ser("QTc"), age_years, **evkw("QTc"))

    temp_s = ser("T1")
    temp_events = _temp_events(temp_s, baseline=_baseline(temp_s),
                               handoff_start=handoff_start)
    planned_line = None
    if planned_hypo:
        # 계획된 저체온: 이벤트 나열이 아니라 상태 한 줄 (§7b.5)
        low_evs = [e for e in temp_events if e["kind"] == "hypothermia"]
        if low_evs and not temp_s.empty:
            last = temp_s[pd.DatetimeIndex(temp_s.index) >= handoff_start] \
                if handoff_start is not None else temp_s
            end_t = float(last.median()) if len(last) else float(temp_s.iloc[-1])
            planned_line = (f"- 계획된 저체온(체외순환): 최저 {temp_s.min():.1f}°C, "
                            f"종료 시 {end_t:.1f}°C")
        temp_events = [e for e in temp_events if e["kind"] != "hypothermia"]
    events.extend(temp_events)

    for ev in events:
        classify_event(ev, age_years, interventions, use_intervention_link)

    reportable = [e for e in events if e["reportable"]]
    minor = [e for e in events if not e["reportable"]]

    # 총량 항목 (episode 아님 — 종료 시점 상태이므로 flag되면 REPORTABLE)
    total_lines = []
    for kind, (txt, flagged) in (("blood_loss", _ebl_line(sub, age_years, weight_kg)),
                                 ("oliguria", _uo_line(sub, weight_kg))):
        if not txt:
            continue
        ev = dict(item=kind, kind=kind, label=txt, tier2=not flagged,
                  reportable=bool(flagged), reasons=["R3"] if flagged else [],
                  n=None, total_min=None, longest_min=0.0, max_run=0, ext=None,
                  delta=None, low=True, unit="", digits=0, n_episodes=0,
                  threshold=None, baseline=None, baseline_drop_ok=True,
                  persists_at_handoff=bool(flagged), interventions=[])
        if flagged:
            total_lines.append("- " + txt)
            reportable.append(ev)
        else:
            minor.append(ev)

    # 정렬: 개입 동반 → 위험선 → 지속 → 그 외
    order = {"R1": 0, "R4": 1, "R3": 2, "R2": 3}
    reportable.sort(key=lambda e: min((order.get(r, 9) for r in e["reasons"]), default=9))

    lines = [f"(연령군: {AGE_GROUP_LABELS[age_group(age_years)]})", HDR_REPORTABLE]
    body = ([planned_line] if planned_line else []) \
        + [render_event(e) for e in reportable] + total_lines
    lines += body if body else [NO_REPORTABLE]
    hb = _handoff_block(ser, handoff_start, interventions)
    if hb:
        lines += [HDR_HANDOFF] + hb

    text = "\n".join(lines)
    if not return_audit:
        return text
    audit = dict(
        sid=sid, age_years=age_years, weight_kg=weight_kg,
        planned_hypothermia=planned_hypo,
        n_interventions=len(interventions),
        reportable=[_audit_ev(e) for e in reportable],
        minor=[_audit_ev(e) for e in minor],
        minor_text=" · ".join(render_minor(e) for e in minor if e.get("n")),
    )
    return text, audit


def _audit_ev(ev) -> dict:
    """감사용 직렬화 (datetime 제거 — pkl/json 안전)."""
    keep = ("item", "kind", "label", "threshold", "low", "unit", "digits", "tier2",
            "n", "total_min", "longest_min", "max_run", "ext", "delta",
            "baseline", "baseline_drop_ok", "persists_at_handoff",
            "n_episodes", "reportable", "reasons")
    out = {k: ev.get(k) for k in keep}
    out["interventions"] = [f"{c}@{t}" for c, t in (ev.get("interventions") or [])]
    return out


def _find_weight_map(emr_df: pd.DataFrame, sid_series) -> dict:
    """EMR에서 체중 컬럼을 찾아 {수술ID: kg}. 못 찾으면 빈 dict."""
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


def _find_record_map(emr_df: pd.DataFrame, sid_series) -> dict:
    """EMR에서 마취기록 본문을 찾아 {수술ID: text}. R1 개입 연동에 필요하다."""
    target = None
    for col in emr_df.columns:
        if isinstance(col, tuple) and len(col) >= 2 and \
                col[0] == "마취기록" and col[1] == "기록":
            target = col
            break
    if target is None:
        print("[build_vital_map] ⚠ 마취기록 본문 컬럼을 찾지 못했다 — "
              "R1(개입 연동)과 계획된 저체온 판정이 꺼진다.")
        return {}
    out = {}
    for s, v in zip(sid_series, emr_df[target]):
        if pd.notna(s) and isinstance(v, str):
            out[int(s)] = v
    print(f"[build_vital_map] 마취기록 본문 {len(out)}건")
    return out


def build_vital_map(vital_df: pd.DataFrame, emr_df: pd.DataFrame,
                    with_audit: bool = False):
    """{수술ID(int): vital_summary_str} (with_audit=True 면 (map, audit_map))."""
    age_map, weight_map, record_map = {}, {}, {}
    try:
        sid_series = emr_df["수술 ID"].astype("Int64")
        age_series = emr_df[("수술", "수술당시나이", "")]
        age_map = {int(s): float(a) for s, a in zip(sid_series, age_series)
                   if pd.notna(s) and pd.notna(a)}
        weight_map = _find_weight_map(emr_df, sid_series)
        record_map = _find_record_map(emr_df, sid_series)
    except Exception as e:
        print(f"[build_vital_map] 나이/기록 매핑 실패: {e}, 기본값 5세 사용")

    if not weight_map:
        print("[build_vital_map] ⚠ 체중 컬럼을 찾지 못했다 — "
              "EBL의 EBV 대비 %와 UO 핍뇨 판정이 생략된다.")

    vital_sids = vital_df[VIT_SID_COL].unique()
    vital_map, audit_map = {}, {}
    n_age, n_w, n_rec = 0, 0, 0

    for i, sid in enumerate(vital_sids):
        sid_int = int(sid)
        age = age_map.get(sid_int)
        w = weight_map.get(sid_int)
        rec = record_map.get(sid_int, "")
        n_age += age is not None
        n_w += w is not None
        n_rec += bool(rec)
        res = summarize_vitals(vital_df, sid_int, 5.0 if age is None else age, w,
                               record_text=rec, return_audit=True)
        vital_map[sid_int], audit_map[sid_int] = res
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(vital_sids)}] 처리 중...")

    n = len(vital_map)
    print(f"[build_vital_map] 완료: {n}건 (나이 {n_age}/{n} · 체중 {n_w}/{n} · "
          f"마취기록 {n_rec}/{n})")
    if n and n_age < n:
        print(f"  ⚠ 나이 미매핑 {n - n_age}건은 기본값 5세 — 연령별 임계값이 틀어진다")
    if n and not n_rec:
        print("  ⚠ 마취기록이 전혀 매핑되지 않았다 — R1(개입 연동)이 전부 꺼진 상태다")
    return (vital_map, audit_map) if with_audit else vital_map


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 자체 검증 (합성 데이터 — PHI 없음, GPU 불필요)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _synth(sid, rows, t0="2026-01-01 09:00"):
    """rows: [(item, [값...])] — 5분 간격 시계열 DataFrame 생성."""
    base = pd.Timestamp(t0)
    recs = []
    for item, vals in rows:
        for k, v in enumerate(vals):
            recs.append({VIT_SID_COL: sid,
                         VIT_TIME_COL: base + pd.Timedelta(minutes=5 * k),
                         VIT_ITEM_COL: f"{item} <- {item} <- 마취기록",
                         VIT_VAL_COL: v})
    return pd.DataFrame(recs)


def selftest(verbose=True):
    """규칙 R1~R4 · baseline · 계획된 저체온이 의도대로 동작하는지 검증."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'✓' if cond else '✗'} {name}")

    # ── 케이스 1: 지속 저혈압 + 승압제 (R1·R2) / 1표본 고혈압 (MINOR)
    n = 24
    sbp = [95] * 6 + [58, 55, 57, 60, 62] + [95] * (n - 11)
    sbp[20] = 140                     # 1표본 고혈압 → tier2 · MINOR 기대
    df = _synth(1, [("SBP", sbp),
                    ("HR", [110] * n),
                    ("SpO2", [99] * n),
                    ("T1", [36.5] * n)])
    rec = ("▶09:00 Start monitoring / 적극적 가온적용\n"
           "▶09:35 BP 55/30 → Ephedrine 3mg ivs\n"
           "▶11:00 Extubation / Transfer to PACU")
    txt, aud = summarize_vitals(df, 1, age_years=5.0, weight_kg=18.0,
                                record_text=rec, return_audit=True)
    if verbose:
        print("\n[케이스1] 지속 저혈압 + 승압제\n" + txt)
    kinds = {e["label"].split("(")[0]: e for e in aud["reportable"]}
    check("저혈압이 REPORTABLE", "저혈압" in kinds)
    check("저혈압에 R1(개입) 포함", "R1" in kinds.get("저혈압", {}).get("reasons", []))
    check("저혈압에 R2(지속) 포함", "R2" in kinds.get("저혈압", {}).get("reasons", []))
    check("1표본 고혈압은 MINOR",
          any("고혈압" in e["label"] for e in aud["minor"]))
    check("AT HANDOFF 블록 존재", HDR_HANDOFF in txt)

    # ── 케이스 2: 1표본 저혈압만 (아무 규칙도 통과 못 함 → MINOR)
    sbp2 = [95] * 24
    sbp2[10] = 60
    df2 = _synth(2, [("SBP", sbp2), ("HR", [110] * 24)])
    txt2, aud2 = summarize_vitals(df2, 2, age_years=5.0, record_text="", 
                                  return_audit=True)
    if verbose:
        print("\n[케이스2] 1표본 저혈압만\n" + txt2)
    check("1표본 저혈압은 REPORTABLE 아님",
          not any("저혈압" in e["label"] for e in aud2["reportable"]))
    check("REPORTABLE 없음 문구", NO_REPORTABLE in txt2)

    # ── 케이스 3: 청색성 심질환 — 기저 SpO2 75% (절대<90 상시 위반, baseline으로 억제)
    df3 = _synth(3, [("SpO2", [75, 76, 74, 75, 76] + [75] * 19),
                     ("HR", [120] * 24)])
    txt3, aud3 = summarize_vitals(df3, 3, age_years=1.0, record_text="",
                                  return_audit=True)
    if verbose:
        print("\n[케이스3] 기저 SpO2 75% (청색성)\n" + txt3)
    check("기저 저산소는 REPORTABLE 아님(baseline 게이트)",
          not any(e["kind"] == "desaturation" for e in aud3["reportable"]))
    # 같은 환자가 68%까지 떨어지면(baseline -7%p) 등재돼야 한다
    df3b = _synth(4, [("SpO2", [75] * 6 + [68, 67, 68] + [75] * 15),
                      ("HR", [120] * 24)])
    _, aud3b = summarize_vitals(df3b, 4, age_years=1.0, record_text="",
                                return_audit=True)
    check("baseline 대비 큰 하락은 REPORTABLE",
          any(e["kind"] == "desaturation" for e in aud3b["reportable"]))

    # ── 케이스 4: DHCA — 계획된 저체온은 이벤트가 아니라 상태 한 줄
    df4 = _synth(5, [("T1", [36.4] * 4 + [30, 24, 20.1, 20.3, 24, 30] + [36.2] * 14),
                     ("HR", [120] * 24)])
    txt4, aud4 = summarize_vitals(df4, 5, age_years=0.5,
                                  record_text="▶09:30 CPB on / DHCA 20C",
                                  return_audit=True)
    if verbose:
        print("\n[케이스4] DHCA 계획된 저체온\n" + txt4)
    check("계획된 저체온 상태 표기", "계획된 저체온" in txt4)
    check("저체온 이벤트는 나열되지 않음",
          not any(e["kind"] == "hypothermia" for e in aud4["reportable"]))

    # ── 케이스 5: R4 절대 위험선 — SpO2 78% 1표본
    sp = [99] * 24
    sp[12] = 78
    df5 = _synth(6, [("SpO2", sp), ("HR", [110] * 24)])
    _, aud5 = summarize_vitals(df5, 6, age_years=5.0, record_text="",
                               return_audit=True)
    check("1표본이라도 위험선(<80) 넘으면 REPORTABLE",
          any("R4" in e["reasons"] for e in aud5["reportable"]))

    print(f"\n{'전부 통과' if ok else '실패 있음'} · 진단 카운터: {RULE_DIAG}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="바이탈 요약 (v3.2)")
    ap.add_argument("--selftest", action="store_true",
                    help="합성 데이터로 규칙 검증 (PHI·GPU 불필요)")
    ap.add_argument("--sample", type=int, default=0,
                    help="실데이터 미리보기 케이스 수")
    a = ap.parse_args()

    if a.selftest:
        _sys.exit(0 if selftest() else 1)

    from config import VITAL_PKL
    print("Vital pkl 로드 중...")
    vital_df = pd.read_pickle(VITAL_PKL)
    print(f"  shape: {vital_df.shape}")
    for sid in vital_df[VIT_SID_COL].unique()[:max(1, a.sample)]:
        print(f"\n{'=' * 60}\n수술ID: {sid}")
        print(summarize_vitals(vital_df, int(sid), age_years=5.0))
