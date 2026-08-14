"""
vital_thresholds.py — 소아 바이탈 임계값 단일 출처 (교과서 근거)

모든 숫자는 아래 두 교과서에서 직접 인용했다. 근거 없는 숫자는 넣지 않는다.
근거를 못 찾은 항목(Ppeak)은 임계값을 만들지 않고 통계만 출력한다.

  [S] Smith's Anesthesia for Infants and Children, 9th ed. (Elsevier, 2021)
  [M] Miller's Anesthesia, 10th ed. (Elsevier, 2024)
       → docs/references/ (gitignored, 로컬 참조 전용). 페이지는 PDF 페이지 번호.

핵심 설계: **2-tier 판정**
  - normal 이탈 (`abnormal`) : 연령별 정상 참조범위를 벗어남 → 통계적 이상
  - 임상적 유의 (`critical`) : 소생/개입 기준을 넘음 → 인계문에 반드시 들어가야 함
  v1~v3에서 QTc 연장·일시적 SpO2 저하 같은 "범위 이탈이지만 임상적으로 무의미"한
  이벤트가 gold checklist를 오염시킨 문제(PROMPTS_V3.md L235·L249)를 이 분리로 해결한다.

전거 요약
---------
HR    [S] Ch.18 Table 18.1 "Normal Resting Heart Rates of Infants and Children" (PDF p.380)
      [S] Ch.57 Table 57.3 "Adequate Vital Signs for Children" — AHA/ILCOR PALS (PDF p.1400)
SBP   [S] Table 57.3 (저혈압, PALS)  /  Ch.18 Table 18.2 NHLBI 4th Task Force 95th %ile (PDF p.380)
DBP   [S] Ch.18 Table 18.3 NHLBI 4th Task Force 95th %ile (PDF p.380)
MAP   [M] Ch.4 성인 술중저혈압 정의 MAP <65 mmHg 또는 baseline 대비 20% 하강 (PDF p.226)
      [S] Ch.18 MAP = DBP + 1/3(SBP-DBP) (PDF p.380) · Ch.21 TAXI "MAP not <2SD below
          normal mean for age" (PDF p.491)
SpO2  [S] Ch.57 ROSC 후 pulse oximetry 목표 94–99% (PDF p.1401)
Temp  [S] Ch.21 "safe range for a child's core temperature is between 35.5°C and 37.5°C" (PDF p.467)
      [S] Ch.7  술중 저체온 정의 = core temp <36°C (PDF p.193)
QTc   [S] Ch.5  QT 0.47 s(신생아) → 0.44 s(연장아), Bazett 보정 (PDF p.133)
      [M] QTc ≥480 ms(여) / ≥470 ms(남) = 연장 (PDF p.1175)
UO    [M] Ch.24 핍뇨(oliguria) = urine flow rate <0.5 mL/kg/h (PDF p.544)
EBL   [S] Ch.21 Table 21.6 "Estimates of Circulating Blood Volume" (PDF p.467)
      [S] Ch.18 급성 실혈 >10% EBV / 총 실혈 >50% EBV → 동맥압 감시 적응증 (PDF p.381)
Ppeak 소아 일반마취용 임계값 없음 (CDH 신생아 맥락의 25·28 cmH2O만 존재) → 판정 안 함
"""

from typing import Optional, Tuple

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 연령 그룹 — 표기·리포트용. 임계값 자체는 아래에서 연속 나이로 계산한다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEONATE_MAX_YEARS = 28 / 365.0      # [S] Table 57.3 "term neonate"

AGE_GROUP_LABELS = {
    "neonate": "신생아(<28일)",
    "infant": "영아(28일–1세)",
    "toddler": "유아(1–2세)",
    "preschool": "학령전(3–5세)",
    "school": "학령기(6–11세)",
    "adolescent": "청소년(≥12세)",
}


def age_group(age: float) -> str:
    if age < NEONATE_MAX_YEARS:
        return "neonate"
    if age < 1:
        return "infant"
    if age < 3:
        return "toddler"
    if age < 6:
        return "preschool"
    if age < 12:
        return "school"
    return "adolescent"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [S] Table 18.1 — 안정 시 심박수 mean 및 range(±2 SD). (age_upper_exclusive, low, high)
# 원표의 연령 구간을 그대로 옮겼다. v1의 5구간 근사(infant 100–160 등)는 이 표와
# 크게 어긋나 정상 영아를 "빈맥"으로, 정상 학령전아를 "서맥"으로 오표기하고 있었다.
HR_NORMAL_BANDS = [
    (1 / 365.0, 94, 145),      # 0–24 hr
    (8 / 365.0, 100, 175),     # 1–7 days
    (31 / 365.0, 115, 190),    # 8–30 days
    (3 / 12.0, 124, 190),      # 1–3 months
    (1.0, 111, 179),           # 3–12 months
    (3.0, 98, 163),            # 1–3 years
    (5.0, 65, 132),            # 3–5 years
    (8.0, 70, 115),            # 5–8 years
    (float("inf"), 55, 105),   # 8–16 years
]

# [S] Table 57.3 (PALS) — brady는 흉부압박을 고려하는 심박수, tachy는 증상이
# 통상 발생하기 시작하는 추정 심박수. (age_upper_exclusive, brady_lt, tachy_gt)
HR_CRITICAL_BANDS = [
    (NEONATE_MAX_YEARS, 60, 220),   # term neonate
    (1.0, 60, 190),                 # infant
    (10.0, 60, 180),                # child
    (float("inf"), 60, 150),        # >10 yr
]


def _pick(bands, age: float):
    for upper, *vals in bands:
        if age < upper:
            return tuple(vals)
    return tuple(bands[-1][1:])


def hr_normal_range(age: float) -> Tuple[float, float]:
    """[S] Table 18.1 — 연령별 정상 안정시 HR (±2 SD)."""
    return _pick(HR_NORMAL_BANDS, age)


def hr_critical_range(age: float) -> Tuple[float, float]:
    """[S] Table 57.3 (PALS) — (서맥 기준, 빈맥 기준)."""
    return _pick(HR_CRITICAL_BANDS, age)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 혈압
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def sbp_hypotension(age: float) -> float:
    """[S] Table 57.3 (PALS) — 저혈압 정의 SBP.
    신생아 <60 / 영아 <70 / 소아 <70+(2×나이) / >10세 <90."""
    if age < NEONATE_MAX_YEARS:
        return 60.0
    if age < 1:
        return 70.0
    if age <= 10:
        return 70.0 + 2.0 * age
    return 90.0


# [S] Table 18.2 — 남/녀 1–12세, 신장 50th %ile 기준 SBP 95th %ile.
# 원표는 남녀 범위(예: 103–104)로 주어져 있어 상한을 취해 과잉 flag를 줄였다.
SBP_P95_BY_YEAR = {1: 104, 2: 106, 3: 109, 4: 111, 5: 112, 6: 114,
                   7: 115, 8: 116, 9: 118, 10: 119, 11: 121, 12: 123}
SBP_P95_INFANT = 100.0      # Table 18.2 미수록 구간 — 영아 고혈압 관례값
SBP_P95_ADULT = 130.0       # ≥13세는 AAP 2017 성인 기준(130/80) 적용

DBP_P95_BY_YEAR = {1: 58, 2: 63, 3: 67, 4: 70, 5: 72, 6: 74,
                   7: 76, 8: 78, 9: 79, 10: 80, 11: 80, 12: 81}
DBP_P95_INFANT = 65.0
DBP_P95_ADULT = 80.0


def _p95(table, age: float, infant_val: float, adult_val: float) -> float:
    if age < 1:
        return infant_val
    if age >= 13:
        return adult_val
    return float(table[min(12, max(1, int(age)))])


def sbp_hypertension(age: float) -> float:
    """[S] Table 18.2 — 연령별 SBP 95th %ile (이 값 초과 = 고혈압)."""
    return _p95(SBP_P95_BY_YEAR, age, SBP_P95_INFANT, SBP_P95_ADULT)


def dbp_hypertension(age: float) -> float:
    """[S] Table 18.3 — 연령별 DBP 95th %ile."""
    return _p95(DBP_P95_BY_YEAR, age, DBP_P95_INFANT, DBP_P95_ADULT)


# DBP 하한: 두 교과서 어디에도 소아 이완기 저혈압 수치 기준이 없다.
# PALS·TAXI 모두 관류 지표로 SBP/MAP만 쓴다 → DBP 저혈압 이벤트는 판정하지 않는다.
# (v1은 infant 30 / adolescent 45 등 출처 없는 값을 쓰고 있었다.)
DBP_HAS_LOWER_THRESHOLD = False

MAP_LOW_ADULT = 65.0        # [M] Ch.4 술중저혈압 절대 기준 MAP <65 mmHg
MAP_RELATIVE_DROP = 0.20    # [M] Ch.4 상대 기준 — baseline 대비 20% 하강


def map_hypotension(age: float) -> float:
    """MAP 저혈압 기준.

    소아 술중 MAP 절대 기준표는 두 교과서에 없다([S]는 de Graaff 2016 MPOG
    참조범위를 '그림'으로만 제시). 그래서 널리 쓰이는 소아 관례식
    `1.5×나이+40`을 쓰되, 성인 기준([M] MAP<65)을 상한으로 씌워 연속적으로
    성인값에 수렴시킨다 (16.7세에 65에 도달).
    """
    return min(1.5 * age + 40.0, MAP_LOW_ADULT)


def map_relative_hypotension(baseline_map: float) -> float:
    """[M] Ch.4 — baseline MAP 대비 20% 하강 기준."""
    return baseline_map * (1.0 - MAP_RELATIVE_DROP)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SpO2 · 체온 · QTc
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPO2_TARGET_LOW = 94.0      # [S] Ch.57 ROSC 후 목표 94–99% → 미만은 목표 이탈
SPO2_CRIT = 90.0            # [S] 탈포화(desaturation) 통용 기준 <90%

TEMP_SAFE_LOW = 35.5        # [S] Ch.21 안전범위 하한
TEMP_SAFE_HIGH = 37.5       # [S] Ch.21 안전범위 상한
TEMP_HYPOTHERMIA = 36.0     # [S] Ch.7 술중 저체온 정의 <36°C
TEMP_FEVER = 38.0           # 발열 통용 기준 (MH 감별 필요 구간)

QTC_UPPER_NEONATE = 470.0   # [S] Ch.5 신생아 QT 0.47 s
QTC_UPPER_CHILD = 440.0     # [S] Ch.5 연장아 QT 0.44 s
QTC_PROLONGED = 480.0       # [M] QTc ≥480 ms = 연장 (성별 미상 시 보수적으로 480)


def qtc_upper_normal(age: float) -> float:
    """[S] Ch.5 — 연령별 QTc 정상 상한."""
    return QTC_UPPER_NEONATE if age < NEONATE_MAX_YEARS else QTC_UPPER_CHILD


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 소변량 · 실혈량
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UO_OLIGURIA = 0.5           # [M] Ch.24 핍뇨 <0.5 mL/kg/hr

# [S] Table 21.6 — 순환혈액량 추정치(mL/kg). 원표 범위의 중앙을 취했다.
# (age_upper_exclusive, mL/kg)
EBV_BANDS = [
    (3 / 12.0, 85.0),        # 만삭 신생아 80–90 (미숙아 90–100은 재태주수 정보 필요)
    (1.0, 78.0),             # 3개월–1세 75–80
    (6.0, 73.0),             # 3–6세 70–75 (1–3세는 원표 공백 → 인접 구간 적용)
    (float("inf"), 68.0),    # >6세 65–70
]

EBL_SIGNIFICANT_PCT = 10.0   # [S] Ch.18 급성 실혈 >10% EBV → 동맥압 감시 적응증
EBL_MASSIVE_PCT = 50.0       # [S] Ch.18 총 실혈 >50% EBV

# [S] Ch.21 제한적 수혈전략 — Hb 7 g/dL 이상이면 수혈 불필요(신생아·청색성
# 심질환 제외), 수혈 후 목표 7–9.5 g/dL. checklist "수혈·수액" 항목 판정 근거.
HB_TRANSFUSION_TRIGGER = 7.0
HB_POSTTRANSFUSION_GOAL = (7.0, 9.5)


def estimated_blood_volume_ml_per_kg(age: float) -> float:
    """[S] Table 21.6 — 연령별 추정 순환혈액량(mL/kg)."""
    return _pick(EBV_BANDS, age)[0]


def ebl_percent_ebv(ebl_ml: float, age: float,
                    weight_kg: Optional[float]) -> Optional[float]:
    """실혈량을 순환혈액량 대비 %로 환산. 체중 없으면 None.

    소아에서 절대 mL은 의미가 없다 — 5 kg 영아의 100 mL는 EBV의 ~24%다.
    """
    if not weight_kg or weight_kg <= 0:
        return None
    ebv = estimated_blood_volume_ml_per_kg(age) * weight_kg
    if ebv <= 0:
        return None
    return 100.0 * ebl_ml / ebv
