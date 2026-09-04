"""
anesthetic_record.py — 마취기록 타임라인에서 '개입(intervention)' 추출 (v3.2)

왜 필요한가 (docs/PIPELINE_V3.2.md §7b.4 R1)
  "이 바이탈 이상이 실제로 문제였나"의 가장 좋은 대리지표는 **의사가 반응했는지**다.
  저혈압 시각에 승압제가 들어갔다면 그건 인계해야 할 사건이고, 아무 처치 없이
  한 표본 스쳐 지났다면 대개 인계 가치가 없다. 그 정보가 이미 마취기록에 있다.

마취기록 포맷 (실데이터)
  ▶12:05 Started monitoring ECG, NIBP, SpO2 / 활력 징후 이상없음 / 적극적 가온적용
  ▶12:10 Intubation with AuraGain #3→4 without difficulty / B.B.S O.K
  ▶13:37 Bridion 110mg ivs / Extubation / Transfer to PACU

의도적으로 **제외**한 것 (오탐 원인)
  - 가온(forced air warming) : 거의 모든 케이스에 상시 적용된다 → 저체온을 전부 승격시킨다.
  - 최초 기관내삽관(intubation) : 유도 절차이지 '문제에 대한 대응'이 아니다.
    재삽관·후두경련·기관지경련 등 **구조(rescue)** 만 잡는다.

이 모듈은 pandas·GPU 없이 동작한다 (테스트 가능).
"""

import re

# 개입 분류 → 키워드 (소문자 매칭). **교수님 확인 대상** (§7b.9-1)
INTERVENTION_KEYWORDS = {
    "vasopressor": [
        "ephedrine", "epinephrine", "epi ", "norepi", "phenylephrine", "dopamine",
        "dobutamine", "vasopressin", "milrinone", "승압", "에페드린", "에피네프린",
    ],
    "anticholinergic": ["atropine", "glycopyrrolate", "robinul", "아트로핀"],
    "transfusion": [
        "prbc", "p-rbc", "rbc", "ffp", "platelet", "plt ", "cryo", "transfus",
        "수혈", "혈액제제",
    ],
    "fluid_bolus": [
        "bolus", "volume loading", "fluid challenge", "albumin", "볼루스", "수액부하",
        "volume expan",
    ],
    "airway_rescue": [
        "reintubat", "re-intubat", "재삽관", "laryngospasm", "후두경련",
        "bronchospasm", "기관지경련", "croup", "mask ventilation", "cpap",
        "recruitment", "suction ", "desat",
    ],
    "cpr": [
        "cpr", "chest compression", "compression started", "defibrillat",
        "cardioversion", "제세동", "심장마사지",
    ],
    "antiarrhythmic": [
        "amiodarone", "lidocaine ivs", "esmolol", "adenosine", "digoxin",
    ],
}

# 이벤트 종류 → 이 개입이 있으면 R1 성립
EVENT_INTERVENTION_MAP = {
    "hypotension": ["vasopressor", "fluid_bolus", "transfusion"],
    "bradycardia": ["anticholinergic", "cpr", "vasopressor"],
    "tachycardia": ["antiarrhythmic"],
    "hypertension": [],                 # 대응 처치가 기록에 잘 안 남는다 → R1 미적용
    "desaturation": ["airway_rescue", "cpr"],
    "hypothermia": [],                  # 가온은 상시 적용 → R1 미적용 (위 주석)
    "fever": [],
    "qtc": ["antiarrhythmic", "cpr"],
    "blood_loss": ["transfusion"],
    "oliguria": ["fluid_bolus", "transfusion"],
}

_RE_TIMELINE = re.compile(r"[▶▷►]?\s*(\d{1,2}):(\d{2})\s*([^\n▶▷►]*)")


def parse_timeline(record_text: str):
    """마취기록 → [(minute_of_day, text), ...]. 시각 표기가 없으면 빈 리스트."""
    out = []
    for m in _RE_TIMELINE.finditer(record_text or ""):
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            continue
        out.append((h * 60 + mi, m.group(3).strip()))
    return out


def extract_interventions(record_text: str):
    """[(minute_of_day, category, snippet), ...] — 시각이 붙은 개입만."""
    found = []
    for minute, text in parse_timeline(record_text):
        low = text.lower()
        for cat, kws in INTERVENTION_KEYWORDS.items():
            if any(k in low for k in kws):
                found.append((minute, cat, text[:60]))
    return found


def _circ_diff(a: int, b: int) -> int:
    """분 단위 시각 차이 (자정 넘김 고려, 0~720)."""
    d = abs(a - b) % 1440
    return min(d, 1440 - d)


def fmt_minute(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def match_interventions(event_kind: str, event_minutes, interventions,
                        window_min: float = 10.0):
    """이벤트 시각들과 개입 목록을 매칭 → [(category, "HH:MM"), ...] (중복 제거).

    event_minutes : 이벤트 episode의 시각(minute_of_day) 목록
    interventions : extract_interventions() 결과
    """
    cats = EVENT_INTERVENTION_MAP.get(event_kind, [])
    if not cats or not event_minutes or not interventions:
        return []
    hits, seen = [], set()
    for minute, cat, _snip in interventions:
        if cat not in cats:
            continue
        if any(_circ_diff(minute, em) <= window_min for em in event_minutes):
            key = (cat, minute)
            if key not in seen:
                seen.add(key)
                hits.append((cat, fmt_minute(minute)))
    return hits


INTERVENTION_LABELS = {
    "vasopressor": "승압제", "anticholinergic": "아트로핀/항콜린",
    "transfusion": "수혈", "fluid_bolus": "수액부하",
    "airway_rescue": "기도 처치", "cpr": "심폐소생/제세동",
    "antiarrhythmic": "항부정맥제",
}


def render_interventions(hits) -> str:
    """[(cat,"13:20"), ...] → '승압제 13:20, 수혈 13:40'."""
    return ", ".join(f"{INTERVENTION_LABELS.get(c, c)} {t}" for c, t in hits)
