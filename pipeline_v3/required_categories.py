"""
required_categories.py — 인계문 **필수 항목군** 단일 출처

인계문(handoff)에 반드시 포함되어야 하는 6개 항목군 정의. 생성 프롬프트,
gold checklist 추출, coverage 채점이 전부 이 파일 하나를 참조한다.

  1. comorbidity_medication  기저질환·약물
  2. airway_management       기도관리
  3. intraop_event           수술 중 이벤트 및 처치
  4. transfusion_fluid       수혈·수액
  5. preop_abnormal_test     수술 전 검사이상
  6. uri_status              감기 유무

판정 원칙 (**조건부 필수**)
---------------------------
"필수"는 *EMR에 해당 소견이 있으면 반드시 인계문에 들어가야 한다*는 뜻이다.
소견이 없는 항목군까지 "없음"이라고 쓰라는 뜻이 아니다 — 그렇게 하면
"이상 소견만 짧게"라는 인계문의 대전제(brevity 축)가 무너진다.
단 하나의 예외가 `uri_status`인데, 감기는 **없다는 사실 자체가 술후 호흡기
합병증 위험 평가에 쓰이므로** gold가 언급했을 때만 항목이 된다는 점은 같지만
채점 시 별도 그룹으로 분리해 누락을 눈에 띄게 한다.

근거 (docs/references/, PDF 페이지)
-----------------------------------
전반   [S] Ch.16 Table 16.1 "review of systems and their preoperative
       implications" — 술전 평가에서 빠뜨리면 안 되는 계통별 항목 (PDF p.323)
1  기저질환·약물   [S] Ch.16 Box 16.1 마취 전 상담이 필요한 복잡 환자군 (PDF p.323)
2  기도관리        [S] Ch.19 소아 어려운 기도·후두경련·발관 후 croup (PDF p.365 이하),
                   [S] Ch.16 Tait 위험인자: 5세 미만 삽관·기도수술 (PDF p.330)
3  수술 중 이벤트   [S] Ch.57 Table 57.3 소생 기준 (PDF p.1400),
                   [M] Ch.4 술중저혈압 MAP<65 / baseline 20% 하강 (PDF p.226)
4  수혈·수액      [S] Ch.21 Table 21.6 EBV, MABL 계산식, 제한적 수혈전략
                   Hb 7 g/dL (PDF p.467–468), TAXI 결정도 (PDF p.491)
5  수술 전 검사이상 [S] Ch.16 술전 검사 원칙 (PDF p.323),
                   [S] Ch.36 Box 36.2 술전 검사·ECG·PFT 적응증 (PDF p.1002)
6  감기 유무       [S] Ch.16 Table 16.6 COLDS score + URI 후 2–4주 호흡기
                   부작용 증가, 기관지 과민성 4–6주 지속 (PDF p.329–330)
"""

# (id, 한글 라벨, 판정 설명 — 프롬프트에 그대로 삽입된다)
REQUIRED_CATEGORIES = [
    (
        "comorbidity_medication",
        "기저질환·약물",
        "선천기형·증후군·심폐질환 등 주요 기저질환, 복용/투여 중인 약물과 그 "
        "술후 영향(잔여 근이완, 마약성 진통제, 항경련제, 스테로이드 등), 알레르기.",
    ),
    (
        "airway_management",
        "기도관리",
        "기관튜브/LMA 종류와 크기, 어려운 기도·삽관 시도 횟수, 발관 여부와 "
        "발관 후 상태, 후두경련·기관지경련·croup·재삽관, 술후 산소 요구.",
    ),
    (
        "intraop_event",
        "수술 중 이벤트 및 처치",
        "저혈압·서맥·부정맥·탈포화·저체온 등 실제 발생한 이벤트와 그에 대한 "
        "처치(승압제·수액부하·아트로핀·수혈 등), 유치된 line/device, 체위 관련 문제. "
        "바이탈 이벤트는 지속시간과 기준 대비 편차(최저/최고값)를 반드시 포함.",
    ),
    (
        "transfusion_fluid",
        "수혈·수액",
        "실혈량(가능하면 EBV 대비 %), 수혈한 혈액제제 종류와 양, 투여 수액 "
        "총량, 소변량/핍뇨, 수술 종료 시점의 혈역학·용적 상태.",
    ),
    (
        "preop_abnormal_test",
        "수술 전 검사이상",
        "술전 혈액검사·응고·전해질·영상·ECG·심초음파·PFT 등에서 확인된 "
        "이상 소견 중 술후 관리에 영향을 주는 것.",
    ),
    (
        "uri_status",
        "감기 유무",
        "최근 상기도감염(URI) 증상 유무와 발생 시점(2주 이내/2–4주). 술후 "
        "호흡기 부작용 위험 평가에 직결되므로 gold가 언급했다면 반드시 전달.",
    ),
]

CATEGORY_IDS = [c[0] for c in REQUIRED_CATEGORIES]
CATEGORY_LABELS = {c[0]: c[1] for c in REQUIRED_CATEGORIES}
CATEGORY_DESCRIPTIONS = {c[0]: c[2] for c in REQUIRED_CATEGORIES}

FALLBACK_CATEGORY = "other"
ALL_CATEGORY_IDS = CATEGORY_IDS + [FALLBACK_CATEGORY]

# v1/v2 category → 6개 항목군 매핑. 기존 gold_checklist JSON을 재추출 없이
# 새 그룹으로 읽을 수 있게 한다.
LEGACY_CATEGORY_MAP = {
    "airway": "airway_management",
    "respiratory": "airway_management",
    "hemodynamics": "intraop_event",
    "intraop_event": "intraop_event",
    "lines_devices": "intraop_event",
    "cooperation_agitation": "intraop_event",
    "bleeding_transfusion": "transfusion_fluid",
    "congenital_major_disease": "comorbidity_medication",
    "drug_effect": "comorbidity_medication",
    "other": FALLBACK_CATEGORY,
}


def normalize_category(raw) -> str:
    """LLM이 뱉은 category 문자열 → 표준 항목군 id. 미상은 'other'."""
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in CATEGORY_LABELS:
        return key
    return LEGACY_CATEGORY_MAP.get(key, FALLBACK_CATEGORY)


def prompt_block(indent: str = "") -> str:
    """프롬프트에 삽입할 항목군 설명 블록 (한국어)."""
    lines = []
    for i, (cid, label, desc) in enumerate(REQUIRED_CATEGORIES, 1):
        lines.append(f"{indent}{i}. {label} ({cid}) — {desc}")
    return "\n".join(lines)


def group_by_category(items) -> dict:
    """checklist items → {category_id: [item, ...]}."""
    out = {c: [] for c in ALL_CATEGORY_IDS}
    for it in items or []:
        out[normalize_category(it.get("category"))].append(it)
    return out
