# 인계문 필수 항목군 (Required Categories)

구현: [pipeline_v3/required_categories.py](../pipeline_v3/required_categories.py)
근거 문헌 표기(**[S]** / **[M]**)와 PDF 페이지 규칙은 [THRESHOLDS.md](THRESHOLDS.md)와 동일.

## 6개 항목군

| # | id | 라벨 | 무엇을 담나 |
|---|---|---|---|
| 1 | `comorbidity_medication` | 기저질환·약물 | 선천기형·증후군·심폐질환 등 주요 기저질환, 투여 약물과 술후 영향(잔여 근이완, 마약성 진통제, 항경련제, 스테로이드), 알레르기 |
| 2 | `airway_management` | 기도관리 | 기관튜브/LMA 종류·크기, 어려운 기도·삽관 시도 횟수, 발관 여부와 발관 후 상태, 후두경련·기관지경련·croup·재삽관, 술후 산소 요구 |
| 3 | `intraop_event` | 수술 중 이벤트 및 처치 | 저혈압·서맥·부정맥·탈포화·저체온 등 실제 이벤트와 처치(승압제·수액부하·아트로핀·수혈), 유치 line/device, 체위 관련 문제 |
| 4 | `transfusion_fluid` | 수혈·수액 | 실혈량(가능하면 EBV 대비 %), 혈액제제 종류·양, 수액 총량, 소변량/핍뇨, 종료 시점 혈역학·용적 상태 |
| 5 | `preop_abnormal_test` | 수술 전 검사이상 | 술전 혈액·응고·전해질·영상·ECG·심초음파·PFT 이상 소견 중 술후 관리에 영향을 주는 것 |
| 6 | `uri_status` | 감기 유무 | 최근 상기도감염(URI) 증상 유무와 발생 시점 |

`other`는 fallback이며 그룹 채점의 분모에 들어가지 않는다.

## 판정 원칙 — **조건부 필수**

"필수"는 *EMR에 해당 소견이 있으면 반드시 인계문에 들어가야 한다*는 뜻이다.
소견이 없는 항목군까지 "없음"이라고 쓰라는 뜻이 **아니다** — 그렇게 하면 "이상 소견만 짧게"라는
인계문의 대전제(brevity 축)와 `특이사항 없음` 규칙이 무너진다.

따라서 채점에서도 **gold가 실제로 다룬 항목군만 분모**에 넣는다
(`checklist[sid]["required_categories"]`).

## 근거

### 전반
[S] Ch.16 **Table 16.1** "review of systems and their preoperative implications" (PDF p.323) —
술전 계통별 문진에서 마취 계획에 영향을 주는 항목. 6개 군은 이 표와 술후 인계에서 실제로
필요한 정보의 교집합이다.

### 1. 기저질환·약물
[S] Ch.16 **Box 16.1** (PDF p.323) — 마취 전 상담이 필요한 복잡 환자군 목록.

### 2. 기도관리
- [S] Ch.19 소아 어려운 기도, 후두경련, 발관 후 croup (PDF p.365 이하)
- [S] Ch.16 (PDF p.330) Tait 2001 술후 호흡기 합병증 독립 위험인자: **5세 미만 삽관**,
  reactive airway disease, 부모 흡연, 미숙아, **기도 수술**, 다량 분비물 + 코막힘
- [S] (PDF p.435 부근) 발관 후 stridor·croup, 음압성 폐부종

### 3. 수술 중 이벤트 및 처치
- [S] Ch.57 **Table 57.3** (PDF p.1400) — 소생 기준 vital
- [M] Ch.4 (PDF p.226) — 술중저혈압 정의 MAP < 65 또는 baseline 20% 하강
- 판정 임계값은 전부 [THRESHOLDS.md](THRESHOLDS.md)

### 4. 수혈·수액
- [S] Ch.21 **Table 21.6** 순환혈액량, MABL 계산식, 제한적 수혈전략 Hb 7 g/dL (PDF p.467–468)
- [S] TAXI RBC 수혈 결정도 (PDF p.491)
- [M] Ch.24 핍뇨 < 0.5 mL/kg/h (PDF p.544)

### 5. 수술 전 검사이상
- [S] Ch.16 술전 검사 원칙 (PDF p.323)
- [S] Ch.36 **Box 36.2** (PDF p.1002) — 술전 검사·ECG·심초음파·PFT 적응증. 예: 신경근육성
  척추측만에서 FVC < 30% 예측치 또는 최대흡기압 < 30 cmH₂O면 술후 삽관 유지 위험 증가.

### 6. 감기 유무 — 왜 독립 항목군인가
[S] Ch.16 (PDF p.329–330):

- URI **도중 또는 이후 4주까지** 마취하면 후두경련·기관지경련·탈포화·심한 기침 발생률이 증가.
  Tait & Knight: 합병증은 감기 "도중"이 아니라 **감기 후 2주 이내**에 가장 높다.
- 기관지 과민성은 **4–6주** 지속.
- 경증 URI는 **2주** 연기가 안전한 타협안.
- **COLDS score** ([S] Table 16.6) — Current signs/Onset/Lung disease/Device/Surgery.
  18점에 근접하면 술후 호흡기 부작용 50%, 19점 초과면 통상 수술 취소.

| | 1점 | 2점 | 3점 |
|---|---|---|---|
| **C** current signs | 없음 | 경증(코막힘·콧물·인후통·재채기·미열·마른기침) | 중등–중증(농성 분비물, 습성 기침, 이상 폐음, 기면, toxic, 고열) |
| **O** onset | > 4주 전 | 2–4주 전 | < 2주 전 |
| **L** lung disease | 없음 | 경증(RSV 병력, 경증 간헐 천식, >1세 BPD, 심한 코골이, 간접흡연) | 중등–중증(중등 지속 천식, 영아 BPD, OSA, 폐고혈압) |
| **D** airway device | 없음/마스크 | LMA·성문상 기도유지기 | 기관튜브 |
| **S** surgery | 기타(환기관 포함) | 소기도(T&A, 누관 탐침, 굴곡 기관지경, 발치) | 대기도(구개열, 경성 기관지경, 악안면) |

→ 감기 유무와 **발생 시점**은 PACU/ICU에서 호흡기 감시 강도를 정하는 직접 근거이므로
다른 군에 묻지 않고 독립 항목군으로 둔다.

## 반영 지점

| 단계 | 파일 | 무엇이 바뀌나 |
|---|---|---|
| 생성 | [pipeline_v3/prompt_utils.py](../pipeline_v3/prompt_utils.py) | SYSTEM/USER 프롬프트에 6군 명시 + "소견 있는 군은 절대 누락 금지 / 없는 군은 쓰지 말 것" |
| checklist 추출 | [pipeline_v3/eval_v3/checklist.py](../pipeline_v3/eval_v3/checklist.py) | `category` 값을 6군으로 강제, `normalize_category()`로 정규화, 케이스별 `required_categories` 기록 |
| Coverage 채점 | [pipeline_v3/eval_v3/metrics.py](../pipeline_v3/eval_v3/metrics.py) | judge 프롬프트에 항목별 소속 군 표시, `category_coverage` / `missed_categories` 산출 |

기존 gold checklist JSON은 재추출 없이도 읽힌다 — `LEGACY_CATEGORY_MAP`이 v1/v2 category
(`airway`, `hemodynamics`, `bleeding_transfusion` 등)를 6군으로 매핑한다.

## 산출 지표

`parse_coverage()`가 케이스마다 아래를 낸다.

```json
"category_coverage": {
  "airway_management": {"n": 2, "score": 0.75, "missed": [],   "label": "기도관리"},
  "transfusion_fluid": {"n": 1, "score": 1.0,  "missed": [],   "label": "수혈·수액"},
  "uri_status":        {"n": 1, "score": 0.0,  "missed": ["c4"], "label": "감기 유무"}
},
"missed_categories": ["uri_status"]
```

`missed_categories`(그 군의 항목을 **전부** 놓친 경우)는 케이스 note에
`필수 항목군 전부 누락: 감기 유무` 형태로 남는다. composite 점수 자체는 낮추지 않는다 —
안전게이트(`missed_abnormal`)와 달리 이건 진단용 지표이고, 게이트로 승격할지는
전문의 검수 결과를 보고 정한다.
