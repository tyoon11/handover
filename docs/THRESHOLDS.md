# 바이탈 임계값 근거 (Threshold Evidence)

구현: [utils/vital_thresholds.py](../utils/vital_thresholds.py) · 적용: [utils/vital_summarizer.py](../utils/vital_summarizer.py)
규칙 문서: [VITAL_SUMMARY_RULES.md](../VITAL_SUMMARY_RULES.md)

## 참고 교과서

| 표기 | 문헌 | 로컬 파일 |
|---|---|---|
| **[S]** | Davis PJ, Cladis FP. *Smith's Anesthesia for Infants and Children*, 9th ed. Elsevier, 2021 | `docs/references/smiths_anesthesia_infants_children_2021.pdf` |
| **[M]** | Gropper MA (ed). *Miller's Anesthesia*, 10th ed. Elsevier, 2024 | `docs/references/millers_anesthesia_2024.pdf` |

> PDF 원본은 저작권 때문에 `.gitignore` 처리했다(`docs/references/`). 아래 페이지 번호는 **PDF 페이지**
> (인쇄 페이지가 아님) 기준이라 로컬 파일에서 그대로 열어 대조할 수 있다.

## 설계 원칙

1. **근거 없는 숫자는 넣지 않는다.** 교과서에서 못 찾은 항목(Ppeak, DBP 하한)은 임계값을
   만들지 않고 통계만 출력하거나 판정을 생략한다.
2. **2-tier 판정.** 인계문에서 중요한 것은 "참조범위를 벗어났다"가 아니라 "개입이 필요했다"다.
   - `정상범위 이탈` — 연령별 참조범위 밖 (통계적 이상)
   - `⚑` — 소생·개입 기준 초과 (임상적으로 유의)
   v1의 단일 임계값 때문에 QTc 연장·일시적 SpO2 저하가 대량 생성되어 gold checklist를
   오염시키던 문제([PROMPTS_V3.md](../PROMPTS_V3.md) L235·L249)를 이 분리로 해결한다.
3. **두 tier는 서로 배타적으로 센다.** 유의 기준을 넘은 값은 "정상범위 이탈"에 이중계수하지 않는다.
4. **횟수만으로 보고하지 않는다.** 모든 이벤트는 `횟수 / 지속시간(최장 episode) / 기준 대비 편차`를
   함께 낸다 — 15분 연속 저혈압과 5분씩 흩어진 저혈압은 임상적 의미가 다르다.
   [M] Ch.4(PDF p.226)가 술중저혈압을 *"time or fraction of the case below threshold"* 로
   정량화하라고 권하는 방식이다. 지속시간은 이산 표본을 구간으로 환산한 추정치이며
   산정 규칙은 [VITAL_SUMMARY_RULES.md](../VITAL_SUMMARY_RULES.md) §2에 있다.

---

## 1. 심박수 (HR)

### 정상 참조범위 — [S] Ch.18 **Table 18.1** "Normal Resting Heart Rates of Infants and Children" (PDF p.380)

원표의 mean ± 2 SD 범위를 **연령 구간 그대로** 옮겼다.

| 연령 | mean | range (±2 SD) |
|---|---|---|
| 0–24 hr | 119 | 94–145 |
| 1–7 days | 133 | 100–175 |
| 8–30 days | 163 | 115–190 |
| 1–3 months | 152 | 124–190 |
| 3–12 months | 140 | 111–179 |
| 1–3 years | 126 | 98–163 |
| 3–5 years | 98 | 65–132 |
| 5–8 years | 96 | 70–115 |
| 8–16 years | 77 | 55–105 |

### 유의 기준 — [S] Ch.57 **Table 57.3** "Adequate Vital Signs for Children" (AHA/ILCOR PALS, PDF p.1400)

| 연령 | 서맥 < | 빈맥 > |
|---|---|---|
| term neonate | 60 | 220 |
| infant | 60 | 190 |
| child (1–10 y) | 60 | 180 |
| > 10 y | 60 | 150 |

> 원표 각주: 서맥 기준은 *흉부압박을 고려해야 하는* 심박수이고, 빈맥 기준은 *증상이 통상
> 발생하기 시작하는* 추정 심박수다.

### v1 대비 변경

| 연령군 | v1 (서맥/빈맥) | v3 정상범위 | 문제 |
|---|---|---|---|
| infant | 100 / 160 | 111–190 | **정상 영아(HR 165)를 빈맥으로 오탐** |
| toddler | 90 / 150 | 98–163 | 정상 유아(HR 155)를 빈맥으로 오탐 |
| preschool | 80 / 140 | 65–132 | **정상 학령전아(HR 70)를 서맥으로 오탐** |
| school | 70 / 130 | 70–115 (5–8y) | 상한이 느슨해 빈맥 누락 |
| adolescent | 60 / 110 | 55–105 | 대체로 일치 |

v1의 5구간 근사는 출처가 없었고 Table 18.1과 양방향으로 어긋났다.

---

## 2. 혈압

### 저혈압 (SBP) — [S] **Table 57.3** (PALS, PDF p.1400)

| 연령 | 저혈압 |
|---|---|
| term neonate | SBP < 60 |
| infant | SBP < 70 |
| child (1–10 y) | SBP < 70 + (2 × 나이) |
| > 10 y | SBP < 90 |

v1은 신생아 행이 없어 신생아에게 영아 기준(<70)을 적용하고 있었다 → 신생아 행 추가.

### 고혈압 (SBP/DBP) — [S] Ch.18 **Table 18.2 / 18.3** (NHLBI 4th Task Force, 신장 50th %ile, PDF p.380)

95th percentile 초과를 고혈압으로 본다. 원표가 남/녀 범위(예: `103–104`)로 주어져 있어
**상한**을 취해 과잉 flag를 줄였다.

| 나이 | SBP 95th | DBP 95th |
|---|---|---|
| 1 | 104 | 58 |
| 2 | 106 | 63 |
| 3 | 109 | 67 |
| 4 | 111 | 70 |
| 5 | 112 | 72 |
| 6 | 114 | 74 |
| 7 | 115 | 76 |
| 8 | 116 | 78 |
| 9 | 118 | 79 |
| 10 | 119 | 80 |
| 11 | 121 | 80 |
| 12 | 123 | 81 |

- **< 1세**: Table 18.2/18.3에 없는 구간 → 관례값 SBP 100 / DBP 65 사용 (근거 약함으로 명시).
- **≥ 13세**: AAP 2017 기준에 따라 성인값 130/80.
- v1의 DBP 상한(toddler 70)은 이 표의 1–2세 95th(58–63)보다 높아 고혈압을 놓치고 있었다.

### DBP 하한 — **판정하지 않음**

두 교과서 어디에도 소아 이완기 저혈압 수치 기준이 없다. PALS도 TAXI도 관류 지표로
SBP/MAP만 쓴다. v1은 `infant 30 / adolescent 45` 같은 출처 없는 값을 쓰고 있었으므로 제거했다.
DBP는 통계와 상한 초과만 보고한다.

### MAP 저혈압

소아 **술중** MAP 절대 기준표는 두 교과서에 없다 — [S] Ch.18(PDF p.381)과 Ch.21(PDF p.446)은
de Graaff 2016 MPOG 참조범위를 **그림(Fig. 21.7)으로만** 제시한다. 그래서:

```
map_hypotension(age) = min(1.5 × age + 40, 65)
```

- `1.5 × age + 40` — 널리 쓰이는 소아 관례식 (v1에서 그대로 계승; 교과서 직접 근거 아님)
- 상한 `65` — [M] Ch.4 (PDF p.226) 술중저혈압 절대 기준 **MAP < 65 mmHg**.
  16.7세에서 두 값이 만나므로 성인값으로 연속적으로 수렴한다.
- [M] Ch.4는 상대 기준 **baseline 대비 20% 하강**도 제시한다 → `map_relative_hypotension()`으로
  노출해 두었으나, 현재 vital 요약은 baseline MAP을 확보하지 못해 사용하지 않는다.
- 참고 원칙: [S] Ch.21 TAXI(PDF p.491) "Hemodynamically stable = MAP not < 2 SD below
  normal mean for age".

---

## 3. SpO2 — [S] Ch.57 (PDF p.1401)

ROSC 후 pulse oximetry 목표는 **94–99%**.

- `⚑ SpO2 < 90%` — 탈포화 (유의)
- `목표미달 90–93%` — 목표 이탈

v1의 warn 임계 95는 근거가 없었고 목표 하한 94와 1% 어긋나 있었다.

## 4. 체온 — [S] Ch.21 (PDF p.467), Ch.7 (PDF p.193)

> "The safe range for a child's core temperature is between 35.5°C and 37.5°C." — [S] Ch.21
> 술중 저체온의 정의는 core temperature < 36°C — [S] Ch.7

| 표기 | 기준 |
|---|---|
| `⚑ 저체온` | < 35.5 (안전범위 하한 미만) |
| `저체온` | < 36.0 (술중 저체온 정의) |
| `안전범위 초과` | > 37.5 |
| `⚑ 발열` | > 38.0 (MH 감별 필요 구간) |

체온만 소수 1자리로 출력한다 — 35.3과 35.8은 tier가 다르다.

## 5. QTc — [S] Ch.5 (PDF p.133), [M] (PDF p.1175)

> QT interval은 신생아 0.47 s에서 연장아 0.44 s로 변한다 (Bazett 보정) — [S] Ch.5
> QTc ≥ 480 ms (여) / ≥ 470 ms (남) = 연장 — [M]

| 표기 | 기준 |
|---|---|
| `정상상한 초과` | > 470 (신생아) / > 440 (그 외) |
| `⚑ 연장` | > 480 (성별 미상이므로 보수적으로 480 사용) |

v1의 단일 기준(450/460)은 출처가 불명확했고, 임상적으로 무의미한 "QTc 연장"을 대량
생성해 checklist를 오염시켰다. 2-tier로 분리하면 `⚑`만 인계 대상이 된다.

## 6. 소변량 (UO) — [M] Ch.24 (PDF p.544)

> "By consensus, oliguria is defined as a urine flow rate less than 0.5 mL/kg/h."

- `⚑ 핍뇨` — rate < 0.5 mL/kg/hr

**속도 계산을 고쳤다.** v1은 `rate = total / (weight × 기록건수 / 60)`, 즉 *기록 1건 = 1분*을
가정했다(VITAL_SUMMARY_RULES §7 B8). 실제 기록 간격은 5분 안팎이라 속도가 5배가량
과대평가되어 핍뇨가 전혀 검출되지 않았다. v3는 UO 행의 **실제 타임스탬프 경과 시간**을 쓴다
(UO 행이 1건뿐이면 케이스 전체 기록 구간으로 대체).

## 7. 실혈량 (EBL) — [S] Ch.21 Table 21.6 (PDF p.467), Ch.18 (PDF p.381)

절대 mL은 소아에서 의미가 없다 — 5 kg 영아의 100 mL는 순환혈액량의 약 24%다.
체중을 알면 **EBV 대비 %**를 함께 출력한다.

### 순환혈액량 추정 — Table 21.6

| 환자 | mL/kg (원표) | 사용값 |
|---|---|---|
| 미숙아 | 90–100 | (재태주수 정보 필요 — 미사용) |
| 만삭 신생아 | 80–90 | 85 |
| 3개월–1세 | 75–80 | 78 |
| 3–6세 | 70–75 | 73 (1–3세는 원표 공백 → 인접 구간 적용) |
| > 6세 | 65–70 | 68 |

### 판정 — [S] Ch.18 (PDF p.381)

동맥압 감시 적응증으로 제시된 기준을 그대로 쓴다.

- `⚑ 유의 실혈` — > 10% EBV (급성 실혈 >10% EBV)
- `⚑ 대량 실혈` — > 50% EBV (총 실혈 >50% EBV)

### 수혈 기준 (checklist 판정용, 요약문에서는 미사용)

- 제한적 수혈전략: Hb **7 g/dL** 미만에서 수혈 고려. 신생아기·청색성 심질환은 예외.
  수혈 후 목표 7–9.5 g/dL — [S] Ch.21 (PDF p.467–468), TAXI 결정도 (PDF p.491)
- MABL = EBV × (시작 Hct − 목표 Hct) / 시작 Hct — [S] Ch.21 (PDF p.467)

## 8. Ppeak — **임계값 없음**

두 교과서에 소아 일반마취용 최고기도압 기준이 없다. [S]에 있는 25 / 28 cmH₂O는
**선천성 횡격막 탈장 신생아**의 수술 연기·ECMO 적응증 맥락 전용이라 일반화할 수 없다.
→ 통계만 출력하고 이벤트 판정은 하지 않는다.

---

## 미해결 / 후속

- **MAP 소아 절대 기준**: de Graaff 2016 (*Anesthesiology* 125:904–913) 원논문의 수치표를
  확보하면 `map_hypotension()`의 관례식을 교체할 수 있다. 현재는 관례식임을 코드에 명시.
- **영아 고혈압 기준**: Table 18.2/18.3이 1세부터라 <1세는 관례값(100/65)을 쓴다.
- **미숙아**: Table 21.6의 미숙아 EBV(90–100 mL/kg)와 [S] Table 18.5 미숙아 혈압 참조범위는
  재태주수·일령 정보가 EMR에서 확보되면 적용 가능.
- **baseline 대비 상대 저혈압**: [M] Ch.4의 20% 하강 기준은 마취 유도 전 baseline MAP이
  vital 시계열에서 분리 가능해지면 도입.
