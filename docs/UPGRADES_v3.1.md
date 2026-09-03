# v3.1 고도화 내역

기간: 2026-08-11 ~ 2026-08-24 · 대상 커밋: `68ed347` · `5136ac2` · `79b1ff6` · `1f4f4a4`

v3.1은 "돌아가는 파이프라인"을 **임상적으로 신뢰할 수 있는 근거 기반 파이프라인**으로
끌어올린 릴리스다. 핵심은 세 가지 — (1) 모든 바이탈 임계값을 교과서 전거로 재정립하고,
(2) 이상 소견을 "몇 회"에서 "얼마나 오래·얼마나 벗어남"으로 정량화했으며,
(3) 필수 항목군·교수님 재검수 GT로 평가 기준을 실제 임상에 맞췄다. 여기에 프로덕션 버그 수정과
학습 안정성 확보가 붙는다.

---

## 1. 바이탈 임계값 교과서 근거 재정립 `5136ac2`

전거 문서: **[THRESHOLDS.md](THRESHOLDS.md)** (Smith's Anesthesia 2021 / Miller's Anesthesia 2024 표·페이지 단위)
결과 요약: **[VITAL_SUMMARY_RULES.md](../VITAL_SUMMARY_RULES.md) §4**

| 항목 | v1 (출처 불명) | v3.1 (교과서 근거) |
|------|----------------|--------------------|
| **HR** | infant 100/160, preschool 80/140 … | Smith Table 18.1 원 9구간 + Table 57.3(PALS) 2-tier |
| **SBP/DBP 고혈압** | 임의 상수 | Table 18.2/18.3 연령별 95th %ile |
| **저혈압** | 임의 상수 | PALS 기준 (신생아 <60 / 영아 <70 / 1–10세 70+2×나이 / >10세 90) |
| **DBP 하한·Ppeak** | 판정함 | **판정 제거** — 두 교과서에 소아 기준 없음 (출처 없는 값 삭제) |
| **QTc** | 450/460 단일 | 480 / 440·470 **2-tier** |
| **신생아 연령군** | 없음 | **신설** (SBP<60, QTc 상한 470) |

**왜 중요한가**: v1 임계값은 정상 영아를 빈맥으로, 정상 학령전아를 서맥으로 오표기하고 있었다.
QTc는 단일 기준이라 임상적으로 무의미한 "연장"을 대량 생성해 gold checklist를 오염시켰다.

---

## 2. 바이탈 이상 소견 정량화 `5136ac2`

이상 소견을 **2-tier + 지속시간 + 편차**로 표기하도록 전면 개편했다.

- **2-tier**: `[유의]` = 소생·개입 기준 초과(임상적으로 유의) / `[유의]` 없음 = 연령별 정상범위 이탈(통계적 이상).
  두 tier는 배타적으로 카운트해 이중계수를 막는다.
- **지속시간**: 모든 이벤트에 최장 episode·총 지속시간을 병기 (Miller Ch.4의 *time below threshold* 방식).
- **편차**: 기준 대비 최저/최고값 차이 병기.

예: `[유의]저혈압(<70): 4회 20분(최장 15분), 최저 55 (기준-15)`

> 생성 프롬프트·brevity judge·gen_pairs 루브릭을 **함께** 수정한 점이 핵심이다.
> 안 고치면 coverage와 brevity가 서로 반대로 당기고, DPO가 정량화 표현을 오히려 학습에서 빼버린다.

---

## 3. 필수 항목군 6개 도입 `5136ac2`

정의 문서: **[REQUIRED_CATEGORIES.md](REQUIRED_CATEGORIES.md)**

기저질환·약물 / 기도관리 / 수술 중 이벤트 및 처치 / 수혈·수액 / 수술 전 검사이상 / 감기 유무
— **조건부 필수**(소견 있는 군만)로 정의.

- 생성 프롬프트 · checklist 추출 · coverage 채점 **3곳**에 일관 반영.
- `LEGACY_CATEGORY_MAP`으로 기존 checklist JSON 하위호환 유지.

---

## 4. 교수님 재검수 GT 반영 `5136ac2`

- `scripts/import_khs_review.py`: 검수본 → 최종 GT 산출 (병합 방침 4종).
- `scripts/build_gold_override.py` → `khs_gold_override.json` (PHI, gitignored).
- `load_khs_gold()`가 기존 c10 위에 덮어쓴다. 파일 없으면 기존 동작 그대로.

---

## 5. 프로덕션 버그 수정 `5136ac2`

세 건 모두 "코드는 맞는데 프로덕션에서 조용히 빠지던" 종류다.

1. **UO 속도 5배 과대평가** — v1은 `rate = total / (weight × 기록건수 / 60)`, 즉 *기록 1건=1분*을
   가정했다. 실제 기록 간격이 5분 안팎이라 속도가 ~5배 부풀어 **핍뇨(<0.5 mL/kg/hr)가 전혀
   검출되지 않았다.** → 실제 타임스탬프 경과시간으로 수정.
2. **`build_vital_map`이 `weight_kg`를 안 넘김** — EBL의 EBV 대비 %와 핍뇨 판정이 프로덕션에서
   통째로 빠지고 있었다. → 나이/체중 매핑 배선 + 매핑 실패 시 명시적 경고.
3. **EBL 절대 mL의 무의미성** — 소아에서 5 kg 영아의 100 mL ≈ EBV의 24%. EBV 대비 %를
   병기하도록 수정 (Smith Table 21.6).

---

## 6. 학습 전멸 원인 수정 — CUDA 스트림 race `1f4f4a4`

260824 run에서 llama·qwen SFT 4변형이 전부 죽어 21변형 중 13종만 산출됐다.
OOM이 아니라 **비동기 CUDA 스트림 race**(loss 계산 중 unspecified launch failure)였고,
대응 env(`CUDA_LAUNCH_BLOCKING=1`)가 `GEMMA4_BASES`에만 걸려 있어 gemma4만 7변형 전부 살아남았다.

- → **전 모델로 확대** 적용. 끄려면 `HANDOVER_NO_LAUNCH_BLOCKING=1` (속도↔안정성 트레이드오프).

---

## 7. 측정오류(아티팩트) 좁은 배제 `1f4f4a4`

생성 인계문에 `T1 최저 20.3°C`, `SpO2 42%`가 실려 나왔다. 하지만 **이 코호트는 소아 심장수술을
포함**해 DHCA 저체온 18~20°C와 청색성 심질환 SpO2 40~70%가 **실제 값**이다.
임상적으로 드문 값을 지우면 정작 인계해야 할 소견이 사라진다.

- → 물리적으로 불가능하거나 센서 이탈로만 설명되는 값만 **좁게** 배제
  (`HR>300`, `SpO2>100`, `T1<14 또는 >45`, `BP<5` 등). 애매한 값은 남긴다.
- 배제 건수는 `DROPPED_ARTIFACTS`에 남고 `rebuild_vital_map`이 출력한다.
- 정의: `utils/vital_thresholds.py`의 `PLAUSIBLE_RANGE`.

---

## 8. 프롬프트 토큰 예산 관리 `79b1ff6`

v3.1에서 필수 항목군·정량화 지시로 지시문이 커지자, few-shot이 붙는 `gen_pairs`에서
EMR을 전부 비워도 `MAX_PROMPT_TOKENS`(2560)를 넘겨 `PromptTruncationError`가 났다.

- SYSTEM/USER 프롬프트 압축 (6개 항목군은 SYSTEM에만 상세, USER는 한 줄 참조):
  지시문 **2641 → 1968자**. 학습·추론에서도 EMR 좌측절단이 덜 일어난다.
- few-shot 예시 EMR 길이 매직넘버 1500 → `PAIRGEN["fewshot_emr_chars"]=700`: few-shot **3360 → 1760자**.
- `scripts/check_prompt_budget.py`: 실제 tokenizer로 모델별 고정비용·EMR 여유를 **사전** 측정
  (몇 시간 돌린 뒤 죽는 걸 방지).

---

## 9. 검토·재실행 도구

| 도구 | 목적 | 커밋 |
|------|------|------|
| `export_gold_cases_full.py` | gold 케이스별 **완전 EMR**(캡 없음) md/xlsx 추출. 리포트 HTML의 2600자 캡·좌측절단과 달리 소스 pkl에서 `build_emr_text`로 재생성 | `68ed347` |
| `scripts/invalidate_v3.py` | 임계값 변경 시 캐시 무효화 | `5136ac2` |
| `scripts/rebuild_vital_map.py` | vital_summary_map 재생성 (`--skip_done`이 새 임계값을 조용히 무시하던 문제 대응) | `5136ac2`·`1f4f4a4` |
| **[RERUN_RUNBOOK.md](RERUN_RUNBOOK.md)** | 위 도구들로 재실험하는 절차서 | `5136ac2` |

> 주의: `export_gold_cases_full.py`가 만드는 완전 EMR과, 모델 실제 입력(앞을 좌측절단)은 서로
> 다르다. 전자는 **사람 검토용 원본**이다.

---

## 관련 문서

- **[THRESHOLDS.md](THRESHOLDS.md)** — 임계값 전거 (Smith/Miller 표·페이지)
- **[VITAL_SUMMARY_RULES.md](../VITAL_SUMMARY_RULES.md)** — 바이탈 요약 생성규칙 (v3.1 반영)
- **[REQUIRED_CATEGORIES.md](REQUIRED_CATEGORIES.md)** — 필수 항목군 6개 정의
- **[RERUN_RUNBOOK.md](RERUN_RUNBOOK.md)** — 재실험 절차
- **[PIPELINE_V3.md](../PIPELINE_V3.md)** · **[PROMPTS_V3.md](../PROMPTS_V3.md)** — 파이프라인·프롬프트 전체 레퍼런스
