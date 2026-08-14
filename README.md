# 소아수술실 인계요약지 생성 파이프라인

수술 후 OR→PACU/ICU 초간결 인계문(1~5문장 한국어) 생성 — SFT/RLAIF(DPO·SimPO) 비교 연구.

> **현재 유효 코드는 `pipeline_v3/` 하나다.**
> v1(`pipeline/`, `config.py`)·v2(`config_v2.py`, `pipeline/eval_v2/`)는 legacy 봉인 —
> 결함 목록과 재설계 근거는 `CODE_REVIEW_V3_PROPOSAL.md`, v3 프로토콜은 `PIPELINE_V3.md` 참고.
> **v1 sum_score 기반 순위는 어떤 보고에도 인용 금지** (평가셋 유출·judge 순환·절단 버그 중첩).

## 시작하기 (폐쇄망 서버)

```bash
git pull
bash scripts/install_hooks.sh        # pre-commit PHI 가드 (필수)
pip install -r requirements.txt      # 내부 미러 기준

# 경로가 기본값과 다르면:
export HANDOVER_BASE_DIR=/home/coder/workspace/data/handover
export HANDOVER_MODEL_DIR=/home/coder/workspace/data/local_models

# vLLM GLIBCXX/zmq 이슈 방지 (세션당 1회 — 안 하면 vLLM이 조용히 HF로 폴백해 매우 느려짐)
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

필요 데이터(서버 `DATA_DIR`에 있어야 함): v1과 동일한 pkl들
(`gold_sampled_251008.pkl`, `jsft_251008.pkl`, `selfjudge_251008.pkl`, `rlhf_251008.pkl`,
`vital_summary_map.pkl`) + `gold_sampled/인계요약지_gold_sampled_251002_KHS.xlsx`

+ `gold_sampled/인계요약지_SY.xlsx` + **`preprocessed/khs_gold_remap.json`**
  (수술ID remap — PHI라서 repo에 없음; 기존 작업 PC의 `data/preprocessed/`에서 복사).

> **v3.1 재실행 중이라면 [docs/RERUN_RUNBOOK.md](docs/RERUN_RUNBOOK.md) 를 따르세요.**
> 임계값·프롬프트·gold가 모두 바뀌어서 아래 순서를 `--skip_done` 으로 그냥 돌리면
> 옛 산출물(특히 `vital_summary_map.pkl`)을 재사용해 조용히 틀린 결과가 나옵니다.

## 실행 순서 (요약 — 상세는 PIPELINE_V3.md §6)

```bash
# 1회 준비물
python -m pipeline_v3.make_fewshot_bank        --gpus 0,1,2,3
python -m pipeline_v3.build_gold_checklist_v3  --gpus 0,1,2,3
python -m pipeline_v3.eval_v3.calibrate        --gpus 0,1,2,3

# SFT 타깃 생성 (1회 공유)
python -m pipeline_v3.gen_pairs --split sft --models llama qwen --gpus 0,1,2,3

# 학습 → on-policy 쌍 → RLAIF → 추론 → dev 평가
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 --skip_done

# 최종 1회 (gold 22 개봉 — dev로 선택 끝난 뒤에만)
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 --skip_done --final

# 다린(기존 연구) 병기 최종 gold 리포트: 다린 재추론(gold sid) → --final 리포트에 병기
#   경로는 config 기본값(HANDOVER_DARIN_DIR)에서 자동 — 필요 시 --out_root/--darin_root 로 override
python reinfer_darin_on_v3sids.py --gpus 0,1 --split gold --skip_done
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 \
    --skip_done --final --include_source --include_darin
#   → outputs_v3/<run>/report/results_gold_v3_source_darin.html (EMR·GT·v3·다린 병기, PHI 포함=외부공유 금지)
```

## 실험 매트릭스 (모델당 7변형)

raw / rlaif_dpo / rlaif_simpo / sft_1ep / sft_3ep / sft_1ep_dpo / sft_3ep_dpo
— 결과는 `outputs_v3/<run>/report/results_{dev,gold}_v3.md`
(3축+CI, raw 대비 permutation p(Holm), judge 일치도, 제외 케이스 표).

## 평가 요약 (v3)

| 축                 | 정의                                       | judge                    |
| ------------------ | ------------------------------------------ | ------------------------ |
| coverage (0.5)     | 전문의 gold checklist recall (macro+micro) | gemma4_31b + qwen35 교차 |
| faithfulness (0.3) | claim의 EMR entailment (주입 방어 구분자)  | 〃                       |
| brevity (0.2)      | 과설명/행정 노이즈 감점                    | 〃                       |

- 이상소견 케이스에 "특이사항 없음" → composite 0 (안전게이트, `gate=missed_abnormal`)
- judge 실패/gold 부재는 점수가 아니라 **제외**로 집계 (유효비율 <80%면 평가 실패 처리)
- 선호쌍 생성 judge는 prometheus — 평가 judge와 분리(순환 금지)

## 보안 / PHI (필독)

- 환자 데이터가 들어가는 확장자(`*.pkl, *.xlsx, *.html, *.jsonl, *.log`)와 `data/`,
  `outputs*/`는 gitignore + pre-commit 훅이 이중 차단한다. **훅 설치 필수.**
- 실제 수술ID는 코드/문서에 쓰지 않는다 — 필요하면 `data/` 밑 JSON(예: `khs_gold_remap.json`).
- `utils/download_models.py`의 TLS 우회는 `HANDOVER_INSECURE_SSL=1`일 때만 동작.
- **남은 사람 작업**: ① GitHub PAT(.env) 즉시 revoke 후 파일 삭제,
  ② repo private 전환 또는 `git filter-repo`로 과거 이력의 수술ID 스크럽
  (이력에 이미 push된 P0-3 항목은 새 커밋으로는 지워지지 않는다).

## 바이탈 threshold (v3.1 — 교과서 근거로 전면 재설정)

전거는 **Smith's Anesthesia for Infants and Children 9e (2021)** 와
**Miller's Anesthesia 10e (2024)**. 표·페이지 단위 근거와 v1 대비 변경 내역은
**[docs/THRESHOLDS.md](docs/THRESHOLDS.md)**, 코드 단일 출처는
[utils/vital_thresholds.py](utils/vital_thresholds.py).

판정은 **2-tier** — `⚑`(소생·개입 기준 초과 = 임상적 유의) / 표시 없음(연령별 참조범위 이탈).

| 항목 | 유의(`⚑`) 기준 | 정상범위 기준 | 전거 |
|---|---|---|---|
| HR | 서맥 <60 · 빈맥 >220/190/180/150 | 연령별 mean±2SD (9구간) | Smith Table 57.3 / Table 18.1 |
| SBP | 신생아<60·영아<70·1–10세<70+2×age·>10세<90 | 고혈압 = 95th pct 초과 | Smith Table 57.3 / Table 18.2 |
| MBP | `min(1.5×age+40, 65)` 미만 | — | 관례식 + Miller Ch.4 (MAP<65) |
| DBP | — (하한 기준 문헌 없음 → 판정 안 함) | 고혈압 = 95th pct 초과 | Smith Table 18.3 |
| SpO2 | <90% | 목표미달 90–93% (목표 94–99%) | Smith Ch.57 |
| T1 | <35.5°C · >38.0°C | 저체온 <36.0 · 안전범위 초과 >37.5 | Smith Ch.21 / Ch.7 |
| QTc | >480 ms | 정상상한 초과 >470(신생아)/>440 | Miller / Smith Ch.5 |
| UO | 핍뇨 <0.5 mL/kg/hr | — | Miller Ch.24 |
| EBL | >10% EBV · >50% EBV | — | Smith Table 21.6 / Ch.18 |
| Ppeak | — (소아 일반마취 기준 문헌 없음 → 판정 안 함) | — | — |

> 교과서 PDF는 `docs/references/`에 두되 저작권 때문에 gitignore 처리했다.

## 인계문 필수 항목군

인계문이 반드시 다뤄야 할 6개 항목군 — 정의·근거는
**[docs/REQUIRED_CATEGORIES.md](docs/REQUIRED_CATEGORIES.md)**, 코드 단일 출처는
[pipeline_v3/required_categories.py](pipeline_v3/required_categories.py).

기저질환·약물 / 기도관리 / 수술 중 이벤트 및 처치 / 수혈·수액 / 수술 전 검사이상 / 감기 유무

**조건부 필수** — EMR에 소견이 있는 군은 반드시 전달하고, 없는 군은 "없음"조차 쓰지 않는다
(brevity 축과 `특이사항 없음` 규칙 보호). 생성 프롬프트·checklist 추출·coverage 채점 세 곳에
동시에 반영돼 있고, coverage는 항목군별 recall(`category_coverage` / `missed_categories`)을 함께 낸다.

## IRB / DRB

- IRB: E-2601-138-1712 (텍스트+바이탈 멀티모달)
- DRB: DRB-E(I)-2026-02-04
- ※ 연구자 명단 수정 필요 (IRB·DRB 모두)
