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

## 바이탈 threshold 출처 (v1에서 유지)

| 항목       | 기준                      | 출처                    |
| ---------- | ------------------------- | ----------------------- |
| HR         | 연령별 서맥/빈맥          | 대한소아청소년과학회    |
| SBP 저혈압 | 70+2×age(yr)             | PALS/ATLS               |
| SBP 고혈압 | 연령별 95th pct           | 대한소아청소년학회 2008 |
| MBP 저혈압 | 1.5×age(yr)+40           | PMID 17273118           |
| DBP 저/고  | 연령별 5th/90th pct       | AAP 4th Report          |
| QTc        | <8세 >450ms, ≥8세 >460ms | PMID 16482041           |
| SpO2       | <95% / <90%               | —                      |
| T1         | <35.5°C / >38.0°C       | —                      |

## IRB / DRB

- IRB: E-2601-138-1712 (텍스트+바이탈 멀티모달)
- DRB: DRB-E(I)-2026-02-04
- ※ 연구자 명단 수정 필요 (IRB·DRB 모두)
