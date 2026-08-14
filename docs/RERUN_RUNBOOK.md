# v3.1 재실행 런북

임계값 개정([THRESHOLDS.md](THRESHOLDS.md)) · 필수 항목군 도입([REQUIRED_CATEGORIES.md](REQUIRED_CATEGORIES.md)) ·
바이탈 이벤트 정량화 · 교수님 재검수 GT 반영 이후 전체 재실행 절차.

## 왜 그냥 `--skip_done` 으로 돌리면 안 되나

`--skip_done` 은 "산출물이 있으면 건너뛴다"이다. 이번 변경은 **입력과 프롬프트가 전부 바뀌었으므로**
그대로 돌리면 옛 산출물을 재사용해 조용히 잘못된 결과가 나온다. 특히:

| 산출물 | 왜 무효인가 | 재사용 시 증상 |
|---|---|---|
| `vital_summary_map.pkl` | 임계값·이벤트 포맷 전면 개정 | **새 임계값이 전혀 반영 안 됨** (v1 preprocess가 파일 있으면 그냥 로드) |
| `gold_checklist_v3.json` | category 6항목군 + 재검수 GT | coverage가 옛 기준으로 채점됨 |
| `fewshot_bank_v3.json` | 생성 프롬프트 변경 | few-shot 예시가 옛 스타일 |
| `calibration_v3.csv` | brevity 루브릭 변경 | judge 보정값 불일치 |
| `pairs/` | 생성·쌍생성 루브릭 변경 | 옛 프롬프트로 만든 학습 타깃 |
| `sft/`, `rlaif/` | 옛 프롬프트로 학습됨 | 학습·추론 프롬프트 불일치 |
| `inference/`, `evaluation/`, `report/` | 위 전부의 하류 | — |

---

## 0. 사전 확인

```bash
cd <repo>
git pull
bash scripts/install_hooks.sh          # PHI 가드
export HANDOVER_BASE_DIR=/home/coder/workspace/data/handover
export HANDOVER_MODEL_DIR=/home/coder/workspace/data/local_models
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6   # vLLM 폴백 방지
```

## 1. 교수님 재검수 GT 반영

### 병합 방침 (연구책임자 확정, 2026-08-14)

| # | 검토 의견('GT 수정·보완 사항') | 최종 GT |
|---|---|---|
| 1 | 비어 있음 | 기존 전문의 GT 유지 |
| 2 | 기존 GT를 **보완**하는 내용 | 기존 GT + 보완 내용 **병합** |
| 3 | 통째로 다시 쓴 문안 | 그것으로 **교체** |
| 4 | 기존 GT를 **부정**하는 문안 | 검토 의견 우선 → **교체** |

2와 4는 문장만으로 기계 판정이 안 된다. 같은 "수술 중 특이사항 없었음"이라도 기존 GT가
기저질환·기도 정보면 **보완**이고, 기존 GT가 수술 중 처치 기록이면 **부정**이다.
그래서 기본은 병합이고, 합친 결과에 실질 소견과 '없음' 주장이 공존하면 `merged_conflict`
로 표시해 확정을 보류한다. `--prefer-review-on-conflict` 를 주면 방침 4로 일괄 교체한다.

```bash
# ① 추출 + 분류 + 최종 GT 산출 + 확정용 워크시트
python scripts/import_khs_review.py output_케이스별_khs_reviewed.xlsx \
    --prefer-review-on-conflict --show
#   → data/preprocessed/khs_review_<stamp>.json
#   → data/gold_sampled/khs_gt_merge_worksheet.xlsx

# ② 워크시트에서 '최종 GT' 확인 — 고칠 게 있으면 직접 수정.
#    '확정?'이 Y 인 행만 override 로 나간다. (분류가 conflict 인 행은 빨간색)

# ③ 확정본 → override JSON
python scripts/build_gold_override.py
#   → data/preprocessed/khs_gold_override.json   ({sid: 확정 GT})
```

2026-08-14 검수본 기준 결과: `replaced 6 · replaced_conflict 2 · merged 5 · kept 9` = 22건.

> **워크시트가 최종 GT의 단일 출처다.** 실제 수술ID는 PHI라 repo 코드에 둘 수 없으므로
> (pre-commit 훅이 `1005xxxxx` 패턴을 차단) 케이스별 예외는 전부 이 gitignored 파일에만 있다.

`load_khs_gold()` 가 이 override를 기존 c10 gold 위에 덮어쓴다. 파일이 없으면 옛 gold 그대로
동작하므로, 확정 전에도 나머지 단계는 진행할 수 있다.

> 로컬에서 만들었다면 서버 `$HANDOVER_BASE_DIR/data/preprocessed/` 로 복사해야 한다.

## 2. 무효 산출물 정리

```bash
python scripts/invalidate_v3.py                       # dry-run — 무엇이 지워질지 확인
python scripts/invalidate_v3.py --apply --archive     # .bak_<stamp>/ 로 보관 (권장)
```

`--archive` 없이 `--apply` 하면 바로 삭제된다. 이전 run과 수치를 비교할 계획이면 `--archive`.
일부만 지우려면 `--scope vital,checklist` 처럼 지정한다.

## 3. 바이탈 요약 재생성 (가장 먼저)

```bash
python scripts/rebuild_vital_map.py                   # 미리보기만
python scripts/rebuild_vital_map.py --apply
```

출력 끝의 자가진단을 확인한다:

- `체중 컬럼 사용: (...)` — 안 뜨면 EBL의 EBV 대비 %와 UO 핍뇨 판정이 **생략**된다.
- `나이 매핑 n/N` — 미매핑 케이스는 기본값 5세로 판정되어 연령별 임계값이 틀어진다.
- `⚑ 유의 이벤트 포함 n/N` — 0이면 임계값이 반영되지 않은 것이니 중단하고 원인부터 확인.
- `지속시간 표기 포함 n/N` — 0이면 정량화가 반영되지 않은 것.

## 4. 준비물 재생성

```bash
python -m pipeline_v3.make_fewshot_bank        --gpus 0,1,2,3
python -m pipeline_v3.build_gold_checklist_v3  --gpus 0,1,2,3 --accept
python -m pipeline_v3.eval_v3.calibrate        --gpus 0,1,2,3
```

`build_gold_checklist_v3` 로그에서 확인할 것:

- `교수님 재검수 GT 적용 n건 (override)` — 1단계를 했다면 반드시 떠야 한다.
- `검수상태: reviewed=0` 경고 — `--accept` 는 `accepted_without_review` 만 세운다.
  **정식 gold 로 보고하려면 전문의가 JSON을 직접 검수하고 `reviewed:true` 로 바꿔야 한다.**

## 5. 학습 → 추론 → dev 평가

```bash
python -m pipeline_v3.gen_pairs --split sft --models llama qwen --gpus 0,1,2,3
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 --skip_done
```

여기서의 `--skip_done` 은 2단계로 옛 산출물을 이미 지웠으므로 **중단 후 이어달리기용**이다.

## 6. 최종 gold (dev로 선택이 끝난 뒤에만)

```bash
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 \
    --skip_done --final

# 다린(기존 연구) 병기 리포트
python reinfer_darin_on_v3sids.py --gpus 0,1 --split gold --skip_done
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 --gpus_per_job 2 \
    --skip_done --final --include_source --include_darin
```

## 7. 검토본 재수출

```bash
python export_gold_cases_full.py
```

Threshold 시트/마크다운은 이제 `utils/vital_thresholds.py` 에서 값을 직접 뽑으므로
임계값과 검토 자료가 어긋나지 않는다.

---

## 결과 해석 시 주의

- **v3.0 수치와 직접 비교 금지.** 임계값·프롬프트·gold가 동시에 바뀌었으므로 composite 변화는
  모델 개선이 아니라 측정 기준 변화를 포함한다. 비교하려면 v3.0 산출물을 `--archive` 로 남겨두고
  "기준 변경 전/후"로 따로 보고한다.
- **gold 22건은 최종 test 셋(봉인).** dev로 변형 선택이 끝난 뒤 1회만 개봉한다.
- `missed_categories` 는 진단 지표이고 composite을 낮추지 않는다. 안전게이트로 승격할지는
  전문의 검수 후 결정한다.
