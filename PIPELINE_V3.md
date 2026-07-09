# 파이프라인 v3 — 논문급 재실험 프로토콜

> CODE_REVIEW_V3_PROPOSAL.md 로드맵(Phase 0~4)의 구현.
> v1(`config.py`, `pipeline/`)·v2(`config_v2.py`, `pipeline/eval_v2/`)는 **legacy로 봉인** — 수정 금지, 보고 금지.
> v3 코드는 전부 `pipeline_v3/` 패키지. 폐쇄망 서버에서 `git pull` 후 아래 순서대로 실행.

---

## 0. 재설계 3원칙

1. **실패는 크게** — 생성/파싱/매칭/추출의 어떤 실패도 0.0/1.0 점수로 변환하지 않는다.
   `None` + 플래그 + '제외' 집계. 유효 케이스 비율 < 80%면 평가 자체가 exit 1.
2. **한 벌만** — 프롬프트 빌드·절단·chat template·모델 키·태그 파싱은 `pipeline_v3` 안에 단 한 벌.
3. **순환 금지** — 데이터 생성 judge(**prometheus**) ≠ 평가 judge(**gemma4_31b + qwen35**)
   ≠ 평가 대상 계열. 평가 대상과 같은 family의 judge 점수는 공식 점수에서 제외.
   gold checklist는 전문의 검수(`reviewed:true`) 전엔 항상 "잠정" 표기.

---

## 1. 데이터 설계 (유출 차단 — T1/T9/M1)

| split | 크기 | 역할 | 비고 |
|---|---|---|---|
| GOLD | 22 | **최종 test — 봉인** | `--final`에서만 추론/평가. 모델·하이퍼 선택 금지 |
| SFT_SRC (jsft) | 2736 | SFT 타깃 생성 | `gen_pairs --split sft` |
| SELFJUDGE_SRC | 2736 | **DPO 선호쌍 생성** | SFT와 분리(T9). on-policy 후보 |
| RLHF | 150 | few-shot pool(40) + **dev set(110)** | seed=42 고정 분할 |

- **모든 split 로드는 `data_splits.load_splits()`** — gold와의 교집합이 있으면 `LeakageError`로 즉시 중단.
- few-shot 예시(T1): gold가 아니라 **held-out 40건에 대해 gemma4_31b가 zero-shot 작성한
  초안**(`fewshot_bank_v3.json`, 전문의 수정 가능). 행마다 `random_state=row_idx` 로테이션 —
  특정 케이스 스타일이 전 학습셋에 박히지 않음.
- run마다 `split_manifest.json`에 split 지문(sid sha1 해시, PHI 없음) 기록.

## 2. 선호쌍/SFT 타깃 생성 (`gen_pairs.py` — T2/T3/T4/T8)

- 후보: 모델당 greedy 1 + temperature(0.8) 샘플 3 → `clean_v3` 통과분만, 정규화 dedup.
  생성 예외/붕괴는 **후보에서 제외** (v1처럼 "특이사항 없음"으로 치환하지 않음).
- 채점: **prometheus-8x7b, reference-free** — 남의 환자 기준답안 폐기(T2).
  해당 케이스 EMR+vital을 근거로 두 루브릭:
  - `coverage`(가중 2배): EMR의 조치가능 이상소견을 모두 언급하면 5,
    **이상소견이 있는데 '특이사항 없음'이면 1** (T3의 "없음 최적해" 붕괴 차단)
  - `fidelity+brevity`: EMR 근거 없는 창작/장황이면 1
- judge 입력: 8192 예산, **EMR-only 좌측절단** (`fit_text_prompt`) — 응답/루브릭은 절대 안 잘림(T4).
- 점수 파싱: `[RESULT] n`만 신뢰. 실패 → 해당 후보 drop (준랜덤 점수 주입 금지).
- 쌍 선정: chosen=최고점, rejected=최저점, **chosen≠rejected + margin ≥ 2.0(만점 15)** 강제(T8).
  탈락 사유별 통계는 `pairs_*_meta.json`, 후보 전 점수는 `*_audit.jsonl`.

## 3. 학습 (T5/T6/T9)

- 토큰 예산 한 벌: `MAX_PROMPT_TOKENS=2560`, `MAX_SEQ_TOKENS=3072` —
  SFT collator·**DPO(`max_prompt_length=2560`)**·SimPO 모두 동일(T5: TRL 기본 512 절단 제거).
- 프롬프트 절단: `prompt_utils.fit_user_content` — EMR 좌측만 자르고
  `### OUTPUT` 생존을 assert. 학습·추론·쌍생성 동일 규칙(T6).
- SFT: `pairs_sft.pkl`의 chosen을 타깃으로. judge-augmented(A/B) 샘플은 기본 유지
  (`--no_judge_aug`로 ablation 가능).
- DPO/SimPO: **정책별 on-policy 쌍** —
  `rlaif_*`는 base 정책 쌍(`pairs_dpo_{model}_raw.pkl`),
  `sft_Xep_dpo`는 해당 SFT 정책 쌍(`pairs_dpo_{model}_sftXep.pkl`). SFT 데이터 재사용 금지(T9).

## 4. 추론 (`inference.py` — T6/B2/B3)

- vLLM 우선(엔진·max_model_len=4096 통일), HF fallback은 left-truncation/left-padding.
- 레코드에 `engine` 필드 기록 — **최종 비교는 전 변형 동일 엔진** (fallback 섞이면 경고).
- LoRA merge 캐시 키 = adapter **내용 해시** — 재학습하면 자동 무효화(B3).
- gold split은 `--allow_gold` 없이는 실행 거부(봉인).

## 5. 평가 (`eval_v3/` — E2~E8)

### 5.1 3축 + 자연 게이트
- coverage(0.5) / faithfulness(0.3) / brevity(0.2). 가중치는 calibration 회귀로 검증.
- **이상소견 케이스 + "특이사항 없음"** → coverage=0 **그리고 faithfulness=0**
  (그 주장 자체가 EMR과 모순) → composite=0.0, `gate="missed_abnormal"` (E2 — 상수 캡 없이 게이트 구현).
- 생성 붕괴(빈/반복/garbage/leak) → 0.0, `gate="degenerate"`.
- normal 케이스 + 실질 보고 → coverage 정의불가(None), faith+brev 재정규화.

### 5.2 실패는 점수가 아니다 (E3)
| 상황 | v2 (버그) | v3 |
|---|---|---|
| judge JSON 실패 | coverage 0.0 | 축=None, 케이스 '제외' + `judge_failed` |
| coverage verdict에 항목 누락 | 누락항목=no 처리 | judge 실패로 간주, 제외 (id 대조검증, E7) |
| sid 매칭 실패 | normal-case 간주 → 1.0 가능 | **하드 에러** (파이프라인 중단) |
| checklist 추출 실패 | items=[] → coverage 1.0 | `gold_llm_failed` → 케이스 제외 |
| gold 없음(no_gold) | faith+brev로 채점 | 케이스 제외 (별도 표) |

유효 케이스 비율 < `MIN_VALID_CASE_RATIO(0.8)` → 평가 exit 1 (성적이 아니라 인프라 문제).

### 5.3 교차 judge (T7)
- 모든 출력을 gemma4_31b·qwen35 **둘 다**로 채점.
- 공식 점수 = **평가 대상과 다른 family의 judge만** 평균
  (gemma 계열 모델 → qwen35만, qwen 계열 → gemma4_31b만, llama → 둘 평균).
- 두 judge 일치도(Spearman/MAD)를 변형별로 항상 보고.
- 선호쌍 judge는 prometheus(mixtral 계열) — 학습 신호와 평가 지표의 judge가 다르므로
  v1의 "평가 judge에 직접 최적화(reward hacking)" 해석 불능 문제가 제거됨.

### 5.4 주입 방어 (E7)
- faithfulness premise = **EMR 원문 + vital summary만** (생성 지시문 미포함).
- 모델 출력은 `<<<MODEL_HANDOFF>>>` 구분자로 감싸 "지시가 아니라 채점 대상 텍스트"로 명시.

### 5.5 정제 오탐 수정 (E4/E5)
- 조사-끝 휴리스틱: 0점 → `maybe_truncated` 플래그(정상 채점).
- 문자벽에서 `-=#*_~.` 마크다운 구분선 제외. 허용 문자셋에 곱슬따옴표·전각부호·단위 추가.
- `is_no_issue_v3`: **전체 일치만** — "특이사항 없음. intraop VT 발생"은 no-issue가 아님.

### 5.6 통계 (E8/B4)
- 축별 bootstrap 95% CI(케이스 단위, B=10k).
- 변형 vs `{model}_raw`: **paired permutation test**(공통 유효 케이스, 20k) + **Holm 보정**.
  보조로 t(임계값 `t.ppf`, B4)·Wilcoxon(전부-0 가드).
- coverage는 macro(케이스 평균) + **micro(항목 pooled)** 병기.

### 5.7 calibration (`eval_v3/calibrate.py`)
- SY 엑셀 전문의 168행을 **두 judge 모두, 배치로** 재채점 →
  judge별 × 생성모델 family별 Spearman 표 (`calibration_v3.csv`).
- 방향 불일치 축이 있으면 해당 루브릭 프롬프트를 고치고 재실행 — gold 개봉 전 필수 관문.

### 5.8 gold checklist (E6)
- `build_gold_checklist_v3`: KHS c10(교수님 피드백)만 정답 소스. remap은
  `data/preprocessed/khs_gold_remap.json`(PHI — repo 밖).
- `reviewed`는 **사람만** 설정. `--accept`는 `accepted_without_review=true` 별도 플래그.
  모든 리포트에 "전문의검수 n / 잠정채택 n / 잠정 n"이 그대로 표기된다.

## 6. 실행 순서 (폐쇄망 서버)

```bash
# (0) 훅 설치 + 데이터 준비물 확인
bash scripts/install_hooks.sh
# data/preprocessed/khs_gold_remap.json 을 서버 DATA_DIR에 복사 (로컬 data/에 있음, repo엔 없음)

# ★ vLLM GLIBCXX/zmq 이슈 방지 — python 시작 전에 셸에서 export (세션당 1회).
#   LD_PRELOAD는 프로세스 시작 시점에만 유효하므로 반드시 셸에서 export할 것.
#   (run_all_v3는 자식 프로세스에 자동 주입하지만, 수동 스텝은 이 export가 필요)
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
#   확인: strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep -c GLIBCXX_3.4.29  → 1 이상이어야 함

# (1) 1회 준비 (judge/writer GPU 4장)
python -m pipeline_v3.make_fewshot_bank --gpus 0,1,2,3
python -m pipeline_v3.build_gold_checklist_v3 --gpus 0,1,2,3     # (+검수 후 reviewed:true)
python -m pipeline_v3.eval_v3.calibrate --gpus 0,1,2,3           # judge↔전문의 정렬 확인

# (2) SFT 타깃 생성 (1회 공유)
python -m pipeline_v3.gen_pairs --split sft --models llama qwen --gpus 0,1,2,3

# (3) 학습→쌍→RLAIF→추론→dev 평가 (모델 선택은 여기까지로만)
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 \
    --gpus_per_job 2 --skip_done

# (4) 최종 1회 — gold 개봉 (checklist 검수·calibration 확인 후)
python -m pipeline_v3.run_all_v3 --models llama qwen --gpus 0,1,2,3 \
    --gpus_per_job 2 --skip_done --final
```

경로가 다르면 환경변수로: `HANDOVER_BASE_DIR`, `HANDOVER_DATA_DIR`,
`HANDOVER_WORK_DIR`, `HANDOVER_MODEL_DIR`.

## 7. 산출물

```
{WORK_DIR}/outputs_v3/<run_id>/
├── split_manifest.json                  # split 지문 (sid 해시)
├── pairs/pairs_sft.pkl, pairs_dpo_*.pkl, *_audit.jsonl, *_meta.json
├── sft/<model>_<ep>ep/{final/, .done}
├── rlaif/<model>_<...>/{final/, .done}
├── inference/<model>_<exp>/{dev,gold}_results.jsonl (+.done, engine 기록)
├── evaluation/<model>_<exp>/{dev,gold}_results_scores_v3.jsonl + _summary_v3.json
├── report/results_{dev,gold}_v3.{csv,md,html}
└── provenance_v3.json                   # 내용해시 + 동일출력 경고 + git rev
```

## 8. 보고 체크리스트 (논문/교수님 보고 전)

- [ ] v1 `sum_score` 순위는 어디에도 인용하지 않는다 (T1/T2/T7 중첩 오염).
- [ ] gold checklist `reviewed` 상태를 표에 명시 (잠정이면 잠정이라 쓴다).
- [ ] calibration 표(judge↔전문의 Spearman, family별) 첨부.
- [ ] 두 judge 일치도(rho) + 공식 점수의 judge 배정 규칙 명시.
- [ ] 제외 케이스 수와 사유(judge 실패/no_gold) 명시 — 조용히 빼지 않는다.
- [ ] CI와 Holm 보정 p — "n=22 평균 한 줄"로 결론 내지 않는다.
- [ ] 전 변형 동일 추론 엔진(vLLM) 확인 (`engine` 필드).
- [ ] v3 점수를 RLAIF 보상으로 직접 쓰지 않는다 (쌍 생성은 prometheus 경로 유지).

## 9. 위협 요인 (논문 Limitations에 쓸 것)

- gold n=22 — CI가 넓다. dev(110)로 선택하고 gold는 확증에만 사용.
- gold checklist가 LLM 부트스트랩(전문의 검수 전) 상태면 계통 편향 가능.
- gemma4_31b 평가 대상(gemma 계열)은 qwen35 단독 judge — judge 1개 분산.
- few-shot bank가 모델 초안(스타일 앵커) — 전문의 검수로 업그레이드 권장.
- vital summarizer의 나이 파싱/UO 가정(B7/B8)은 v1 자산 재사용 — vital 재생성 시
  `utils/vital_summarizer.py` 개정 필요(문서화만, v3 범위 밖).
```
