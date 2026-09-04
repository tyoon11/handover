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

### 모델 다운로드 (병원 프록시 TLS 포함)

병원망은 TLS 를 가로채므로 HF 다운로드가 `CERTIFICATE_VERIFY_FAILED: self-signed
certificate in certificate chain` 로 실패한다. 프록시 CA 를 뽑아 지정하면 검증을 유지한 채
해결된다.

원인은 접속 차단이 아니다 — 이 컨테이너는 `REQUESTS_CA_BUNDLE`이 시스템 번들
(`/etc/ssl/certs/ca-certificates.crt`, 병원 CA 포함)을 가리키고 있어서 curl과 예전
requests 기반 huggingface_hub는 통과했다. 그런데 **httpx는 `REQUESTS_CA_BUNDLE`도
`SSL_CERT_FILE`도 읽지 않고 certifi를 쓴다.** hub가 httpx로 바뀌면서 파이썬만 실패한다.

```bash
python utils/download_models.py --fix-certifi     # certifi에 시스템 CA 덧붙임(백업·재실행 안전)
python utils/download_models.py --probe           # 통과 확인
export HF_TOKEN=<토큰>                             # gated 모델용

python utils/download_models.py --group v32 --check    # 계획·디스크 확인
python utils/download_models.py --group v32            # 받기 (v3.2 teacher·judge 후보)
```

`--fix-certifi` 는 이 환경의 **모든 파이썬 HTTPS**(pip·datasets·vllm 다운로드 포함)를 함께
고친다. 권한이 없어 certifi를 못 고치면 매번 지정하는 방식으로 우회한다:

```bash
export HANDOVER_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
python utils/download_models.py --probe --ca-bundle /etc/ssl/certs/ca-certificates.crt
```

시스템 번들에도 CA가 없으면 `--extract-ca ~/hf_proxy_ca.pem` 으로 프록시가 제시하는 체인을
직접 뽑아 `--ca-bundle` 로 지정한다. 그래도 안 되면 `--insecure`(= `HANDOVER_INSECURE_SSL=1`)로
검증을 끌 수 있다 —
토큰이 미검증 연결로 나가므로 신뢰된 병원망 안에서만. 대상 목록은
`pipeline_v3/config_v3.py` 의 `MODELS` 가 단일 소스다.

**TLS 를 통과했는데 대용량 샤드만 0바이트로 멈출 때** — TLS 문제와 별개의 증상이다.
작은 파일(`config.json`·tokenizer)은 받아지는데 `*.safetensors` 의 `.incomplete` 가
0바이트로 고정되고 `du` 가 안 늘면, 파일 종류에 따라 **호스트가 다른 것**이 원인이다:
대용량은 Xet 스토리지(`cas-bridge`/`transfer.xethub.hf.co`)로 리다이렉트되고 자체 청크
프로토콜을 써서 프록시를 잘 통과하지 못한다.

먼저 **CDN 도달성**을 확인한다 — 이게 막혀 있으면 타임아웃·워커 조정으로는 절대 안 된다:

```bash
python utils/download_models.py --probe-cdn
#   ① API 도달 OK
#   ② 대용량 리다이렉트 → us.aws.cdn.hf.co
#   ③ CDN 실패(20.0s): TimeoutError   ← 여기서 걸리면 프록시 allowlist 문제
```

`huggingface.co` 는 열려 있는데 CDN 만 막히는 이유: HF 가 대용량 전송을 **다른 도메인**으로
옮겼다. 예전에는 `cdn-lfs*.huggingface.co` 라서 `*.huggingface.co` 허용 규칙에 걸렸지만,
지금은 `us.aws.cdn.hf.co` · `cas-bridge.xethub.hf.co` 등 **`hf.co` 도메인**이다.
→ 프록시 allowlist 에 **`*.hf.co`** 추가를 요청해야 한다. (그래서 "예전엔 됐는데 지금 안 됨"이 된다)

CDN 이 열려 있는데도 멈추면 프록시 env 감지 시 스크립트가 자동으로 거는
`HF_HUB_DISABLE_XET=1`(Xet 대신 LFS CDN 경로)과 `HF_HUB_DOWNLOAD_TIMEOUT=120`이 먼저 붙고,
그다음 동시연결을 낮춘다:

```bash
python utils/download_models.py --models qwen35_122b --max-workers 2
```

`hf_transfer`(`HF_HUB_ENABLE_HF_TRANSFER=1`)는 쓰지 말 것 — Rust 다운로더가 자체 TLS
루트를 써서 시스템 CA(병원 CA)를 보지 않는다.

**프록시 허용 정책 확인** — `403`(즉시)과 `무응답`(타임아웃)은 원인이 다르고 요청 대상도 다르다:

```bash
python utils/download_models.py --probe-proxy
```

| CONNECT 응답 | 원인 | 요청할 곳 |
|---|---|---|
| `200 Connection established` | 허용 + 연결 성공 | — |
| `403` (+ `X-Squid-Error`) | squid ACL 거부 (허용목록에 없음) | 프록시 담당 — `*.hf.co` 추가 |
| `407` | 프록시 인증 필요 | 프록시 담당 — 자격증명 |
| `503`/`504` | squid 는 허용, 상단 연결·DNS 실패 | 네트워크 보안 담당 |
| 무응답(타임아웃) | squid 통과 후 방화벽 drop 추정 | 네트워크 보안 담당 |

`example.com` 결과가 기본정책을 알려준다 — 통과하면 allowlist 방식이 **아니므로**
"허용목록 추가"가 아니라 "해당 호스트/대역 차단 해제"로 요청해야 한다.

### 로컬 PC에서 받아 옮기기 (프록시가 안 열릴 때)

프록시가 `us.aws.cdn.hf.co` 를 막으면 폐쇄망에서 받을 방법이 없다. 인터넷 되는 PC에서 받아
외장 디스크로 옮기고 **해시로 검증**한다 (수십 GB 샤드는 잘려 들어와도 로드는 되고 추론만
이상해지므로 크기 비교로는 부족하다).

받는 쪽 (예: macOS + 외장 SSD):

```bash
pip install -U huggingface_hub hf_transfer
export HANDOVER_MODEL_DIR=/Volumes/T7/local_models
export HF_HUB_ENABLE_HF_TRANSFER=1        # 직결 인터넷에서만 (프록시 환경에선 쓰지 말 것)

M="qwen35_122b mprometheus llama70b"      # 148GB. teacher 선발전까지 하려면 + qwen72b
python utils/download_models.py --models $M --check
python utils/download_models.py --models $M
python utils/download_models.py --models $M --checksums write
```

옮긴 쪽 (폐쇄망 서버):

```bash
rsync -a --info=progress2 /mnt/T7/local_models/ /home/coder/workspace/data/local_models/
export HANDOVER_MODEL_DIR=/home/coder/workspace/data/local_models
python utils/download_models.py --models $M --checksums verify   # 불일치는 exit 1
python utils/download_models.py --group all --check
```

- 외장 디스크가 **FAT32면 4GB 파일 제한**에 걸린다. exFAT/APFS/ext4 를 쓸 것
  (`diskutil info /Volumes/T7 | grep -i personality`).
- `--checksums verify` 는 누락·해시불일치를 파일 단위로 지적한다. 불일치 파일만 재전송하면 된다.
- gated 모델(llama·gemma4·gemma4_31b·medgemma27b)은 HF 라이선스 동의 + `HF_TOKEN` 이 필요하지만,
  현재 미보유 5종(teacher 3후보·mprometheus·llama70b)은 **전부 public** 이라 토큰이 필요 없다.

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

판정은 **2-tier** — `[유의]`(소생·개입 기준 초과 = 임상적 유의) / 표시 없음(연령별 참조범위 이탈).

| 항목 | 유의(`[유의]`) 기준 | 정상범위 기준 | 전거 |
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
