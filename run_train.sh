#!/bin/bash
# ============================================================
# run_train.sh — GPU 4,5,6,7 병렬 학습 스크립트
#
# 실험 매트릭스 (README 기준):
#   SFT  llama 1ep / 3ep
#   SFT  qwen  1ep / 3ep
#   RLAIF llama raw  DPO / SimPO        (SFT 불필요)
#   RLAIF llama sft1ep DPO              (llama 1ep 필요)
#   RLAIF llama sft3ep DPO              (llama 3ep 필요)
#   RLAIF qwen  sft3ep DPO              (qwen  3ep 필요)
#
# GPU 배분: 잡당 2 GPU, 2잡 병렬 → 총 5 라운드
# ============================================================

set -euo pipefail

# ── 설정 ─────────────────────────────────────────────────────
RUN_ID="$(date +%m%d)"
export HANDOVER_RUN_ID="$RUN_ID"

WORK_DIR="/home/coder/workspace/data/handover/0511/handover"
SFT_BASE="${WORK_DIR}/outputs/${RUN_ID}/sft"

LOG_DIR="${WORK_DIR}/logs_${RUN_ID}"
mkdir -p "$LOG_DIR"

PY="python"

echo "========================================================"
echo " HANDOVER 학습 시작"
echo " RUN_ID  : ${RUN_ID}"
echo " LOG_DIR : ${LOG_DIR}"
echo " SFT_BASE: ${SFT_BASE}"
echo "========================================================"


# ── 유틸 함수 ─────────────────────────────────────────────────
run_job() {
    # run_job <label> <log_file> <cmd...>
    local label="$1"; local logfile="$2"; shift 2
    echo "[START] ${label}"
    "$@" > "$logfile" 2>&1 &
    echo $!   # 반환값: PID
}

wait_jobs() {
    # wait_jobs <label_a> <pid_a> <label_b> <pid_b>
    local la="$1" pa="$2" lb="$3" pb="$4"
    local ok=0

    wait "$pa"; local sa=$?
    wait "$pb"; local sb=$?

    [ $sa -eq 0 ] && echo "  [✓] ${la}" || { echo "  [✗] ${la} (exit $sa) — 로그 확인: ${LOG_DIR}"; ok=1; }
    [ $sb -eq 0 ] && echo "  [✓] ${lb}" || { echo "  [✗] ${lb} (exit $sb) — 로그 확인: ${LOG_DIR}"; ok=1; }

    if [ $ok -ne 0 ]; then
        echo "[ERROR] 위 잡 실패. 계속하려면 Enter, 중단하려면 Ctrl+C"
        read -r
    fi
}

wait_single() {
    local label="$1" pid="$2"
    wait "$pid"; local s=$?
    [ $s -eq 0 ] && echo "  [✓] ${label}" || { echo "  [✗] ${label} (exit $s)"; exit $s; }
}


# ════════════════════════════════════════════════════════════
# Phase 1 — SFT
# ════════════════════════════════════════════════════════════
echo ""
echo "━━━ Phase 1: SFT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Round 1: llama 1ep (GPU 4,5) + qwen 1ep (GPU 6,7)
echo ""
echo "[Round 1] SFT llama 1ep  ||  SFT qwen 1ep"
PID_A=$(run_job "SFT llama 1ep" "${LOG_DIR}/sft_llama_1ep.log" \
    $PY 02_sft_train.py --base llama --epochs 1 --gpus 4,5)

PID_B=$(run_job "SFT qwen 1ep"  "${LOG_DIR}/sft_qwen_1ep.log" \
    $PY 02_sft_train.py --base qwen  --epochs 1 --gpus 6,7)

wait_jobs "SFT llama 1ep" "$PID_A" "SFT qwen 1ep" "$PID_B"

# Round 2: llama 3ep (GPU 4,5) + qwen 3ep (GPU 6,7)
echo ""
echo "[Round 2] SFT llama 3ep  ||  SFT qwen 3ep"
PID_C=$(run_job "SFT llama 3ep" "${LOG_DIR}/sft_llama_3ep.log" \
    $PY 02_sft_train.py --base llama --epochs 3 --gpus 4,5)

PID_D=$(run_job "SFT qwen 3ep"  "${LOG_DIR}/sft_qwen_3ep.log" \
    $PY 02_sft_train.py --base qwen  --epochs 3 --gpus 6,7)

wait_jobs "SFT llama 3ep" "$PID_C" "SFT qwen 3ep" "$PID_D"


# ════════════════════════════════════════════════════════════
# Phase 2 — RLAIF
# ════════════════════════════════════════════════════════════
echo ""
echo "━━━ Phase 2: RLAIF ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Round 3: llama raw DPO (GPU 4,5) + llama raw SimPO (GPU 6,7)
#           SFT 의존성 없으므로 Phase 1 완료 후 바로 실행
echo ""
echo "[Round 3] RLAIF llama raw DPO  ||  RLAIF llama raw SimPO"
PID_E=$(run_job "RLAIF llama raw DPO"   "${LOG_DIR}/rlaif_llama_raw_dpo.log" \
    $PY 03_rlaif_train.py --base llama --loss dpo   --gpus 4,5)

PID_F=$(run_job "RLAIF llama raw SimPO" "${LOG_DIR}/rlaif_llama_raw_simpo.log" \
    $PY 03_rlaif_train.py --base llama --loss simpo --gpus 6,7)

wait_jobs "RLAIF llama raw DPO" "$PID_E" "RLAIF llama raw SimPO" "$PID_F"

# Round 4: llama sft1ep DPO (GPU 4,5) + qwen sft3ep DPO (GPU 6,7)
echo ""
echo "[Round 4] RLAIF llama sft1ep DPO  ||  RLAIF qwen sft3ep DPO"
PID_G=$(run_job "RLAIF llama sft1ep DPO" "${LOG_DIR}/rlaif_llama_sft1ep_dpo.log" \
    $PY 03_rlaif_train.py --base llama --loss dpo \
        --sft_ckpt "${SFT_BASE}/llama_1ep/final" --gpus 4,5)

PID_H=$(run_job "RLAIF qwen sft3ep DPO"  "${LOG_DIR}/rlaif_qwen_sft3ep_dpo.log" \
    $PY 03_rlaif_train.py --base qwen  --loss dpo \
        --sft_ckpt "${SFT_BASE}/qwen_3ep/final"  --gpus 6,7)

wait_jobs "RLAIF llama sft1ep DPO" "$PID_G" "RLAIF qwen sft3ep DPO" "$PID_H"

# Round 5: llama sft3ep DPO — GPU 4개 전체 사용 (최고 성능 기대 실험)
echo ""
echo "[Round 5] RLAIF llama sft3ep DPO  (GPU 4,5,6,7 전체)"
PID_I=$(run_job "RLAIF llama sft3ep DPO" "${LOG_DIR}/rlaif_llama_sft3ep_dpo.log" \
    $PY 03_rlaif_train.py --base llama --loss dpo \
        --sft_ckpt "${SFT_BASE}/llama_3ep/final" --gpus 4,5,6,7)

wait_single "RLAIF llama sft3ep DPO" "$PID_I"


# ════════════════════════════════════════════════════════════
# 완료
# ════════════════════════════════════════════════════════════
echo ""
echo "========================================================"
echo " 모든 학습 완료  (RUN_ID: ${RUN_ID})"
echo ""
echo " 다음 단계:"
echo "   python 04_inference.py --model_path \${SFT_BASE}/llama_1ep/final --out_tag llama_sft1ep --gpus 4,5"
echo "   python 05_evaluate.py  --result_file outputs/.../gold_results.jsonl"
echo ""
echo " 로그 위치: ${LOG_DIR}/"
echo "========================================================"