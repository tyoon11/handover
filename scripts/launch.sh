#!/bin/bash
# ============================================================
# launch.sh — tmux 세션에서 run_train.sh 실행
#
# 사용법:
#   bash launch.sh          # 새 tmux 세션 생성 후 학습 시작
#   bash launch.sh attach   # 실행 중인 세션에 접속
#   bash launch.sh kill     # 세션 강제 종료
#
# tmux 기본 명령:
#   Ctrl+b d   → 세션 detach (학습은 계속 실행)
#   Ctrl+b [   → 스크롤 모드 (q로 나가기)
# ============================================================

SESSION="handover_train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-start}" in
    attach)
        tmux attach -t "$SESSION"
        ;;
    kill)
        tmux kill-session -t "$SESSION" 2>/dev/null && echo "세션 종료됨" || echo "세션 없음"
        ;;
    *)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "[ERROR] 이미 '${SESSION}' 세션이 실행 중."
            echo "  접속: bash launch.sh attach"
            echo "  종료: bash launch.sh kill"
            exit 1
        fi

        echo "tmux 세션 '${SESSION}' 시작..."
        tmux new-session -d -s "$SESSION" -x 220 -y 50
        tmux send-keys -t "$SESSION" \
            "cd '${SCRIPT_DIR}' && bash run_train.sh 2>&1 | tee run_train_$(date +%m%d_%H%M).log" Enter

        echo ""
        echo "학습이 백그라운드에서 시작됐어."
        echo ""
        echo "  접속 (실시간 확인): tmux attach -t ${SESSION}   또는   bash launch.sh attach"
        echo "  분리 (학습 유지):   Ctrl+b → d"
        echo "  로그 직접 확인:     tail -f run_train_*.log"
        echo ""
        ;;
esac