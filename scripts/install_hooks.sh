#!/usr/bin/env bash
# git hook 설치 — clone/pull 직후 1회 실행.
# .githooks/pre-commit(PHI 가드)을 이 repo의 hook 경로로 등록한다.
set -e
cd "$(dirname "$0")/.."
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
echo "[install_hooks] core.hooksPath=.githooks 등록 완료 (pre-commit PHI 가드 활성)"
