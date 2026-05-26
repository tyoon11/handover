#!/usr/bin/env python3
"""
monitor.py — 학습 진행 상황 요약 출력

사용법:
    python monitor.py                  # 한 번 출력
    watch -n 60 python monitor.py      # 60초마다 갱신
"""

import os
import re
import glob
from pathlib import Path
from datetime import datetime

WORK_DIR = Path("/home/coder/workspace/data/handover/0511/handover")

# 오늘 날짜 기준 로그 폴더 찾기
today = datetime.now().strftime("%m%d")
log_dirs = sorted(WORK_DIR.glob(f"logs_{today}*")) + sorted(WORK_DIR.glob("logs_*"))
LOG_DIR = log_dirs[0] if log_dirs else None

JOBS = [
    ("sft_llama_1ep", "SFT  llama 1ep"),
    ("sft_qwen_1ep", "SFT  qwen  1ep"),
    ("sft_llama_3ep", "SFT  llama 3ep"),
    ("sft_qwen_3ep", "SFT  qwen  3ep"),
    ("rlaif_llama_raw_dpo", "RLAIF llama raw   DPO"),
    ("rlaif_llama_raw_simpo", "RLAIF llama raw   SimPO"),
    ("rlaif_llama_sft1ep_dpo", "RLAIF llama sft1ep DPO"),
    ("rlaif_qwen_sft3ep_dpo", "RLAIF qwen  sft3ep DPO"),
    ("rlaif_llama_sft3ep_dpo", "RLAIF llama sft3ep DPO  ← 최고성능"),
]


def parse_log(logfile: Path):
    """로그에서 마지막 step/loss 추출."""
    if not logfile.exists():
        return None, None, None

    size = logfile.stat().st_size
    if size == 0:
        return "시작 전", None, None

    # 끝에서 4KB만 읽기
    with open(logfile, "rb") as f:
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", errors="ignore")

    # transformers Trainer 로그 패턴: "{'loss': 0.123, ..., 'epoch': 1.5}"
    step_matches = re.findall(r"'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)", tail)
    # 또는 단순 step 카운터
    step_num = re.findall(r"\b(\d+)/(\d+)\s*\[", tail)

    if step_matches:
        loss, epoch = step_matches[-1]
        return "학습 중", f"loss={loss}", f"epoch={epoch}"
    elif step_num:
        cur, total = step_num[-1]
        pct = int(cur) / int(total) * 100
        return "학습 중", f"{cur}/{total} ({pct:.0f}%)", None
    elif "완료" in tail or "Saving" in tail:
        return "완료", None, None
    elif "Error" in tail or "Traceback" in tail:
        # 마지막 에러 줄 추출
        err_lines = [l for l in tail.splitlines() if "Error" in l or "error" in l]
        return "실패", err_lines[-1][:60] if err_lines else "에러", None
    else:
        return "실행 중", None, None


def gpu_status():
    """nvidia-smi로 GPU 메모리 사용량."""
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        lines = []
        for line in out.strip().splitlines():
            idx, used, total, util = line.split(", ")
            if int(idx) in (4, 5, 6, 7):
                lines.append(f"  GPU{idx}: {used:>6}/{total} MB  util={util}%")
        return lines
    except Exception:
        return ["  (nvidia-smi 사용 불가)"]


print("=" * 65)
print(f" HANDOVER 학습 모니터  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
if LOG_DIR:
    print(f" LOG_DIR: {LOG_DIR}")
else:
    print(" LOG_DIR: (로그 없음 — 학습 미시작?)")
print("=" * 65)

if LOG_DIR:
    for key, label in JOBS:
        logfile = LOG_DIR / f"{key}.log"
        status, info1, info2 = parse_log(logfile)

        icon = {
            "완료": "✓",
            "실패": "✗",
            "학습 중": "▶",
            "실행 중": "▶",
            "시작 전": "○",
        }.get(status or "", "?")
        parts = [f"  {icon} {label:<30}  {status or '?'}"]
        if info1:
            parts.append(f"  {info1}")
        if info2:
            parts.append(f"  {info2}")
        print("".join(parts))

print()
print("── GPU 상태 ────────────────────────────────────────────")
for line in gpu_status():
    print(line)
print("=" * 65)
