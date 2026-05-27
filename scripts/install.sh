#!/bin/bash
# install.sh — 인계요약지 파이프라인 의존성 설치
# 실행: bash install.sh
# 또는 특정 단계만: bash install.sh --skip-torch

set -e

CONDA_ENV="py310"
SKIP_TORCH=false
for arg in "$@"; do
  [[ "$arg" == "--skip-torch" ]] && SKIP_TORCH=true
done

echo "======================================================"
echo " 인계요약지 파이프라인 라이브러리 설치"
echo " conda env: $CONDA_ENV"
echo "======================================================"
echo ""

# ── 현재 CUDA 버전 확인 ────────────────────────────────────────────────
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "[CUDA] 버전: ${CUDA_VER:-'감지 실패 (nvcc 없음)'}"
python -c "import torch; print(f'[torch] 현재: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "[torch] 미설치"
echo ""

# ── 1. PyTorch (CUDA 12.1 기준 — A6000 지원) ──────────────────────────
if [ "$SKIP_TORCH" = false ]; then
  echo "[1/6] PyTorch 설치 (CUDA 12.1)..."
  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet
  python -c "import torch; print(f'  → torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
else
  echo "[1/6] PyTorch 스킵 (--skip-torch)"
fi

# ── 2. HuggingFace 핵심 ────────────────────────────────────────────────
echo ""
echo "[2/6] HuggingFace 핵심 라이브러리..."
pip install \
  transformers==4.51.3 \
  tokenizers==0.21.1 \
  datasets==3.6.0 \
  huggingface_hub==0.30.0 \
  accelerate==1.6.0 \
  safetensors==0.4.4 \
  sentencepiece==0.2.0 \
  --quiet
echo "  → transformers 4.51.3 (Qwen3 지원), datasets, accelerate, sentencepiece 완료"

# ── 3. PEFT + TRL ──────────────────────────────────────────────────────
echo ""
echo "[3/6] PEFT + TRL (LoRA / DPO / SFT)..."
pip install \
  peft==0.12.0 \
  trl==0.10.1 \
  --quiet
echo "  → peft, trl 완료"

# ── 4. bitsandbytes (8bit/4bit 양자화, 선택적) ───────────────────────
echo ""
echo "[4/6] bitsandbytes (양자화 지원)..."
pip install bitsandbytes==0.43.3 --quiet
echo "  → bitsandbytes 완료"

# ── 5. 데이터 처리 / 유틸 ─────────────────────────────────────────────
echo ""
echo "[5/6] 데이터 처리 유틸..."
pip install \
  pandas==2.1.4 \
  numpy==1.26.4 \
  scikit-learn==1.5.1 \
  tqdm==4.66.5 \
  scipy==1.13.1 \
  --quiet
echo "  → pandas, numpy, scikit-learn, tqdm, scipy 완료"

# ── 6. Flash Attention 2 (A6000에서 권장) ─────────────────────────────
echo ""
echo "[6/6] Flash Attention 2 (빌드에 수 분 소요)..."
pip install flash-attn==2.6.3 --no-build-isolation --quiet && \
  echo "  → flash-attn 완료" || \
  echo "  [WARN] flash-attn 설치 실패 (없어도 동작하나 느릴 수 있음)"

# ── 최종 확인 ─────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " 설치 완료 — 버전 확인"
echo "======================================================"
python - << 'PYCHECK'
import importlib, sys

pkgs = {
    "torch":          "torch",
    "transformers":   "transformers",
    "datasets":       "datasets",
    "accelerate":     "accelerate",
    "peft":           "peft",
    "trl":            "trl",
    "bitsandbytes":   "bitsandbytes",
    "pandas":         "pandas",
    "numpy":          "numpy",
    "scikit-learn":   "sklearn",
    "tqdm":           "tqdm",
    "flash_attn":     "flash_attn",
}

ok, fail = [], []
for name, mod in pkgs.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        ok.append(f"  ✓ {name:<20} {ver}")
    except ImportError:
        fail.append(f"  ✗ {name}")

for l in ok:   print(l)
for l in fail: print(l)

import torch
print(f"\n  GPU 수:  {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
PYCHECK
