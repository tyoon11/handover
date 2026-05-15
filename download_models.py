"""
download_models.py — 실험에 필요한 HuggingFace 모델 다운로드

다운로드 대상 (총 8개):
  [학습 베이스]
  - meta-llama/Llama-3.1-8B-Instruct
  - Qwen/Qwen3-8B
  - google/gemma-4-E4B-it
  - Qwen/Qwen3.5-9B
  - snuh/hari-q3-8b
  [평가]
  - prometheus-eval/prometheus-7b-v2.0
  - google/flan-t5-large
  - google/flan-t5-xl

실행:
  python download_models.py                              # 전체
  python download_models.py --models gemma4 qwen35 hari # 특정 모델만
  python download_models.py --check                      # 상태 확인만
  python download_models.py --token <HF_TOKEN>           # 토큰 직접 지정
  python download_models.py --force                      # 이미 있어도 재다운로드
"""

import argparse, os, sys
from pathlib import Path

MODEL_DIR = Path("/home/coder/workspace/data/local_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 다운로드 대상 모델 ─────────────────────────────────────────────────────
MODELS = {
    # 학습 베이스
    "llama": {
        "repo_id":   "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": MODEL_DIR / "Llama-3.1-8B-Instruct",
        "desc":      "학습 베이스 (SFT/RLAIF)",
        "gated":     True,   # Meta 라이선스 동의 + HF 토큰 필요
    },
    "qwen": {
        "repo_id":   "Qwen/Qwen3-8B",
        "local_dir": MODEL_DIR / "Qwen3-8B",
        "desc":      "학습 베이스 (SFT/RLAIF)",
        "gated":     False,
    },
    "gemma4": {
        "repo_id":   "google/gemma-4-E4B-it",
        "local_dir": MODEL_DIR / "gemma-4-E4B-it",
        "desc":      "비교 베이스 (Google Gemma 4 E4B)",
        "gated":     True,   # Google 라이선스 동의 필요
    },
    "qwen35": {
        "repo_id":   "Qwen/Qwen3.5-9B",
        "local_dir": MODEL_DIR / "Qwen3.5-9B",
        "desc":      "비교 베이스 (Qwen 3.5 9B)",
        "gated":     False,
    },
    "hari": {
        "repo_id":   "snuh/hari-q3-8b",
        "local_dir": MODEL_DIR / "hari-q3-8b",
        "desc":      "서울대병원 특화 모델",
        "gated":     False,
    },
    # 평가
    "prometheus": {
        "repo_id":   "prometheus-eval/prometheus-7b-v2.0",
        "local_dir": MODEL_DIR / "prometheus-7b-v2.0",
        "desc":      "LLM-as-Judge 평가",
        "gated":     False,
    },
    "flan-large": {
        "repo_id":   "google/flan-t5-large",
        "local_dir": MODEL_DIR / "flan-t5-large",
        "desc":      "SCALE factuality 평가 (large)",
        "gated":     False,
    },
    "flan-xl": {
        "repo_id":   "google/flan-t5-xl",
        "local_dir": MODEL_DIR / "flan-t5-xl",
        "desc":      "SCALE factuality 평가 (xl)",
        "gated":     False,
    },
}

IGNORE_PATTERNS = [
    "*.msgpack", "*.h5", "flax_model*", "tf_model*",
    "rust_model*", "coreml*", "onnx*",
]


def check_status(local_dir: Path):
    if not local_dir.exists():
        return False, "폴더 없음"
    if not (local_dir / "config.json").exists():
        return False, "config.json 없음"
    weights = (list(local_dir.rglob("*.safetensors")) +
               list(local_dir.rglob("*.bin")))
    if not weights:
        return False, "weight 파일 없음"
    total_gb = sum(f.stat().st_size for f in weights) / 1e9
    return True, f"{len(weights)}개 파일  {total_gb:.1f} GB"


def print_status(targets):
    print(f"\n[모델 상태]  저장경로: {MODEL_DIR}")
    print(f"  {'키':<14} {'HF repo_id':<42} {'상태'}")
    print("  " + "-" * 72)
    for key in targets:
        info = MODELS[key]
        ok, detail = check_status(info["local_dir"])
        mark = "✓" if ok else "✗"
        status = f"완료 ({detail})" if ok else f"미완료 ({detail})"
        gated = " 🔒" if info["gated"] else ""
        print(f"  {mark} {key:<14} {info['repo_id']:<42}{gated}")
        print(f"    {'':<14} {status}")
    print()


def download_one(key: str, token, force: bool) -> bool:
    from huggingface_hub import snapshot_download

    info      = MODELS[key]
    repo_id   = info["repo_id"]
    local_dir = info["local_dir"]

    ok, detail = check_status(local_dir)
    if ok and not force:
        print(f"  [SKIP] {key}: 이미 완료 ({detail})")
        return True

    if info["gated"] and not token:
        print(f"  [WARN] {key}: gated 모델이라 HF 토큰 필요 (--token 또는 HF_TOKEN)")

    print(f"\n  ▶ [{key}]  {repo_id}")
    print(f"    {info['desc']}")
    print(f"    → {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            ignore_patterns=IGNORE_PATTERNS,
            resume_download=True,
        )
        ok2, detail2 = check_status(local_dir)
        if ok2:
            print(f"    ✓ 완료: {detail2}")
            return True
        else:
            print(f"    ✗ 검증 실패: {detail2}")
            return False
    except Exception as e:
        print(f"    ✗ 오류: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 모델 다운로드")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help=f"다운로드할 모델 키 (기본: 전체)\n선택지: {list(MODELS.keys())}")
    parser.add_argument("--check",  action="store_true", help="상태 확인만")
    parser.add_argument("--force",  action="store_true", help="이미 있어도 재다운로드")
    parser.add_argument("--token",  type=str, default=None, help="HuggingFace 토큰")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print(f"[인증] HF 토큰: {token[:8]}...")
    else:
        print("[인증] 토큰 없음 — public 모델만 가능")

    targets = args.models
    print(f"[대상] {', '.join(targets)}")

    print_status(targets)

    if args.check:
        return

    # 다운로드
    results = {}
    for key in targets:
        results[key] = download_one(key, token, args.force)

    # 요약
    print("\n" + "=" * 50)
    ok_list   = [k for k, v in results.items() if v]
    fail_list = [k for k, v in results.items() if not v]
    if ok_list:
        print(f"✓ 성공: {', '.join(ok_list)}")
    if fail_list:
        print(f"✗ 실패: {', '.join(fail_list)}")
        print()
        print("  팁:")
        print("  - gated 모델(llama, gemma4): HF 라이선스 동의 후 --token <HF_TOKEN>")
        print("  - 네트워크 차단 환경: export HF_ENDPOINT=https://hf-mirror.com")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
