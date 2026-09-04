#!/usr/bin/env python3
"""
invalidate_v3.py — 임계값·프롬프트·GT 변경으로 무효가 된 v3 산출물 정리

`--skip_done` 은 "파일이 있으면 건너뛴다"이므로, 임계값이나 프롬프트를 고친 뒤
그냥 재실행하면 **옛 산출물을 그대로 재사용**한다. 특히 다음 두 개가 위험하다.

  - `vital_summary_map.pkl` : pipeline/00_preprocess.build_and_save_vital_map 이
    파일이 있으면 그냥 로드한다 → 새 임계값이 절대 반영되지 않는다.
  - `gold_checklist_v3.json`: category 6항목군·재검수 GT가 반영되지 않는다.

기본은 **dry-run**. 실제로 지우려면 --apply, 지우는 대신 보관하려면 --archive.

  python scripts/invalidate_v3.py                 # 무엇이 지워질지만 출력
  python scripts/invalidate_v3.py --apply
  python scripts/invalidate_v3.py --apply --archive   # 삭제 대신 .bak_<stamp>/ 로 이동
  python scripts/invalidate_v3.py --apply --scope vital,checklist   # 일부만
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_v3.config_v3 import (      # noqa: E402
    EVAL_OUT, GOLD_CHECKLIST_JSON, INFER_OUT, PAIRS_OUT, REPORT_OUT,
    RLAIF_OUT, SFT_OUT, VITAL_AUDIT_PKL, VITAL_MAP_PKL, DATA_DIR,
)

PREP = DATA_DIR / "preprocessed"

# scope → (설명, 경로들, 무효 사유)
SCOPES = {
    "vital": (
        "바이탈 요약 맵 + 이벤트 감사본",
        [VITAL_MAP_PKL, VITAL_AUDIT_PKL],
        "v3.2 요약 재설계(REPORTABLE/MINOR 3블록 · R1~R4 · baseline 게이트) — 재생성 필수",
    ),
    "checklist": (
        "gold checklist",
        [GOLD_CHECKLIST_JSON],
        "category 6항목군 도입 + 교수님 재검수 GT override",
    ),
    "fewshot": (
        "few-shot bank",
        [PREP / "fewshot_bank_v3.json"],
        "생성 프롬프트 변경(필수 항목군·정량화) — 예시 재생성",
    ),
    "calib": (
        "judge calibration",
        [PREP / "calibration_v3.csv", PREP / "calibration_v3_report.txt"],
        "brevity 루브릭 변경(정량화 감점 제외)",
    ),
    "pairs": (
        "SFT 타깃 / 선호쌍",
        [PAIRS_OUT],
        "생성 프롬프트·쌍생성 루브릭 변경",
    ),
    "train": (
        "SFT / RLAIF 체크포인트",
        [SFT_OUT, RLAIF_OUT],
        "옛 프롬프트로 학습됨 — 재학습 필요",
    ),
    "infer": (
        "추론 결과",
        [INFER_OUT],
        "옛 프롬프트·옛 바이탈 요약 기반",
    ),
    "eval": (
        "평가 결과 / 리포트",
        [EVAL_OUT, REPORT_OUT],
        "checklist·judge 루브릭 변경",
    ),
}

DEFAULT_ORDER = ["vital", "checklist", "fewshot", "calib",
                 "pairs", "train", "infer", "eval"]


def _size(p: Path) -> str:
    if not p.exists():
        return "-"
    if p.is_file():
        n = p.stat().st_size
    else:
        n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제로 삭제/이동")
    ap.add_argument("--archive", action="store_true",
                    help="삭제 대신 .bak_<stamp>/ 로 이동")
    ap.add_argument("--scope", type=str, default=",".join(DEFAULT_ORDER),
                    help=f"쉼표 구분. 가능: {','.join(DEFAULT_ORDER)}")
    args = ap.parse_args()

    scopes = [s.strip() for s in args.scope.split(",") if s.strip()]
    bad = [s for s in scopes if s not in SCOPES]
    if bad:
        sys.exit(f"알 수 없는 scope: {bad}\n가능: {list(SCOPES)}")

    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    arc = DATA_DIR.parent / f".bak_{stamp}"
    total, targets = 0, []

    print(f"{'scope':10s} {'크기':>8s}  경로")
    print("-" * 78)
    for s in scopes:
        desc, paths, why = SCOPES[s]
        for p in paths:
            exists = p.exists()
            print(f"{s:10s} {_size(p):>8s}  {p}" + ("" if exists else "   (없음)"))
            if exists:
                targets.append((s, p))
                total += 1
        print(f"{'':10s} {'':>8s}  └ {desc}: {why}")
    print("-" * 78)

    if not targets:
        print("무효화할 산출물이 없습니다 — 새로 돌리면 됩니다.")
        return
    if not args.apply:
        print(f"[dry-run] {total}개 대상. 실제 실행: --apply "
              f"(보관하려면 --apply --archive)")
        return

    for s, p in targets:
        if args.archive:
            dest = arc / s / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            print(f"  이동 {p} → {dest}")
        else:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            print(f"  삭제 {p}")
    print(f"\n완료: {total}개 처리" + (f" (보관: {arc})" if args.archive else ""))


if __name__ == "__main__":
    main()
