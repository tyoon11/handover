"""
build_gold_checklist_v3.py — gold checklist 빌드 CLI (v3)

v2 대비:
  - `--accept`는 accepted_without_review=true만 설정. reviewed는 사람 전용 (E6).
  - 수술ID remap은 DATA_DIR/preprocessed/khs_gold_remap.json 에서 로드 (PHI, P0-3).
  - 추출 실패 케이스는 source="gold_llm_failed"로 남아 평가에서 '제외' 집계 (E3).

실행:
  python -m pipeline_v3.build_gold_checklist_v3 --gpus 0,1,2,3
  python -m pipeline_v3.build_gold_checklist_v3 --accept   # 검수 불가 시 잠정 채택
"""

import argparse
import os


def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    g = p.parse_known_args()[0].gpus
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = g
        print(f"[GPU] CUDA_VISIBLE_DEVICES={g}")


_early_gpus()

from .config_v3 import (      # noqa: E402
    CHECKLIST_EXTRACTOR, GOLD_CHECKLIST_JSON, GOLD_KHS_XLSX, GOLD_PKL, model_path,
)
from .eval_v3 import checklist as CK      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--accept", action="store_true",
                    help="전문의 검수 불가 시 잠정 채택 (accepted_without_review=true; "
                         "reviewed는 절대 자동 설정하지 않음)")
    ap.add_argument("--extractor", type=str, default=CHECKLIST_EXTRACTOR,
                    help="checklist 구조화에 쓸 로컬 모델 키 (채점 judge 아님)")
    args = ap.parse_args()

    import pandas as pd
    gold_df = pd.read_pickle(GOLD_PKL)

    if not GOLD_KHS_XLSX.exists():
        raise FileNotFoundError(f"KHS gold 엑셀 없음: {GOLD_KHS_XLSX}")
    gold_refs, _drafts = CK.load_khs_gold(GOLD_KHS_XLSX, gold_df)
    print(f"[build] 교수님 gold(c10) {len(gold_refs)}건 → 이것만 정답으로 사용")

    from .eval_v3.engine import JudgeEngine
    engine = JudgeEngine(model_path(args.extractor), backend=args.backend)
    checklist = CK.build_checklist(engine, gold_df, gold_refs)
    engine.unload()

    if args.accept:
        checklist = CK.accept_provisional(
            checklist, note="전문의 직접 검수 불가 — c10 기반 자동추출본 잠정 채택")

    CK.save_checklist(checklist, GOLD_CHECKLIST_JSON)
    st = CK.review_status(checklist)
    n_items = sum(len(v.get("items", [])) for v in checklist.values())
    n_normal = sum(1 for v in checklist.values() if v.get("is_normal_case"))
    n_nogold = sum(1 for v in checklist.values() if v.get("source") == "no_gold")
    n_failed = sum(1 for v in checklist.values() if v.get("source") == "gold_llm_failed")
    print(f"\n[build] 저장: {GOLD_CHECKLIST_JSON}")
    print(f"  케이스 {st['n']}건 / 항목 {n_items}개 / normal {n_normal} / "
          f"no_gold {n_nogold} / 추출실패 {n_failed}")
    print(f"  검수상태: reviewed={st['reviewed']} 잠정채택={st['accepted_without_review']} "
          f"잠정={st['provisional']}")
    if n_failed:
        print("  ✗ 추출 실패 케이스는 평가에서 '제외'로 집계된다 — 재실행 또는 수기 작성 필요")
    if st["reviewed"] == 0:
        print("  ⚠ 전문의 검수 0건 — JSON을 직접 열어 검수 후 reviewed:true로 바꿔야 정식 gold")


if __name__ == "__main__":
    main()
