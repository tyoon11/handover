"""
build_gold_checklist.py — Gold actionable checklist 빌더 (개선안 1·6)

전문의 gold(인계요약지_SY.xlsx)를 1차 소스로, gemma-4-31B가 케이스별 actionable
checklist를 구조화 추출 → 전문의가 직접 수정 가능한 JSON(gold_checklist_v2.json)으로 저장.

모드:
  --inspect       SY 엑셀 컬럼/자동감지 결과만 출력하고 종료 (빌드 안 함)
  (기본)          checklist 빌드 후 저장
  --calibrate     이미 만든 checklist로 SY의 모델출력을 v2로 채점 →
                  전문의 점수와 상관(Spearman)을 내어 'judge가 사람과 맞는가' 검증

실행:
  python build_gold_checklist.py --inspect
  python build_gold_checklist.py --gpus 0,1,2,3
  python build_gold_checklist.py --calibrate --gpus 0,1,2,3
"""

import sys as _sys, os, argparse, json, pickle
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))


def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    return p.parse_known_args()[0].gpus


_g = _early_gpus()
if _g is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _g
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_g}")

import pandas as pd
from config import GOLD_PKL, VITAL_MAP_PKL
from config_v2 import (
    EVAL_V2_LLM, EVAL_V2_BACKEND, EVAL_V2_GEN,
    GOLD_CHECKLIST_JSON, XLSX_COL_HINTS, SY_COLS, SY_HEADER_ROWS, gold_sy_path,
    GOLD_KHS_XLSX, KHS_SHEET, KHS_COLS, KHS_HEADER_ROWS, KHS_GOLD_REMAP,
)
from pipeline.eval_v2 import checklist as CK


def _sid_of(gold_df, idx):
    try:
        s = gold_df.iloc[idx]["수술 ID"]
        return int(s.iloc[0]) if hasattr(s, "iloc") else int(s)
    except Exception:
        return -1


def do_inspect():
    path = gold_sy_path()
    print(f"[inspect] SY gold 파일: {path} (exists={path.exists()})")
    try:
        CK.inspect_xlsx(path, XLSX_COL_HINTS)
    except Exception as e:
        print(f"  inspect 실패: {e}")
    print("\n[SY_COLS 위치 기반 매핑]", SY_COLS)


def do_build(args):
    gold_df = pd.read_pickle(GOLD_PKL)

    # gold 소스 = KHS c10(교수님 피드백)만 사용 (SY 무시).
    khs_gold, khs_draft = {}, {}
    if GOLD_KHS_XLSX.exists():
        khs_gold, khs_draft = CK.load_khs(
            GOLD_KHS_XLSX, KHS_SHEET, KHS_COLS, KHS_HEADER_ROWS, gold_df,
            remap=KHS_GOLD_REMAP)
    else:
        print(f"[build] ⚠ KHS 파일 없음: {GOLD_KHS_XLSX}")
    print(f"[build] 교수님 gold(c10) {len(khs_gold)}건 → 이것만 정답으로 사용")

    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    from pipeline.eval_v2.engine import EvalEngine
    engine = EvalEngine(EVAL_V2_LLM, backend=args.backend, gen_cfg=EVAL_V2_GEN)

    checklist = CK.build_checklist(engine, gold_df, vital_map,
                                   gold_refs=khs_gold, context_refs=khs_draft)

    # 교수님 직접 검수 불가 → --accept 시 c10 기반 추출본을 정식 gold로 '채택'
    # (gold 있는 케이스만 reviewed=true; no_gold는 그대로 둠)
    if getattr(args, "accept", False):
        for v in checklist.values():
            if v.get("source") in ("gold_llm", "gold_normal"):
                v["reviewed"] = True
                v["review_note"] = "교수님 직접 검수 불가 — c10 기반 자동추출본을 정식 gold로 채택"

    CK.save_checklist(checklist, GOLD_CHECKLIST_JSON)

    n_items = sum(len(v["items"]) for v in checklist.values())
    n_normal = sum(1 for v in checklist.values() if v.get("is_normal_case"))
    n_nogold = sum(1 for v in checklist.values() if v.get("source") == "no_gold")
    print(f"\n[build] 저장: {GOLD_CHECKLIST_JSON}")
    print(f"  케이스 {len(checklist)}건, 항목 총 {n_items}개, "
          f"normal {n_normal}건, gold없음(수기필요) {n_nogold}건")
    if n_nogold:
        miss = [v["sid"] for v in checklist.values() if v.get("source") == "no_gold"]
        print(f"  ⚠ gold 없는 케이스(c10 공란) sid={miss} — coverage 평가 제외(중립). "
              f"(DNET craniotomy는 c10 자체가 없음)")
    if getattr(args, "accept", False):
        print("  ✓ --accept: gold 보유 케이스를 reviewed=true(채택)로 확정")
    else:
        print("  ⚠ reviewed=false. 검수 불가 시 --accept로 재실행해 확정하세요.")


def do_calibrate(args):
    """SY 모델출력(전문의 점수 보유)을 v2로 채점 → 사람 점수와 상관 검증."""
    from scipy import stats
    path = gold_sy_path()
    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)

    checklist = CK.load_checklist(GOLD_CHECKLIST_JSON)
    if not checklist:
        print("[calibrate] checklist 없음 — 먼저 빌드하세요."); return
    gold_by_idx, scored = CK.load_sy(path, SY_COLS, SY_HEADER_ROWS)

    from pipeline.eval_v2.engine import EvalEngine, clean_v2
    from pipeline.eval_v2 import metrics as M
    from config import build_emr_text, build_user_prompt
    engine = EvalEngine(EVAL_V2_LLM, backend=args.backend, gen_cfg=EVAL_V2_GEN)

    rows = []
    for row in scored:
        idx = row["idx"]
        sid = _sid_of(gold_df, idx)
        entry = checklist.get(str(sid)) or {"items": [], "is_normal_case": True}
        gen, status = clean_v2(row["gen"])
        emr = build_emr_text(gold_df.iloc[idx]) if idx < len(gold_df) else ""
        prem = build_user_prompt(emr, vital_map.get(sid, ""))

        if status != "ok":
            sc = M.degenerate_scores(status)
        else:
            cov = (M.parse_coverage(
                engine.chat_json([M.build_coverage_prompt(gen, entry)[1]],
                                  system=M.build_coverage_prompt(gen, entry)[0])[0], entry)
                if entry.get("items") else dict(coverage=1.0, missed=[], high_missed=False))
            fa = M.parse_faithfulness(
                engine.chat_json([M.build_faithfulness_prompt(prem, gen)[1]],
                                 system=M.build_faithfulness_prompt(prem, gen)[0])[0])
            br = M.parse_brevity(
                engine.chat_json([M.build_brevity_prompt(gen)[1]],
                                 system=M.build_brevity_prompt(gen)[0])[0])
            sc = M.composite(cov, fa, br, entry, gen)
        rows.append({**row, **{k: sc.get(k) for k in
                     ("coverage", "faithfulness", "brevity", "composite", "safety_violation")}})

    df = pd.DataFrame(rows)
    out_csv = GOLD_CHECKLIST_JSON.parent / "calibration_v2.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[calibrate] 저장: {out_csv}  (n={len(df)})")

    # 상관 (전문의 점수 방향 가정: brevity/critical 높을수록 좋음;
    #        err/over/under 높을수록 나쁨 → v2와 음의 상관 기대)
    pairs = [
        ("coverage", "h_critical", "+"), ("coverage", "h_under", "-"),
        ("faithfulness", "h_err", "-"),
        ("brevity", "h_brevity", "+"), ("brevity", "h_over", "-"),
    ]
    print("\n  ── Spearman (v2 vs 전문의, n≥5만) ──")
    print("  방향: +면 양의상관 기대, -면 음의상관 기대")
    for v2c, hc, exp in pairs:
        sub = df[[v2c, hc]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 5:
            print(f"    {v2c:13s} vs {hc:11s}: n={len(sub)} (부족)")
            continue
        rho, p = stats.spearmanr(sub[v2c], sub[hc])
        flag = "✓" if ((rho > 0) == (exp == "+")) else "✗방향불일치"
        print(f"    {v2c:13s} vs {hc:11s}: rho={rho:+.3f} p={p:.3g} n={len(sub)} {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--backend", type=str, default=EVAL_V2_BACKEND)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--accept", action="store_true",
                    help="교수님 검수 불가 시: c10 기반 추출본을 정식 gold로 채택(reviewed=true)")
    ap.add_argument("--gold_note", action="store_true",
                    help="기존 checklist에 gold_note(실제 인계지) 추가 (재추출 안 함, 기존 필드 보존)")
    args = ap.parse_args()

    if args.inspect:
        do_inspect()
    elif args.calibrate:
        do_calibrate(args)
    elif args.gold_note:
        do_gold_note(args)
    else:
        do_build(args)


def do_gold_note(args):
    """기존 gold_checklist_v2.json을 보존한 채 gold_note만 LLM으로 합성해 덧붙인다."""
    checklist = CK.load_checklist(GOLD_CHECKLIST_JSON)
    if not checklist:
        print("[gold_note] checklist 없음 — 먼저 빌드하세요."); return
    from pipeline.eval_v2.engine import EvalEngine
    engine = EvalEngine(EVAL_V2_LLM, backend=args.backend, gen_cfg=EVAL_V2_GEN)
    checklist = CK.compose_gold_notes(engine, checklist)
    CK.save_checklist(checklist, GOLD_CHECKLIST_JSON)
    n = sum(1 for v in checklist.values() if v.get("gold_note"))
    print(f"[gold_note] gold_note {n}건 추가 → {GOLD_CHECKLIST_JSON}")


if __name__ == "__main__":
    main()
