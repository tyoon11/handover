"""
eval_v3/calibrate.py — judge ↔ 전문의(168행) 정렬 검증 (T7/개선안 Phase1-4)

인계요약지_SY.xlsx '데이터' 시트의 전문의 채점 168행(22케이스 × llama/qwen × prompt 1~4)을
v3 judge들로 재채점해 Spearman 상관을 낸다. v2와의 차이:
  - 모든 judge 호출이 배치 (v2 do_calibrate의 행별 순차 호출 제거)
  - EVAL_JUDGES 두 judge 모두 실행 → judge별 + 생성모델 family별 상관표
    (계열 편향: 'gemma judge가 gemma 출력에 후한가'를 수치로 보고)
  - 가중치(V3_WEIGHTS) 근거: 전문의 종합 선호와 composite의 상관을 함께 출력

실행:
  python -m pipeline_v3.eval_v3.calibrate --gpus 0,1,2,3
"""

import argparse
import json
import os


def _early_gpus():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    g = p.parse_known_args()[0].gpus
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = g


_early_gpus()

from ..config_v3 import (      # noqa: E402
    DATA_DIR, EVAL_JUDGES, GOLD_CHECKLIST_JSON, GOLD_SY_XLSX, ensure_dir, model_path,
)
from ..prompt_utils import build_emr_text, get_sid     # noqa: E402
from . import metrics as M                              # noqa: E402
from .checklist import load_checklist                   # noqa: E402
from .cleaning import clean_v3                          # noqa: E402

# SY 엑셀 (위치 기반, 병합셀 ffill) — v2 검증된 매핑
SY_COLS = dict(idx=0, dept=1, recovery=2, anrec=3, preop=4, premed=5, gold=6,
               prompt=7, model=8, gen=9,
               brevity=10, critical=11, free=12, err=13, over=14, under=15,
               read=16, comment=17)
SY_HEADER_ROWS = 3


def load_sy_scored(xlsx_path):
    """전문의 채점행 로드 → [{idx, model, gen, h_*}, ...]"""
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name="데이터", header=None)
    data = df.iloc[SY_HEADER_ROWS:].reset_index(drop=True)
    inv = {v: k for k, v in SY_COLS.items()}
    named = data.rename(columns=inv)
    for role in ("idx", "gold"):
        named[role] = named[role].ffill()

    rows = []
    for _, r in named.iterrows():
        try:
            cidx = int(float(r["idx"]))
        except Exception:
            continue
        gen = r.get("gen")
        if gen is None or isinstance(gen, float) or not str(gen).strip():
            continue

        def num(role):
            try:
                return float(r.get(role))
            except Exception:
                return None

        rows.append(dict(
            idx=cidx, model=str(r.get("model", "")).strip().lower(),
            gen=str(gen).strip(),
            h_brevity=num("brevity"), h_critical=num("critical"),
            h_err=num("err"), h_over=num("over"), h_under=num("under"),
            h_read=num("read"),
        ))
    print(f"[calibrate] SY 채점행 {len(rows)}건 로드")
    return rows


def _family_of_gen_model(name: str) -> str:
    n = (name or "").lower()
    if "llama" in n:
        return "llama"
    if "qwen" in n:
        return "qwen"
    if "gemma" in n:
        return "gemma"
    return "other"


def run_judge(judge_key, rows, gold_df, vital_map, checklist, backend):
    """단일 judge로 168행 배치 채점 → rows에 <judge>_cov/faith/brev/comp 추가."""
    from .engine import JudgeEngine
    engine = JudgeEngine(model_path(judge_key), backend=backend)

    sid_by_idx = {i: get_sid(gold_df.iloc[i]) for i in range(len(gold_df))}
    prompts = dict(cov=[], fa=[], br=[])
    index = dict(cov=[], fa=[], br=[])
    systems = {}
    meta = []
    for k, row in enumerate(rows):
        gen, status, _flags = clean_v3(row["gen"])
        sid = sid_by_idx.get(row["idx"], -1)
        entry = checklist.get(str(sid), {"items": [], "is_normal_case": False,
                                         "source": "no_gold"})
        emr = build_emr_text(gold_df.iloc[row["idx"]]) if row["idx"] < len(gold_df) else ""
        vital = vital_map.get(sid, "")
        meta.append(dict(gen=gen, status=status, entry=entry))
        if status != "ok":
            continue
        fast = M.fast_path(gen, status, entry)
        if fast is not None:
            meta[k]["fast"] = fast
            continue
        if entry.get("items"):
            s, u = M.build_coverage_prompt(gen, entry)
            systems["cov"] = s; prompts["cov"].append(u); index["cov"].append(k)
        s, u = M.build_faithfulness_prompt(emr, vital, gen)
        systems["fa"] = s; prompts["fa"].append(u); index["fa"].append(k)
        s, u = M.build_brevity_prompt(gen)
        systems["br"] = s; prompts["br"].append(u); index["br"].append(k)

    res = {}
    for ax in ("cov", "fa", "br"):
        if prompts[ax]:
            out = engine.chat_json(prompts[ax], system=systems[ax], want="obj", retries=2)
            res[ax] = dict(zip(index[ax], out))
        else:
            res[ax] = {}
    engine.unload()

    for k, row in enumerate(rows):
        mt = meta[k]
        if mt["status"] != "ok":
            sc = M._base(coverage=0.0, faithfulness=0.0, brevity=0.0, composite=0.0,
                         gate="degenerate")
        elif "fast" in mt:
            sc = mt["fast"]
        else:
            cov = M.parse_coverage(res["cov"].get(k), mt["entry"]) if k in res["cov"] else \
                dict(coverage=None, covered=[], partial=[], missed=[],
                     judge_failed=False, reason="no_items")
            fa = M.parse_faithfulness(res["fa"].get(k))
            br = M.parse_brevity(res["br"].get(k))
            sc = M.composite_from_axes(cov, fa, br, mt["entry"])
        for ax in ("coverage", "faithfulness", "brevity", "composite"):
            row[f"{judge_key}.{ax}"] = sc.get(ax)
    return rows


# 방향 기대: (v3축, 전문의 축, 기대부호)
PAIRS = [
    ("coverage", "h_critical", "+"), ("coverage", "h_under", "-"),
    ("faithfulness", "h_err", "-"),
    ("brevity", "h_brevity", "+"), ("brevity", "h_over", "-"),
    ("composite", "h_critical", "+"),
]


def report(rows, judges):
    import pandas as pd
    from scipy import stats as S
    df = pd.DataFrame(rows)
    df["gen_family"] = df["model"].map(_family_of_gen_model)

    lines = []
    for jk in judges:
        lines.append(f"\n══ judge = {jk} ══")
        for scope, sub in [("전체", df)] + [(f"family={f}", g)
                                            for f, g in df.groupby("gen_family")]:
            for v3c, hc, exp in PAIRS:
                col = f"{jk}.{v3c}"
                s = sub[[col, hc]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(s) < 10:
                    continue
                rho, p = S.spearmanr(s[col], s[hc])
                ok = "✓" if ((rho > 0) == (exp == "+")) else "✗방향불일치"
                lines.append(f"  [{scope:12s}] {v3c:13s} vs {hc:10s}: "
                             f"rho={rho:+.3f} p={p:.3g} n={len(s)} {ok}")
    txt = "\n".join(lines)
    print(txt)
    return df, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--backend", type=str, default="auto")
    ap.add_argument("--checklist", type=str, default=str(GOLD_CHECKLIST_JSON))
    args = ap.parse_args()

    import pickle
    from ..config_v3 import GOLD_PKL, VITAL_MAP_PKL
    import pandas as pd
    gold_df = pd.read_pickle(GOLD_PKL)
    with open(VITAL_MAP_PKL, "rb") as f:
        vital_map = pickle.load(f)
    checklist = load_checklist(args.checklist)
    if not checklist:
        print("[calibrate] checklist 없음 — 먼저 build_gold_checklist_v3 실행")
        return

    rows = load_sy_scored(GOLD_SY_XLSX)
    for jk in EVAL_JUDGES:
        rows = run_judge(jk, rows, gold_df, vital_map, checklist, args.backend)

    df, txt = report(rows, EVAL_JUDGES)
    out_dir = ensure_dir(DATA_DIR / "preprocessed")
    out_csv = out_dir / "calibration_v3.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    (out_dir / "calibration_v3_report.txt").write_text(txt, encoding="utf-8")
    print(f"\n[calibrate] 저장: {out_csv}")


if __name__ == "__main__":
    main()
