"""
diagnose_v2.py — "학습하면 왜 망가지나 / raw는 왜 편차 크나" 3대 가설 일괄 검증

검증 항목:
  [1] SFT 데이터 구성 (A-1, A-2): 1행→3샘플 중 'A'/'B' 단일토큰 타깃 비율 +
      chosen 초단문("특이사항 없음") 편중.  → 생성분포 붕괴/mode collapse 원인.
  [2] 추론 출력의 think/누출 (B-2, A-4): generated_raw에서 <think>·//thought·
      judge토큰·프롬프트누출·문자벽 비율. 특히 raw·thinking 모델(qwen/hari).
  [3] RLAIF 로그 (A-3): trainer_state.json의 reward margin·loss·kl 폭주/발산 여부.
      (덤: SFT loss=0 붕괴 점검)

실행:
  python diagnose_v2.py --run_id <RUN_ID>
GPU 불필요. 서버에서 실행.
"""

import sys, os, argparse, json, re, statistics
from pathlib import Path


def _early_run_id():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run_id", type=str, default=None)
    rid = p.parse_known_args()[0].run_id
    if rid:
        os.environ["HANDOVER_RUN_ID"] = rid
    return rid


_RID = _early_run_id()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import SYNTH_PKL, OUTPUT_BASE, INFER_OUT, SFT_OUT, RLAIF_OUT

BAR = "=" * 72


# ── 공통 ────────────────────────────────────────────────────────────────────
_NO_ISSUE = ["특이사항 없음", "특이 사항 없음", "특이사항없음", "이상 없음", "none", "no issues"]


def is_no_issue(t):
    if not t:
        return True
    c = str(t).strip().lower().replace(" ", "").replace(".", "")
    return any(c == p.replace(" ", "").lower() for p in _NO_ISSUE)


# ── [1] SFT 데이터 구성 ──────────────────────────────────────────────────────
def check1_sft_data():
    print(f"\n{BAR}\n[1] SFT 데이터 구성 (A-1 단일토큰 / A-2 초단문 편중)\n{BAR}")
    if not Path(SYNTH_PKL).exists():
        print(f"  SKIP: {SYNTH_PKL} 없음"); return
    df = pd.read_pickle(SYNTH_PKL)
    n = len(df)
    cols = list(df.columns)
    has = "chosen" in cols and "rejected" in cols
    print(f"  파일: {SYNTH_PKL}  행수: {n}")
    if not has:
        print(f"  ⚠ chosen/rejected 컬럼 없음. 컬럼={cols[:20]}"); return

    # 02_sft_train.build_dataset: 1행 → 3샘플(생성1 + 'A' + 'B')
    total = n * 3
    print(f"\n  ▶ build_dataset(02_sft_train)는 1행→3샘플: 생성1 + judge 'A' + judge 'B'")
    print(f"    총 학습샘플 {total} = 생성 {n}({100*n/total:.0f}%) + "
          f"단일토큰 'A'/'B' {2*n}({100*2*n/total:.0f}%)")
    print(f"    → 학습의 2/3가 '한 글자' 출력 타깃  [A-1 확정: 코드+데이터]")

    # chosen 초단문 편중
    chosen = df["chosen"].astype(str)
    lens = chosen.str.len()
    no_issue = chosen.apply(is_no_issue).sum()
    short = (lens < 15).sum()
    print(f"\n  ▶ chosen(생성 타깃) 길이/편중:")
    print(f"    '특이사항 없음'류: {no_issue}/{n} ({100*no_issue/n:.1f}%)")
    print(f"    15자 미만 초단문:  {short}/{n} ({100*short/n:.1f}%)")
    print(f"    길이 char: 중앙값 {int(lens.median())}, 평균 {lens.mean():.0f}, "
          f"25%={int(lens.quantile(.25))} 75%={int(lens.quantile(.75))}")
    verdict = "높음(mode collapse 위험 큼)" if (no_issue/n > 0.2 or short/n > 0.3) else "보통"
    print(f"    → A-2 초단문 편중: {verdict}")


# ── [2] 추론 출력의 think/누출 ───────────────────────────────────────────────
_PATTERNS = {
    "think태그": re.compile(r"</?think", re.I),
    "thought누출": re.compile(r"(?://|_|\|)\s*thought|ownthought|thought\b", re.I),
    "judge토큰(A/B/RESULT)": re.compile(r"\[RESULT\]|Assistant [AB]\b|^\s*[AB]\s*$", re.I | re.M),
    "프롬프트누출": re.compile(r"Constraint|System:|###\s|You are an anesthesiolog", re.I),
    "문자벽": re.compile(r"(.)\1{7,}"),
    "영문Summary덤프": re.compile(r"\*\*Summary|Post-?Op(?:erative)? (?:Findings|Events)", re.I),
}
_MODEL_KEYS = ["gemma4_31b", "gemma4", "qwen35", "llama", "qwen", "hari"]


def _split_tag(tag):
    for mk in _MODEL_KEYS:
        if tag == mk or tag.startswith(mk + "_"):
            return mk, (tag[len(mk) + 1:] or "?")
    p = tag.split("_", 1)
    return p[0], (p[1] if len(p) > 1 else "?")


def check2_leak():
    print(f"\n{BAR}\n[2] 추론 출력 think/누출/붕괴 (B-2 thinking누출 / A-4 토큰누출)\n{BAR}")
    files = sorted(INFER_OUT.rglob("gold_results.jsonl"))
    if not files:
        print(f"  SKIP: {INFER_OUT}에 추론 결과 없음"); return
    print(f"  추론 파일 {len(files)}개 | 각 출력의 generated_raw에서 패턴 탐지\n")
    print(f"  {'model/exp':28s} {'n':>3}  " + "  ".join(f"{k[:8]:>8s}" for k in _PATTERNS))
    rows = []
    for f in files:
        tag = f.parent.name
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        raws = [str(r.get("generated_raw") or r.get("generated", "")) for r in recs]
        n = len(raws) or 1
        counts = {k: sum(1 for t in raws if pat.search(t)) for k, pat in _PATTERNS.items()}
        rows.append((tag, len(raws), counts))
    # thinking 모델/ raw 먼저
    def keyf(row):
        m, e = _split_tag(row[0])
        return (m not in ("qwen35", "qwen", "hari"), m, e)
    for tag, n, counts in sorted(rows, key=keyf):
        cells = "  ".join(f"{counts[k]:>8d}" for k in _PATTERNS)
        print(f"  {tag:28s} {n:>3}  {cells}")
    # 요약: thinking 모델 raw의 think/thought 누출
    print("\n  ▶ 해석: thinking 모델(qwen/qwen35/hari)의 think/thought 열이 높으면 B-2, "
          "\n    judge토큰(A/B) 열이 높으면 A-1의 학습누출, 문자벽/Summary덤프는 생성붕괴.")


# ── [3] RLAIF/SFT 학습 로그 ─────────────────────────────────────────────────
_LOG_KEYS = ["loss", "rewards/margins", "rewards/accuracies", "rewards/chosen",
             "rewards/rejected", "kl", "logps/chosen", "grad_norm"]


def _summ(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    nan = any(v != v for v in vals)  # NaN
    fin = [v for v in vals if v == v]
    if not fin:
        return "all-NaN"
    return (f"first={fin[0]:.3g} last={fin[-1]:.3g} "
            f"min={min(fin):.3g} max={max(fin):.3g}" + (" ⚠NaN있음" if nan else ""))


def check3_train_logs():
    print(f"\n{BAR}\n[3] 학습 로그: RLAIF margin/kl 폭주 + SFT loss=0 (A-3 / A-4)\n{BAR}")
    states = sorted(OUTPUT_BASE.rglob("trainer_state.json"))
    if not states:
        print(f"  SKIP: trainer_state.json 없음 (report_to=none이라 checkpoint에만 생성). "
              f"\n    경로 예: {RLAIF_OUT}/<tag>/checkpoint-*/trainer_state.json"); return
    # tag별로 log_history가 가장 긴 state 하나만
    best = {}
    for s in states:
        try:
            st = json.loads(s.read_text())
        except Exception:
            continue
        # tag = sft/rlaif 폴더명 (checkpoint 상위)
        tag = s.parent.name
        if tag.startswith("checkpoint"):
            tag = s.parent.parent.name
        lh = st.get("log_history", [])
        if tag not in best or len(lh) > len(best[tag][1]):
            best[tag] = (s, lh)

    for tag, (s, lh) in sorted(best.items()):
        kind = "SFT" if str(SFT_OUT) in str(s) else ("RLAIF" if str(RLAIF_OUT) in str(s) else "?")
        print(f"\n  [{kind}] {tag}  (log {len(lh)} steps)")
        keys_present = [k for k in _LOG_KEYS if any(k in e for e in lh)]
        # 그 외 숫자 키도 자동 포착
        extra = set()
        for e in lh:
            for k, v in e.items():
                if isinstance(v, (int, float)) and k not in _LOG_KEYS and k not in ("epoch", "step", "learning_rate"):
                    extra.add(k)
        for k in keys_present + sorted(extra):
            vals = [e[k] for e in lh if k in e]
            sm = _summ(vals)
            if sm:
                flag = ""
                if "loss" == k and any((e.get("loss") == 0) for e in lh):
                    flag = "  ⚠loss=0(학습안됨)"
                if "margins" in k:
                    mx = max((e[k] for e in lh if k in e and e[k] == e[k]), default=0)
                    if mx > 10:
                        flag = "  ⚠margin 폭주(과최적화)"
                print(f"      {k:22s} {sm}{flag}")
    print("\n  ▶ 해석: RLAIF의 rewards/margins가 계속 커지고 loss→0인데 출력이 깨지면 "
          "\n    KL 앵커 이탈(과최적화)=A-3. SFT loss가 0/NaN이면 라벨마스킹 실패=A-4.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default=_RID)
    ap.parse_args()
    print(f"\n진단 대상 run: {_RID}  (OUTPUT_BASE={OUTPUT_BASE})")
    check1_sft_data()
    check2_leak()
    check3_train_logs()
    print(f"\n{BAR}\n완료. [1] 단일토큰/초단문, [2] 누출패턴, [3] 학습로그 를 종합해 원인 확정.\n{BAR}")


if __name__ == "__main__":
    main()
