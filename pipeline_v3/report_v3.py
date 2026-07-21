"""
report_v3.py — 결과 요약 (통계 포함) + 리치 HTML 리포트

E8 반영:
  - 변형별 3축 평균 + bootstrap 95% CI
  - baseline({model}_raw) 대비 paired permutation test (공통 유효 케이스만) + Holm 보정
  - macro/micro coverage 병기, 제외 케이스(judge 실패/no_gold) 별도 표
  - gold checklist 검수 상태('전문의 검수 n / 잠정채택 n / 잠정 n') 명시 (E6)

HTML 리포트(자체완결·외부의존 없음, 라이트/다크 대응):
  1) 지표 설명 카드   2) 시각 대시보드(모델색·composite 막대+CI)
  3) 요약표          4) 제외/안전게이트 상세표
  5) 케이스별 예시 출력 토글(<details>) — 모델 생성문·축별 점수·게이트

출력: {OUTPUT_BASE}/report/results_{split}_v3.{csv,md,html}
※ HTML에는 EMR 원문 미포함 (PHI 최소화) — 모델 출력·checklist 항목·점수만.
"""

import argparse
import html as ihtml
import json

from .config_v3 import DARIN_INFER_OUT, EVAL_OUT, REPORT_OUT, RUN_ID, STATS, ensure_dir
from .eval_v3.stats import holm_correction, paired_tests

# 모델 식별 색(카테고리) — dataviz 검증 팔레트 slot 1..8 (라이트/다크). 순환 금지.
_SERIES_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                 "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_SERIES_DARK = ["#3987e5", "#199e70", "#c98500", "#008300",
                "#9085e9", "#e66767", "#d55181", "#d95926"]

# 지표 설명 (제목, 본문) — 리포트 상단 카드
_METRIC_DOCS = [
    ("composite (종합)",
     "0.5·coverage + 0.3·faithfulness + 0.2·brevity 의 가중합. "
     "dev split은 케이스별 전문의 checklist가 없어 <b>coverage를 측정하지 않으며</b>, "
     "faithfulness·brevity를 0.6/0.4로 재정규화해 산출한다. "
     "coverage를 포함한 완전한 3축 종합은 <b>gold split(--final)</b>에서만 나온다."),
    ("coverage (충실성·gold 전용)",
     "전문의 gold checklist 대비 recall. <b>macro</b>=케이스별 recall 평균, "
     "<b>micro</b>=전체 항목 pooled recall. dev에선 '—'."),
    ("faithfulness (사실성)",
     "생성문의 각 claim이 EMR·바이탈에 의해 뒷받침되는 비율(entailment). "
     "환각·근거 없는 서술은 감점. 프롬프트 주입 방어 구분자 사용."),
    ("brevity (간결성)",
     "과설명·행정 노이즈에 대한 감점. 임상적으로 필요한 정보를 담되 짧고 "
     "정보밀도가 높을수록 높다."),
    ("안전게이트",
     "<b>missed_abnormal</b>: 이상소견이 있는데 '특이사항 없음'류로 뭉갠 경우 → composite 0 강제. "
     "<b>degenerate</b>: 빈·조각 출력. 둘 다 임상 안전 실패로 별도 집계."),
    ("제외 (n 유효/제외)",
     "judge 실패 또는 gold 부재 케이스는 <b>점수가 아니라 인프라 문제</b>로 따로 센다. "
     "유효비율이 80% 미만이면 해당 변형의 평가는 신뢰 불가로 본다."),
    ("CI · vs raw · 유의성(✓)",
     "CI = bootstrap 95% 신뢰구간. vs raw = 같은 모델의 {model}_raw 대비 "
     "<b>공통 유효 케이스</b> paired permutation test, Holm 다중비교 보정. "
     "✓ = 보정 후에도 유의(α=" + str(STATS.get("alpha", 0.05)) + ")."),
    ("judge (교차·순환 방지)",
     "평가 judge는 gemma4_31b + qwen35 교차 채점. 대상 모델과 <b>같은 family의 judge는 "
     "채점에서 제외</b>(자기채점 순환 방지). 선호쌍 생성 judge(prometheus)와도 분리."),
]


def _load(split: str):
    """EVAL_OUT/*/ {split}_results_summary_v3.json + scores jsonl 로드."""
    rows, scores = {}, {}
    if not EVAL_OUT.exists():
        return rows, scores
    for d in sorted(EVAL_OUT.iterdir()):
        sf = d / f"{split}_results_summary_v3.json"
        jf = d / f"{split}_results_scores_v3.jsonl"
        if not sf.exists():
            continue
        summary = json.loads(sf.read_text(encoding="utf-8"))
        rows[summary["tag"]] = summary
        if jf.exists():
            scores[summary["tag"]] = [
                json.loads(l) for l in jf.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    return rows, scores


def _valid_map(recs):
    """{sid: composite} — official이 유효(excluded 아님)한 케이스만."""
    out = {}
    for r in recs:
        off = r.get("official") or {}
        if not off.get("excluded") and off.get("composite") is not None:
            out[r["sid"]] = off["composite"]
    return out


def pairwise_vs_raw(rows, scores):
    """모델별 {model}_raw 대비 각 변형의 paired permutation + Holm."""
    results = {}
    models = sorted({s["model"] for s in rows.values()})
    for m in models:
        base_tag = f"{m}_raw"
        if base_tag not in scores:
            continue
        base = _valid_map(scores[base_tag])
        pvals, details = {}, {}
        for tag, recs in scores.items():
            if not tag.startswith(m + "_") or tag == base_tag:
                continue
            comp = _valid_map(recs)
            common = sorted(set(base) & set(comp))
            if len(common) < 5:
                pvals[tag] = None
                details[tag] = dict(n_common=len(common))
                continue
            t = paired_tests([comp[s] for s in common], [base[s] for s in common],
                             iters=STATS["permutation_iters"], seed=STATS["seed"])
            pvals[tag] = t["permutation"]["p"]
            details[tag] = dict(n_common=len(common), mean_diff=t["mean_diff"],
                                permutation_p=t["permutation"]["p"],
                                t=t["t"], wilcoxon=t["wilcoxon"])
        holm = holm_correction(pvals, alpha=STATS["alpha"])
        for tag in details:
            details[tag]["holm"] = holm.get(tag)
        results[m] = details
    return results


def _fmt_ci(x):
    if not x or x.get("mean") is None:
        return "-"
    if x.get("lo") is None:
        return f"{x['mean']:.3f}"
    return f"{x['mean']:.3f} [{x['lo']:.3f},{x['hi']:.3f}]"


def _num(v, nd=3):
    """None/숫자 → 표시 문자열."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return ihtml.escape(str(v))


# ── 소스(EMR/GT) 로드 — --include_source 일 때만 (PHI 최소화 기본값 유지) ──────
_EMR_DISPLAY_CAP = 2600      # per-case EMR 표시 상한(문자) — 파일 비대 방지


def _scrub_phi(text):
    """직접식별자(병록번호/등록번호/환자명 라벨) 스크럽. build_emr_text는 이미
    이름·병록번호 '필드'를 미포함하므로, 자유텍스트에 라벨과 함께 박힌 잔여분 방어선."""
    import re
    if not text:
        return ""
    t = str(text)
    t = re.sub(r"(병\s*록\s*번\s*호|등\s*록\s*번\s*호|chart\s*(?:no\.?|number)|MRN)"
               r"\s*[:：]?\s*\S+", "[식별번호 제거]", t, flags=re.I)
    t = re.sub(r"(환자\s*(?:성명|이름|명)|성\s*명|환자명)\s*[:：]?\s*\S+",
               "[이름 제거]", t, flags=re.I)
    return t


def _load_sources(split):
    """{sid: {'emr','vital','gt'}} — 실패 시 {} (호출부에서 안내). EMR은 build_emr_text
    (임상 필드만; 이름·병록번호 미포함) + _scrub_phi. GT(사람 gold)는 gold split만."""
    out = {}
    try:
        import pickle
        from .config_v3 import VITAL_MAP_PKL
        from .data_splits import load_splits
        from .prompt_utils import build_emr_text, get_sid
        need = ("gold",) if split == "gold" else ("dev",)
        splits = load_splits(need=need)
        df = splits.get(split) if splits.get(split) is not None else splits[need[0]]
        with open(VITAL_MAP_PKL, "rb") as f:
            vital_map = pickle.load(f)

        gt_map = {}
        if split == "gold":
            try:
                import pandas as pd
                from .config_v3 import GOLD_KHS_XLSX, GOLD_PKL
                from .eval_v3 import checklist as CK
                gold_df = pd.read_pickle(GOLD_PKL)
                gold_refs, _drafts = CK.load_khs_gold(GOLD_KHS_XLSX, gold_df)
                for i in range(len(gold_df)):
                    s = get_sid(gold_df.iloc[i])
                    if s != -1 and i in gold_refs:
                        gt_map[s] = gold_refs[i]
            except Exception as e:
                print(f"[report_v3] GT(사람 gold) 로드 실패 → GT 생략 ({type(e).__name__}: {e})")

        for i in range(len(df)):
            row = df.iloc[i]
            s = get_sid(row)
            if s == -1:
                continue
            emr = _scrub_phi(build_emr_text(row))
            if len(emr) > _EMR_DISPLAY_CAP:
                emr = emr[:_EMR_DISPLAY_CAP] + " …(이하 생략)"
            out[s] = dict(emr=emr,
                          vital=_scrub_phi(str(vital_map.get(s, ""))),
                          gt=(_scrub_phi(gt_map.get(s, "")) or None))
    except Exception as e:
        print(f"[report_v3] source 로드 실패 → EMR/GT 생략 ({type(e).__name__}: {e})")
        return {}
    return out


# ── 다린(기존 연구) 모델 출력 로드 — --include_darin 일 때만 (비교 병기용) ─────────
#   data/inferenced/{raw,jsft,self_judge}/{model}.pkl 에 생성텍스트 컬럼 '인계요약지',
#   sid 컬럼 '수술 ID'. base 모델·평가셋이 v3와 달라 '같은 sid의 다른 모델 출력' 병기다.
_DARIN_SELFJUDGE = {"dpo": "rlaif_dpo", "simpo": "rlaif_simpo",
                    "dpo_jsft_1epoch": "sft_1ep_dpo", "dpo_jsft_3epoch": "sft_3ep_dpo"}


def _darin_tag(setting: str, stem: str):
    """(설명dir, 파일stem) → v3 tag. 예: ('self_judge','llama_dpo_jsft_1epoch')
    → 'llama_sft_1ep_dpo'. 매칭 불가 시 None."""
    if stem.startswith("llama"):
        fam, rest = "llama", stem[len("llama"):].lstrip("_")
    elif stem.startswith("qwen"):
        fam, rest = "qwen35", stem[len("qwen"):].lstrip("_")   # 다린 qwen → v3 qwen35
    else:
        return None
    if setting == "raw":
        return f"{fam}_raw"
    if setting == "jsft":
        return f"{fam}_sft_3ep" if rest == "3epoch" else f"{fam}_sft_1ep"
    if setting == "self_judge":
        v = _DARIN_SELFJUDGE.get(rest)
        return f"{fam}_{v}" if v else None
    return None


def _darin_sid_and_text_cols(df):
    """MultiIndex/flat 모두 대응: (sid_col, text_col) 반환. 못 찾으면 (None,None)."""
    def find(pred):
        for c in df.columns:
            top = c[0] if isinstance(c, tuple) else c
            if pred(str(top)):
                return c
        return None
    sid_c = find(lambda t: t.replace(" ", "") == "수술ID")
    txt_c = find(lambda t: t == "인계요약지")
    return sid_c, txt_c


def _load_darin_outputs(darin_root):
    """{v3_tag: {sid(int): 생성텍스트}} — darin_root = .../data/inferenced. 실패 시 {}."""
    import re
    from pathlib import Path
    root = Path(darin_root)
    if not root.exists():
        print(f"[report_v3] 다린 출력 경로 없음: {root} → 병기 생략")
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}
    out = {}
    n_files = 0
    for setting in ("raw", "jsft", "self_judge"):
        d = root / setting
        if not d.is_dir():
            continue
        for fp in sorted(d.glob("*.pkl")):
            tag = _darin_tag(setting, fp.stem)
            if not tag:
                print(f"[report_v3] 다린 파일 매핑 실패 skip: {setting}/{fp.name}")
                continue
            try:
                df = pd.read_pickle(fp)
            except Exception as e:
                print(f"[report_v3] 다린 pkl 로드 실패 {fp.name}: {type(e).__name__}")
                continue
            sid_c, txt_c = _darin_sid_and_text_cols(df)
            if sid_c is None or txt_c is None:
                print(f"[report_v3] 다린 컬럼(수술 ID/인계요약지) 못 찾음: {fp.name}")
                continue
            m = {}
            for _, row in df.iterrows():
                sv, tv = row[sid_c], row[txt_c]
                try:
                    sid = int(sv.iloc[0] if hasattr(sv, "iloc") else
                              (sv[0] if isinstance(sv, (list, tuple)) else sv))
                except (TypeError, ValueError):
                    continue
                if tv is None or (isinstance(tv, float) and tv != tv):
                    continue
                m[sid] = str(tv).strip()
            if m:
                out[tag] = m
                n_files += 1
    print(f"[report_v3] 다린 출력 로드: {n_files}개 변형, "
          f"{sum(len(v) for v in out.values())}건")
    return out


def write_outputs(split, rows, pw, scores=None, include_source=False,
                  darin_root=None):
    scores = scores or {}
    out_dir = ensure_dir(REPORT_OUT)
    # CSV
    import csv
    csv_path = out_dir / f"results_{split}_v3.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tag", "model", "experiment", "n_valid", "n_excluded",
                    "composite", "composite_lo", "composite_hi",
                    "coverage", "micro_coverage", "faithfulness", "brevity",
                    "gate_missed_abnormal", "gate_degenerate", "no_issue_rate",
                    "judge_agreement_rho", "vs_raw_diff", "vs_raw_p_holm",
                    "significant"])
        for tag, s in sorted(rows.items()):
            d = pw.get(s["model"], {}).get(tag, {})
            holm = d.get("holm") or {}
            w.writerow([
                tag, s["model"], s["experiment"], s["n_valid"], s["n_excluded"],
                s["composite"]["mean"], s["composite"]["lo"], s["composite"]["hi"],
                s["coverage"]["mean"], s["micro"]["micro_coverage"],
                s["faithfulness"]["mean"], s["brevity"]["mean"],
                s["gates"]["missed_abnormal"], s["gates"]["degenerate"],
                s["no_issue_rate"], s["judge_agreement"]["spearman"],
                d.get("mean_diff"), holm.get("p_adj"), holm.get("significant"),
            ])

    # MD
    md = [f"# v3 결과 요약 — split={split}\n",
          f"run_id: `{RUN_ID}`\n"]
    rev = next(iter(rows.values()), {}).get("checklist_review")
    if rev:
        md.append(f"\n**gold checklist 상태**: 전문의검수 {rev['reviewed']} / "
                  f"잠정채택 {rev['accepted_without_review']} / 잠정 {rev['provisional']} "
                  f"(검수 0이면 결과는 잠정치)\n")
    md.append("\n| tag | n(유효/제외) | composite [95% CI] | coverage(macro/micro) "
              "| faith | brev | 안전게이트 | vs raw Δ (Holm p) |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for tag, s in sorted(rows.items(), key=lambda kv: (kv[1]["model"],
                                                       -(kv[1]["composite"]["mean"] or 0))):
        d = pw.get(s["model"], {}).get(tag, {})
        holm = d.get("holm") or {}
        sig = " **✓**" if holm.get("significant") else ""
        vs = (f"{d.get('mean_diff'):+0.3f} (p={holm.get('p_adj')}){sig}"
              if d.get("mean_diff") is not None else "-")
        micro = s["micro"]["micro_coverage"]
        md.append(
            f"| {tag} | {s['n_valid']}/{s['n_excluded']} | {_fmt_ci(s['composite'])} "
            f"| {s['coverage']['mean']}/{micro} | {s['faithfulness']['mean']} "
            f"| {s['brevity']['mean']} | 누락게이트 {s['gates']['missed_abnormal']}건 "
            f"| {vs} |\n")
    md.append("\n- composite = 0.5·coverage + 0.3·faithfulness + 0.2·brevity "
              "(교차 judge 평균; 대상 모델과 같은 family judge는 제외)\n"
              "- '제외'는 judge 실패/gold 부재 케이스 — 점수가 아니라 인프라 문제로 따로 센다\n"
              "- vs raw: 공통 유효 케이스 paired permutation test, Holm 보정\n")
    (out_dir / f"results_{split}_v3.md").write_text("".join(md), encoding="utf-8")

    # HTML (리치 — 대시보드 + 케이스 토글). --include_source면 EMR/GT(비식별) 포함.
    sources = _load_sources(split) if include_source else {}
    darin = _load_darin_outputs(darin_root) if darin_root else {}
    html = _build_html(split, rows, pw, scores, rev, sources, darin)
    base = f"results_{split}_v3" + ("_source" if include_source else "") \
        + ("_darin" if darin else "")
    html_name = base + ".html"
    (out_dir / html_name).write_text(html, encoding="utf-8")

    print(f"[report_v3] 저장: {out_dir}/results_{split}_v3.(csv|md) + {html_name}"
          + (" [EMR/GT 포함 — 외부공유 금지]" if include_source else "")
          + (" [다린 출력 병기]" if darin else ""))


# ── 리치 HTML 빌더 ────────────────────────────────────────────────────────────
def _model_colors(models):
    """{model: (light_hex, dark_hex)} — 정렬 순서로 slot 배정(순환 금지, 8개 초과는 회색)."""
    out = {}
    for i, m in enumerate(models):
        if i < len(_SERIES_LIGHT):
            out[m] = (_SERIES_LIGHT[i], _SERIES_DARK[i])
        else:
            out[m] = ("#898781", "#898781")   # slot 소진 → 무채색(식별은 라벨로)
    return out


def _css(model_colors):
    per_model = "\n".join(
        f".m-{m}{{--c:{lt}}}" for m, (lt, dk) in model_colors.items())
    per_model_dark = "\n".join(
        f".m-{m}{{--c:{dk}}}" for m, (lt, dk) in model_colors.items())
    return f"""
:root{{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --track:#eeede9; --good:#006300; --warn:#b26a00; --crit:#c0362f;
}}
@media (prefers-color-scheme:dark){{
  :root{{
    --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --track:#2c2c2a; --good:#0ca30c; --warn:#fab219; --crit:#e66767;
  }}
{per_model_dark}
}}
{per_model}
*{{box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
  background:var(--plane);max-width:1160px;margin:0 auto;padding:24px 20px 80px;
  line-height:1.5;font-size:14px}}
h1{{font-size:22px;margin:0 0 2px}} h2{{font-size:17px;margin:34px 0 12px;
  padding-bottom:6px;border-bottom:1px solid var(--grid)}}
.sub{{color:var(--ink2);font-size:13px;margin:0 0 4px}}
.chk{{display:inline-block;margin-top:8px;padding:6px 10px;border:1px solid var(--border);
  border-radius:8px;background:var(--surface);font-size:12.5px;color:var(--ink2)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}}
.card{{border:1px solid var(--border);border-radius:10px;background:var(--surface);
  padding:11px 13px}}
.card h3{{font-size:13px;margin:0 0 4px}} .card p{{margin:0;font-size:12.5px;color:var(--ink2)}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:2px 0 14px;font-size:12.5px;color:var(--ink2)}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}
.sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.mgroup{{font-size:12.5px;color:var(--muted);margin:14px 0 4px;font-weight:600}}
.bar-row{{display:flex;align-items:center;gap:10px;margin:3px 0}}
.bar-label{{flex:0 0 168px;font-size:12px;text-align:right;color:var(--ink2);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bar-track{{position:relative;flex:1 1 auto;height:18px;background:var(--track);
  border-radius:4px}}
.bar-ci{{position:absolute;top:6px;height:6px;background:var(--border);border-radius:3px}}
.bar-fill{{position:absolute;top:0;left:0;height:18px;border-radius:4px;background:var(--c)}}
.bar-val{{flex:0 0 118px;font-size:12px;font-variant-numeric:tabular-nums;color:var(--ink)}}
.bar-val .sig{{color:var(--good);font-weight:700}}
.bar-val .dn{{color:var(--crit)}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;background:var(--surface)}}
.wrap{{overflow-x:auto}}
td,th{{border:1px solid var(--grid);padding:5px 8px;text-align:right}}
th{{background:var(--track);color:var(--ink2);font-weight:600;position:sticky;top:0}}
td.l,th.l{{text-align:left}} td{{font-variant-numeric:tabular-nums}}
tr:hover td{{background:rgba(127,127,127,.06)}}
.pill{{padding:1px 7px;border-radius:20px;font-size:11px;font-weight:600}}
.pill.ok{{color:var(--good);border:1px solid var(--good)}}
.pill.no{{color:var(--muted);border:1px solid var(--axis)}}
.g-miss{{color:var(--crit);font-weight:600}} .g-deg{{color:var(--warn);font-weight:600}}
details{{border:1px solid var(--border);border-radius:10px;background:var(--surface);
  margin:8px 0;overflow:hidden}}
details>summary{{cursor:pointer;padding:10px 13px;font-size:13px;font-weight:600;
  list-style:none;display:flex;justify-content:space-between;gap:12px;align-items:center}}
details>summary::-webkit-details-marker{{display:none}}
details>summary:hover{{background:rgba(127,127,127,.06)}}
summary .meta{{font-weight:400;color:var(--ink2);font-size:12px;
  font-variant-numeric:tabular-nums}}
.gen{{text-align:left;max-width:520px;white-space:pre-wrap;color:var(--ink)}}
.exc{{color:var(--muted);font-style:italic}}
.case{{border-top:1px solid var(--grid);padding:11px 13px}}
.case:first-child{{border-top:none}}
.case-hd{{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;font-size:12px;
  color:var(--ink2);font-variant-numeric:tabular-nums;margin-bottom:8px}}
.case-hd .sid{{font-weight:700;color:var(--ink)}} .case-hd b{{color:var(--ink)}}
.panes{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
@media(max-width:720px){{.panes{{grid-template-columns:1fr}}}}
.pane{{border:1px solid var(--border);border-radius:8px;background:var(--plane);
  padding:7px 9px;min-width:0}}
.pane.emr{{grid-column:1/-1}}
.pane.out{{background:var(--surface);box-shadow:inset 3px 0 0 var(--good)}}
.pane.gt{{box-shadow:inset 3px 0 0 var(--axis)}}
.pane.darin{{box-shadow:inset 3px 0 0 var(--muted)}}
.darinout{{color:var(--ink2);border-left:2px solid var(--grid)}}
.plabel{{font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);
  margin-bottom:4px}}
.ptext{{font-size:12px;white-space:pre-wrap;word-break:break-word;color:var(--ink);
  max-height:230px;overflow:auto;line-height:1.5}}
.srcnote{{margin:2px 0 12px;padding:7px 11px;border:1px solid var(--border);border-radius:8px;
  background:var(--surface);color:var(--ink2);font-size:12px}}
.foot{{margin-top:26px;color:var(--muted);font-size:12px}}
"""


def _dashboard(rows, pw, model_colors):
    """모델별 그룹, composite 내림차순 수평막대(+CI whisker, 직접라벨)."""
    out = ['<div class="legend">']
    for m, (lt, _dk) in model_colors.items():
        out.append(f'<span><span class="sw m-{m}" style="background:var(--c)"></span>{ihtml.escape(m)}</span>')
    out.append('</div>')
    by_model = {}
    for tag, s in rows.items():
        by_model.setdefault(s["model"], []).append((tag, s))
    for m in sorted(by_model):
        out.append(f'<div class="mgroup">{ihtml.escape(m)}</div>')
        grp = sorted(by_model[m], key=lambda kv: -(kv[1]["composite"]["mean"] or 0))
        for tag, s in grp:
            c = s["composite"]
            mean = c.get("mean") or 0
            lo, hi = c.get("lo"), c.get("hi")
            d = pw.get(m, {}).get(tag, {})
            holm = d.get("holm") or {}
            diff = d.get("mean_diff")
            sig = holm.get("significant")
            ci = ""
            if lo is not None and hi is not None:
                ci = (f'<div class="bar-ci" style="left:{lo*100:.1f}%;'
                      f'width:{max(0,(hi-lo))*100:.1f}%"></div>')
            vs = ""
            if diff is not None:
                cls = "sig" if sig else ("dn" if diff < 0 else "")
                mark = " ✓" if sig else ""
                vs = f' <span class="{cls}">({diff:+.3f}{mark})</span>'
            title = f"{tag}: composite {mean:.3f}"
            if lo is not None:
                title += f"  CI[{lo:.3f}, {hi:.3f}]"
            out.append(
                f'<div class="bar-row"><div class="bar-label" title="{ihtml.escape(tag)}">'
                f'{ihtml.escape(tag)}</div>'
                f'<div class="bar-track m-{m}" title="{ihtml.escape(title)}">{ci}'
                f'<div class="bar-fill" style="width:{mean*100:.1f}%"></div></div>'
                f'<div class="bar-val">{mean:.3f}{vs}</div></div>')
    return "".join(out)


def _summary_table(rows, pw):
    head = ("<div class='wrap'><table><tr>"
            "<th class='l'>tag</th><th>n(유효/제외)</th><th>composite [95% CI]</th>"
            "<th>coverage</th><th>micro</th><th>faith</th><th>brev</th>"
            "<th>누락게이트</th><th>붕괴</th><th>judge ρ</th><th>vs raw (Holm)</th></tr>")
    body = []
    for tag, s in sorted(rows.items(), key=lambda kv: (kv[1]["model"],
                                                       -(kv[1]["composite"]["mean"] or 0))):
        d = pw.get(s["model"], {}).get(tag, {})
        holm = d.get("holm") or {}
        diff = d.get("mean_diff")
        if diff is None:
            vs = "—"
        else:
            mark = ' <span class="pill ok">✓</span>' if holm.get("significant") else ""
            vs = f"{diff:+.3f} (p={_num(holm.get('p_adj'))}){mark}"
        gm = s["gates"]["missed_abnormal"]
        gd = s["gates"]["degenerate"]
        gm_s = f'<span class="g-miss">{gm}</span>' if gm else "0"
        gd_s = f'<span class="g-deg">{gd}</span>' if gd else "0"
        rho = (s.get("judge_agreement") or {}).get("spearman")
        body.append(
            f"<tr><td class='l'>{ihtml.escape(tag)}</td>"
            f"<td>{s['n_valid']}/{s['n_excluded']}</td>"
            f"<td>{_fmt_ci(s['composite'])}</td>"
            f"<td>{_num(s['coverage']['mean'])}</td>"
            f"<td>{_num(s['micro']['micro_coverage'])}</td>"
            f"<td>{_num(s['faithfulness']['mean'])}</td>"
            f"<td>{_num(s['brevity']['mean'])}</td>"
            f"<td>{gm_s}</td><td>{gd_s}</td><td>{_num(rho)}</td>"
            f"<td class='l'>{vs}</td></tr>")
    return head + "".join(body) + "</table></div>"


def _exclusions_table(rows, scores):
    """제외 사유·안전게이트가 있는 변형만 상세 집계."""
    lines = []
    for tag in sorted(rows):
        s = rows[tag]
        recs = scores.get(tag, [])
        reasons = {}
        for r in recs:
            off = r.get("official") or {}
            if off.get("excluded"):
                reasons[off.get("exclude_reason") or "?"] = \
                    reasons.get(off.get("exclude_reason") or "?", 0) + 1
        gm, gd = s["gates"]["missed_abnormal"], s["gates"]["degenerate"]
        if not reasons and not gm and not gd and not s["n_excluded"]:
            continue
        rs = ", ".join(f"{ihtml.escape(str(k))}×{v}" for k, v in reasons.items()) or "—"
        lines.append(
            f"<tr><td class='l'>{ihtml.escape(tag)}</td><td>{s['n_excluded']}</td>"
            f"<td class='l'>{rs}</td>"
            f"<td>{('<span class=g-miss>'+str(gm)+'</span>') if gm else 0}</td>"
            f"<td>{('<span class=g-deg>'+str(gd)+'</span>') if gd else 0}</td></tr>")
    if not lines:
        return "<p class='sub'>제외·안전게이트 발동 케이스 없음 (모든 변형 전건 유효).</p>"
    return ("<div class='wrap'><table><tr><th class='l'>tag</th><th>제외 n</th>"
            "<th class='l'>제외 사유</th><th>누락게이트</th><th>붕괴</th></tr>"
            + "".join(lines) + "</table></div>")


def _gate_span(gate):
    if gate == "missed_abnormal":
        return '<span class="g-miss">missed_abnormal</span>'
    if gate == "degenerate":
        return '<span class="g-deg">degenerate</span>'
    return "—" if not gate else ihtml.escape(gate)


def _case_table(split, recs, dmap=None):
    """소스 미포함 컴팩트 표 (PHI-free 기본). dmap: {sid: 다린출력} 있으면 열 추가."""
    is_gold = (split == "gold")
    show_darin = dmap is not None
    head = ("<div class='wrap'><table><tr><th>sid</th><th>comp</th>"
            + ("<th>cov</th>" if is_gold else "")
            + "<th>faith</th><th>brev</th><th>gate</th>"
            + ("<th class='l'>missed(gold)</th>" if is_gold else "")
            + "<th class='l'>생성 인계문 (v3)</th>"
            + ("<th class='l'>다린(기존) 출력</th>" if show_darin else "")
            + "</tr>")
    body = []
    for r in recs:
        off = r.get("official") or {}
        sid_raw = r.get("sid", "")
        sid = ihtml.escape(str(sid_raw))
        darin_c = ""
        if show_darin:
            dtxt = dmap.get(sid_raw) or (dmap.get(int(sid_raw))
                                         if str(sid_raw).isdigit() else None)
            darin_c = (f"<td class='gen darinout'>{ihtml.escape(dtxt)}</td>" if dtxt
                       else "<td class='darinout'>—</td>")
        if off.get("excluded"):
            reason = ihtml.escape(str(off.get("exclude_reason") or "제외"))
            cols = 6 + (2 if is_gold else 0) + (1 if show_darin else 0)
            body.append(f"<tr><td>{sid}</td><td class='l exc' colspan='{cols-1}'>"
                        f"제외: {reason}</td></tr>")
            continue
        cov_c = f"<td>{_num(off.get('coverage'))}</td>" if is_gold else ""
        missed_c = ""
        if is_gold:
            miss = off.get("missed") or []
            mtxt = ihtml.escape(", ".join(str(x) for x in miss)[:200]) if miss else "—"
            missed_c = f"<td class='l'>{mtxt}</td>"
        gen = ihtml.escape(str(r.get("generated", "")))
        body.append(
            f"<tr><td>{sid}</td><td>{_num(off.get('composite'))}</td>{cov_c}"
            f"<td>{_num(off.get('faithfulness'))}</td>"
            f"<td>{_num(off.get('brevity'))}</td><td>{_gate_span(off.get('gate') or '')}</td>"
            f"{missed_c}<td class='gen'>{gen or '<span class=exc>(빈 출력)</span>'}</td>"
            f"{darin_c}</tr>")
    return head + "".join(body) + "</table></div>"


def _case_blocks(split, recs, sources, dmap=None):
    """소스 포함 블록 — 케이스마다 입력(EMR·vital)·GT·모델출력 나란히.
    dmap: {sid: 다린출력} 있으면 v3 출력 옆에 다린 출력 패널 추가."""
    is_gold = (split == "gold")
    show_darin = dmap is not None
    out = []
    for r in recs:
        off = r.get("official") or {}
        sid = r.get("sid")
        sid_s = ihtml.escape(str(sid))
        if off.get("excluded"):
            reason = ihtml.escape(str(off.get("exclude_reason") or "제외"))
            out.append(f"<div class='case'><div class='case-hd'><span class='sid'>sid {sid_s}"
                       f"</span><span class='exc'>제외: {reason}</span></div></div>")
            continue
        hd = [f"<span class='sid'>sid {sid_s}</span>",
              f"<span>comp <b>{_num(off.get('composite'))}</b></span>"]
        if is_gold:
            hd.append(f"<span>cov {_num(off.get('coverage'))}</span>")
        hd.append(f"<span>faith {_num(off.get('faithfulness'))}</span>")
        hd.append(f"<span>brev {_num(off.get('brevity'))}</span>")
        if off.get("gate"):
            hd.append(f"<span>gate {_gate_span(off.get('gate'))}</span>")
        src = sources.get(sid) or sources.get(int(sid) if str(sid).isdigit() else sid) or {}
        panes = []
        if src.get("emr"):
            panes.append(f"<div class='pane emr'><div class='plabel'>입력 · EMR "
                         f"(비식별)</div><div class='ptext'>{ihtml.escape(src['emr'])}</div></div>")
        if src.get("vital"):
            panes.append(f"<div class='pane'><div class='plabel'>입력 · Intraop Vital</div>"
                         f"<div class='ptext'>{ihtml.escape(src['vital'])}</div></div>")
        if src.get("gt"):
            panes.append(f"<div class='pane gt'><div class='plabel'>정답 · 전문의 GT</div>"
                         f"<div class='ptext'>{ihtml.escape(str(src['gt']))}</div></div>")
        gen = ihtml.escape(str(r.get("generated", "")))
        panes.append(f"<div class='pane out'><div class='plabel'>모델 출력 (v3)</div>"
                     f"<div class='ptext'>{gen or '<span class=exc>(빈 출력)</span>'}</div></div>")
        if show_darin:
            dtxt = dmap.get(sid) or (dmap.get(int(sid))
                                     if str(sid).isdigit() else None)
            panes.append(
                f"<div class='pane darin'><div class='plabel'>다린(기존 모델) 출력</div>"
                f"<div class='ptext'>{ihtml.escape(dtxt) if dtxt else '<span class=exc>(해당 sid 출력 없음)</span>'}"
                f"</div></div>")
        out.append(f"<div class='case'><div class='case-hd'>{''.join(hd)}</div>"
                   f"<div class='panes'>{''.join(panes)}</div></div>")
    return "".join(out)


def _case_toggles(split, rows, scores, sources, darin=None):
    """변형별 <details> 토글 — 케이스별 점수·게이트·(옵션)입력/GT·모델출력·다린출력."""
    if not scores:
        return "<p class='sub'>per-case 점수 파일(scores jsonl) 없음 — 케이스 상세 생략.</p>"

    def _key(r):
        off = r.get("official") or {}
        return (bool(off.get("excluded")), -(off.get("composite") or 0))

    out = []
    if sources:
        note = ("입력 EMR은 <b>임상 필드만</b> 구성(병록번호·환자이름 미포함) + 라벨 식별자 스크럽한 "
                "<b>비식별</b> 텍스트입니다. 그래도 자유텍스트 잔여 위험이 있으니 <b>외부 공유·커밋 금지</b>. "
                + ("전문의 GT(사람 gold) 병기." if split == "gold"
                   else "dev split은 전문의 GT가 없어 입력·출력만 표시."))
        out.append(f"<div class='srcnote'>⚠ {note}</div>")
    for tag, s in sorted(rows.items(), key=lambda kv: (kv[1]["model"],
                                                       -(kv[1]["composite"]["mean"] or 0))):
        recs = scores.get(tag, [])
        if not recs:
            continue
        recs = sorted(recs, key=_key)
        dmap = darin.get(tag) if darin else None
        inner = _case_blocks(split, recs, sources, dmap) if sources \
            else _case_table(split, recs, dmap)
        meta = (f"composite {_num(s['composite']['mean'])} · "
                f"유효 {s['n_valid']}/{s['n_valid']+s['n_excluded']}")
        out.append(f"<details><summary><span>{ihtml.escape(tag)}</span>"
                   f"<span class='meta'>{meta}</span></summary>{inner}</details>")
    return "".join(out)


def _build_html(split, rows, pw, scores, rev, sources=None, darin=None):
    sources = sources or {}
    darin = darin or {}
    models = sorted({s["model"] for s in rows.values()})
    model_colors = _model_colors(models)
    cards = "".join(
        f"<div class='card'><h3>{ihtml.escape(t)}</h3><p>{b}</p></div>"
        for t, b in _METRIC_DOCS)
    chk = ""
    if rev:
        chk = (f"<div class='chk'><b>gold checklist</b> — 전문의검수 {rev['reviewed']} / "
               f"잠정채택 {rev['accepted_without_review']} / 잠정 {rev['provisional']} "
               f"(검수 0이면 결과는 잠정치)</div>")
    return (
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>v3 결과 — {ihtml.escape(split)} ({ihtml.escape(RUN_ID)})</title>"
        f"<style>{_css(model_colors)}</style>"
        f"<h1>v3 결과 — split={ihtml.escape(split)}</h1>"
        f"<p class='sub'>run_id <b>{ihtml.escape(RUN_ID)}</b> · 변형 {len(rows)}종 · "
        f"모델 {', '.join(ihtml.escape(m) for m in models)}</p>{chk}"
        "<h2>지표 설명</h2>"
        f"<div class='cards'>{cards}</div>"
        "<h2>대시보드 — composite (막대) + 95% CI (whisker), 모델색</h2>"
        f"{_dashboard(rows, pw, model_colors)}"
        "<h2>요약표</h2>"
        f"{_summary_table(rows, pw)}"
        "<h2>제외 · 안전게이트 상세</h2>"
        f"{_exclusions_table(rows, scores)}"
        "<h2>케이스별 예시 출력 (변형 클릭해 펼치기)</h2>"
        f"{_case_toggles(split, rows, scores, sources, darin)}"
        "<p class='foot'>composite = 0.5·coverage + 0.3·faithfulness + 0.2·brevity "
        "(dev는 coverage 미측정→faith/brev 재정규화). "
        + ("입력 EMR·GT는 비식별 처리(병록번호·이름 제외) — 외부 공유 금지."
           if sources else "EMR 원문은 PHI 최소화로 미포함(--include_source로 비식별 EMR/GT 표시).")
        + "</p>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["gold", "dev"], default="dev")
    ap.add_argument("--include_source", action="store_true",
                    help="케이스 토글에 비식별 EMR·GT 병기(별도 파일 *_source.html). "
                         "기본 off — 자동 리포트는 PHI-free 유지.")
    ap.add_argument("--include_darin", action="store_true",
                    help="케이스별로 다린(기존 연구) 모델 출력 병기. --darin_root 미지정 시 "
                         "config DARIN_INFER_OUT 사용. 파일명에 _darin 접미사.")
    ap.add_argument("--darin_root", default=None,
                    help="다린 inferenced 디렉토리 경로 "
                         "(raw/jsft/self_judge 하위폴더의 *.pkl(컬럼 '수술 ID','인계요약지')을 읽음). "
                         f"미지정 시 config 기본값 {DARIN_INFER_OUT} 사용.")
    args = ap.parse_args()
    darin_root = args.darin_root or (str(DARIN_INFER_OUT) if args.include_darin else None)
    rows, scores = _load(args.split)
    if not rows:
        print(f"[report_v3] {args.split} 평가 요약 없음 — evaluate 먼저 실행")
        return
    pw = pairwise_vs_raw(rows, scores)
    write_outputs(args.split, rows, pw, scores, include_source=args.include_source,
                  darin_root=(darin_root if args.include_darin else None))


if __name__ == "__main__":
    main()
