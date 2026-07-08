"""
eval_v3/stats.py — n≈20 소표본에 맞는 통계 (E8, B4)

제공:
  - bootstrap_ci      : 케이스 단위 percentile bootstrap 95% CI
  - paired_permutation: 부호반전(paired sign-flip) permutation test — 정규성 가정 없음
  - paired_tests      : permutation + Wilcoxon(전부-0 diff 가드) + t (t.ppf로 임계값, B4)
  - micro_coverage    : 케이스별 항목수 편차를 뭉개지 않는 pooled(micro) coverage 병기
  - holm_correction   : 다중비교 보정
  - judge_agreement   : 2-judge Spearman/불일치 요약
"""

import numpy as np


def _rng(seed):
    return np.random.RandomState(seed)


def bootstrap_ci(values, iters=10000, alpha=0.05, seed=42):
    """percentile bootstrap CI. 반환 (mean, lo, hi, n). n<2면 CI=None."""
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    n = len(vals)
    if n == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    mean = float(vals.mean())
    if n < 2:
        return dict(mean=mean, lo=None, hi=None, n=n)
    rng = _rng(seed)
    idx = rng.randint(0, n, size=(iters, n))
    boots = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=round(mean, 4), lo=round(float(lo), 4),
                hi=round(float(hi), 4), n=n)


def paired_permutation(a, b, iters=20000, seed=42):
    """paired sign-flip permutation test (양측). a,b: 같은 케이스 순서의 점수 리스트."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    d = a - b
    n = len(d)
    if n == 0:
        return dict(p=None, mean_diff=None, n=0)
    obs = abs(d.mean())
    if np.allclose(d, 0):
        return dict(p=1.0, mean_diff=0.0, n=n)
    rng = _rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(iters, n))
    null = np.abs((signs * d).mean(axis=1))
    p = float((np.sum(null >= obs - 1e-12) + 1) / (iters + 1))
    return dict(p=round(p, 5), mean_diff=round(float(d.mean()), 4), n=n)


def paired_tests(a, b, iters=20000, seed=42, alpha=0.05):
    """permutation(주) + Wilcoxon/t(보조). B4: t 임계값은 자유도 기반 t.ppf."""
    from scipy import stats as S
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    out = dict(n=n, mean_diff=round(float(d.mean()), 4) if n else None)
    out["permutation"] = paired_permutation(a, b, iters=iters, seed=seed)

    if n >= 2 and d.std(ddof=1) > 0:
        t_stat, p_t = S.ttest_rel(a, b)
        se = d.std(ddof=1) / np.sqrt(n)
        tcrit = float(S.t.ppf(1 - alpha / 2, n - 1))
        out["t"] = dict(t=round(float(t_stat), 4), p=round(float(p_t), 5),
                        ci=[round(float(d.mean() - tcrit * se), 4),
                            round(float(d.mean() + tcrit * se), 4)],
                        dz=round(float(d.mean() / d.std(ddof=1)), 4))
    else:
        out["t"] = None

    if n >= 2 and not np.allclose(d, 0):
        try:
            w_stat, p_w = S.wilcoxon(a, b, zero_method="wilcox")
            out["wilcoxon"] = dict(W=float(w_stat), p=round(float(p_w), 5))
        except ValueError:
            out["wilcoxon"] = None
    else:
        out["wilcoxon"] = None       # 전부-0 diff 크래시 가드 (B4)
    return out


def holm_correction(pvals: dict, alpha=0.05) -> dict:
    """{name: p} → {name: dict(p, p_adj, significant)} (Holm-Bonferroni)."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev = 0.0
    for rank, (k, p) in enumerate(items):
        p_adj = min(1.0, max(prev, (m - rank) * p))
        prev = p_adj
        out[k] = dict(p=p, p_adj=round(p_adj, 5), significant=bool(p_adj < alpha))
    for k, v in pvals.items():
        if v is None:
            out[k] = dict(p=None, p_adj=None, significant=None)
    return out


def micro_coverage(records) -> dict:
    """항목 pooled(micro) coverage — 케이스별 항목수 1~9개 편차 보정 병기용.
    records: score dict 목록 (covered/partial/missed 필드 사용)."""
    num = den = 0.0
    for r in records:
        if r.get("coverage") is None:
            continue
        c, p, m = len(r.get("covered", [])), len(r.get("partial", [])), len(r.get("missed", []))
        num += c + 0.5 * p
        den += c + p + m
    return dict(micro_coverage=round(num / den, 4) if den else None,
                n_items=int(den))


def judge_agreement(scores_a, scores_b, axis="composite") -> dict:
    """같은 출력에 대한 두 judge 점수의 일치도 (Spearman + 평균 절대차)."""
    from scipy import stats as S
    pairs = [(a.get(axis), b.get(axis)) for a, b in zip(scores_a, scores_b)
             if a.get(axis) is not None and b.get(axis) is not None]
    if len(pairs) < 5:
        return dict(n=len(pairs), spearman=None, p=None, mad=None)
    xs, ys = zip(*pairs)
    rho, p = S.spearmanr(xs, ys)
    mad = float(np.mean(np.abs(np.array(xs) - np.array(ys))))
    return dict(n=len(pairs), spearman=round(float(rho), 4),
                p=round(float(p), 5), mad=round(mad, 4))
