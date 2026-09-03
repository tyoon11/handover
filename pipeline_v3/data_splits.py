"""
data_splits.py — split 로드 + 유출 차단 (T1/M1, Phase 3 dev-test 분리)

역할 고정 (한 split = 한 역할):
  GOLD (22)        : v3.2에서 둘로 쪼갠다 (docs/PIPELINE_V3.2.md §3)
                     - GOLD_GT_FEWSHOT (6): 전문의 GT를 few-shot 예시로 제공하기 위한 몫.
                       SFT 타깃 생성(teacher)에서만 쓰이고 추론·평가에는 절대 들어가지 않는다.
                     - GOLD_TEST (16)     : 최종 test. 봉인 — run_all_v3 --final 에서만.
                     load_splits()["gold"] 는 **test 16** 을 돌려준다 (평가가 자동으로 16만 보게).
                     22 전체가 필요하면 ["gold_full"], few-shot 몫은 ["gold_gt_fewshot"].
  SFT_SRC (2736)   : SFT 타깃 생성 (gen_pairs --split sft).
  SELFJUDGE (2736) : DPO 선호쌍 생성 (gen_pairs --split dpo) — SFT와 분리 (T9).
  RLHF (150)       : few-shot pool(40) + dev set(110).
                     few-shot은 행별 로테이션으로만 사용 (T1: gold 유출 차단).

모든 로드는 load_splits() 하나로 — 로드 시점에 유출 assert가 강제된다.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from .config_v3 import (
    GOLD_PKL, SFT_SRC_PKL, SELFJUDGE_SRC_PKL, RLHF_PKL, VITAL_MAP_PKL,
    SPLIT_SEED, N_FEWSHOT_POOL, N_GOLD_GT_FEWSHOT, ensure_dir,
    is_no_issue_v3, load_khs_gold_override,
)
from .prompt_utils import get_sid


class LeakageError(RuntimeError):
    """평가셋이 학습/few-shot 경로에 유출되면 즉시 실패."""


def sid_set(df: pd.DataFrame) -> set:
    sids = set()
    for _, row in df.iterrows():
        s = get_sid(row)
        if s == -1:
            raise LeakageError(
                "수술ID를 추출할 수 없는 행 발견 — sid 없는 행은 어느 split에도 넣을 수 없다."
            )
        sids.add(s)
    return sids


def assert_disjoint(a: set, b: set, name_a: str, name_b: str):
    inter = a & b
    if inter:
        raise LeakageError(
            f"[유출] {name_a} ∩ {name_b} = {len(inter)}건 — "
            f"평가/학습 경계가 깨졌다. 파이프라인을 중단한다."
        )


def split_rlhf(rlhf_df: pd.DataFrame, seed: int = SPLIT_SEED,
               n_fewshot: int = N_FEWSHOT_POOL):
    """RLHF 150 → few-shot pool + dev set. sid 정렬 후 고정 seed 셔플 (재현 가능)."""
    order = rlhf_df.copy()
    order["_sid"] = [get_sid(r) for _, r in order.iterrows()]
    order = order.sort_values("_sid").reset_index(drop=True)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(order))
    fewshot = order.iloc[perm[:n_fewshot]].drop(columns=["_sid"])
    dev = order.iloc[perm[n_fewshot:]].drop(columns=["_sid"])
    return fewshot.reset_index(drop=True), dev.reset_index(drop=True)


def split_gold(gold_df: pd.DataFrame, seed: int = SPLIT_SEED,
               n_fewshot: int = N_GOLD_GT_FEWSHOT):
    """gold 22 → (GT few-shot 몫, 최종 test 몫). 결정적이고 층화된 분할 (v3.2 §3).

    층화 기준: GT가 '특이사항 없음'인 정상 케이스 비율을 두 집합에서 보존한다.
    (n=22에서 무작위로 6을 뽑으면 정상 케이스가 몰려 test 16의 구성이 크게 흔들린다.)
    GT override JSON이 없으면 층화 없이 seed 셔플만 하고 경고한다.
    """
    order = gold_df.copy()
    order["_sid"] = [get_sid(r) for _, r in order.iterrows()]
    order = order.sort_values("_sid").reset_index(drop=True)

    gt = load_khs_gold_override()
    if gt:
        normal = np.array([is_no_issue_v3(gt.get(str(s), "")) for s in order["_sid"]])
    else:
        print("[splits] ⚠ khs_gold_override.json 없음 — gold 분할 층화 불가(seed 셔플만). "
              "GT few-shot 6건에 정상 케이스가 몰릴 수 있다.")
        normal = np.zeros(len(order), dtype=bool)

    rng = np.random.RandomState(seed)
    idx_norm = np.where(normal)[0]
    idx_abn = np.where(~normal)[0]
    # 정상 케이스 몫을 비율대로 배정 (최소 0, 최대 정상 케이스 수)
    n_norm_pick = int(round(n_fewshot * len(idx_norm) / max(1, len(order))))
    n_norm_pick = min(n_norm_pick, len(idx_norm))
    n_abn_pick = min(n_fewshot - n_norm_pick, len(idx_abn))
    pick = np.concatenate([
        rng.permutation(idx_norm)[:n_norm_pick],
        rng.permutation(idx_abn)[:n_abn_pick],
    ])
    pick_set = set(int(i) for i in pick)
    fewshot = order.iloc[sorted(pick_set)].drop(columns=["_sid"]).reset_index(drop=True)
    test = order.iloc[[i for i in range(len(order)) if i not in pick_set]] \
                .drop(columns=["_sid"]).reset_index(drop=True)
    print(f"[splits] gold {len(order)} → GT few-shot {len(fewshot)} / test {len(test)} "
          f"(정상 케이스 {int(normal.sum())}건 중 {n_norm_pick}건을 few-shot으로)")
    return fewshot, test


def load_splits(need=("gold", "sft", "dpo", "fewshot", "dev"), verbose=True) -> dict:
    """모든 split 로드 + 유출 검증. 반환: dict(name → DataFrame), 'vital_map' 포함.

    어떤 조합을 요청하든 gold와의 분리는 항상 검증한다.
    """
    out = {}
    gold_df = pd.read_pickle(GOLD_PKL)
    gold_sids = sid_set(gold_df)        # 유출 검증은 항상 22건 전체 기준
    gt_fewshot_df, gold_test_df = split_gold(gold_df)
    assert_disjoint(sid_set(gt_fewshot_df), sid_set(gold_test_df),
                    "gold_gt_fewshot", "gold_test")
    out["gold_full"] = gold_df
    out["gold_gt_fewshot"] = gt_fewshot_df
    out["gold"] = gold_test_df          # 평가·추론이 보는 gold = test 16

    if "sft" in need or "dpo" in need:
        if "sft" in need:
            sft_df = pd.read_pickle(SFT_SRC_PKL)
            assert_disjoint(gold_sids, sid_set(sft_df), "gold", "sft_src")
            out["sft"] = sft_df
        if "dpo" in need:
            dpo_df = pd.read_pickle(SELFJUDGE_SRC_PKL)
            assert_disjoint(gold_sids, sid_set(dpo_df), "gold", "selfjudge_src")
            out["dpo"] = dpo_df

    if "fewshot" in need or "dev" in need:
        rlhf_df = pd.read_pickle(RLHF_PKL)
        rlhf_sids = sid_set(rlhf_df)
        assert_disjoint(gold_sids, rlhf_sids, "gold", "rlhf")
        fewshot_df, dev_df = split_rlhf(rlhf_df)
        assert_disjoint(sid_set(fewshot_df), sid_set(dev_df), "fewshot_pool", "dev")
        out["fewshot"] = fewshot_df
        out["dev"] = dev_df

    import pickle
    with open(VITAL_MAP_PKL, "rb") as f:
        out["vital_map"] = pickle.load(f)

    if verbose:
        sizes = {k: (len(v) if hasattr(v, "__len__") else "-") for k, v in out.items()}
        print(f"[splits] 로드+유출검증 완료: {sizes}")
    return out


def _sid_digest(sids: set) -> str:
    """split 지문 — 실제 sid(PHI)는 기록하지 않고 정렬해시만 남긴다."""
    joined = ",".join(str(s) for s in sorted(sids))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def write_split_manifest(out_dir, splits: dict):
    """run 산출물에 split 지문 기록 (재현성 증빙; PHI 없는 해시만)."""
    manifest = {}
    for name, df in splits.items():
        if name == "vital_map" or not isinstance(df, pd.DataFrame):
            continue
        sids = sid_set(df)
        manifest[name] = dict(n=len(df), sid_sha1=_sid_digest(sids))
    manifest["split_seed"] = SPLIT_SEED
    manifest["n_fewshot_pool"] = N_FEWSHOT_POOL
    path = ensure_dir(out_dir) / "split_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[splits] manifest 저장: {path}")
    return manifest
