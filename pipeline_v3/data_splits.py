"""
data_splits.py — split 로드 + 유출 차단 (T1/M1, Phase 3 dev-test 분리)

역할 고정 (한 split = 한 역할):
  GOLD (22)        : 최종 test. 봉인 — run_all_v3 --final 에서만 추론/평가.
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
    SPLIT_SEED, N_FEWSHOT_POOL, ensure_dir,
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


def load_splits(need=("gold", "sft", "dpo", "fewshot", "dev"), verbose=True) -> dict:
    """모든 split 로드 + 유출 검증. 반환: dict(name → DataFrame), 'vital_map' 포함.

    어떤 조합을 요청하든 gold와의 분리는 항상 검증한다.
    """
    out = {}
    gold_df = pd.read_pickle(GOLD_PKL)
    gold_sids = sid_set(gold_df)
    out["gold"] = gold_df

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
