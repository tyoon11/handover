"""
merging.py — LoRA merge 캐시 (vLLM용) — adapter '내용 해시'로 캐시 무효화 (B3)

v1의 결함: merged 캐시가 있으면 무조건 재사용 → adapter를 재학습해도 옛 가중치로 평가.
v3: merge 디렉토리 이름에 adapter 내용 해시를 넣는다. adapter가 바뀌면 경로가 바뀐다.
"""

from pathlib import Path

from .provenance import dir_hash


def merge_lora_for_vllm(model_path: str, base_model_path: str) -> str:
    """PEFT adapter면 base와 merge해 캐시 dir 반환. full model이면 그대로 반환."""
    model_path = Path(model_path)
    if not (model_path / "adapter_config.json").exists():
        return str(model_path)
    if not base_model_path:
        raise ValueError("PEFT 체크포인트인데 base 모델 경로 미지정")

    a_hash = dir_hash(model_path)
    merged = model_path / f"_merged_{a_hash}"
    if (merged / "config.json").exists():
        print(f"  [merge] 캐시 사용 (adapter hash={a_hash}): {merged}")
        return str(merged)

    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  [merge] LoRA merge 중 (CPU, adapter hash={a_hash})... base={base_model_path}")
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path), dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    m = PeftModel.from_pretrained(base, str(model_path))
    m = m.merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(merged))
    tok.save_pretrained(str(merged))
    del m, base
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  [merge] 완료: {merged}")
    return str(merged)
