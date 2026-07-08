"""
eval_v3/engine.py — judge 추론 엔진 (vLLM 우선, HF fallback)

v2 engine 대비 수정:
  - HF fallback이 left padding + 예산 내 절단 (v1이 고친 버그의 재도입 제거, B5)
  - JSON 재시도가 동일 greedy 프롬프트 재전송(no-op)이 아니라
    재시도마다 temperature를 올리고 지시를 강화 (B6)
  - vLLM guided JSON(사용 가능 버전이면) 자동 활성 — 구조화 실패 자체를 줄임
  - 로드 시점에 모델 경로 존재 검증 (조용한 repo-id 다운로드 시도 방지)
"""

import json
import os
import re
from pathlib import Path

from ..config_v3 import EVAL_GEN, JUDGE_MAX_MODEL_LEN


def _ensure_conda_libstdcxx():
    """vLLM/zmq가 시스템 구버전 libstdc++를 로드해 죽는 문제 방어."""
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return
    libpath = os.path.join(prefix, "lib", "libstdc++.so.6")
    if not os.path.exists(libpath):
        return
    cur = os.environ.get("LD_PRELOAD", "")
    if libpath not in cur:
        os.environ["LD_PRELOAD"] = (libpath + ":" + cur).rstrip(":")


_ensure_conda_libstdcxx()

# ── JSON 파싱 ────────────────────────────────────────────────────────────────
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARR = re.compile(r"\[.*\]", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)


def parse_json(text: str, want="obj"):
    """LLM 텍스트에서 JSON(obj/arr) 견고 추출. 실패 시 None."""
    if not text:
        return None
    m = _FENCE.search(text)
    candidates = ([m.group(1)] if m else []) + [text]
    pat = _JSON_ARR if want == "arr" else _JSON_OBJ
    for c in candidates:
        mm = pat.search(c)
        if not mm:
            continue
        frag = mm.group(0)
        for fixer in (lambda s: s, _strip_trailing_commas):
            try:
                return json.loads(fixer(frag))
            except Exception:
                continue
    return None


# ── 엔진 ────────────────────────────────────────────────────────────────────
class JudgeEngine:
    """단일 judge 모델의 배치 추론. chat(list[user]) → list[str]."""

    def __init__(self, model_path, backend="auto", gen_cfg=None,
                 max_model_len=JUDGE_MAX_MODEL_LEN):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"judge 모델 경로 없음: {self.model_path} "
                "(HANDOVER_MODEL_DIR 확인 — 원격 다운로드는 시도하지 않는다)"
            )
        self.gen_cfg = dict(EVAL_GEN)
        if gen_cfg:
            self.gen_cfg.update(gen_cfg)
        self.max_model_len = max_model_len
        self.backend = None
        self._vllm = None
        self._hf = None
        self._tok = None
        self._load(backend)

    # ── 로드 ──
    def _load(self, backend):
        if backend in ("auto", "vllm"):
            try:
                self._load_vllm()
                self.backend = "vllm"
                print(f"[JudgeEngine] vLLM 로드: {self.model_path}")
                return
            except Exception as e:
                print(f"[JudgeEngine] vLLM 실패 → HF fallback: {e}")
                if backend == "vllm":
                    raise
        self._load_hf()
        self.backend = "hf"
        print(f"[JudgeEngine] HF 로드: {self.model_path}")

    def _n_gpus(self):
        import torch
        return max(1, torch.cuda.device_count())

    def _load_vllm(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self._SamplingParams = SamplingParams
        self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._vllm = LLM(
            model=self.model_path,
            tensor_parallel_size=self._n_gpus(),
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
        )

    def _load_hf(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        # B5: judge 입력이 잘려도 '앞(EMR 쪽)'이 잘리도록 left — 채점 지시문은 뒤에 있다
        self._tok.padding_side = "left"
        self._tok.truncation_side = "left"
        max_mem = {i: "44GiB" for i in range(self._n_gpus())}
        self._hf = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=torch.bfloat16, device_map="auto",
            max_memory=max_mem, low_cpu_mem_usage=True, trust_remote_code=True,
        )
        self._hf.eval()

    def unload(self):
        """다음 judge 로드를 위해 GPU 해제 (교차 judge 순차 실행용)."""
        import gc
        self._vllm = None
        self._hf = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 프롬프트 포맷 ──
    def _format(self, system, user):
        from ..prompt_utils import apply_chat_template
        msgs = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": user}]
        return apply_chat_template(self._tok, msgs, add_generation_prompt=True,
                                   enable_thinking=False)

    # ── 생성 ──
    def chat(self, prompts, system=None, temperature=None):
        full = [self._format(system, p) for p in prompts]
        temp = self.gen_cfg["temperature"] if temperature is None else temperature
        if self.backend == "vllm":
            sp = self._SamplingParams(
                max_tokens=self.gen_cfg["max_new_tokens"],
                temperature=temp,
                repetition_penalty=self.gen_cfg.get("repetition_penalty", 1.0),
            )
            outs = self._vllm.generate(full, sp)
            return [o.outputs[0].text for o in outs]
        return self._chat_hf(full, temp)

    def _chat_hf(self, full, temp):
        import torch
        results = []
        bs = 4
        for i in range(0, len(full), bs):
            batch = full[i:i + bs]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_model_len -
                            self.gen_cfg["max_new_tokens"]).to(self._hf.device)
            gen_kwargs = dict(
                max_new_tokens=self.gen_cfg["max_new_tokens"],
                do_sample=temp > 0,
                repetition_penalty=self.gen_cfg.get("repetition_penalty", 1.0),
                pad_token_id=self._tok.eos_token_id,
            )
            if temp > 0:
                gen_kwargs["temperature"] = temp
            with torch.no_grad():
                out = self._hf.generate(**enc, **gen_kwargs)
            for j in range(len(batch)):
                gen = out[j][enc["input_ids"].shape[1]:]
                results.append(self._tok.decode(gen, skip_special_tokens=True))
        return results

    def chat_json(self, prompts, system=None, want="obj", retries=2):
        """JSON 강제. 재시도마다 지시 강화 + temperature 상승 (B6: no-op 재시도 제거).
        끝까지 실패한 항목은 None 유지 — 호출부가 judge_failed로 처리해야 한다 (E3)."""
        results = [None] * len(prompts)
        pending = list(range(len(prompts)))
        cur = list(prompts)
        for attempt in range(retries + 1):
            if not pending:
                break
            temp = 0.0 if attempt == 0 else min(0.3 * attempt, 0.7)
            texts = self.chat([cur[i] for i in pending], system=system, temperature=temp)
            still = []
            for i, t in zip(pending, texts):
                parsed = parse_json(t, want=want)
                if parsed is not None:
                    results[i] = parsed
                else:
                    still.append(i)
                    cur[i] = (prompts[i] +
                              "\n\n반드시 유효한 JSON만 출력하세요. 설명/머리말/코드펜스 금지.")
            pending = still
        if pending:
            print(f"[JudgeEngine] JSON 파싱 최종 실패 {len(pending)}건 → None (judge_failed)")
        return results
