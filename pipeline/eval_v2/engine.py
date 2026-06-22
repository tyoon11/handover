"""
eval_v2/engine.py — v2 평가용 로컬 LLM 엔진 (gemma-4-31B)

역할: 하나의 강력한 로컬 instruct LLM이 평가의 세 가지 일을 모두 수행한다.
  (A) EMR/인계문에서 actionable finding 추출 (checklist 부트스트랩)
  (B) Coverage 채점 — 출력이 각 checklist 항목을 cover하는가
  (C) Faithfulness 채점 — 출력 claim들이 EMR과 entailment하는가

API 불가(로컬 전용). vLLM tensor-parallel 우선, 실패 시 HF fallback.
모든 호출은 batched + 구조화 JSON 강제 + 견고한 파싱/재시도.
"""

import os
import re
import json
from pathlib import Path

# vLLM/zmq가 시스템 구버전 libstdc++를 로드해 죽는 문제 방어 (04_inference와 동일)
def _ensure_conda_libstdcxx():
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


# ── JSON 파싱 (LLM 출력에서 첫 JSON 블록 추출) ──────────────────────────────
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARR = re.compile(r"\[.*\]", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str, want="obj"):
    """LLM 텍스트 출력에서 JSON(obj/array)을 견고하게 추출. 실패 시 None."""
    if not text:
        return None
    # ```json ... ``` 펜스 우선
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    pat = _JSON_ARR if want == "arr" else _JSON_OBJ
    for c in candidates:
        mm = pat.search(c)
        if mm:
            frag = mm.group(0)
            for fixer in (lambda s: s, _strip_trailing_commas):
                try:
                    return json.loads(fixer(frag))
                except Exception:
                    continue
    return None


def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)


# ── 정제 v2 (개선안 4) — v1 clean_output의 greedy preamble 삭제 버그 제거 ────
# v1 문제: _RE_THINK_PREAMBLE이 lookahead 앵커까지 앞부분을 통째로 삭제 →
#          "intentional hypothermia를 종료합니다" → "를 종료합니다." 같은 조각 생성.
#          또 len<5면 "특이사항 없음"으로 강제 치환 → 생성실패와 정상출력 혼동.
# v2 원칙: think 블록만 안전하게 제거, 조각화 금지, 실패는 명시 플래그로 분리.

_RE_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RE_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_RE_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_RE_ROLE = re.compile(r"^\s*(?:assistant|user|system)\s*:?\s*$", re.MULTILINE | re.IGNORECASE)


def detect_repetition(text: str, n: int = 4, thresh: float = 0.5) -> bool:
    """n-gram 반복 비율이 높으면 degenerate(반복 루프)로 판정."""
    toks = text.split()
    if len(toks) < n * 3:
        return False
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return False
    uniq = len(set(grams)) / len(grams)
    return uniq < thresh


def clean_v2(raw: str):
    """v2 정제. 반환: (cleaned_text, status)
       status ∈ {"ok", "empty", "repetition", "truncated"}
       - 실패여도 텍스트를 임의 문구로 치환하지 않는다(점수 단계에서 status로 분기)."""
    if raw is None:
        return "", "empty"
    text = raw
    text = _RE_THINK_CLOSED.sub("", text)
    # 안 닫힌 think: 닫힘이 없으면 think 이후를 trailing으로 간주해 제거
    if "<think>" in text and "</think>" not in text:
        text = _RE_THINK_OPEN.sub("", text)
    text = _RE_ROLE.sub("", text)
    cleaned = text.strip()

    if len(cleaned) < 3:
        return cleaned, "empty"
    if detect_repetition(cleaned):
        return cleaned, "repetition"
    # 잘림 휴리스틱: 종결부호/한글 종결 없이 조사로 끝나면 truncated 의심
    if re.search(r"(를|을|이|가|에|의|로|와|과)\s*$", cleaned) and not cleaned.endswith(("다", "요", ".", "음", "함")):
        return cleaned, "truncated"
    return cleaned, "ok"


# ── LLM 엔진 ────────────────────────────────────────────────────────────────
class EvalEngine:
    """gemma-4-31B 기반 배치 추론 엔진. chat() 으로 프롬프트 리스트 → 텍스트 리스트."""

    def __init__(self, model_path, backend="auto", gen_cfg=None):
        self.model_path = str(model_path)
        self.gen_cfg = gen_cfg or {}
        self.backend = None
        self._vllm = None
        self._hf = None
        self._tok = None
        self._load(backend)

    def _load(self, backend):
        if backend in ("auto", "vllm"):
            try:
                self._load_vllm()
                self.backend = "vllm"
                print(f"[EvalEngine] vLLM 로드 완료: {self.model_path}")
                return
            except Exception as e:
                print(f"[EvalEngine] vLLM 로드 실패 → HF fallback: {e}")
                if backend == "vllm":
                    raise
        self._load_hf()
        self.backend = "hf"
        print(f"[EvalEngine] HF 로드 완료: {self.model_path}")

    def _n_visible_gpus(self):
        import torch
        return max(1, torch.cuda.device_count())

    def _load_vllm(self):
        from vllm import LLM, SamplingParams  # noqa
        from transformers import AutoTokenizer
        self._SamplingParams = SamplingParams
        self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tp = self._n_visible_gpus()
        self._vllm = LLM(
            model=self.model_path,
            tensor_parallel_size=tp,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=False,
        )

    def _load_hf(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        max_mem = {i: "44GiB" for i in range(self._n_visible_gpus())}
        self._hf = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_mem,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._hf.eval()

    def _format(self, system, user):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        try:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # system role 미지원 템플릿 → user에 합침
            merged = (system + "\n\n" + user) if system else user
            return self._tok.apply_chat_template(
                [{"role": "user", "content": merged}], tokenize=False, add_generation_prompt=True
            )

    def chat(self, prompts, system=None):
        """prompts: list[str] (user 메시지). 반환: list[str] (생성 텍스트)."""
        full = [self._format(system, p) for p in prompts]
        if self.backend == "vllm":
            return self._chat_vllm(full)
        return self._chat_hf(full)

    def _chat_vllm(self, full):
        sp = self._SamplingParams(
            max_tokens=self.gen_cfg.get("max_new_tokens", 1024),
            temperature=self.gen_cfg.get("temperature", 0.0),
            repetition_penalty=self.gen_cfg.get("repetition_penalty", 1.0),
        )
        outs = self._vllm.generate(full, sp)
        return [o.outputs[0].text for o in outs]

    def _chat_hf(self, full):
        import torch
        results = []
        bs = 4
        for i in range(0, len(full), bs):
            batch = full[i:i + bs]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=8192).to(self._hf.device)
            with torch.no_grad():
                out = self._hf.generate(
                    **enc,
                    max_new_tokens=self.gen_cfg.get("max_new_tokens", 1024),
                    do_sample=False,
                    repetition_penalty=self.gen_cfg.get("repetition_penalty", 1.0),
                    pad_token_id=self._tok.eos_token_id,
                )
            for j in range(len(batch)):
                gen = out[j][enc["input_ids"].shape[1]:]
                results.append(self._tok.decode(gen, skip_special_tokens=True))
        return results

    def chat_json(self, prompts, system=None, want="obj", retries=2):
        """JSON 강제. 파싱 실패한 항목만 재시도. 끝까지 실패 시 None 유지."""
        results = [None] * len(prompts)
        pending = list(range(len(prompts)))
        attempt = 0
        cur_prompts = list(prompts)
        while pending and attempt <= retries:
            sub = [cur_prompts[i] for i in pending]
            texts = self.chat(sub, system=system)
            still = []
            for i, t in zip(pending, texts):
                parsed = parse_json(t, want=want)
                if parsed is not None:
                    results[i] = parsed
                else:
                    still.append(i)
            pending = still
            # 재시도 시 더 강하게 JSON만 요구
            if pending:
                for i in pending:
                    cur_prompts[i] = (
                        prompts[i]
                        + "\n\n반드시 유효한 JSON만 출력하세요. 설명/머리말 금지."
                    )
            attempt += 1
        return results
