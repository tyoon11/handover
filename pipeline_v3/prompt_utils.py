"""
prompt_utils.py — 프롬프트 빌드·chat template·절단의 유일한 소스 (T4/T5/T6)

원칙:
  - 절단은 항상 'EMR 본문만' 좌측(오래된 쪽)에서 자른다.
    시스템 프롬프트·지시문·### OUTPUT·generation header는 절대 잘리지 않는다.
  - 절단 후 반드시 assert_prompt_intact()로 생존 검증 (조용한 절단 금지).
  - 학습/추론/쌍생성/평가 모두 이 모듈만 사용 (v1의 3벌 중복 제거).
"""

from .config_v3 import MAX_PROMPT_TOKENS

# ── EMR 텍스트 빌드 (v1 build_emr_text와 동일 로직 — 검증된 부분 유지) ────────
EMR_SID_COL = "수술 ID"      # EMR MultiIndex level0 (공백 있음 — Vital의 '수술ID'와 다름)
EMR_HANDOFF_COL = ("마취기록", "기록", "")
EMR_PREOP_SUM_COL = ("마취전평가", "마취 전 상태 평가",
                     "마취 전 상태평가 -> 마취전 환자상태 요약")
EMR_PREMED_COL = ("마취전평가", "마취 전 상태 평가",
                  "마취 전 상태평가 -> 수술전 준비사항과 Premedication -> Premedication")


def _is_na(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float):
        try:
            return v != v  # NaN
        except Exception:
            return False
    return False


def safe_get(row, col) -> str:
    """MultiIndex Series에서 컬럼 값을 안전하게 추출 (dict/NaN 처리)."""
    try:
        v = row[col]
    except (KeyError, IndexError, TypeError):
        return ""
    if _is_na(v):
        return ""
    if isinstance(v, dict):
        vals = []
        for vlist in v.values():
            if isinstance(vlist, list):
                vals.extend(str(x) for x in vlist if x is not None)
            else:
                vals.append(str(vlist))
        return " ".join(vals)
    return str(v)


def get_sid(row) -> int:
    """행에서 수술ID 추출. 실패 시 -1 (호출부에서 반드시 검사할 것 — 조용한 fallback 금지)."""
    try:
        v = row[EMR_SID_COL]
        return int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
    except Exception:
        return -1


def build_emr_text(row) -> str:
    """Pre-anesthetic Summary + Premedication + TOTALS + Record (v1과 동일 포맷)."""
    preop = safe_get(row, EMR_PREOP_SUM_COL)
    premed = safe_get(row, EMR_PREMED_COL)
    anrec = safe_get(row, EMR_HANDOFF_COL)

    totals_lines = []
    try:
        items = row.items()
    except Exception:
        items = []
    for k, v in items:
        if (isinstance(k, tuple) and len(k) >= 2
                and k[0] == "마취기록" and k[1] == "마취기록TOTALS" and not _is_na(v)):
            label = k[2] if len(k) > 2 else str(k)
            totals_lines.append(f"{label}: {v}")
    totals_str = "\n".join(totals_lines)

    parts = [
        f"- Pre-anesthetic Patient Conditions Summary\n{preop}",
        f"- Preoperative Preparations and Premedication\n{premed}",
    ]
    if totals_str:
        parts.append(f"- Anesthetic TOTALS\n{totals_str}")
    parts.append(f"- Anesthetic Record\n{anrec}")
    return "\n\n".join(parts)


# ── 프롬프트 (v1과 동일 지시문 — 학습 연속성 유지) ───────────────────────────
SYSTEM_PROMPT = """You are an anesthesiologist giving an ultra-brief OR to PACU/ICU handoff AFTER surgery has fully ended.
Do NOT ask for or suggest any intraoperative checks; only summarize key post-op relevant findings from the EMR.

RULES:
- If no clinically meaningful abnormal findings, output EXACTLY and ONLY "특이사항 없음".
- If any exist, output 1-5 VERY short sentences in formal Korean.
- NEVER include normal/stable findings, routine vitals, surgery steps, administrative drug info, or transfer phrases.
- NO lists, NO explanations, NO repetition.

REPORT EVERY ONE of these six groups that has an abnormal finding; write nothing for a group
that has none. Omitting a group that HAS a finding is the worst possible error.
1 기저질환·약물 (comorbidity, drugs and post-op effect, allergy)
2 기도관리 (ETT/LMA size, difficult airway, extubation, laryngospasm/bronchospasm/croup, O2 need)
3 수술 중 이벤트 및 처치 (events and what was done, lines/devices)
4 수혈·수액 (blood loss, products, fluids, urine output)
5 수술 전 검사이상 (abnormal pre-op labs/ECG/echo/PFT affecting post-op care)
6 감기 유무 (recent URI and its timing)
Combine groups into one sentence rather than exceeding 5 sentences.

QUANTIFY vital abnormalities: state how long each lasted and its nadir/peak vs threshold, taken
from the vital summary. "저혈압" is insufficient; "20분간 저혈압(최저 55mmHg)" is correct.
Events marked ⚑ there are clinically significant and must appear.

Pediatric airway, fluids, and drug sensitivity are especially important."""

OUTPUT_HEADER = "### OUTPUT"


def build_user_prompt(emr_text: str, vital_summary: str = "") -> str:
    vital_section = (
        f"\n\n### INTRAOPERATIVE VITAL SUMMARY\n{vital_summary}" if vital_summary else ""
    )
    return f"""Using the EMR below, generate an ultra-brief PACU/ICU handoff.

- If there are NO clinically meaningful issues, output exactly and only "특이사항 없음".
- If there ARE issues, output 1-5 very short sentences in formal Korean.
- Do NOT include normal findings, routine or administrative details, or any request to re-check intraoperative events.

필수 6군(기저질환·약물 / 기도관리 / 수술 중 이벤트 및 처치 / 수혈·수액 /
수술 전 검사이상 / 감기 유무) 중 이상 소견이 있는 군은 빠뜨리지 마세요.
바이탈 이상은 지속시간과 최저/최고값을 함께: "20분간 저혈압(최저 55mmHg)".

### EMR
{emr_text}{vital_section}

{OUTPUT_HEADER}
"""


# ── chat template 한 벌 ─────────────────────────────────────────────────────
def apply_chat_template(tokenizer, msgs, add_generation_prompt=True,
                        enable_thinking=False) -> str:
    """모든 스크립트가 쓰는 유일한 chat template 헬퍼.
    thinking 미지원 tokenizer는 자동 fallback, system 미지원 템플릿은 user에 병합."""
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        pass
    try:
        return tokenizer.apply_chat_template(msgs, **kwargs)
    except Exception:
        # system role 미지원 (gemma 등) → user에 병합
        merged, sys_txt = [], ""
        for m in msgs:
            if m["role"] == "system":
                sys_txt = m["content"]
            elif m["role"] == "user":
                content = (sys_txt + "\n\n" + m["content"]) if sys_txt else m["content"]
                merged.append({"role": "user", "content": content})
                sys_txt = ""
            else:
                merged.append(m)
        return tokenizer.apply_chat_template(merged, **kwargs)


def n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ── EMR-only 절단 (T4/T6 핵심 수정) ──────────────────────────────────────────
class PromptTruncationError(RuntimeError):
    """절단 후에도 프롬프트 구조가 깨졌을 때 — 조용히 넘어가지 않는다."""


def assert_prompt_intact(prompt_text: str):
    """### OUTPUT 헤더가 살아있는지 검증. 잘렸으면 즉시 실패."""
    if OUTPUT_HEADER not in prompt_text:
        raise PromptTruncationError(
            f"프롬프트에서 '{OUTPUT_HEADER}'가 사라짐 — 절단 로직 오류 또는 예산 부족. "
            "지시문/생성헤더는 절대 잘리면 안 된다."
        )


def _fit_user_and_prompt(tokenizer, emr_text: str, vital_summary: str,
                         system: str, budget: int, enable_thinking: bool,
                         fewshot_block: str):
    """(truncated_user_str, rendered_prompt) — EMR-only 좌측절단 이진 탐색 공통 구현.

    예산 초과 시 'EMR 본문 좌측'만 잘라낸다 (최신 기록·TOTALS·지시문·OUTPUT 보존).
    잘린 경우 '[... EMR 앞부분 생략 ...]' 마커 삽입. assert_prompt_intact() 통과 보장.
    """
    def make_user(emr: str) -> str:
        user = build_user_prompt(emr, vital_summary)
        if fewshot_block:
            user = fewshot_block + "\n\n" + user
        return user

    def render(user: str) -> str:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        return apply_chat_template(tokenizer, msgs, add_generation_prompt=True,
                                   enable_thinking=enable_thinking)

    user = make_user(emr_text)
    full = render(user)
    if n_tokens(tokenizer, full) <= budget:
        assert_prompt_intact(full)
        return user, full

    marker = "[... EMR 앞부분 생략(길이 초과) ...]\n"
    lo, hi = 0, len(emr_text)          # keep = emr_text[-mid:]
    best = None
    for _ in range(20):                 # 2^20 > 어떤 EMR 길이도 커버
        mid = (lo + hi) // 2
        cand_user = make_user(marker + emr_text[len(emr_text) - mid:])
        cand_full = render(cand_user)
        if n_tokens(tokenizer, cand_full) <= budget:
            best = (cand_user, cand_full)
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    if best is None:
        # EMR을 전부 비워도 초과 → 예산이 지시문조차 못 담는 상황. 크게 실패.
        raise PromptTruncationError(
            f"토큰 예산({budget})이 지시문+few-shot조차 담지 못함 — "
            f"few-shot 수를 줄이거나 budget을 늘려야 한다."
        )
    assert_prompt_intact(best[1])
    return best


def fit_chat_prompt(tokenizer, emr_text: str, vital_summary: str = "",
                    system: str = SYSTEM_PROMPT, budget: int = MAX_PROMPT_TOKENS,
                    enable_thinking: bool = False,
                    fewshot_block: str = "") -> str:
    """토큰 예산에 맞춘 최종 chat 프롬프트 문자열 (추론/후보생성용)."""
    return _fit_user_and_prompt(tokenizer, emr_text, vital_summary, system,
                                budget, enable_thinking, fewshot_block)[1]


def fit_user_content(tokenizer, emr_text: str, vital_summary: str = "",
                     system: str = SYSTEM_PROMPT, budget: int = MAX_PROMPT_TOKENS,
                     enable_thinking: bool = False,
                     fewshot_block: str = "") -> str:
    """토큰 예산에 맞춘 'user 메시지 문자열' (학습용 — 같은 절단 규칙 공유, T6)."""
    return _fit_user_and_prompt(tokenizer, emr_text, vital_summary, system,
                                budget, enable_thinking, fewshot_block)[0]


def fit_text_prompt(tokenizer, render_fn, emr_text: str, budget: int,
                    marker: str = "[... EMR 앞부분 생략(길이 초과) ...]\n") -> str:
    """임의 템플릿(render_fn: emr → 전체 프롬프트 문자열)에 대한 EMR-only 좌측절단.

    judge 프롬프트(rubric/후보/EMR 혼합) 등 chat 프롬프트가 아닌 경우에 사용 (T4).
    render_fn이 만든 프롬프트가 예산 내면 그대로, 아니면 EMR 좌측을 잘라 맞춘다.
    EMR을 다 비워도 초과면 PromptTruncationError (rubric/response는 절대 안 자름).
    """
    full = render_fn(emr_text)
    if n_tokens(tokenizer, full) <= budget:
        return full
    lo, hi = 0, len(emr_text)
    best = None
    for _ in range(20):
        mid = (lo + hi) // 2
        candidate = render_fn(marker + emr_text[len(emr_text) - mid:])
        if n_tokens(tokenizer, candidate) <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    if best is None:
        raise PromptTruncationError(
            f"토큰 예산({budget})이 EMR을 전부 제거해도 초과 — "
            "템플릿(rubric/response)이 예산보다 크다."
        )
    return best


def training_texts(tokenizer, emr_text: str, vital_summary: str, target: str,
                   system: str = SYSTEM_PROMPT, budget: int = MAX_PROMPT_TOKENS,
                   enable_thinking: bool = False):
    """학습용 (prompt_text, full_text) 쌍 — 같은 절단 규칙 공유 (학습/추론 일치, T6).

    prompt_text = fit_chat_prompt 결과 (generation header까지),
    full_text   = prompt_text + target + eos 는 호출부(collator)에서 토큰 결합.
    """
    prompt_text = fit_chat_prompt(tokenizer, emr_text, vital_summary,
                                  system=system, budget=budget,
                                  enable_thinking=enable_thinking)
    return prompt_text, target
