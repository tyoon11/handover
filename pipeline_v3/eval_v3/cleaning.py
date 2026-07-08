"""
eval_v3/cleaning.py — 생성 출력 정제 (v2 clean_v2의 오탐 수정판, E4)

v2 대비 수정:
  - '조사로 끝나면 truncated → 0점' 폐기. truncated는 하드 실패가 아니라
    flags=["maybe_truncated"]로만 기록하고 정상 채점한다.
    ("…저혈당 주의"처럼 명사로 끝나는 정상 한국어 인계문이 0점 받던 문제)
  - 문자벽 판정에서 마크다운 구분선 문자(-=#*_~.공백)는 제외.
    ("--------" 구분선이 repetition 0점 받던 문제)
  - 외국문자 비율 판정의 허용셋 확장: 곱슬따옴표·전각문장부호·단위기호·호환자모.
  - 어떤 경우에도 텍스트를 임의 문구로 치환하지 않는다 (실패는 status/flags로 분리).

반환: (cleaned_text, status, flags)
  status ∈ {"ok", "empty", "repetition", "garbage", "leak"}   ← 하드 실패만
  flags  ⊂ {"maybe_truncated", "had_think_block", "had_role_lines"}
"""

import re

_RE_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RE_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_RE_ROLE = re.compile(r"^\s*(?:assistant|user|system)\s*:?\s*$",
                      re.MULTILINE | re.IGNORECASE)
_RE_THOUGHT_LEAK = re.compile(r"(?://|_|\|)\s*thought\b|<\|?thought", re.IGNORECASE)

# 문자벽: 같은 문자 8+ 연속. 단 마크다운 구분선/공백 문자는 제외 (E4)
_WALL_EXEMPT = set("-=#*_~. \t\n·")
_RE_CHAR_RUN = re.compile(r"(.)\1{7,}")

# 허용 문장부호 (외국스크립트 비율 계산용) — v2 셋 + 곱슬따옴표/전각/단위
_ALLOWED_PUNCT = set(
    " \n\t().,/%+-:;#*°<>=~?!'\"[]{}·…&|\\@^_$²³㎎㎍㏖①②③"
    "‘’“”"   # ' ' " "
    "，。？！：；（）【】『』「」、〜ㆍ％℃㎏㎜㎝㎖ℓμΔ±×÷—–‐"
    " 　"               # nbsp, 전각공백
)


def _foreign_ratio(text: str) -> float:
    """한글/영문/숫자/한자/허용부호가 아닌 문자 비율 (공백 제외)."""
    total = other = 0
    for c in text:
        if c.isspace():
            continue
        total += 1
        o = ord(c)
        is_hangul = 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F
        is_ascii = c.isascii() and (c.isalnum() or c in _ALLOWED_PUNCT)
        is_cjk = 0x4E00 <= o <= 0x9FFF
        if not (is_hangul or is_ascii or is_cjk or c in _ALLOWED_PUNCT):
            other += 1
    return (other / total) if total else 0.0


def _has_char_wall(text: str) -> bool:
    for m in _RE_CHAR_RUN.finditer(text):
        if m.group(1) not in _WALL_EXEMPT:
            return True
    return False


def detect_repetition(text: str, n: int = 4, thresh: float = 0.5) -> bool:
    """반복 루프: 동일 단어 3연속 또는 n-gram 다양성 붕괴."""
    toks = text.split()
    for i in range(len(toks) - 2):
        if toks[i] and toks[i] == toks[i + 1] == toks[i + 2]:
            return True
    if len(toks) < n * 3:
        return False
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return (len(set(grams)) / len(grams)) < thresh


# 잘림 '의심' 신호 — 하드 실패 아님 (플래그만)
_RE_TRAILING_JOSA = re.compile(r"(?:을|를|이|가|에|의|로|와|과|은|는)\s*$")
_OK_ENDINGS = ("다", "요", ".", "음", "함", "됨", "임", ")", "%")


def clean_v3(raw):
    """반환 (cleaned, status, flags). 실패해도 텍스트 치환 없음."""
    flags = []
    if raw is None:
        return "", "empty", flags
    text = str(raw)

    if "<think>" in text.lower():
        flags.append("had_think_block")
    text = _RE_THINK_CLOSED.sub("", text)
    if "<think>" in text.lower() and "</think>" not in text.lower():
        text = _RE_THINK_OPEN.sub("", text)
    if _RE_ROLE.search(text):
        flags.append("had_role_lines")
        text = _RE_ROLE.sub("", text)
    cleaned = text.strip()

    if len(cleaned) < 3:
        return cleaned, "empty", flags
    if _has_char_wall(cleaned):
        return cleaned, "repetition", flags
    if _foreign_ratio(cleaned) > 0.15:
        return cleaned, "garbage", flags
    if _RE_THOUGHT_LEAK.search(cleaned):
        return cleaned, "leak", flags
    if detect_repetition(cleaned):
        return cleaned, "repetition", flags

    if _RE_TRAILING_JOSA.search(cleaned) and not cleaned.endswith(_OK_ENDINGS):
        flags.append("maybe_truncated")   # 정상 채점 + 플래그 (0점 아님, E4)
    return cleaned, "ok", flags
