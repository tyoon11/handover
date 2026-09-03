"""
prompts_pairgen.py — 선호쌍 생성 judge 프롬프트 단일 소스 (v3.2)

gen_pairs.py 에 있던 judge 템플릿/루브릭을 분리했다. 이유 두 가지:
  - 프롬프트는 '한 벌만' — 지문(prompt_registry.fingerprint)을 뜨려면 GPU/torch import
    없이 로드되는 모듈에 있어야 한다 (gen_pairs 는 import 시점에 CUDA_VISIBLE_DEVICES 를 만진다).
  - 프롬프트 동결 게이트(docs/PIPELINE_V3.2.md §4)에서 생성·평가 프롬프트를 함께 해시한다.

언어 규칙: 지시·루브릭은 영어, 채점 대상 텍스트와 출력은 한국어 (prompt_utils 참조).
"""

JUDGE_TMPL = """###Task Description:
An EMR-grounded evaluation. A source EMR (with intraoperative vital summary), a handoff
response to evaluate, and a score rubric are given.
1. Write brief feedback strictly based on the score rubric, judging ONLY against the EMR.
2. After feedback, write a score (integer 1-5).
3. Output format: "(feedback) [RESULT] (score)"
4. No other opening or closing.

###Source EMR (ground truth for this patient):
{emr}

###Intraoperative vital summary:
{vital}

###Response to evaluate:
{response}

###Score Rubric:
{rubric}

###Feedback:"""

# T3 교정 핵심: '전부 놓치고 특이사항 없음'이 최고점이 되지 않도록 coverage 루브릭 신설
COVERAGE_RUBRIC = (
    "Clinical coverage (recall of actionable abnormal findings). "
    "Score 5: mentions ALL clinically actionable abnormal findings present in the EMR/vitals, "
    "covering every one of the six mandatory groups that HAS a finding — comorbidity/medication, "
    "airway management, intraoperative events and interventions, transfusion/fluids, abnormal "
    "pre-op tests, recent URI (cold) status. Vital abnormalities must be QUANTIFIED with how long "
    "they lasted and how far they deviated from the threshold (nadir/peak), e.g. "
    "'20분간 저혈압(최저 55mmHg)'; a bare '저혈압' is incomplete. "
    "Score 3: mentions the most critical findings but misses some, or reports vital abnormalities "
    "without duration and nadir/peak. "
    "Score 1: says '특이사항 없음' despite clear abnormal findings in the EMR, or misses most. "
    "If the EMR truly has no meaningful abnormal findings, an exact '특이사항 없음' scores 5."
)

FIDELITY_BREVITY_RUBRIC = (
    "Fidelity and exception-based brevity. "
    "Score 5: every statement is supported by the EMR/vitals, only abnormal findings are "
    "mentioned, in 1-5 short formal Korean sentences. Duration and nadir/peak numbers attached to "
    "an ABNORMAL vital finding are required content, NOT verbosity — never penalize them. "
    "Score 3: minor unsupported details or mild verbosity. "
    "Score 1: fabricates findings not in the EMR, states r/o as confirmed, or is dominated by "
    "normal/routine/administrative content."
)

