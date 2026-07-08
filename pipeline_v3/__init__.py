# pipeline_v3 — 논문급 재실험 파이프라인 (CODE_REVIEW_V3_PROPOSAL.md 로드맵 구현)
#
# 원칙 (v3 재설계 3원칙):
#   1. 실패는 크게: 어떤 실패도 0.0/1.0 점수로 변환하지 않는다 (None + 플래그 + 제외 집계).
#   2. 한 벌만: 프롬프트/절단/chat template/모델 키/태그 파싱은 이 패키지가 유일한 소스.
#      v1(config.py, pipeline/0*.py)과 v2(config_v2.py, eval_v2/)는 legacy로 봉인 — 수정 금지.
#   3. 순환 금지: 데이터 생성 judge ≠ 평가 judge ≠ 평가 대상 모델 계열.
#      gold는 사람 검수 전엔 항상 '잠정' 표기.
