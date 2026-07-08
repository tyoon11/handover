"""
inspect_khs.py — KHS gold 엑셀 구조 확인용 (서버에서 실행 → 출력 공유)

실행:
  python inspect_khs.py
  # 경로가 다르면:
  python inspect_khs.py "/home/coder/workspace/data/handover/data/gold_sampled/인계요약지_gold_sampled_251002_KHS.xlsx"
"""

import sys
import pandas as pd

DEFAULT = "/home/coder/workspace/data/handover/data/gold_sampled/인계요약지_gold_sampled_251002_KHS.xlsx"
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT

print("=" * 70)
print("FILE:", path)
print("=" * 70)

xl = pd.ExcelFile(path)
print("SHEETS:", xl.sheet_names)

SHEET = "데이터" if "데이터" in xl.sheet_names else xl.sheet_names[0]
print(f"\n[sheet '{SHEET}']")

# 1) 일반 헤더로 읽기
df = pd.read_excel(path, sheet_name=SHEET)
print("shape:", df.shape)
print("\n── 컬럼 목록 (repr) ──")
for c in df.columns:
    print("  ", repr(c))

# 2) 헤더 없이 앞 3행 — 다중행 헤더 여부 판별
print("\n── header=None 기준 앞 3행 (다중헤더 판별용) ──")
raw = pd.read_excel(path, sheet_name=SHEET, header=None)
for r in range(min(3, len(raw))):
    cells = []
    for c in range(raw.shape[1]):
        v = raw.iloc[r, c]
        if pd.notna(v):
            cells.append(f"c{c}={repr(str(v))[:40]}")
    print(f"  row{r}: " + " | ".join(cells))

# 3) Feedback / LLM-sample / sid 컬럼 자동감지
def find(keys):
    for c in df.columns:
        name = (" ".join(map(str, c)) if isinstance(c, tuple) else str(c)).lower()
        if any(k.lower() in name for k in keys):
            return c
    return None

fb = find(["feedback", "교수", "gold"])
llm = find(["sample_from_llm", "인계요약지_sample", "llm"])
sid = find(["수술id", "수술 id"])
print("\n── 자동감지 ──")
print("  feedback   →", repr(fb))
print("  llm_sample →", repr(llm))
print("  sid        →", repr(sid))

# 4) Feedback 채움 현황 + 샘플
if fb is not None:
    s = df[fb]
    nonempty = s.apply(lambda x: pd.notna(x) and str(x).strip() not in ("", "nan"))
    print(f"\n── Feedback 채워진 행: {int(nonempty.sum())}/{len(df)} ──")
    shown = 0
    for i in range(len(df)):
        if nonempty.iloc[i]:
            print(f"\n  [row {i}] FEEDBACK:\n    {str(s.iloc[i])[:300]}")
            if llm is not None:
                print(f"    (LLM_sample: {str(df[llm].iloc[i])[:150]})")
            shown += 1
            if shown >= 5:
                break
else:
    print("\n⚠ Feedback 컬럼 자동감지 실패 — 위 '컬럼 목록'에서 실제 이름 알려주세요.")
