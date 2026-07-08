"""
khs_to_txt.py — KHS gold 엑셀 전체를 txt로 덤프 (서버에서 실행 → khs_dump.txt 공유)

실행:
  python khs_to_txt.py
  python khs_to_txt.py "<엑셀경로>" "<출력txt경로>"

출력: '데이터' 시트는 케이스별 c9(gemma원안)/c10(교수gold)/c11을 잘림 없이 전체 기록,
      나머지 시트(루브릭 등)도 통째로 덤프.
"""

import os
import sys
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/coder/workspace/data/handover/data/gold_sampled/인계요약지_gold_sampled_251002_KHS.xlsx"
# 출력은 PHI 포함 — 반드시 gitignored 디렉토리(data/)로만 덤프한다 (CWD 금지)
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_OUT_DIR, exist_ok=True)
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_OUT_DIR, "khs_dump.txt")

HEADER_ROWS = 3
# 데이터 시트 위치 컬럼
COLS = dict(idx=0, pid=1, sid=2, dept_list=3, dept=4, recovery=5,
            anrec=6, preop=7, premed=8, llm=9, feedback=10, newgold=11)


def cell(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


lines = []
xl = pd.ExcelFile(PATH)
lines.append(f"FILE: {PATH}")
lines.append(f"SHEETS: {xl.sheet_names}")

# ── '데이터' 시트: 케이스별 정리 ──────────────────────────────────────────
SHEET = "데이터" if "데이터" in xl.sheet_names else xl.sheet_names[0]
raw = pd.read_excel(PATH, sheet_name=SHEET, header=None)
data = raw.iloc[HEADER_ROWS:].reset_index(drop=True)
lines.append("\n" + "=" * 80)
lines.append(f"[시트 '{SHEET}']  데이터 행수: {len(data)}")
lines.append("=" * 80)

for r in range(len(data)):
    row = data.iloc[r]
    idx = cell(row[COLS["idx"]])
    sid = cell(row[COLS["sid"]])
    if not (idx or sid or cell(row[COLS["feedback"]]) or cell(row[COLS["llm"]])):
        continue
    lines.append(f"\n{'#'*80}")
    lines.append(f"# CASE idx={idx}  sid={sid}  dept={cell(row[COLS['dept']])}  "
                 f"recovery={cell(row[COLS['recovery']])}")
    lines.append("#" * 80)
    lines.append(f"\n[마취전 환자상태 요약 (c7)]\n{cell(row[COLS['preop']])}")
    lines.append(f"\n[Premedication (c8)]\n{cell(row[COLS['premed']])}")
    lines.append(f"\n[마취기록 (c6)]\n{cell(row[COLS['anrec']])}")
    lines.append(f"\n[c9 = gemma-3-27b 원안 (피드백 대상)]\n{cell(row[COLS['llm']])}")
    lines.append(f"\n[c10 = 교수님 피드백 = GOLD] ★\n{cell(row[COLS['feedback']])}")
    lines.append(f"\n[c11 = 인계요약지(새작성, 대개 공란)]\n{cell(row[COLS['newgold']])}")

# ── 나머지 시트 통째 덤프 ──────────────────────────────────────────────────
for sh in xl.sheet_names:
    if sh == SHEET:
        continue
    df = pd.read_excel(PATH, sheet_name=sh, header=None)
    lines.append("\n" + "=" * 80)
    lines.append(f"[시트 '{sh}']  shape={df.shape}")
    lines.append("=" * 80)
    for r in range(len(df)):
        cells = [cell(df.iloc[r, c]) for c in range(df.shape[1])]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"저장: {OUT}  (줄 수: {len(lines)})")
