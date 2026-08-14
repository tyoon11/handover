#!/usr/bin/env python3
"""
build_gold_override.py — 병합 확정 워크시트 → khs_gold_override.json

scripts/import_khs_review.py 가 만든 `khs_gt_merge_worksheet.xlsx` 에서
'확정?' 열이 Y 인 행만 골라 `{sid: final_gt}` 오버라이드를 만든다.
이 JSON 은 checklist.load_khs_gold() 가 c10(기존 gold) 위에 덮어쓴다.

  python scripts/build_gold_override.py
  python scripts/build_gold_override.py --worksheet <경로> --out <경로>

'확정?'이 Y 가 아닌 행은 **조용히 빠진다** — 미확정 GT가 gold 로 새는 걸 막기 위해
일부러 fail-loud 하지 않고, 대신 빠진 sid 를 전부 출력한다.
"""

import argparse
import json
import sys
from pathlib import Path

import openpyxl

COL_SID, COL_TYPE, COL_FINAL, COL_OK = 1, 2, 6, 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worksheet", type=Path,
                    default=Path("data/gold_sampled/khs_gt_merge_worksheet.xlsx"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/preprocessed/khs_gold_override.json"))
    args = ap.parse_args()

    if not args.worksheet.exists():
        sys.exit(f"워크시트 없음: {args.worksheet}\n"
                 f"  먼저 scripts/import_khs_review.py 를 실행하세요.")

    ws = openpyxl.load_workbook(args.worksheet, data_only=True).active
    override, skipped, empty = {}, [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[COL_SID - 1] is None:
            continue
        sid = int(row[COL_SID - 1])
        mtype = str(row[COL_TYPE - 1] or "")
        final = str(row[COL_FINAL - 1] or "").strip()
        ok = str(row[COL_OK - 1] or "").strip().upper()
        if ok != "Y":
            skipped.append((sid, mtype))
            continue
        if not final:
            empty.append(sid)
            continue
        override[str(sid)] = final

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(override, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[override] 확정 {len(override)}건 → {args.out}")
    if empty:
        print(f"  ⚠ 확정(Y)인데 최종 GT가 비어 있음 {len(empty)}건: {empty}")
    if skipped:
        print(f"  · 미확정으로 제외 {len(skipped)}건: "
              f"{[f'{s}({t})' for s, t in skipped]}")


if __name__ == "__main__":
    main()
