"""LLM 결과에서 특정 라벨로 판정된 항목만 패킷에서 골라 새 패킷을 만든다(교차 확인용).

사용: python subset_packet.py <packet.xlsx> <llm.xlsx> <label> <out.xlsx>
"""

import sys

import pandas as pd


def main(packet, llm, label, out):
    p = pd.read_excel(packet, sheet_name=0)
    l = pd.read_excel(llm)
    ids = set(l[l.llm_label == label]["항목ID"])
    p[p["항목ID"].isin(ids)].to_excel(out, index=False)
    print(f"{label} {len(ids)}건 → {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
