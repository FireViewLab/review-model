"""LLM 라벨을 합쳐 최종 라벨을 만든다. 주 판정자(opus) 라벨을 쓰고, 보조 판정자와 의심 여부가 갈리면 UNCERTAIN으로 돌린다.

사용:
  python consensus.py <out.xlsx> <primary.xlsx> <secondary.xlsx>

출력은 검수 패킷과 같은 컬럼(항목ID/annotation_id, 최종 라벨, 광고 표시(Y/N), 확인된 신호 코드, 근거 메모)이라
agreement.py, collect_labels.py에 그대로 넣을 수 있다. label_source 컬럼에 합의 여부를 남긴다.
"""

import sys

import pandas as pd

ID_COLS = ("항목ID", "annotation_id")


def main(out, primary, secondary):
    p, s = pd.read_excel(primary), pd.read_excel(secondary)
    idc = next(c for c in ID_COLS if c in p.columns)
    m = p.merge(s[[idc, "llm_label"]], on=idc, how="left", suffixes=("", "_2"))
    sus1, sus2 = m.llm_label.eq("SUSPICIOUS"), m.llm_label_2.eq("SUSPICIOUS")
    split = (sus1 != sus2) & m.llm_label_2.notna()
    final = m.llm_label.where(~split, "UNCERTAIN")
    source = pd.Series("agree", index=m.index).where(m.llm_label.eq(m.llm_label_2), "primary")
    source = source.where(~split, "split_to_uncertain")
    pd.DataFrame({
        idc: m[idc],
        "최종 라벨": final,
        "광고 표시(Y/N)": m.llm_disclosed,
        "확인된 신호 코드": m.llm_signal_types,
        "근거 메모": m.llm_evidence_note,
        "label_source": source,
    }).to_excel(out, index=False)
    print(f"{len(m)}건 → {out}")
    print(final.value_counts().to_dict(), source.value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:4])
