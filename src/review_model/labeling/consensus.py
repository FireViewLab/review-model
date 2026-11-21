"""두 LLM 판정을 합쳐 최종 라벨을 만든다.

주 판정자 라벨을 쓰고, 보조 판정자와 의심 여부(SUSPICIOUS인지)가 갈리면 UNCERTAIN으로 돌린다.
보조 판정자가 일부 항목만 봤다면 나머지는 주 판정자 라벨을 그대로 쓴다.
출력은 검수 패킷과 같은 컬럼이라 agreement, dataset.collect에 그대로 넣을 수 있다.

    python -m review_model.labeling.consensus llm/final.xlsx llm/opus.xlsx llm/sonnet.xlsx
"""

import argparse

import pandas as pd

from review_model.constants import ID_COLUMNS


def combine(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    id_col = next(c for c in ID_COLUMNS if c in primary.columns)
    m = primary.merge(secondary[[id_col, "llm_label"]], on=id_col, how="left", suffixes=("", "_2"))
    split = (m.llm_label.eq("SUSPICIOUS") != m.llm_label_2.eq("SUSPICIOUS")) & m.llm_label_2.notna()
    source = pd.Series("primary", index=m.index).where(~m.llm_label.eq(m.llm_label_2), "agree")
    return pd.DataFrame(
        {
            id_col: m[id_col],
            "최종 라벨": m.llm_label.where(~split, "UNCERTAIN"),
            "광고 표시(Y/N)": m.llm_disclosed,
            "확인된 신호 코드": m.llm_signal_types,
            "근거 메모": m.llm_evidence_note,
            "label_source": source.where(~split, "split_to_uncertain"),
        }
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("primary")
    ap.add_argument("secondary")
    a = ap.parse_args(argv)
    result = combine(pd.read_excel(a.primary), pd.read_excel(a.secondary))
    result.to_excel(a.out, index=False)
    print(f"{len(result)}건 → {a.out}")
    print(result["최종 라벨"].value_counts().to_dict(), result.label_source.value_counts().to_dict())


if __name__ == "__main__":
    main()
