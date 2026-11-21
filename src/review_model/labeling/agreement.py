"""두 라벨 묶음의 일치도(일치율, Cohen κ, 의심 여부 κ)를 계산하고 불일치 검토 시트를 만든다.

각 파일의 첫 시트에서 ID 컬럼과 라벨 컬럼(최종 라벨 / gold_label / human_label / llm_label 중 처음
찾은 것)을 읽는다. 보정, 자기 일관성, LLM 의견 비교, 시험 세트 확인에 같이 쓴다.

    python -m review_model.labeling.agreement mine.xlsx gold.xlsx diff.xlsx
"""

import argparse
import sys

import pandas as pd

from review_model.constants import ID_COLUMNS

LABEL_COLUMNS = ("최종 라벨", "gold_label", "human_label", "llm_label")
SIGNAL_COLUMNS = ("확인된 신호 코드", "gold_signal_types", "signal_types", "llm_signal_types")
NOTE_COLUMNS = ("근거 메모", "gold_evidence_note", "evidence_note", "llm_evidence_note")


def load(path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    id_col = next(c for c in ID_COLUMNS if c in df.columns)
    label_col = next(c for c in LABEL_COLUMNS if c in df.columns)
    out = pd.DataFrame({"id": df[id_col].astype(str), "label": df[label_col].astype("string").str.strip()})
    for name, candidates in (("signals", SIGNAL_COLUMNS), ("note", NOTE_COLUMNS)):
        col = next((c for c in candidates if c in df.columns), None)
        out[name] = df[col] if col else None
    return out[out.label.notna() & (out.label != "")]


def kappa(a, b) -> float:
    """Cohen κ. 우연히 같을 확률을 뺀 일치도."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    observed = (a == b).mean()
    expected = sum((a == v).mean() * (b == v).mean() for v in set(a) | set(b))
    return 1.0 if expected >= 1 else (observed - expected) / (1 - expected)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    args = ap.parse_args(argv)
    m = load(args.a).merge(load(args.b), on="id", suffixes=("_a", "_b"))
    if m.empty:
        sys.exit("겹치는 항목이 없습니다. 두 파일 모두 라벨 칸이 채워져 있는지 확인하세요.")
    agree = (m.label_a == m.label_b).mean()
    k = kappa(m.label_a, m.label_b)
    # 학습 목표가 SUSPICIOUS 판별이라 의심 여부만 본 일치도를 따로 본다.
    k_binary = kappa(m.label_a == "SUSPICIOUS", m.label_b == "SUSPICIOUS")
    print(f"비교 {len(m)}건 · 일치 {agree:.1%} · κ {k:.2f} · 의심 여부 κ {k_binary:.2f}")
    print(pd.crosstab(m.label_a, m.label_b, rownames=["A"], colnames=["B"]).to_string())
    diff = m[m.label_a != m.label_b].copy()
    diff["불일치 유형"] = diff.label_a + " vs " + diff.label_b
    diff["최종 합의 라벨(입력)"] = ""
    diff["합의 근거(입력)"] = ""
    diff.rename(
        columns={
            "id": "항목ID",
            "label_a": "A 라벨",
            "label_b": "B 라벨",
            "signals_a": "A 신호",
            "signals_b": "B 신호",
            "note_a": "A 근거",
            "note_b": "B 근거",
        }
    ).to_excel(args.out, index=False, sheet_name="불일치")
    print(f"불일치 {len(diff)}건 → {args.out}")


if __name__ == "__main__":
    main()
