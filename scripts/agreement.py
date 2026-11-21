"""두 라벨 묶음의 일치도를 계산하고 불일치 검토 시트를 만든다.

사용:
  uv run --with openpyxl --with pandas python agreement.py <A.xlsx> <B.xlsx> <out.xlsx>

각 파일은 첫 시트에서 ID 컬럼(항목ID 또는 annotation_id)과 라벨 컬럼
(최종 라벨 / gold_label / human_label / llm_label 중 처음 찾은 것)을 읽는다.
보정(팀 합의 60건 대비), 자기 일관성(같은 항목 재검수), LLM 2차 의견 비교에 같이 쓴다.
"""

import sys

import pandas as pd

ID_COLS = ("항목ID", "annotation_id", "확인ID")
LABEL_COLS = ("최종 라벨", "gold_label", "human_label", "llm_label")
SIGNAL_COLS = ("확인된 신호 코드", "gold_signal_types", "signal_types", "llm_signal_types")
NOTE_COLS = ("근거 메모", "gold_evidence_note", "evidence_note", "llm_evidence_note")


def load(path):
    df = pd.read_excel(path, sheet_name=0)
    idc = next(c for c in ID_COLS if c in df.columns)
    lab = next(c for c in LABEL_COLS if c in df.columns)
    out = pd.DataFrame({"id": df[idc].astype(str), "label": df[lab].astype("string").str.strip()})
    for name, cols in (("signals", SIGNAL_COLS), ("note", NOTE_COLS)):
        col = next((c for c in cols if c in df.columns), None)
        out[name] = df[col] if col else None
    return out[out.label.notna() & (out.label != "")]


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == lab).mean() * (b == lab).mean() for lab in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(pa, pb, out):
    a, b = load(pa), load(pb)
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    if m.empty:
        sys.exit("겹치는 항목이 없습니다.")
    agree = (m.label_a == m.label_b).mean()
    k = kappa(m.label_a.reset_index(drop=True), m.label_b.reset_index(drop=True))
    # 의심 여부만 본 이진 일치도. 학습 목표가 SUSPICIOUS 판별이라 따로 본다.
    sa, sb = m.label_a == "SUSPICIOUS", m.label_b == "SUSPICIOUS"
    kb = kappa(sa.reset_index(drop=True), sb.reset_index(drop=True))
    print(f"비교 {len(m)}건 · 일치 {agree:.1%} · κ {k:.2f} · 의심 여부 κ {kb:.2f}")
    print(pd.crosstab(m.label_a, m.label_b, rownames=["A"], colnames=["B"]).to_string())
    diff = m[m.label_a != m.label_b].copy()
    diff["불일치 유형"] = diff.label_a + " vs " + diff.label_b
    diff["최종 합의 라벨(입력)"] = ""
    diff["합의 근거(입력)"] = ""
    diff.rename(
        columns={"label_a": "A 라벨", "label_b": "B 라벨", "signals_a": "A 신호", "signals_b": "B 신호",
                 "note_a": "A 근거", "note_b": "B 근거", "id": "항목ID"}
    ).to_excel(out, index=False, sheet_name="불일치")
    print(f"불일치 {len(diff)}건 → {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
