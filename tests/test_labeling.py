import pandas as pd

from review_model.labeling.agreement import kappa
from review_model.labeling.consensus import combine


def test_kappa_bounds():
    assert kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0
    assert abs(kappa(["A", "B"], ["B", "A"]) + 1.0) < 1e-9


def test_consensus_demotes_suspicious_split_to_uncertain():
    primary = pd.DataFrame(
        {
            "항목ID": ["1", "2", "3", "4"],
            "llm_label": ["SUSPICIOUS", "SUSPICIOUS", "NORMAL", "NORMAL"],
            "llm_disclosed": ["N"] * 4,
            "llm_signal_types": [""] * 4,
            "llm_evidence_note": [""] * 4,
        }
    )
    # 보조 판정자는 1~2번만 봤다.
    secondary = pd.DataFrame({"항목ID": ["1", "2"], "llm_label": ["SUSPICIOUS", "NORMAL"]})
    out = combine(primary, secondary).set_index("항목ID")
    assert out.loc["1", "최종 라벨"] == "SUSPICIOUS" and out.loc["1", "label_source"] == "agree"
    assert out.loc["2", "최종 라벨"] == "UNCERTAIN" and out.loc["2", "label_source"] == "split_to_uncertain"
    assert out.loc["3", "최종 라벨"] == "NORMAL" and out.loc["3", "label_source"] == "primary"
