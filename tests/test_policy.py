import numpy as np

from review_model.policy import rules


def _evidence(prob, photo=False, absent=False, five=False, length=50, dup=False):
    n = len(prob)
    return rules.Evidence(
        text_prob=np.asarray(prob, dtype=float),
        image_present=np.full(n, photo),
        image_absent_verified=np.full(n, absent),
        rating_five=np.full(n, five),
        rating_missing=np.full(n, False),
        meaningful_length=np.full(n, length),
        cross_duplicate=np.full(n, dup),
    )


CUTS = rules.Cuts(warn=0.3, danger=0.8)


def test_grade_and_index_bands_agree():
    score = np.array([0.0, 0.29, 0.3, 0.79, 0.8, 1.0])
    grades = rules.grade(score, CUTS)
    index = rules.policy_index(score, CUTS)
    assert list(grades) == ["SAFE", "SAFE", "WARN", "WARN", "DANGER", "DANGER"]
    for g, i in zip(grades, index, strict=True):
        low, high = rules.BANDS[g]
        assert low <= i <= high
    assert all(np.diff(index) <= 0)  # 의심 점수가 높을수록 지수는 낮다.


def test_photo_does_not_change_text_first_or_evidence():
    with_photo = _evidence([0.5], photo=True)
    without = _evidence([0.5], photo=False)
    assert rules.text_first(with_photo) == rules.text_first(without)
    assert rules.text_evidence(with_photo, CUTS) == rules.text_evidence(without, CUTS)


def test_evidence_rules_only_raise_score():
    low_info = _evidence([0.1], absent=True, five=True, length=5)
    assert rules.text_evidence(low_info, CUTS)[0] == CUTS.warn
    dup = _evidence([0.1, 0.5], dup=True)
    assert list(rules.text_evidence(dup, CUTS)) == [CUTS.warn, CUTS.danger]


def test_logistic_learns_text_signal():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    prob = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.1, 400), 0.01, 0.99)
    ev = _evidence(prob)
    w = rules.fit_logistic(rules.features(ev), y)
    assert w[0] > 0
    assert rules.learned(ev, w)[y == 1].mean() > rules.learned(ev, w)[y == 0].mean()
