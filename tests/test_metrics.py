import numpy as np

from review_model.training.metrics import (
    average_precision,
    choose_threshold,
    fit_temperature,
    recall_at_precision,
    sigmoid,
)


def test_average_precision_perfect_and_reversed():
    y = np.array([1, 1, 0, 0])
    assert average_precision(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) < 0.6


def test_choose_threshold_meets_precision_target():
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    p = np.linspace(0.95, 0.05, len(y))
    threshold, method = choose_threshold(y, p, 0.75)
    assert method == "precision_target"
    flagged = p >= threshold
    assert y[flagged].mean() >= 0.75


def test_recall_at_precision_zero_when_unreachable():
    assert recall_at_precision(np.array([0, 0, 1]), np.array([0.9, 0.8, 0.1]), 0.9) == 0.0


def test_fit_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    # 방향은 맞지만 지나치게 큰 로짓 → temperature가 1보다 커야 한다.
    logits = (y * 2 - 1) * 8 + rng.normal(0, 6, 500)
    assert fit_temperature(logits, y) > 1.0
    assert sigmoid(0) == 0.5
