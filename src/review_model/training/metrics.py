"""평가 지표와 보정. numpy만 쓴다."""

import math

import numpy as np


def average_precision(y, p) -> float:
    """PR-AUC(평균 정밀도). 점수가 높을수록 양성(의심)으로 본다."""
    order = np.argsort(-np.asarray(p))
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


def pr_points(y, p):
    """점수 내림차순 각 지점의 (임계값, 정밀도, 재현율)."""
    order = np.argsort(-np.asarray(p))
    ys, ps = np.asarray(y)[order], np.asarray(p)[order]
    tp = np.cumsum(ys)
    return ps, tp / np.arange(1, len(ys) + 1), tp / max(ys.sum(), 1)


def recall_at_precision(y, p, target: float) -> float:
    _, precision, recall = pr_points(y, p)
    ok = precision >= target
    return float(recall[ok].max()) if ok.any() else 0.0


def choose_threshold(y, p, target_precision: float):
    """정밀도 목표 이상인 지점 중 재현율이 가장 큰 임계값. 없으면 F1 최대 지점."""
    th, precision, recall = pr_points(y, p)
    ok = precision >= target_precision
    if ok.any():
        return float(th[int(np.argmax(np.where(ok, recall, -1)))]), "precision_target"
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(th[int(np.argmax(f1))]), "max_f1"


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def fit_temperature(logit_diff, y) -> float:
    """로그 손실을 최소화하는 temperature(격자 탐색)."""
    logit_diff, y = np.asarray(logit_diff, dtype=float), np.asarray(y, dtype=float)
    best, best_t = math.inf, 1.0
    for t in np.exp(np.linspace(math.log(0.05), math.log(20), 400)):
        q = np.clip(sigmoid(logit_diff / t), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best:
            best, best_t = nll, float(t)
    return best_t
