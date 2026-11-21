"""등급 규칙 후보.

세 후보 모두 "의심 점수(0~1, 높을수록 의심)"를 만든 뒤 같은 방식으로 등급과 지수를 낸다.
- 등급: 점수 >= danger_cut → DANGER, >= warn_cut → WARN, 그 외 SAFE
- 지수: 등급 구간(SAFE 70~100, WARN 40~69, DANGER 0~39) 안에서 점수 순서를 보존하는 선형 변환.
  등급과 지수가 어긋나지 않는다.

A text_first: v4.0 텍스트 확률만 쓴다. 사진은 등급에 쓰지 않는다.
B text_evidence: A에 확실한 증거만 더한다. 사진은 점수를 낮추지(=SAFE로 올리지) 못한다.
C learned: 텍스트 로짓과 사진 상태·길이·별점·중복을 로지스틱 회귀로 결합한다(검증 세트로 학습).
"""

from dataclasses import dataclass

import numpy as np

from review_model.training.metrics import sigmoid

BANDS = {"SAFE": (70.0, 100.0), "WARN": (40.0, 69.0), "DANGER": (0.0, 39.0)}


@dataclass(frozen=True)
class Evidence:
    """한 리뷰의 등급 판단 재료. 모르는 값은 None으로 둔다(0이나 False로 바꾸지 않는다)."""

    text_prob: np.ndarray  # v4.0 보정 확률
    image_present: np.ndarray  # 사진 1장 이상
    image_absent_verified: np.ndarray  # 사진 0장이 확인됨(컬리)
    rating_five: np.ndarray  # 별점 5점이 확인됨
    rating_missing: np.ndarray
    meaningful_length: np.ndarray
    cross_duplicate: np.ndarray  # 다른 상품에 같은 문장(의미 문자 30자 이상)


@dataclass(frozen=True)
class Cuts:
    warn: float
    danger: float


def grade(score: np.ndarray, cuts: Cuts) -> np.ndarray:
    return np.where(score >= cuts.danger, "DANGER", np.where(score >= cuts.warn, "WARN", "SAFE"))


def policy_index(score: np.ndarray, cuts: Cuts) -> np.ndarray:
    """등급 구간 안에서 의심 점수가 낮을수록 높은 지수(0~100)."""
    score = np.asarray(score, dtype=float)
    out = np.empty_like(score)
    spans = (
        ("SAFE", 0.0, cuts.warn),
        ("WARN", cuts.warn, cuts.danger),
        ("DANGER", cuts.danger, 1.0),
    )
    for name, lo, hi in spans:
        low, high = BANDS[name]
        mask = (score >= lo) & ((score < hi) if name != "DANGER" else (score <= hi))
        position = (score[mask] - lo) / max(hi - lo, 1e-9)
        out[mask] = high - position * (high - low)
    return np.round(out, 1)


def text_first(ev: Evidence) -> np.ndarray:
    return ev.text_prob


def text_evidence(ev: Evidence, cuts: Cuts) -> np.ndarray:
    """확실한 증거로만 점수를 올린다. 사진은 점수를 내리지 않는다."""
    score = ev.text_prob.copy()
    low_info = ev.image_absent_verified & ev.rating_five & (ev.meaningful_length < 10)
    score = np.where(low_info, np.maximum(score, cuts.warn), score)
    # 다른 상품에 같은 긴 문장이 있으면 한 등급 올린다.
    bumped = np.where(score >= cuts.warn, np.maximum(score, cuts.danger), np.maximum(score, cuts.warn))
    return np.where(ev.cross_duplicate, bumped, score)


def features(ev: Evidence) -> np.ndarray:
    p = np.clip(ev.text_prob, 1e-6, 1 - 1e-6)
    return np.column_stack(
        [
            np.log(p / (1 - p)),
            ev.image_present.astype(float),
            ev.image_absent_verified.astype(float),
            np.log1p(ev.meaningful_length),
            ev.rating_five.astype(float),
            ev.rating_missing.astype(float),
            ev.cross_duplicate.astype(float),
        ]
    )


FEATURE_NAMES = (
    "text_logit",
    "image_present",
    "image_absent_verified",
    "log_length",
    "rating_five",
    "rating_missing",
    "cross_duplicate",
)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 50) -> np.ndarray:
    """L2 정규화 로지스틱 회귀(뉴턴법). 반환값의 마지막 원소가 절편이다."""
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = np.column_stack([(x - mean) / std, np.ones(len(x))])
    w = np.zeros(z.shape[1])
    reg = np.eye(z.shape[1]) * l2
    reg[-1, -1] = 0.0
    for _ in range(steps):
        p = sigmoid(z @ w)
        grad = z.T @ (p - y) + reg @ w
        hess = (z * (p * (1 - p))[:, None]).T @ z + reg
        w -= np.linalg.solve(hess, grad)
    # 표준화를 풀어 원래 단위의 가중치로 돌려준다.
    coef = w[:-1] / std
    return np.append(coef, w[-1] - (mean / std) @ w[:-1])


def learned(ev: Evidence, weights: np.ndarray) -> np.ndarray:
    return sigmoid(features(ev) @ weights[:-1] + weights[-1])
