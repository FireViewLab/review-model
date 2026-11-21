"""리뷰 목록을 받아 의심 확률, 텍스트 점수, 정책 지수, 등급을 돌려준다.

확률 계산(모델)과 결과 변환(규칙)을 나눠, 변환은 모델 없이 테스트할 수 있게 한다.
"""

import json
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from review_model.policy.config import PolicyConfig
from review_model.policy.rules import grade, policy_index

ProbabilityFn = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class ReviewResult:
    review_id: str
    text_probability: float  # 보정된 의심 확률(0~1)
    text_score: int  # round((1 - 확률) * 100), 높을수록 신뢰
    policy_index: float  # 등급 구간 안의 지수(SAFE 70~100, WARN 40~69, DANGER 0~39)
    level: str  # SAFE / WARN / DANGER


class Predictor:
    def __init__(self, probability: ProbabilityFn, policy: PolicyConfig, model_version: str):
        self._probability = probability
        self.policy = policy
        self.model_version = model_version

    def analyze(self, reviews: Sequence[dict]) -> list[ReviewResult]:
        """reviews: [{"review_id": str, "content": str}, ...]. 빈 본문은 호출 전에 걸러야 한다."""
        if not reviews:
            return []
        prob = np.clip(np.asarray(self._probability([r["content"] for r in reviews]), dtype=float), 0.0, 1.0)
        levels = grade(prob, self.policy.cuts)
        index = policy_index(prob, self.policy.cuts)
        return [
            ReviewResult(
                review_id=str(r["review_id"]),
                text_probability=round(float(p), 5),
                text_score=round((1 - float(p)) * 100),
                policy_index=float(i),
                level=str(level),
            )
            for r, p, i, level in zip(reviews, prob, index, levels, strict=True)
        ]

    @classmethod
    def from_model_dir(cls, model_dir: str | pathlib.Path, batch: int = 64) -> "Predictor":
        """모델 디렉터리(가중치, inference_config.json, policy.json)에서 만든다. torch가 필요하다."""
        from review_model.training.model import calibrated_probability, device, load_model, logit_diff

        model_dir = pathlib.Path(model_dir)
        dev = device()
        tok, model, config = load_model(model_dir, dev)
        policy = PolicyConfig.load(model_dir / "policy.json")

        def probability(texts: Sequence[str]) -> np.ndarray:
            diff = logit_diff(model, tok, list(texts), dev, config["max_length"], batch=batch)
            return calibrated_probability(diff, config)

        version = json.loads((model_dir / "inference_config.json").read_text()).get("model_family", model_dir.name)
        return cls(probability, policy, version)
