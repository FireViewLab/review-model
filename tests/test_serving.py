import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from review_model.policy.config import PolicyConfig
from review_model.serving.app import create_app
from review_model.serving.predictor import Predictor

POLICY = PolicyConfig(version="text-first-v1", warn_cut=0.3, danger_cut=0.8)


def _fake_probability(texts):
    # 본문에 숫자로 확률을 넣어 결과를 예측할 수 있게 한다.
    return np.array([float(t) for t in texts])


@pytest.fixture
def client():
    predictor = Predictor(_fake_probability, POLICY, "ptext-koelectra-v4.0")
    return TestClient(create_app(predictor, "secret"))


def test_predictor_maps_probability_to_level_and_index():
    results = Predictor(_fake_probability, POLICY, "v").analyze(
        [
            {"review_id": "a", "content": "0.1"},
            {"review_id": "b", "content": "0.5"},
            {"review_id": "c", "content": "0.9"},
        ]
    )
    assert [r.level for r in results] == ["SAFE", "WARN", "DANGER"]
    assert [r.text_score for r in results] == [90, 50, 10]
    assert results[0].policy_index >= 70 and 40 <= results[1].policy_index <= 69 and results[2].policy_index <= 39


def test_analyze_requires_token(client):
    body = {"platform": "kurly", "product_id": "p", "reviews": [{"review_id": "a", "content": "0.2"}]}
    assert client.post("/v4/analyze", json=body).status_code == 401
    ok = client.post("/v4/analyze", json=body, headers={"X-Internal-Token": "secret"})
    assert ok.status_code == 200
    data = ok.json()
    assert data["policy_version"] == "text-first-v1" and data["results"][0]["level"] == "SAFE"


def test_analyze_rejects_duplicates_and_oversized(client):
    headers = {"X-Internal-Token": "secret"}
    dup = {"platform": "k", "product_id": "p", "reviews": [{"review_id": "a", "content": "0.1"}] * 2}
    assert client.post("/v4/analyze", json=dup, headers=headers).status_code == 422
    big = {"platform": "k", "product_id": "p", "reviews": [{"review_id": str(i), "content": "0.1"} for i in range(501)]}
    assert client.post("/v4/analyze", json=big, headers=headers).status_code == 422


def test_health(client):
    assert client.get("/health").json()["model_version"] == "ptext-koelectra-v4.0"


def test_policy_config_roundtrip_and_validation(tmp_path):
    path = tmp_path / "policy.json"
    POLICY.save(path)
    assert PolicyConfig.load(path) == POLICY
    path.write_text(json.dumps({"version": "x", "warn_cut": 0.9, "danger_cut": 0.5}))
    with pytest.raises(ValueError):
        PolicyConfig.load(path)
