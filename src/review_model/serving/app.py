"""v4.0 분석 HTTP 서비스.

    REVIEW_MODEL_DIR=models/ptext-koelectra-v4.0 REVIEW_MODEL_TOKEN=... \\
        uvicorn review_model.serving.app:app --host 127.0.0.1 --port 8100

- POST /v4/analyze: 같은 상품의 리뷰 묶음을 받아 리뷰별 결과를 돌려준다. X-Internal-Token 헤더가 필요하다.
- GET /health: 모델·규칙 버전
"""

import hmac
import os
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from review_model.serving.predictor import Predictor

MAX_REVIEWS = 500


class ReviewIn(BaseModel):
    review_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    platform: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    reviews: list[ReviewIn] = Field(max_length=MAX_REVIEWS)


class ReviewOut(BaseModel):
    review_id: str
    text_probability: float
    text_score: int
    policy_index: float
    level: str


class AnalyzeResponse(BaseModel):
    platform: str
    product_id: str
    model_version: str
    policy_version: str
    results: list[ReviewOut]


def create_app(predictor: Predictor, token: str) -> FastAPI:
    if not token:
        raise ValueError("REVIEW_MODEL_TOKEN이 비어 있습니다.")
    app = FastAPI(title="review-model", version="4.0.0")

    def authorize(x_internal_token: str = Header(default="")) -> None:
        if not hmac.compare_digest(x_internal_token, token):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": predictor.model_version, "policy_version": predictor.policy.version}

    @app.post("/v4/analyze", response_model=AnalyzeResponse, dependencies=[Depends(authorize)])
    def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
        ids = [r.review_id for r in body.reviews]
        if len(ids) != len(set(ids)):
            raise HTTPException(status_code=422, detail="duplicate review_id")
        results = predictor.analyze([r.model_dump() for r in body.reviews])
        return AnalyzeResponse(
            platform=body.platform,
            product_id=body.product_id,
            model_version=predictor.model_version,
            policy_version=predictor.policy.version,
            results=[ReviewOut(**asdict(r)) for r in results],
        )

    return app


def _from_env() -> FastAPI:
    return create_app(
        Predictor.from_model_dir(os.environ["REVIEW_MODEL_DIR"]), os.environ.get("REVIEW_MODEL_TOKEN", "")
    )


# uvicorn review_model.serving.app:app 으로 띄울 때만 모델을 읽는다.
app = _from_env() if os.environ.get("REVIEW_MODEL_DIR") else None
