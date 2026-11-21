"""모델 로딩과 추론 공통 함수."""

import json
import pathlib

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from review_model.training.metrics import sigmoid

MODEL_FAMILY = "ptext-koelectra-v4.0"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_dir: str | pathlib.Path, dev: str):
    """로컬 모델과 inference_config.json을 읽는다."""
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(dev).eval()
    config = json.loads((pathlib.Path(model_dir) / "inference_config.json").read_text())
    return tok, model, config


@torch.no_grad()
def logit_diff(model, tok, texts, dev: str, max_length: int, batch: int = 64, bf16: bool = False) -> np.ndarray:
    """SUSPICIOUS 로짓 - NORMAL 로짓."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(
            dev
        )
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=bf16 and dev == "cuda"):
            logits = model(**enc).logits.float().cpu().numpy()
        out.append(logits[:, 1] - logits[:, 0])
    return np.concatenate(out) if out else np.zeros(0)


def calibrated_probability(diff: np.ndarray, config: dict) -> np.ndarray:
    """v4.0 모델은 저장된 temperature로 보정한다. 그 외 모델(v3.1 등)은 원 확률을 쓴다."""
    temperature = config.get("calibration_temperature", 1.0) if config.get("model_family") == MODEL_FAMILY else 1.0
    return sigmoid(diff / temperature)
