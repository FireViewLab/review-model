"""KoELECTRA 이진 분류(NORMAL/SUSPICIOUS) 파인튜닝. 보정값과 임계값까지 저장한다.

- 손실: 클래스 비율 가중 교차 엔트로피. val PR-AUC가 가장 좋은 epoch를 남긴다.
- 학습률: 전체 스텝의 10% 워밍업 후 선형 감소. bf16 autocast(CUDA).
- 보정: val 로짓에 temperature scaling(NLL 최소). 이 값을 실제 점수 계산에 쓴다.
- 임계값: val에서 정밀도 목표 이상인 지점 중 재현율이 가장 큰 곳.

    python -m review_model.training.train split models/ptext-koelectra-v4.0 [--epochs 4]
"""

import argparse
import hashlib
import json
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from review_model.training.data import load_split
from review_model.training.metrics import average_precision, choose_threshold, fit_temperature, sigmoid
from review_model.training.model import MODEL_FAMILY, device, logit_diff

BASE_MODEL = "monologg/koelectra-base-v3-discriminator"
LABEL_MAP = {"id2label": {0: "NORMAL", 1: "SUSPICIOUS"}, "label2id": {"NORMAL": 0, "SUSPICIOUS": 1}}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dataset_sha256(split_dir: pathlib.Path) -> str:
    sha = hashlib.sha256()
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.csv"
        if path.exists():
            sha.update(path.read_bytes())
    return sha.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split_dir", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--target-precision", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="빠른 점검용: split마다 N건만 쓴다")
    ap.add_argument("--seed", type=int, default=20261121)
    a = ap.parse_args(argv)
    seed_everything(a.seed)
    dev = device()

    train = load_split(a.split_dir / "train.csv", a.limit)
    val = load_split(a.split_dir / "val.csv", a.limit)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.base, num_labels=2, ignore_mismatched_sizes=True, **LABEL_MAP
    ).to(dev)

    positives = max(int(train.y.sum()), 1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, (len(train) - positives) / positives], device=dev))
    loader = DataLoader(
        list(zip(train.content, train.y.tolist(), strict=False)),
        batch_size=a.batch,
        shuffle=True,
        collate_fn=lambda b: list(zip(*b, strict=False)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(loader) * a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * a.warmup), steps)
    print(
        json.dumps(
            {
                "train": len(train),
                "val": len(val),
                "steps_per_epoch": len(loader),
                "total_steps": steps,
                "warmup_steps": int(steps * a.warmup),
                "device": dev,
            }
        ),
        flush=True,
    )

    best_ap, best_state, history = -1.0, None, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for texts, ys in loader:
            enc = tok(list(texts), truncation=True, max_length=a.max_length, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), torch.tensor(ys, device=dev))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        val_ap = average_precision(val.y, sigmoid(logit_diff(model, tok, val.content, dev, a.max_length)))
        history.append({"epoch": epoch, "train_loss": total / max(len(loader), 1), "val_pr_auc": val_ap})
        print(json.dumps(history[-1]), flush=True)
        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    diff = logit_diff(model, tok, val.content, dev, a.max_length)
    temperature = fit_temperature(diff, val.y)
    threshold, method = choose_threshold(val.y, sigmoid(diff / temperature), a.target_precision)

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    config = {
        "model_family": MODEL_FAMILY,
        "threshold": threshold,
        "threshold_space": "calibrated_probability",
        "threshold_method": method,
        "target_precision": a.target_precision,
        "calibration_temperature": temperature,
        "calibration_method": "sigmoid((logit_suspicious-logit_normal)/temperature)",
        "score_formula": "round((1-calibrated_probability)*100)",
        "max_length": a.max_length,
        "base_model": a.base,
        "hyperparameters": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "warmup": a.warmup, "seed": a.seed},
        "val_pr_auc": best_ap,
        "history": history,
        "dataset_sha256": dataset_sha256(a.split_dir),
    }
    (a.out / "inference_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(
        json.dumps({k: config[k] for k in ("threshold", "threshold_method", "calibration_temperature", "val_pr_auc")})
    )


if __name__ == "__main__":
    main()
