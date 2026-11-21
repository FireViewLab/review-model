"""v4.0 텍스트 모델(KoELECTRA 이진 분류)을 학습하고 보정값까지 저장한다. 분석 PC GPU에서 돌린다.

사용:
  python train_v4.py <split_dir> <out_model_dir> [--base monologg/koelectra-base-v3-discriminator]
                     [--epochs 4] [--lr 3e-5] [--batch 32] [--max-length 256] [--target-precision 0.8]
                     [--limit N]   # 빠른 점검용: split마다 N건만 쓴다

- 손실은 클래스 비율 가중 교차 엔트로피다. val PR-AUC가 가장 좋은 epoch를 남긴다.
- 보정: val 로짓에 temperature scaling을 맞춘다(NLL 최소). 저장한 temperature를 실제 점수 계산에 쓴다.
- 임계값: val에서 정밀도 target 이상인 지점 중 재현율이 가장 큰 곳. 없으면 F1 최대.
"""

import argparse
import hashlib
import json
import math
import pathlib
import random

import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def average_precision(y, p):
    order = np.argsort(-p)
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


def pr_points(y, p):
    order = np.argsort(-p)
    ys, ps = np.asarray(y)[order], np.asarray(p)[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    return ps, tp / k, tp / max(ys.sum(), 1)


def choose_threshold(y, p, target):
    th, prec, rec = pr_points(y, p)
    ok = prec >= target
    if ok.any():
        i = int(np.argmax(np.where(ok, rec, -1)))
        return float(th[i]), "precision_target"
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(th[int(np.argmax(f1))]), "max_f1"


def fit_temperature(logit_diff, y):
    best, best_t = math.inf, 1.0
    for t in np.exp(np.linspace(math.log(0.05), math.log(20), 400)):
        q = 1 / (1 + np.exp(-logit_diff / t))
        q = np.clip(q, 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


class Split:
    """csv 한 개(platform, product_id, review_id, content, y). 서빙 환경에 pandas가 없어 표준 csv로 읽는다."""

    def __init__(self, rows):
        self.rows = rows
        self.content = [r["content"] for r in rows]
        self.platform = [r["platform"] for r in rows]
        self.y = np.array([int(r["y"]) for r in rows])

    def __len__(self):
        return len(self.rows)


def load(split_dir, name, limit=0):
    csv.field_size_limit(10**8)
    with open(pathlib.Path(split_dir) / f"{name}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return Split(rows[:limit] if limit else rows)


@torch.no_grad()
def logits_of(model, tok, texts, device, max_length, batch=64):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i: i + batch], truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        out.append(model(**enc).logits.float().cpu().numpy())
    lg = np.concatenate(out) if out else np.zeros((0, 2))
    return lg[:, 1] - lg[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("out")
    ap.add_argument("--base", default="monologg/koelectra-base-v3-discriminator")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--target-precision", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261121)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val = load(a.split_dir, "train", a.limit), load(a.split_dir, "val", a.limit)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.base, num_labels=2, id2label={0: "NORMAL", 1: "SUSPICIOUS"}, label2id={"NORMAL": 0, "SUSPICIOUS": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

    pos = max(int(train.y.sum()), 1)
    weight = torch.tensor([1.0, (len(train) - pos) / pos], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    data = list(zip(train.content, train.y.tolist()))
    loader = DataLoader(data, batch_size=a.batch, shuffle=True, collate_fn=lambda b: list(zip(*b)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(loader) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_ap, best_state, history = -1.0, None, []
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for texts, ys in loader:
            enc = tok(list(texts), truncation=True, max_length=a.max_length, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), torch.tensor(ys, device=device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        d = logits_of(model, tok, val.content, device, a.max_length)
        ap_val = average_precision(val.y, 1 / (1 + np.exp(-d)))
        history.append({"epoch": epoch + 1, "train_loss": total / max(len(loader), 1), "val_pr_auc": ap_val})
        print(json.dumps(history[-1]), flush=True)
        if ap_val > best_ap:
            best_ap = ap_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    d = logits_of(model, tok, val.content, device, a.max_length)
    temperature = fit_temperature(d, val.y)
    calibrated = 1 / (1 + np.exp(-d / temperature))
    threshold, method = choose_threshold(val.y, calibrated, a.target_precision)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    sha = hashlib.sha256()
    for name in ("train", "val", "test"):
        p = pathlib.Path(a.split_dir) / f"{name}.csv"
        if p.exists():
            sha.update(p.read_bytes())
    config = {
        "model_family": "ptext-koelectra-v4.0",
        "threshold": threshold,
        "threshold_space": "calibrated_probability",
        "threshold_method": method,
        "target_precision": a.target_precision,
        "calibration_temperature": temperature,
        "calibration_method": "sigmoid((logit_suspicious-logit_normal)/temperature)",
        "score_formula": "round((1-calibrated_probability)*100)",
        "max_length": a.max_length,
        "base_model": a.base,
        "val_pr_auc": best_ap,
        "history": history,
        "dataset_sha256": sha.hexdigest(),
    }
    (out / "inference_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(json.dumps({k: config[k] for k in ("threshold", "threshold_method", "calibration_temperature", "val_pr_auc")}))


if __name__ == "__main__":
    main()
