"""리뷰 CSV를 v4.0 텍스트 모델로 일괄 채점한다(오프라인 그림자 운영). 분석 PC GPU에서 돌린다.

사용:
  python score_v4.py <model_dir> <reviews.csv> <out.csv> [--batch 128]

입력 CSV에 platform, product_id, review_id, content가 있어야 한다.
출력: platform, product_id, review_id, prob_v4(보정된 의심 확률), text_score_v4(=round((1-prob)*100)),
      suspicious_v4(prob >= threshold)
"""

import argparse
import csv
import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    csv.field_size_limit(10**8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(f"{a.model}/inference_config.json"))
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, local_files_only=True).to(device).eval()
    temp, th, max_len = cfg["calibration_temperature"], cfg["threshold"], cfg["max_length"]

    with open(a.src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    # 길이순으로 묶어 패딩 낭비를 줄인다. 출력은 원래 순서로 쓴다.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["content"]))
    probs = np.zeros(len(rows))
    start = time.time()
    with torch.no_grad():
        for n, i in enumerate(range(0, len(order), a.batch)):
            idx = order[i: i + a.batch]
            enc = tok([rows[j]["content"] for j in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                lg = model(**enc).logits.float().cpu().numpy()
            probs[idx] = 1 / (1 + np.exp(-(lg[:, 1] - lg[:, 0]) / temp))
            if n % 50 == 0:
                done = min(i + a.batch, len(order))
                rate = done / max(time.time() - start, 1e-6)
                print(f"{done:,}/{len(rows):,} · {rate:,.0f}건/초 · 남은 {(len(rows) - done) / rate / 60:.1f}분", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["platform", "product_id", "review_id", "prob_v4", "text_score_v4", "suspicious_v4"])
        for r, p in zip(rows, probs):
            w.writerow([r["platform"], r["product_id"], r["review_id"], f"{p:.5f}", round((1 - p) * 100), int(p >= th)])
    print(f"완료 {len(rows):,}건 → {a.out} · 의심 {int((probs >= th).sum()):,}건 ({(probs >= th).mean():.1%})")


if __name__ == "__main__":
    main()
