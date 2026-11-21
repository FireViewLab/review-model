"""리뷰 CSV를 모델로 일괄 채점한다(오프라인 그림자 운영).

입력 CSV에는 platform, product_id, review_id, content가 있어야 한다. 길이순으로 묶어 패딩 낭비를 줄이고
출력은 입력 순서대로 쓴다.

    python -m review_model.training.score models/ptext-koelectra-v4.0 shadow/sample.csv shadow/scores_v4.csv

출력: platform, product_id, review_id, prob_v4(보정된 의심 확률), text_score_v4(=round((1-prob)*100)),
      suspicious_v4(prob >= threshold)
"""

import argparse
import csv
import time

import numpy as np

from review_model.io import read_rows
from review_model.training.model import calibrated_probability, device, load_model, logit_diff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args(argv)
    dev = device()
    tok, model, config = load_model(a.model, dev)
    rows = read_rows(a.src)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["content"]))
    probs = np.zeros(len(rows))
    start = time.time()
    for n, i in enumerate(range(0, len(order), a.batch)):
        idx = order[i : i + a.batch]
        diff = logit_diff(
            model, tok, [rows[j]["content"] for j in idx], dev, config["max_length"], batch=a.batch, bf16=True
        )
        probs[idx] = calibrated_probability(diff, config)
        if n % 50 == 0:
            done = min(i + a.batch, len(order))
            rate = done / max(time.time() - start, 1e-6)
            print(
                f"{done:,}/{len(rows):,} · {rate:,.0f}건/초 · 남은 {(len(rows) - done) / rate / 60:.1f}분", flush=True
            )

    threshold = config["threshold"]
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["platform", "product_id", "review_id", "prob_v4", "text_score_v4", "suspicious_v4"])
        for row, p in zip(rows, probs, strict=False):
            w.writerow(
                [
                    row["platform"],
                    row["product_id"],
                    row["review_id"],
                    f"{p:.5f}",
                    round((1 - p) * 100),
                    int(p >= threshold),
                ]
            )
    flagged = probs >= threshold
    print(f"완료 {len(rows):,}건 → {a.out} · 의심 {int(flagged.sum()):,}건 ({flagged.mean():.1%})")


if __name__ == "__main__":
    main()
