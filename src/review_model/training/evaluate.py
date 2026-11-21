"""모델을 시험 세트와 프로브 문장으로 평가하고 기준 모델(v3.1)과 비교한다.

관문: 시험 PR-AUC가 기준 모델보다 높아야 한다. 정밀도 0.8에서의 재현율, 플랫폼별 PR-AUC 편차
(양성 5건 이상 플랫폼만), 프로브 점수(참고용)를 같이 보고한다.

    python -m review_model.training.evaluate models/ptext-koelectra-v4.0 split/test.csv eval.json \\
        --baseline /path/to/ptext-koelectra-v3.1-b
"""

import argparse
import json

import numpy as np

from review_model.training import probes
from review_model.training.data import load_split
from review_model.training.metrics import average_precision, recall_at_precision
from review_model.training.model import calibrated_probability, device, load_model, logit_diff


def report(model_dir, test, dev) -> dict:
    tok, model, config = load_model(model_dir, dev)
    max_length = config.get("max_length", 256)

    def prob(texts):
        return calibrated_probability(logit_diff(model, tok, list(texts), dev, max_length), config)

    p, y = prob(test.content), test.y
    threshold = config.get("threshold", 0.5)
    predicted = p >= threshold
    tp = int((predicted & (y == 1)).sum())
    platforms = np.array(test.platform)
    per_platform = {
        pf: round(average_precision(y[platforms == pf], p[platforms == pf]), 4)
        for pf in sorted(set(platforms))
        if y[platforms == pf].sum() >= 5
    }
    s, d, g = prob(probes.SUSPICIOUS), prob(probes.DISCLOSED), prob(probes.GENUINE)
    return {
        "pr_auc": round(average_precision(y, p), 4),
        "threshold": threshold,
        "precision": round(tp / max(predicted.sum(), 1), 4),
        "recall": round(tp / max((y == 1).sum(), 1), 4),
        "recall_at_precision_0.8": round(recall_at_precision(y, p, 0.8), 4),
        "per_platform_pr_auc": per_platform,
        "per_platform_spread": round(max(per_platform.values()) - min(per_platform.values()), 4)
        if len(per_platform) > 1
        else None,
        "probe_suspicious_mean": round(float(s.mean()), 4),
        "probe_disclosed_mean": round(float(d.mean()), 4),
        "probe_genuine_mean": round(float(g.mean()), 4),
        "probe_gap": round(float(s.mean() - g.mean()), 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("test")
    ap.add_argument("out")
    ap.add_argument("--baseline")
    a = ap.parse_args(argv)
    dev = device()
    test = load_split(a.test)
    result = {"n": len(test), "suspicious": int(test.y.sum()), "candidate": report(a.model, test, dev)}
    if a.baseline:
        result["baseline"] = report(a.baseline, test, dev)
    candidate, baseline = result["candidate"], result.get("baseline")
    result["gate"] = {
        "pr_auc_above_baseline": None if not baseline else candidate["pr_auc"] > baseline["pr_auc"],
        "platform_spread": candidate["per_platform_spread"],
        "probe_gap_reference_only": candidate["probe_gap"],
    }
    with open(a.out, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
