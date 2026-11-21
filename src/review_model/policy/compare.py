"""등급 규칙 후보 A·B·C와 v3.1을 같은 라벨로 비교한다.

기준값(WARN·DANGER 경계)과 C의 가중치는 검증 세트로만 정하고, 성능은 시험 세트로 잰다.
운영 표본에 적용했을 때의 플랫폼별 WARN 이상 비율도 같이 낸다.

    python -m review_model.policy.compare \\
        --val split/val.csv --val-scores shadow/val_v4.csv \\
        --test split/test.csv --test-scores shadow/test_v4.csv \\
        --pool pool --v31 llm/labeled_v31.csv \\
        --sample shadow/shadow_sample.csv --sample-scores shadow/scores_v4.csv \\
        --out policy_compare.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from review_model.constants import EMPTY_IMAGES_VERIFIED
from review_model.io import iter_pool
from review_model.policy import rules
from review_model.text import meaningful_length
from review_model.training.metrics import average_precision, pr_points

KEYS = ["platform", "product_id", "review_id"]
WARN_PRECISION, DANGER_PRECISION = 0.5, 0.8


def cut_at_precision(y, score, target) -> float:
    """정밀도 목표를 만족하는 지점 중 재현율이 가장 큰 경계값."""
    th, precision, recall = pr_points(y, score)
    ok = precision >= target
    return float(th[int(np.argmax(np.where(ok, recall, -1)))]) if ok.any() else float(th[0])


def evidence(df: pd.DataFrame) -> rules.Evidence:
    imgs = pd.to_numeric(df.imgs, errors="coerce").fillna(0).to_numpy()
    rating = pd.to_numeric(df.rating, errors="coerce")
    dup = pd.to_numeric(df.get("dup_ids", pd.Series(0, index=df.index)), errors="coerce").fillna(0).to_numpy()
    length = df.content.map(meaningful_length).to_numpy()
    return rules.Evidence(
        text_prob=df.prob_v4.astype(float).to_numpy(),
        image_present=imgs > 0,
        image_absent_verified=(imgs == 0) & df.platform.isin(EMPTY_IMAGES_VERIFIED).to_numpy(),
        rating_five=(rating == 5).to_numpy(),
        rating_missing=rating.isna().to_numpy(),
        meaningful_length=length,
        cross_duplicate=(dup >= 2) & (length >= 30),
    )


def labeled(split_path, scores_path, pool: pd.DataFrame) -> pd.DataFrame:
    split = pd.read_csv(split_path, dtype=str)
    scores = pd.read_csv(scores_path, dtype=str)[[*KEYS, "prob_v4"]]
    df = split.merge(scores, on=KEYS).merge(pool, on=["platform", "review_id"], how="left", suffixes=("", "_pool"))
    df["y"] = df.y.astype(int)
    return df


def summarize(name, y, score, grades, photo) -> dict:
    out = {"pr_auc": round(average_precision(y, score), 4)}
    for g in ("SAFE", "WARN", "DANGER"):
        m = grades == g
        out[f"{g}_share"] = round(float(m.mean()), 3)
        out[f"{g}_suspicious_rate"] = round(float(y[m].mean()), 3) if m.any() else None
    normal = y == 0
    # 정상 리뷰인데 사진 유무만으로 경고 비율이 얼마나 갈리는지.
    warn = grades != "SAFE"
    out["normal_warn_rate_photo"] = round(float(warn[normal & photo].mean()), 3)
    out["normal_warn_rate_no_photo"] = round(float(warn[normal & ~photo].mean()), 3)
    out["suspicious_recall_warn_plus"] = round(float(warn[y == 1].mean()), 3)
    print(
        f"{name:<14} PR-AUC {out['pr_auc']:.3f} · WARN+ 재현율 {out['suspicious_recall_warn_plus']:.2f} · "
        f"정상 경고율 사진 {out['normal_warn_rate_photo']:.2f} / 사진없음 {out['normal_warn_rate_no_photo']:.2f} · "
        f"DANGER 정밀도 {out['DANGER_suspicious_rate']}"
    )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("val", "val-scores", "test", "test-scores", "pool", "v31", "sample", "sample-scores", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)

    pool = pd.DataFrame(iter_pool(a.pool)).drop_duplicates(["platform", "review_id"])
    pool = pool[["platform", "review_id", "imgs", "rating", "dup_ids"]]
    val, test = labeled(a.val, a.val_scores, pool), labeled(a.test, a.test_scores, pool)
    ev_val, ev_test = evidence(val), evidence(test)

    cuts = rules.Cuts(
        cut_at_precision(val.y, ev_val.text_prob, WARN_PRECISION),
        cut_at_precision(val.y, ev_val.text_prob, DANGER_PRECISION),
    )
    weights = rules.fit_logistic(rules.features(ev_val), val.y.to_numpy())
    learned_val = rules.learned(ev_val, weights)
    cuts_c = rules.Cuts(
        cut_at_precision(val.y, learned_val, WARN_PRECISION), cut_at_precision(val.y, learned_val, DANGER_PRECISION)
    )

    candidates = {
        "A_text_first": (lambda ev: rules.text_first(ev), cuts),
        "B_text_evidence": (lambda ev: rules.text_evidence(ev, cuts), cuts),
        "C_learned": (lambda ev: rules.learned(ev, weights), cuts_c),
    }
    y, photo = test.y.to_numpy(), ev_test.image_present
    print(f"시험 세트 {len(test)}건 (의심 {int(y.sum())}) · 경계 A/B warn {cuts.warn:.3f} danger {cuts.danger:.3f}")
    result = {
        "test_n": len(test),
        "test_suspicious": int(y.sum()),
        "cuts": {"A_B": vars(cuts), "C": vars(cuts_c)},
        "C_weights": dict(zip([*rules.FEATURE_NAMES, "intercept"], np.round(weights, 4).tolist(), strict=True)),
        "test": {},
        "sample_warn_plus_by_platform": {},
    }

    v31 = pd.read_csv(a.v31, dtype=str)
    base = test.merge(v31[[*KEYS, "policy_index_v3", "evidence_level_v3"]], on=KEYS, how="left")
    has = base.policy_index_v3.notna().to_numpy()
    result["test"]["v3.1"] = summarize(
        "v3.1",
        y[has],
        1 - base.policy_index_v3[has].astype(float).to_numpy() / 100,
        base.evidence_level_v3[has].to_numpy(),
        photo[has],
    )
    for name, (fn, c) in candidates.items():
        score = fn(ev_test)
        result["test"][name] = summarize(name, y, score, rules.grade(score, c), photo)

    sample = pd.read_csv(a.sample, dtype=str).merge(
        pd.read_csv(a.sample_scores, dtype=str)[[*KEYS, "prob_v4"]], on=KEYS
    )
    ev_sample = evidence(sample)
    rates = {"v3.1": sample.evidence_level_v3.ne("SAFE").groupby(sample.platform).mean()}
    for name, (fn, c) in candidates.items():
        rates[name] = (
            pd.Series(rules.grade(fn(ev_sample), c) != "SAFE", index=sample.index).groupby(sample.platform).mean()
        )
    table = pd.DataFrame(rates).round(3)
    print("\n운영 표본 WARN 이상 비율\n" + table.to_string())
    result["sample_warn_plus_by_platform"] = table.to_dict()
    with open(a.out, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
