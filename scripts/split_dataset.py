"""라벨 표를 누출 없이 train/val/test로 나눈다.

사용:
  python split_dataset.py <labels.csv> <out_dir> [--synthetic synth.csv] [--seed 20261121]

규칙 (v4.0 초안 6장, 결정 ②·③):
- 학습 대상은 NORMAL(0)·SUSPICIOUS(1)이다. UNCERTAIN·INVALID는 뺀다.
  DISCLOSED_PROMO는 텍스트 이진 모델에서 빼고 따로 집계한다.
- 같은 리뷰(플랫폼·review_id)는 한 번만 쓴다.
- 같은 상품, 같은 본문(공백·기호 제거), 같은 작성자 해시(마스킹 제외)로 이어진 리뷰는 한 묶음이다.
  묶음은 통째로 한쪽에만 들어간다.
- 플랫폼·라벨 비율이 세 묶음에서 비슷하도록 묶음을 큰 것부터 배정한다.
- 합성 데이터는 train에만, train의 20% 이하로 넣는다. val·test는 실제 리뷰만이다.
"""

import argparse
import json
import pathlib
import random
import re
from collections import Counter, defaultdict

import pandas as pd

RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}
TARGET = {"NORMAL": 0, "SUSPICIOUS": 1}


def norm(text):
    return re.sub(r"[^0-9a-z가-힣]", "", str(text).lower())


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def groups(df):
    dsu = DSU(len(df))
    for keys in (
        df.platform + "|p|" + df.product_id,
        df.content.map(norm).where(df.content.map(norm).str.len() >= 10),
        df.author_hash.where(df.author_hash.notna() & ~df.author_hash.fillna("").str.startswith("masked-") & (df.author_hash.fillna("") != "")),
    ):
        first = {}
        for i, k in enumerate(keys):
            if pd.isna(k):
                continue
            if k in first:
                dsu.union(i, first[k])
            else:
                first[k] = i
    return [dsu.find(i) for i in range(len(df))]


def assign(df, seed):
    rng = random.Random(seed)
    cells = Counter(zip(df.platform, df.y))
    want = {s: {c: n * r for c, n in cells.items()} for s, r in RATIO.items()}
    have = {s: Counter() for s in RATIO}
    members = defaultdict(list)
    for i, g in enumerate(df.group):
        members[g].append(i)
    order = list(members)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(members[g]))
    split = {}
    for g in order:
        comp = Counter((df.platform.iat[i], df.y.iat[i]) for i in members[g])
        # 가장 모자란 쪽에 넣는다.
        best = max(RATIO, key=lambda s: sum((want[s][c] - have[s][c]) * n for c, n in comp.items()))
        split[g] = best
        have[best].update(comp)
    return df.group.map(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--synthetic")
    ap.add_argument("--seed", type=int, default=20261121)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(a.labels, dtype=str)
    raw = raw.drop_duplicates(["platform", "review_id"], keep="last")
    excluded = raw[~raw.label.isin(TARGET)].label.value_counts().to_dict()
    df = raw[raw.label.isin(TARGET)].reset_index(drop=True)
    df["y"] = df.label.map(TARGET)
    df["group"] = groups(df)
    df["split"] = assign(df, a.seed)

    # 누출 검사: 묶음 키가 두 split에 걸치면 안 된다.
    for name, key in (("product", df.platform + "|" + df.product_id), ("text", df.content.map(norm))):
        spread = df.groupby(key).split.nunique()
        bad = spread[spread > 1]
        if name == "text":
            bad = bad[[len(k) >= 10 for k in bad.index]]
        assert bad.empty, f"{name} 누출 {len(bad)}건"

    parts = {s: df[df.split == s][["platform", "product_id", "review_id", "content", "y"]].copy() for s in RATIO}
    synth_added = 0
    if a.synthetic:
        syn = pd.read_csv(a.synthetic, dtype=str)
        cap = int(len(parts["train"]) * 0.2 / 0.8)
        syn = syn.sample(min(cap, len(syn)), random_state=a.seed)
        syn = syn.assign(platform="synthetic", product_id="synthetic", review_id=[f"syn-{i}" for i in range(len(syn))],
                         y=syn.label.map(TARGET))
        parts["train"] = pd.concat([parts["train"], syn[parts["train"].columns]], ignore_index=True)
        synth_added = len(syn)
    for s, p in parts.items():
        p.sample(frac=1, random_state=a.seed).to_csv(out / f"{s}.csv", index=False)

    summary = {
        "excluded": excluded,
        "synthetic_in_train": synth_added,
        "splits": {s: {"n": len(p), "suspicious": int(p.y.astype(int).sum()),
                       "platforms": p.platform.value_counts().to_dict()} for s, p in parts.items()},
        "groups": int(df.group.nunique()),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
