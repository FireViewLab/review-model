"""라벨 표를 누출 없이 train/val/test로 나눈다.

- 학습 대상은 NORMAL(0)·SUSPICIOUS(1)이다. UNCERTAIN·INVALID·DISCLOSED_PROMO는 빼고 집계만 한다.
- 같은 리뷰(플랫폼·review_id)는 한 번만 쓴다.
- 같은 상품, 같은 정규화 본문(10자 이상), 같은 작성자 해시(마스킹 제외)로 이어진 리뷰는 한 묶음이며
  묶음은 통째로 한 split에만 들어간다.
- 플랫폼·라벨 비율이 세 split에서 비슷하도록 큰 묶음부터 가장 모자란 split에 배정한다.
- 합성 데이터는 train에만, train의 20% 이하로 넣는다. val·test는 실제 리뷰만이다.

    python -m review_model.dataset.split labels.csv split [--synthetic synth.csv]
"""

import argparse
import json
import pathlib
import random
from collections import Counter, defaultdict

import pandas as pd

from review_model.constants import TARGET
from review_model.text import normalize

RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 20261121
SPLIT_COLUMNS = ["platform", "product_id", "review_id", "content", "y"]


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def group_ids(df: pd.DataFrame) -> list[int]:
    """상품·본문·작성자 키 중 하나라도 같으면 같은 묶음으로 잇는다."""
    dsu = DisjointSet(len(df))
    text = df.content.map(normalize)
    author = df.author_hash.fillna("") if "author_hash" in df else pd.Series("", index=df.index)
    keys = (
        df.platform + "|" + df.product_id,
        text.where(text.str.len() >= 10),
        author.where((author != "") & ~author.str.startswith("masked-")),
    )
    for column in keys:
        first: dict[str, int] = {}
        for i, key in enumerate(column):
            if pd.isna(key):
                continue
            if key in first:
                dsu.union(i, first[key])
            else:
                first[key] = i
    return [dsu.find(i) for i in range(len(df))]


def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    cells = Counter(zip(df.platform, df.y, strict=False))
    want = {s: {c: n * r for c, n in cells.items()} for s, r in RATIO.items()}
    have = {s: Counter() for s in RATIO}
    members = defaultdict(list)
    for i, g in enumerate(df.group):
        members[g].append(i)
    order = list(members)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -len(members[g]))
    split = {}
    for g in order:
        comp = Counter((df.platform.iat[i], df.y.iat[i]) for i in members[g])
        best = max(RATIO, key=lambda s: sum((want[s][c] - have[s][c]) * n for c, n in comp.items()))
        split[g] = best
        have[best].update(comp)
    return df.group.map(split)


def check_leakage(df: pd.DataFrame) -> None:
    """같은 상품이나 같은 본문(10자 이상)이 두 split에 걸치면 실패한다."""
    product = df.platform + "|" + df.product_id
    assert (df.groupby(product).split.nunique() <= 1).all(), "상품 누출"
    text = df.content.map(normalize)
    spread = df[text.str.len() >= 10].groupby(text).split.nunique()
    assert (spread <= 1).all(), "본문 누출"


def split_labels(raw: pd.DataFrame, seed: int = SEED):
    raw = raw.drop_duplicates(["platform", "review_id"], keep="last")
    excluded = raw[~raw.label.isin(TARGET)].label.value_counts().to_dict()
    df = raw[raw.label.isin(TARGET)].reset_index(drop=True)
    df["y"] = df.label.map(TARGET)
    df["group"] = group_ids(df)
    df["split"] = assign_splits(df, seed)
    check_leakage(df)
    return df, excluded


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--synthetic", help="합성 데이터 CSV (content, label)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)

    df, excluded = split_labels(pd.read_csv(a.labels, dtype=str), a.seed)
    parts = {s: df[df.split == s][SPLIT_COLUMNS].copy() for s in RATIO}
    synthetic = 0
    if a.synthetic:
        syn = pd.read_csv(a.synthetic, dtype=str)
        cap = int(len(parts["train"]) * 0.2 / 0.8)
        syn = syn.sample(min(cap, len(syn)), random_state=a.seed)
        syn = syn.assign(
            platform="synthetic",
            product_id="synthetic",
            review_id=[f"syn-{i}" for i in range(len(syn))],
            y=syn.label.map(TARGET),
        )
        parts["train"] = pd.concat([parts["train"], syn[SPLIT_COLUMNS]], ignore_index=True)
        synthetic = len(syn)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.sample(frac=1, random_state=a.seed).to_csv(a.out / f"{name}.csv", index=False)
    summary = {
        "excluded": excluded,
        "synthetic_in_train": synthetic,
        "groups": int(df.group.nunique()),
        "splits": {
            s: {"n": len(p), "suspicious": int(p.y.astype(int).sum()), "platforms": p.platform.value_counts().to_dict()}
            for s, p in parts.items()
        },
    }
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
