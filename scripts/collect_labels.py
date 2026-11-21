"""검수를 마친 패킷들을 원본 ID·본문과 합쳐 라벨 표 하나로 만든다.

사용:
  python collect_labels.py <out.csv> <packet.xlsx>:<key.csv> [<packet.xlsx>:<key.csv> ...]

패킷의 항목ID를 key 파일로 원본(platform, product_id, review_id, author_hash)에 잇고,
본문은 후보 풀(pool/*.csv)에서 가져온다. 같은 리뷰가 여러 회차에 나오면 마지막 회차 라벨을 쓴다.
"""

import csv
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
csv.field_size_limit(10**8)


def pool_rows():
    rows = {}
    for f in (ROOT / "pool").glob("*.csv"):
        if f.name.endswith("_counts.csv"):
            continue
        for r in csv.DictReader(f.open()):
            rows[(r["platform"], r["review_id"])] = r
    return rows


def main(out, *pairs):
    pool = pool_rows()
    frames = []
    for order, pair in enumerate(pairs):
        packet, key = pair.split(":")
        p = pd.read_excel(packet, sheet_name=0, dtype=str)
        k = pd.read_csv(key, dtype=str)
        m = p[["항목ID", "최종 라벨", "광고 표시(Y/N)", "확인된 신호 코드"]].merge(k, left_on="항목ID", right_on="item_id")
        m = m[m["최종 라벨"].notna() & (m["최종 라벨"].str.strip() != "")]
        m["round"] = order
        m["source"] = pathlib.Path(packet).name
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["최종 라벨"].str.strip()
    df["disclosed"] = df["광고 표시(Y/N)"].fillna("").str.strip().eq("Y") | df.label.eq("DISCLOSED_PROMO")
    df["signals"] = df["확인된 신호 코드"].fillna("").str.replace(r"[\s,]+", ";", regex=True).str.strip(";")
    df["content"] = [pool.get((p, r), {}).get("content") for p, r in zip(df.platform, df.review_id)]
    missing = df.content.isna().sum()
    if missing:
        sys.exit(f"본문을 찾지 못한 항목 {missing}건이 있습니다. 후보 풀이 바뀌었는지 확인하세요.")
    # 결정 ②: 같은 리뷰(플랫폼·review_id)는 한 번만 센다.
    df = df.sort_values("round").drop_duplicates(["platform", "review_id"], keep="last")
    cols = ["platform", "product_id", "review_id", "author_hash", "content", "label", "disclosed", "signals", "source"]
    df[cols].to_csv(out, index=False)
    print(f"{len(df)}건 → {out}")
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
