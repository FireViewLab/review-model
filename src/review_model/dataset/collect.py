"""검수를 마친 패킷들을 원본 ID·본문과 합쳐 라벨 표 하나로 만든다.

패킷의 항목ID를 키 파일로 원본(platform, product_id, review_id, author_hash)에 잇고, 본문은 후보 풀에서
가져온다. 같은 리뷰가 여러 회차에 나오면 마지막 회차 라벨을 쓴다.

    python -m review_model.dataset.collect labels.csv \\
        llm/pilot400_final.xlsx:packet/v4_pilot_key.csv llm/main5000_final.xlsx:packet/v4_main_5000_key.csv \\
        --pool pool
"""

import argparse
import pathlib
import sys

import pandas as pd

from review_model.io import iter_pool

OUTPUT_COLUMNS = [
    "platform",
    "product_id",
    "review_id",
    "author_hash",
    "content",
    "label",
    "disclosed",
    "signals",
    "source",
]


def load_round(packet: str, key: str, order: int) -> pd.DataFrame:
    labels = pd.read_excel(packet, sheet_name=0, dtype=str)
    keys = pd.read_csv(key, dtype=str)
    m = labels[["항목ID", "최종 라벨", "광고 표시(Y/N)", "확인된 신호 코드"]].merge(
        keys, left_on="항목ID", right_on="item_id"
    )
    m = m[m["최종 라벨"].notna() & (m["최종 라벨"].str.strip() != "")]
    return m.assign(round=order, source=pathlib.Path(packet).name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("rounds", nargs="+", help="<라벨 xlsx>:<키 csv>")
    ap.add_argument("--pool", type=pathlib.Path, default=pathlib.Path("pool"))
    a = ap.parse_args(argv)

    df = pd.concat([load_round(*pair.split(":"), order) for order, pair in enumerate(a.rounds)], ignore_index=True)
    df["label"] = df["최종 라벨"].str.strip()
    df["disclosed"] = df["광고 표시(Y/N)"].fillna("").str.strip().eq("Y") | df.label.eq("DISCLOSED_PROMO")
    df["signals"] = df["확인된 신호 코드"].fillna("").str.replace(r"[\s,]+", ";", regex=True).str.strip(";")
    content = {(r["platform"], r["review_id"]): r["content"] for r in iter_pool(a.pool)}
    df["content"] = [content.get(k) for k in zip(df.platform, df.review_id, strict=False)]
    if df.content.isna().any():
        sys.exit(f"본문을 찾지 못한 항목 {int(df.content.isna().sum())}건이 있습니다. 후보 풀 경로를 확인하세요.")
    # 같은 리뷰(플랫폼·review_id)는 한 번만 센다.
    df = df.sort_values("round").drop_duplicates(["platform", "review_id"], keep="last")
    df[OUTPUT_COLUMNS].to_csv(a.out, index=False)
    print(f"{len(df)}건 → {a.out}")
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()
