import pandas as pd

from review_model.dataset.split import split_labels


def _labels(n=300):
    rows = []
    for i in range(n):
        rows.append(
            {
                "platform": ["kurly", "musinsa", "elevenst"][i % 3],
                "product_id": f"p{i // 4}",  # 상품당 4건
                "review_id": f"r{i}",
                "author_hash": "masked-abc" if i % 5 == 0 else f"a{i % 40}",
                "content": f"리뷰 본문 {i} 입니다 충분히 긴 문장" if i % 7 else "같은 문장이 반복되는 리뷰 본문입니다",
                "label": ["NORMAL", "NORMAL", "SUSPICIOUS", "UNCERTAIN"][i % 4],
            }
        )
    return pd.DataFrame(rows)


def test_split_has_no_product_or_text_leakage():
    df, excluded = split_labels(_labels())
    assert excluded == {"UNCERTAIN": 75}
    assert set(df.split) <= {"train", "val", "test"}
    assert (df.groupby(df.platform + "|" + df.product_id).split.nunique() == 1).all()
    assert df[df.content.str.startswith("같은 문장")].split.nunique() == 1


def test_masked_authors_do_not_link_groups():
    df = _labels(40)
    df["author_hash"] = "masked-same"
    df["content"] = [f"서로 다른 본문 번호 {i} 입니다" for i in range(len(df))]
    out, _ = split_labels(df)
    # 마스킹 닉네임이 같아도 상품이 다르면 같은 묶음이 아니다.
    assert out.group.nunique() == (out.platform + "|" + out.product_id).nunique()


def test_duplicate_review_ids_counted_once():
    df = pd.concat([_labels(20), _labels(20)])
    out, _ = split_labels(df)
    assert out.review_id.is_unique
