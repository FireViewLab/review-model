from review_model.candidates.flags import author_hash, build_pool, flags
from review_model.text import meaningful_length, normalize


def _row(**kw):
    base = {
        "content": "평범한 사용 후기입니다",
        "mlen": "10",
        "rating": "5",
        "imgs": "1",
        "day_c": "1",
        "day_mean": "1",
        "dup_ids": "0",
        "author_n": "",
        "author_prods": "",
        "review_id": "r1",
        "author": "",
        "platform": "kurly",
    }
    return {**base, **kw}


def test_repeat_ignores_laughter():
    assert flags(_row(content="좋아요 좋아요 좋아요 좋아요"))["repeat"]
    assert not flags(_row(content="하하하하하하하하하하"))["repeat"]


def test_disclosed_and_burst():
    assert flags(_row(content="체험단으로 제공받아 작성했어요"))["disclosed"]
    assert flags(_row(day_c="12", day_mean="2"))["burst"]
    assert not flags(_row(day_c="12", day_mean="5"))["burst"]


def test_author_hash_marks_masked_and_hides_name():
    h = author_hash("kurly", "김**", b"salt")
    assert h.startswith("masked-") and "김" not in h
    assert author_hash("kurly", "", b"salt") == ""


def test_pool_counts_shared_review_once():
    rows = [_row(review_id="r1", product_id="p1"), _row(review_id="r1", product_id="p2")]
    _, total, _, pool = build_pool(rows, "kurly", b"salt", per_stratum=5, random_size=5)
    assert total == 1
    assert all("author" not in r for r in pool)


def test_text_helpers():
    assert normalize("Good!! 좋아요 :)") == "good좋아요"
    assert meaningful_length("좋아요 !! 👍") == 3
