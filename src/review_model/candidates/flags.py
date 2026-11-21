"""리뷰별 집계 행에 의심 신호를 판정하고 층별 후보 풀을 만든다.

입력은 `sql/candidate_rows.sql`의 출력(CSV, 표준 입력)이다. 작성자는 HMAC 해시로만 저장하고
원래 닉네임은 디스크에 남기지 않는다. 같은 리뷰가 여러 옵션 상품에 저장된 경우는 한 번만 센다.

    psql ... < sql/candidate_rows.sql | python -m review_model.candidates.flags <platform> <out_dir>

출력: `<out_dir>/<platform>.csv`(층마다 최대 `--per-stratum`건 + 무작위 `--random`건),
      `<out_dir>/<platform>_counts.csv`(층별 전체 해당 건수)
"""

import argparse
import csv
import hashlib
import heapq
import hmac
import pathlib
import re
import sys

from review_model.constants import STRATA

SALT_PATH = pathlib.Path.home() / ".config/review-v4/author_salt"

PATTERNS = {
    "promo": re.compile(
        r"(무조건|꼭|강력|적극|다들|모두)\s*(사세요|구매하세요|구입하세요|추천합니다|추천해요|쟁이세요)"
        r"|강추|인생템|쟁여|사야\s*(돼|해)|후회\s*안|안\s*사면\s*손해|품절\s*전에|지금\s*당장"
    ),
    "unused_claim": re.compile(
        r"아직\s*(안|못)\s*(써|사용|먹|발라|입어)|사용\s*전|개봉\s*전|미개봉|안\s*써\s*봤|써보진|써\s*보지는|배송만|받기만"
    ),
    "unused_effect": re.compile(r"효과|좋을\s*것|좋아요|최고|추천|만족"),
    "certain": re.compile(r"100\s*%|백\s*프로|누구나|모든\s*(분|사람)|효과\s*(보장|확실)|확실한\s*효과|무조건\s*효과"),
    "repeat": re.compile(r"([가-힣A-Za-z]{2,})(\s*\1){3,}"),
    "spec": re.compile(r"(성분|함유|효능|특허|추출물|mAh|소재|원단|용량|기능|인증)"),
    "disclosed": re.compile(
        r"체험단|협찬|제공\s*받|원고료|지원\s*받|서포터즈|무상\s*(으로)?\s*제공|광고\s*(입니다|포함)"
        r"|업체(로부터|에서)\s*(제공|지원)|브랜드(로부터|에서)\s*(제공|지원)|소정의|리뷰\s*이벤트"
    ),
}
POOL_FIELDS = [
    "platform",
    "stratum",
    "product_id",
    "review_id",
    "content",
    "rating",
    "author_hash",
    "author_n",
    "author_prods",
    "written_date",
    "option",
    "imgs",
    "helpful_count",
    "mlen",
    "day_c",
    "day_mean",
    "prod_total",
    "dup_ids",
    "dup_prods",
    *[f"s_{s}" for s in STRATA],
]


def _int(value: str | None) -> int:
    return int(value or 0)


def flags(row: dict[str, str]) -> dict[str, bool]:
    """한 리뷰에 대해 층별 신호 해당 여부를 돌려준다."""
    content = row["content"]
    mlen = _int(row["mlen"])
    rating = float(row["rating"]) if row["rating"] else None
    repeat = PATTERNS["repeat"].search(content)
    day_count, day_mean = _int(row["day_c"]), float(row["day_mean"] or 0)
    return {
        "combo": rating == 5 and _int(row["imgs"]) == 0 and mlen < 10,
        "promo": bool(PATTERNS["promo"].search(content)),
        "unused": bool(PATTERNS["unused_claim"].search(content)) and bool(PATTERNS["unused_effect"].search(content)),
        "certain": bool(PATTERNS["certain"].search(content)),
        # "하하하하" 같은 웃음은 반복 단위 글자가 한 종류라 뺀다.
        "repeat": bool(repeat) and len(set(repeat.group(1))) >= 2,
        "spec": mlen >= 80 and len(PATTERNS["spec"].findall(content)) >= 3,
        "disclosed": bool(PATTERNS["disclosed"].search(content)),
        "burst": day_count >= 10 and day_count >= 5 * day_mean and (rating is None or rating == 5),
        "network": _int(row["dup_ids"]) >= 2,
        "author": _int(row["author_n"]) >= 30 and _int(row["author_prods"]) >= 20,
    }


def author_hash(platform: str, author: str, salt: bytes) -> str:
    """닉네임을 복원할 수 없는 12자 해시로 바꾼다. 플랫폼이 이미 가린 닉네임은 접두어로 표시한다."""
    if not author:
        return ""
    digest = hmac.new(salt, f"{platform}|{author}".encode(), hashlib.sha256).hexdigest()[:12]
    return ("masked-" if "*" in author else "") + digest


def _order_key(review_id: str, tag: str) -> int:
    return -int(hashlib.md5(f"{review_id}{tag}".encode()).hexdigest(), 16)


def build_pool(rows, platform: str, salt: bytes, per_stratum: int = 200, random_size: int = 600):
    """층마다 결정적인 의사 난수 순서로 상위 N건을 남긴다. (counts, total, any_count, pool_rows)를 돌려준다."""
    heaps = {s: [] for s in (*STRATA, "random")}
    counts = dict.fromkeys(STRATA, 0)
    seen, total, any_count = set(), 0, 0

    def keep(stratum, row, tag, limit):
        item = (_order_key(row["review_id"], tag), total, dict(row, stratum=stratum))
        (heapq.heappush if len(heaps[stratum]) < limit else heapq.heappushpop)(heaps[stratum], item)

    for row in rows:
        if row["review_id"] in seen:
            continue
        seen.add(row["review_id"])
        total += 1
        hit = flags(row)
        row["author_hash"] = author_hash(platform, row.pop("author", "") or "", salt)
        for s in STRATA:
            row[f"s_{s}"] = "t" if hit[s] else "f"
        any_count += any(hit[s] for s in STRATA if s != "disclosed")
        for s in STRATA:
            if hit[s]:
                counts[s] += 1
                keep(s, row, "v4-pool", per_stratum)
        keep("random", row, "v4-rand", random_size)
    pool = [r for h in heaps.values() for _, _, r in sorted(h, reverse=True)]
    return counts, total, any_count, pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("platform")
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--per-stratum", type=int, default=200)
    ap.add_argument("--random", type=int, default=600)
    a = ap.parse_args(argv)
    csv.field_size_limit(10**8)
    salt = SALT_PATH.read_text().strip().encode()
    counts, total, any_count, pool = build_pool(csv.DictReader(sys.stdin), a.platform, salt, a.per_stratum, a.random)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / f"{a.platform}_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["count_total", *STRATA, "any_suspicious"])
        w.writerow([total, *(counts[s] for s in STRATA), any_count])
    with open(a.out_dir / f"{a.platform}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=POOL_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(pool)
    print(f"{a.platform} total {total} pool {len(pool)}")


if __name__ == "__main__":
    main()
