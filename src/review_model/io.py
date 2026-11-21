"""CSV 입출력. 리뷰 본문이 길어 기본 필드 크기 제한을 넘는 경우가 있다."""

import csv
import pathlib
from collections.abc import Iterator

csv.field_size_limit(10**8)


def read_rows(path: str | pathlib.Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def iter_pool(pool_dir: str | pathlib.Path) -> Iterator[dict[str, str]]:
    """후보 풀 디렉터리의 플랫폼별 CSV를 차례로 읽는다(집계 파일 제외)."""
    for path in sorted(pathlib.Path(pool_dir).glob("*.csv")):
        if path.name.endswith("_counts.csv"):
            continue
        with open(path, newline="") as fh:
            yield from csv.DictReader(fh)
