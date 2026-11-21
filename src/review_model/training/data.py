"""split CSV(platform, product_id, review_id, content, y) 읽기."""

import pathlib
from dataclasses import dataclass

import numpy as np

from review_model.io import read_rows


@dataclass
class Split:
    content: list[str]
    platform: list[str]
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.content)


def load_split(path: str | pathlib.Path, limit: int = 0) -> Split:
    rows = read_rows(path)
    if limit:
        rows = rows[:limit]
    return Split(
        content=[r["content"] for r in rows],
        platform=[r["platform"] for r in rows],
        y=np.array([int(r["y"]) for r in rows]),
    )
