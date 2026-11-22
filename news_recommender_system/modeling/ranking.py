"""Helpers shared by all models for producing prediction files."""

import json
from pathlib import Path
from typing import Iterable, Sequence


def rank_positions(ranked_news: Sequence[str], impression_news: Sequence[str]) -> list[int]:
    """Return the 1-based rank of each candidate, in the original impression order."""
    return [ranked_news.index(news_id) + 1 for news_id in impression_news]


def split_impression(impressions: str) -> tuple[list[str], list[int]]:
    """Split an impressions string ``"N1-0 N2-1"`` into article ids and click labels."""
    entries = [entry.rsplit("-", 1) for entry in impressions.split()]
    return [news_id for news_id, _ in entries], [int(label) for _, label in entries]


def write_predictions(rows: Iterable[tuple[object, list[int]]], output_path: Path) -> None:
    """Write ``(impression_id, ranks)`` pairs as ``<impression_id> <json_list>`` lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for impression_id, ranks in rows:
            f.write(f"{impression_id} {json.dumps(ranks)}\n")
    print(f"Prediction file written to {output_path}")
