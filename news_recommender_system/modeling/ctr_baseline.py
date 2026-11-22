"""CTR baseline: rank candidates by their click-through rate in a rolling time window.

The CTR of an article is ``clicks / impressions`` within the window (24 hours by default).
The statistics are warmed up with the training impressions that fall into the window before
the first validation impression, then updated with each processed validation impression.
"""

from collections import deque
from pathlib import Path
from typing import Iterator

import pandas as pd

from news_recommender_system.modeling.ranking import (
    rank_positions,
    split_impression,
    write_predictions,
)

NewsStats = dict[str, dict[str, deque]]


def _empty_stats() -> dict[str, deque]:
    return {"clicks": deque(), "impressions": deque()}


def initialize_news_stats(
    behaviors_train_val: pd.DataFrame, behaviors_val: pd.DataFrame, window_seconds: float
) -> NewsStats:
    """Build CTR statistics from impressions in ``[val_start - window, val_start)``."""
    news_stats: NewsStats = {}
    first_impression_time = behaviors_val.iloc[0]["Timestamp"]

    in_window = behaviors_train_val[
        (behaviors_train_val["Timestamp"] >= first_impression_time - window_seconds)
        & (behaviors_train_val["Timestamp"] < first_impression_time)
    ]
    for timestamp, impressions in zip(in_window["Timestamp"], in_window["Impressions"]):
        if impressions == "-":
            continue
        news_ids, labels = split_impression(impressions)
        for news_id, label in zip(news_ids, labels):
            stats = news_stats.setdefault(news_id, _empty_stats())
            stats["impressions"].append(timestamp)
            if label == 1:
                stats["clicks"].append(timestamp)
    return news_stats


def update_news_stats(
    news_stats: NewsStats,
    current_time: float,
    previous: tuple[list[str], list[str], float] | None,
    window_seconds: float,
) -> None:
    """Drop expired events and add the clicks and impressions of the previous impression."""
    for news_id in list(news_stats.keys()):
        stats = news_stats[news_id]
        for key in ("clicks", "impressions"):
            while stats[key] and stats[key][0] < current_time - window_seconds:
                stats[key].popleft()
        if not stats["clicks"] and not stats["impressions"]:
            del news_stats[news_id]

    if previous:
        shown, clicked, timestamp = previous
        for news_id in shown:
            news_stats.setdefault(news_id, _empty_stats())["impressions"].append(timestamp)
        for news_id in clicked:
            news_stats.setdefault(news_id, _empty_stats())["clicks"].append(timestamp)


def ctr_rankings(
    behaviors_train_val: pd.DataFrame,
    behaviors_val: pd.DataFrame,
    window_seconds: float = 24 * 60 * 60,
) -> Iterator[tuple[object, list[int]]]:
    """Yield ``(impression_id, ranks)`` for the time-sorted validation impressions."""
    news_stats = initialize_news_stats(behaviors_train_val, behaviors_val, window_seconds)
    previous: tuple[list[str], list[str], float] | None = None

    for impression_id, current_time, impressions in zip(
        behaviors_val["ImpressionId"], behaviors_val["Timestamp"], behaviors_val["Impressions"]
    ):
        update_news_stats(news_stats, current_time, previous, window_seconds)

        news_ids, labels = split_impression(impressions)
        scored = []
        for news_id in news_ids:
            stats = news_stats.get(news_id)
            n_impressions = len(stats["impressions"]) if stats else 0
            ctr = len(stats["clicks"]) / n_impressions if n_impressions > 0 else 0.0
            scored.append((news_id, ctr))
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [news_id for news_id, _ in scored]
        yield impression_id, rank_positions(ranked, news_ids)

        clicked = [news_id for news_id, label in zip(news_ids, labels) if label == 1]
        previous = (news_ids, clicked, current_time)


def generate_ctr_predictions(
    behaviors_train_val: pd.DataFrame,
    behaviors_val: pd.DataFrame,
    output_path: Path,
    window_seconds: float = 24 * 60 * 60,
) -> None:
    """Write CTR-based predictions for the validation impressions."""
    write_predictions(
        ctr_rankings(behaviors_train_val, behaviors_val, window_seconds), output_path
    )
