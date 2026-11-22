"""Popularity baseline: rank candidates by their number of recent clicks.

Impressions are processed in time order. The click counts of an article are the clicks
seen within a rolling time window (24 hours by default). The clicks of an impression are
only added once the next impression is processed, so no label is used for its own ranking.
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

DEFAULT_WINDOW_SECONDS = 24 * 60 * 60


def _drop_expired_clicks(
    news_clicks: dict[str, deque], current_time: float, window_seconds: float
) -> None:
    """Remove clicks that fall outside the rolling window and delete empty entries."""
    for news_id in list(news_clicks.keys()):
        while news_clicks[news_id] and news_clicks[news_id][0] < current_time - window_seconds:
            news_clicks[news_id].popleft()
        if not news_clicks[news_id]:
            del news_clicks[news_id]


def popularity_rankings(
    behaviors: pd.DataFrame, window_seconds: float = DEFAULT_WINDOW_SECONDS
) -> Iterator[tuple[object, list[int]]]:
    """Yield ``(impression_id, ranks)`` for time-sorted ``behaviors``.

    ``behaviors`` needs the columns ``ImpressionId``, ``Timestamp`` and ``Impressions``.
    """
    news_clicks: dict[str, deque] = {}
    previous_clicks: tuple[list[str], float] | None = None

    for impression_id, current_time, impressions in zip(
        behaviors["ImpressionId"], behaviors["Timestamp"], behaviors["Impressions"]
    ):
        _drop_expired_clicks(news_clicks, current_time, window_seconds)
        if previous_clicks is not None:
            clicked_ids, click_time = previous_clicks
            for news_id in clicked_ids:
                news_clicks.setdefault(news_id, deque()).append(click_time)

        news_ids, labels = split_impression(impressions)
        scored = [(news_id, len(news_clicks.get(news_id, ()))) for news_id in news_ids]
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [news_id for news_id, _ in scored]
        yield impression_id, rank_positions(ranked, news_ids)

        clicked = [news_id for news_id, label in zip(news_ids, labels) if label == 1]
        previous_clicks = (clicked, current_time)


def generate_popularity_predictions(
    behaviors_val: pd.DataFrame,
    output_path: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> None:
    """Write popularity-based predictions for the validation impressions."""
    write_predictions(popularity_rankings(behaviors_val, window_seconds), output_path)
