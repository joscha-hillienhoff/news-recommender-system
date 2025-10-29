"""
CTR-based (Click-Through Rate) Baseline Model.

This module implements a simple, time-aware baseline for news recommendation.
Articles are ranked by their CTR (clicks / impressions) within a rolling
time window (e.g., 24h, 48h, or 7 days).

Usage
-----
from news_recommender_system.models.baseline_ctr import generate_ctr_baseline_predictions

generate_ctr_baseline_predictions(
    behaviors_train_val=behaviors_train_val,
    behaviors_val=behaviors_val,
    time_window_seconds=7 * 24 * 60 * 60,
    output_file="prediction_val_baseline_ctr_1w.txt"
)
"""

from collections import defaultdict, deque
import json
from pathlib import Path
from typing import DefaultDict, Deque, Dict

from loguru import logger
import pandas as pd
import typer

from news_recommender_system.config import INTERIM_DATA_DIR, PROJ_ROOT

# Global state for rolling CTR statistics
news_stats: DefaultDict[str, Dict[str, Deque[int]]] = defaultdict(
    lambda: {"clicks": deque(), "impressions": deque()}
)


def initialize_news_stats(behaviors_train_val, behaviors_val, time_window_seconds: int):
    """Initialize the CTR statistics using impressions from the time window before validation starts."""
    first_impression_time = behaviors_val.iloc[0]["Timestamp"]

    # Filter training interactions within [val_start - window, val_start)
    warmup_df = behaviors_train_val[
        (behaviors_train_val["Timestamp"] >= first_impression_time - time_window_seconds)
        & (behaviors_train_val["Timestamp"] < first_impression_time)
    ]

    for _, row in warmup_df.iterrows():
        if row["Impressions"] == "-":
            continue
        for news in row["Impressions"].split():
            news_id, label = news.split("-")
            news_stats[news_id]["impressions"].append(row["Timestamp"])
            if label == "1":
                news_stats[news_id]["clicks"].append(row["Timestamp"])


def update_news_stats(
    current_time, past_clicked_articles, past_impressed_articles, time_window_seconds
):
    """Maintain the rolling time window.

    - Remove outdated clicks/impressions
    - Add new ones from the previous impression
    """
    for news_id in list(news_stats.keys()):
        # Remove old clicks
        while (
            news_stats[news_id]["clicks"]
            and news_stats[news_id]["clicks"][0] < current_time - time_window_seconds
        ):
            news_stats[news_id]["clicks"].popleft()

        # Remove old impressions
        while (
            news_stats[news_id]["impressions"]
            and news_stats[news_id]["impressions"][0] < current_time - time_window_seconds
        ):
            news_stats[news_id]["impressions"].popleft()

        # Remove entry if empty
        if not news_stats[news_id]["clicks"] and not news_stats[news_id]["impressions"]:
            del news_stats[news_id]

    # Add new interactions from last impression
    if past_impressed_articles:
        impression_list, timestamp = past_impressed_articles

        # Add impressions
        for news_id in impression_list:
            news_stats[news_id]["impressions"].append(timestamp)

        # Add clicks
        for news_id in past_clicked_articles:
            news_stats[news_id]["clicks"].append(timestamp)


def rank_news(
    user_impressions,
    current_time,
    past_clicked_articles,
    past_impressed_articles,
    time_window_seconds,
):
    """Rank news articles by their CTR (clicks / impressions) within the current time window."""
    update_news_stats(
        current_time, past_clicked_articles, past_impressed_articles, time_window_seconds
    )

    ranked = []
    for news_id in user_impressions:
        stats = news_stats.get(news_id, {"clicks": deque(), "impressions": deque()})
        clicks = len(stats["clicks"])
        impressions = len(stats["impressions"])
        ctr = clicks / impressions if impressions > 0 else 0.0
        ranked.append((news_id, ctr))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked]


def rank_submission_format(
    user_impressions,
    current_time,
    past_clicked_articles,
    past_impressed_articles,
    time_window_seconds,
):
    """Return rank positions in the same order as the original impression list."""
    ranked_news = rank_news(
        user_impressions,
        current_time,
        past_clicked_articles,
        past_impressed_articles,
        time_window_seconds,
    )
    return [ranked_news.index(nid) + 1 for nid in user_impressions]


def generate_ctr_baseline_predictions(
    behaviors_train_val,
    behaviors_val,
    time_window_seconds: int = 1 * 24 * 60 * 60,
    output_file: str = "prediction_val_baseline_ctr.txt",
):
    """Generate CTR-based ranking predictions for validation impressions.

    Parameters
    ----------
    behaviors_train_val : pd.DataFrame
        Combined training and validation impressions with 'Timestamp' and 'Impressions'.
    behaviors_val : pd.DataFrame
        Validation impressions to predict for.
    time_window_seconds : int
        Rolling window length (e.g., 24h = 86400 seconds).
    output_file : str
        Output file path for writing predictions.
    """
    initialize_news_stats(behaviors_train_val, behaviors_val, time_window_seconds)

    past_clicked_articles: list[str] = []
    past_impressed_articles: tuple[list[str], int] | None = None

    output_path = PROJ_ROOT / "models" / "predictions" / output_file

    with open(output_path, "w") as f:
        for _, row in behaviors_val.iterrows():
            impression_id = row["ImpressionId"]
            current_time = row["Timestamp"]

            impression_entries = row["Impressions"].split()
            user_impressions = [news.split("-")[0] for news in impression_entries]

            ranked_positions = rank_submission_format(
                user_impressions,
                current_time,
                past_clicked_articles,
                past_impressed_articles,
                time_window_seconds,
            )

            f.write(f"{impression_id} {json.dumps(ranked_positions)}\n")

            past_clicked_articles = [
                news.split("-")[0] for news in impression_entries if news.split("-")[1] == "1"
            ]
            past_impressed_articles = (user_impressions, current_time)

    print(f"✅ CTR-based prediction file '{output_file}' successfully created.")


app = typer.Typer()


@app.command()
def main(
    train_val_path: Path = INTERIM_DATA_DIR / "behaviors_train_val.parquet",
    val_path: Path = INTERIM_DATA_DIR / "behaviors_val.parquet",
    output_file: str = "prediction_val_baseline_ctr.txt",
    time_window_hours: int = 24,
):
    """CLI entry point for the CTR baseline model.

    Loads training + validation behaviors, computes rolling CTRs over the
    given time window, and writes the validation predictions.
    """
    logger.info("Loading data...")
    behaviors_train_val = pd.read_parquet(train_val_path)
    behaviors_val = pd.read_parquet(val_path)

    time_window_seconds = time_window_hours * 3600
    logger.info(
        f"Running CTR baseline with window = {time_window_hours}h ({time_window_seconds:,}s)."
    )

    generate_ctr_baseline_predictions(
        behaviors_train_val,
        behaviors_val,
        time_window_seconds,
        output_file,
    )

    logger.success("CTR baseline modeling complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
