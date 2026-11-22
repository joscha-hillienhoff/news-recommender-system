"""Collaborative filtering with Bayesian Personalized Ranking (BPR) matrix factorization.

Implicit feedback is built from the clicks of the training impressions. The model is fit
on the clicks before the 90th percentile of the impression time. Candidates of a
validation impression are scored with the dot product of user and item factors;
unknown users or articles get the score 0.
"""

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from news_recommender_system.modeling.ranking import (
    rank_positions,
    split_impression,
    write_predictions,
)

SEED = 42
NUM_FACTORS = 200
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
LAMBDA_REG = 0.001
TRAIN_TIME_QUANTILE = 0.9


def build_interactions(behaviors_train: pd.DataFrame) -> pd.DataFrame:
    """Turn training impressions into ``userID, itemID, Time, rating`` click interactions."""
    behaviors = behaviors_train.copy()
    behaviors["Clicked"] = behaviors["Impressions"].apply(
        lambda impressions: [
            news_id for news_id, label in zip(*split_impression(impressions)) if label == 1
        ]
    )
    behaviors["Time"] = pd.to_datetime(behaviors["Time"])
    interactions = behaviors.explode("Clicked")[["User", "Clicked", "Time"]].copy()
    interactions["rating"] = 1
    return interactions.rename(columns={"User": "userID", "Clicked": "itemID"})


def train_bpr(interactions: pd.DataFrame):
    """Fit BPR on the interactions before the 90th time percentile; return the model."""
    import cornac

    threshold = interactions["Time"].quantile(TRAIN_TIME_QUANTILE)
    train = interactions[interactions["Time"] < threshold]
    train_set = cornac.data.Dataset.from_uir(
        train[["userID", "itemID", "rating"]].itertuples(index=False), seed=SEED
    )
    model = cornac.models.BPR(
        k=NUM_FACTORS,
        max_iter=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lambda_reg=LAMBDA_REG,
        verbose=True,
        seed=SEED,
    )
    model.fit(train_set)
    return model


def bpr_rankings(model, behaviors_val: pd.DataFrame) -> Iterator[tuple[object, list[int]]]:
    """Yield ``(impression_id, ranks)`` for every validation impression."""
    user_map = model.train_set.uid_map
    item_map = model.train_set.iid_map

    for impression_id, user_id, impressions in zip(
        behaviors_val["ImpressionId"], behaviors_val["User"], behaviors_val["Impressions"]
    ):
        news_ids, _ = split_impression(impressions)
        scores = [
            model.score(user_map[user_id], item_map[news_id])
            if user_id in user_map and news_id in item_map
            else 0
            for news_id in news_ids
        ]
        ranked = [news_ids[i] for i in np.argsort(scores)[::-1]]
        yield impression_id, rank_positions(ranked, news_ids)


def generate_bpr_predictions(
    behaviors_train: pd.DataFrame, behaviors_val: pd.DataFrame, output_path: Path
) -> None:
    """Train BPR on the training impressions and write validation predictions."""
    model = train_bpr(build_interactions(behaviors_train))
    write_predictions(bpr_rankings(model, behaviors_val), output_path)
