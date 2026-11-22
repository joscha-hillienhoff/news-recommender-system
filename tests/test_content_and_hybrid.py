"""Tests of the ranking helpers of the content-based and hybrid models."""

import numpy as np
import pandas as pd

from news_recommender_system.modeling.content_based import content_rankings, rank_for_user
from news_recommender_system.modeling.hybrid import best_model_labels

NEWS_IDS = ["N1", "N2", "N3"]
SIMILARITY = np.array(
    [
        [1.0, 0.2, 0.9],
        [0.2, 1.0, 0.1],
        [0.9, 0.1, 1.0],
    ]
)
INDEX = {news_id: i for i, news_id in enumerate(NEWS_IDS)}


def test_rank_for_user_prefers_articles_similar_to_the_history():
    assert rank_for_user("N1", ["N2", "N3"], INDEX, SIMILARITY) == ["N3", "N2"]


def test_rank_for_user_keeps_order_without_usable_history():
    assert rank_for_user(np.nan, ["N2", "N3"], INDEX, SIMILARITY) == ["N2", "N3"]
    assert rank_for_user("N9", ["N2", "N3"], INDEX, SIMILARITY) == ["N2", "N3"]


def test_content_rankings_output_is_in_impression_order():
    behaviors = pd.DataFrame(
        {
            "ImpressionId": [7],
            "User": ["U1"],
            "History": ["N1"],
            "Impressions": ["N2-0 N3-1"],
        }
    )
    assert list(content_rankings(behaviors, NEWS_IDS, SIMILARITY)) == [(7, [2, 1])]


def test_best_model_labels_pick_highest_auc_and_break_ties_toward_higher_id(tmp_path):
    truth = tmp_path / "truth.txt"
    truth.write_text("1 [1, 0]\n2 [0, 1]\n3 [0, 0]\n")
    popularity = tmp_path / "popularity.txt"
    popularity.write_text("1 [1, 2]\n2 [1, 2]\n3 [1, 2]\n")  # wrong on impression 2
    content = tmp_path / "content.txt"
    content.write_text("1 [2, 1]\n2 [2, 1]\n3 [1, 2]\n")  # wrong on impression 1
    bpr = tmp_path / "bpr.txt"
    bpr.write_text("1 [2, 1]\n2 [1, 2]\n3 [1, 2]\n")  # wrong on both

    labels = best_model_labels(truth, {1: popularity, 2: content, 3: bpr})
    # Impression 3 has no non-click: all models get AUC 0.5 and the highest id wins.
    assert labels["best_model"].tolist() == [1, 2, 3]
