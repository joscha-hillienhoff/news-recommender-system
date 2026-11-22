"""Smoke tests of the baselines on a few synthetic impressions (no dataset needed)."""

import pandas as pd
import pytest

from news_recommender_system.evaluation import evaluate_model
from news_recommender_system.modeling.ctr_baseline import ctr_rankings
from news_recommender_system.modeling.popularity_baseline import (
    generate_popularity_predictions,
    popularity_rankings,
)
from news_recommender_system.truth import generate_truth_file


@pytest.fixture
def behaviors_val():
    return pd.DataFrame(
        {
            "ImpressionId": [1, 2, 3],
            "Timestamp": [0.0, 10.0, 20.0],
            "Impressions": ["N1-0 N2-1 N3-0", "N1-0 N2-0 N3-1", "N3-0 N1-1 N2-0"],
        }
    )


def test_popularity_ranks_by_clicks_of_previous_impressions(behaviors_val):
    rankings = dict(popularity_rankings(behaviors_val))
    assert rankings == {
        1: [1, 2, 3],  # no clicks seen yet: original order
        2: [2, 1, 3],  # N2 was clicked in impression 1
        3: [1, 3, 2],  # N3 (impression 2) and N2 (impression 1) have one click each
    }


def test_popularity_forgets_clicks_outside_the_window():
    behaviors = pd.DataFrame(
        {
            "ImpressionId": [1, 2, 3],
            "Timestamp": [0.0, 10.0, 1000.0],
            "Impressions": ["N1-0 N2-1", "N1-0 N2-0", "N1-0 N2-0"],
        }
    )
    rankings = dict(popularity_rankings(behaviors, window_seconds=100))
    assert rankings[2] == [2, 1]  # the click at t=0 is within the window at t=10
    assert rankings[3] == [1, 2]  # and expired at t=1000


def test_ctr_ranks_by_click_through_rate():
    train = pd.DataFrame(
        {"ImpressionId": [0], "Timestamp": [-100.0], "Impressions": ["N1-1 N2-0 N3-0"]}
    )
    val = pd.DataFrame(
        {
            "ImpressionId": [1, 2],
            "Timestamp": [0.0, 10.0],
            "Impressions": ["N3-0 N2-1 N1-0", "N3-0 N2-0 N1-0"],
        }
    )
    train_val = pd.concat([train, val], ignore_index=True)

    rankings = dict(ctr_rankings(train_val, val, window_seconds=86400))
    assert rankings == {
        1: [2, 3, 1],  # N1 has CTR 1.0 from the warm-up window; N3 and N2 tie at 0
        2: [3, 1, 2],  # N2 and N1 tie at CTR 0.5; N3 has 0
    }


def test_baseline_predictions_can_be_evaluated(behaviors_val, tmp_path):
    truth_path = tmp_path / "truth.txt"
    prediction_path = tmp_path / "popularity.txt"
    metrics_path = tmp_path / "metrics" / "scores.txt"

    generate_truth_file(behaviors_val, truth_path)
    generate_popularity_predictions(behaviors_val, prediction_path)
    evaluate_model(truth_path, prediction_path, metrics_path)

    assert truth_path.read_text().splitlines()[0] == "1 [0, 1, 0]"
    assert prediction_path.read_text().splitlines()[1] == "2 [2, 1, 3]"
    lines = metrics_path.read_text().splitlines()
    assert [line.split(":")[0] for line in lines] == ["AUC", "MRR", "nDCG@5", "nDCG@10"]
