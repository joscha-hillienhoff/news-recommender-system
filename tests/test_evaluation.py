"""Metric functions on small hand-computed examples."""

import math

import numpy as np
import pytest

from news_recommender_system.evaluation import (
    dcg_score,
    mrr_score,
    ndcg_score,
    parse_line,
    scoring,
)

# Candidates with labels [0, 1, 0, 1] and scores [0.1, 0.9, 0.8, 0.2] are ranked
# [item1 (clicked), item2, item3 (clicked), item0], so the sorted labels are [1, 0, 1, 0].
Y_TRUE = np.array([0, 1, 0, 1])
Y_SCORE = np.array([0.1, 0.9, 0.8, 0.2])


def test_dcg_at_k():
    # 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.5
    assert dcg_score(Y_TRUE, Y_SCORE, k=10) == pytest.approx(1.5)
    assert dcg_score(Y_TRUE, Y_SCORE, k=1) == pytest.approx(1.0)


def test_ndcg_at_k():
    ideal = 1 + 1 / math.log2(3)  # both clicks ranked first
    assert ndcg_score(Y_TRUE, Y_SCORE, k=10) == pytest.approx(1.5 / ideal)
    assert ndcg_score(Y_TRUE, Y_SCORE, k=2) == pytest.approx(1.0 / ideal)
    assert ndcg_score(Y_TRUE, Y_TRUE.astype(float), k=5) == pytest.approx(1.0)


def test_ndcg_without_clicks_is_zero():
    assert ndcg_score(np.array([0, 0, 0]), np.array([0.3, 0.2, 0.1])) == 0.0


def test_mrr_is_mean_reciprocal_rank_of_clicks():
    # Clicks sit at ranks 1 and 3: (1/1 + 1/3) / 2
    assert mrr_score(Y_TRUE, Y_SCORE) == pytest.approx(2 / 3)
    assert mrr_score(np.array([0, 0, 1]), np.array([0.9, 0.5, 0.1])) == pytest.approx(1 / 3)
    assert mrr_score(np.array([0, 0]), np.array([0.9, 0.5])) == 0.0


def test_parse_line():
    assert parse_line("12 [0, 1, 0]\n") == ("12", [0, 1, 0])
    with pytest.raises(ValueError):
        parse_line("12")
    with pytest.raises(ValueError):
        parse_line("12 not-json")


def test_scoring_averages_auc_mrr_and_ndcg():
    truth = ["1 [0, 1, 0, 1]\n", "2 [1, 0, 0]\n"]
    # Ranks reproduce the scores above for impression 1 and a perfect ranking for impression 2.
    submission = ["1 [4, 1, 2, 3]\n", "2 [1, 2, 3]\n"]
    auc, mrr, ndcg5, ndcg10 = scoring(iter(truth), iter(submission))

    ideal = 1 + 1 / math.log2(3)
    assert auc == pytest.approx((0.75 + 1.0) / 2)  # 3 of 4 click/non-click pairs ordered
    assert mrr == pytest.approx((2 / 3 + 1.0) / 2)
    assert ndcg5 == pytest.approx((1.5 / ideal + 1.0) / 2)
    assert ndcg10 == pytest.approx(ndcg5)


def test_scoring_skips_masked_impressions():
    auc, mrr, _, _ = scoring(iter(["1 []\n", "2 [1, 0]\n"]), iter(["1 []\n", "2 [1, 2]\n"]))
    assert (auc, mrr) == (1.0, 1.0)


def test_scoring_rejects_mismatched_impression_ids():
    with pytest.raises(ValueError):
        scoring(iter(["1 [1, 0]\n"]), iter(["2 [1, 2]\n"]))
