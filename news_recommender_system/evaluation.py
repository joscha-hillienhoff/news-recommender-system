"""
Evaluation utilities for recommender submissions.

Reads a ground-truth file and a submission file, computes AUC, MRR, nDCG@5 and
nDCG@10, and writes the results to an output file.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """
    Compute DCG@k.

    Parameters
    ----------
    y_true : np.ndarray
        Binary relevance labels.
    y_score : np.ndarray
        Scores used to rank items (higher is better).
    k : int
        Cutoff.

    Returns
    -------
    float
        Discounted cumulative gain at k.
    """
    order = np.argsort(y_score)[::-1]
    y_true_k = np.take(y_true, order[:k])
    gains = 2**y_true_k - 1
    discounts = np.log2(np.arange(len(y_true_k)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """
    Compute nDCG@k.

    Returns 0.0 if the ideal DCG is 0 to avoid division by zero.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return 0.0 if best == 0 else float(actual / best)


def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Returns 0.0 if there are no positive labels.
    """
    order = np.argsort(y_score)[::-1]
    y_sorted = np.take(y_true, order)
    denom = float(np.sum(y_sorted))
    if denom == 0:
        return 0.0
    rr = y_sorted / (np.arange(len(y_sorted)) + 1)
    return float(np.sum(rr) / denom)


def parse_line(line_text: str) -> Tuple[str, list[int]]:
    """
    Parse a line of the form '<impression_id> <json_list>'.

    Parameters
    ----------
    line_text : str
        Input line.

    Returns
    -------
    Tuple[str, list[int]]
        Impression id and list of integer ranks.

    Raises
    ------
    ValueError
        If the line is malformed or JSON cannot be parsed.
    """
    parts = line_text.strip("\n").split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid line format: {line_text!r} (expected two parts)")

    impid, ranks_json = parts
    try:
        ranks = json.loads(ranks_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON ranks in line: {line_text!r}") from exc
    return impid, ranks


def scoring(truth_f: Iterator[str], sub_f: Iterator[str]) -> tuple[float, float, float, float]:
    """Compute mean AUC, MRR, nDCG@5 and nDCG@10 over all impressions."""
    aucs: list[float] = []
    mrrs: list[float] = []
    ndcg5s: list[float] = []
    ndcg10s: list[float] = []

    for line_index, lt in enumerate(truth_f, start=1):
        ls = next(sub_f, "")
        impid, labels = parse_line(lt)

        # ignore masked impressions
        if not labels:
            continue

        if ls == "":
            # empty line: fill with worst ranks
            sub_impid, sub_ranks = impid, [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(ls)
            except ValueError as exc:
                raise ValueError(f"line-{line_index}: invalid input format") from exc

        if sub_impid != impid:
            raise ValueError(
                f"line-{line_index}: inconsistent impression id {sub_impid} vs {impid}"
            )

        y_true = np.array(labels, dtype="float32")
        y_score = np.array([1.0 / rank for rank in sub_ranks], dtype="float32")

        aucs.append(float(roc_auc_score(y_true, y_score)))
        mrrs.append(mrr_score(y_true, y_score))
        ndcg5s.append(ndcg_score(y_true, y_score, 5))
        ndcg10s.append(ndcg_score(y_true, y_score, 10))

    return (
        float(np.mean(aucs)) if aucs else 0.0,
        float(np.mean(mrrs)) if mrrs else 0.0,
        float(np.mean(ndcg5s)) if ndcg5s else 0.0,
        float(np.mean(ndcg10s)) if ndcg10s else 0.0,
    )


def evaluate_model(truth_path: str, submission_path: str, output_path: str) -> None:
    """Evaluate a submission against truth and write metrics to `output_path`."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(truth_path, "r") as truth_file, open(submission_path, "r") as sub_file:
        auc, mrr, ndcg5, ndcg10 = scoring(truth_file, sub_file)

    with open(output_path, "w") as out:
        out.write(f"AUC:{auc:.4f}\nMRR:{mrr:.4f}\nnDCG@5:{ndcg5:.4f}\n" f"nDCG@10:{ndcg10:.4f}")

    print(f"✅ Saved metrics to {output_path}")
    print(f"AUC={auc:.4f}, MRR={mrr:.4f}, nDCG@5={ndcg5:.4f}, nDCG@10={ndcg10:.4f}")


if __name__ == "__main__":
    # Minimal CLI-like behavior for local runs; reuse evaluate_model so logic is single-sourced.
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    SUBMIT_DIR = os.path.join(INPUT_DIR, "res")
    TRUTH_DIR = os.path.join(INPUT_DIR, "ref")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    evaluate_model(
        truth_path=os.path.join(TRUTH_DIR, "truth_val_sorted.txt"),
        submission_path=os.path.join(SUBMIT_DIR, "bpr.txt"),
        output_path=os.path.join(OUTPUT_DIR, "scores_val_bpr.txt"),
    )
