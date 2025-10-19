import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# Evaluation metrics functions
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best != 0 else 0.0


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def parse_line(l):
    parts = l.strip("\n").split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid line format: {l} (expected two parts)")
    impid, ranks = parts
    ranks = json.loads(ranks)
    return impid, ranks


# Function to compute metrics based on classifier predictions
def scoring_for_classifier(truth_lines, predictions, sub_files_dict):
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []

    for line_index, truth_line in enumerate(truth_lines):
        try:
            impid, labels = parse_line(truth_line)
        except Exception as e:
            print(f"Error parsing truth file line {line_index+1}: {e}")
            continue

        if not labels:
            continue

        pred_model = predictions[line_index]
        sub_lines = sub_files_dict.get(pred_model)
        if sub_lines is None:
            raise ValueError(f"Invalid predicted model {pred_model} at line {line_index+1}")

        submission_line = sub_lines[line_index].strip()
        if submission_line == "":
            sub_impid = impid
            sub_ranks = [1] * len(labels)
        else:
            try:
                sub_impid, sub_ranks = parse_line(submission_line)
            except Exception as e:
                raise ValueError(f"Line-{line_index+1}: Invalid submission line. Error: {e}")

        if sub_impid != impid:
            raise ValueError(
                f"Line-{line_index+1}: Inconsistent Impression Id {sub_impid} vs {impid}"
            )

        y_true = np.array(labels, dtype="float32")
        y_score = []
        for rank in sub_ranks:
            score_rslt = 1.0 / rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError(f"Line-{line_index+1}: score_rslt {score_rslt} out of bounds")
            y_score.append(score_rslt)

        aucs.append(roc_auc_score(y_true, y_score))
        mrrs.append(mrr_score(y_true, y_score))
        ndcg5s.append(ndcg_score(y_true, y_score, k=5))
        ndcg10s.append(ndcg_score(y_true, y_score, k=10))

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)


if __name__ == "__main__":
    base_dir = "outputs_and_truth"
    truth_file_path = os.path.join(base_dir, "truth_val_sorted.txt")

    # Updated submission files with correct filenames
    submission_files = {
        1: os.path.join(base_dir, "baseline.txt"),
        2: os.path.join(base_dir, "cbf.txt"),
        3: os.path.join(base_dir, "bpr.txt"),
    }

    if not os.path.exists(truth_file_path):
        raise FileNotFoundError(f"Truth file not found at: {truth_file_path}")
    for key, file_path in submission_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Submission file for model {key} not found at: {file_path}")

    with open(truth_file_path, "r") as f:
        truth_lines = f.readlines()

    sub_files_dict = {}
    for model_id, file_path in submission_files.items():
        with open(file_path, "r") as f:
            sub_files_dict[model_id] = f.readlines()

    predictions_df = pd.read_csv(
        r"C:\Users\hryad\Desktop\iitm\Even Sem'25\Recommender Systems\Project\TDT4215-Group-Project\hs_work\all_classifier_predictions.csv"
    )
    classifier_columns = [col for col in predictions_df.columns if col != "True_Label"]

    results = {}
    for clf in classifier_columns:
        predictions = predictions_df[clf].tolist()
        if len(predictions) != len(truth_lines):
            raise ValueError(
                f"Number of predictions in {clf} ({len(predictions)}) does not match number of truth lines ({len(truth_lines)})"
            )

        auc, mrr, ndcg5, ndcg10 = scoring_for_classifier(truth_lines, predictions, sub_files_dict)
        results[clf] = {"AUC": auc, "MRR": mrr, "nDCG@5": ndcg5, "nDCG@10": ndcg10}

    for clf, metrics in results.items():
        print(f"Results for {clf}:")
        print("AUC:    {:.4f}".format(metrics["AUC"]))
        print("MRR:    {:.4f}".format(metrics["MRR"]))
        print("nDCG@5: {:.4f}".format(metrics["nDCG@5"]))
        print("nDCG@10:{:.4f}".format(metrics["nDCG@10"]))
        print("-" * 50)
