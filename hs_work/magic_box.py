import os
import json
import numpy as np
import csv
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def parse_line(line):
    """Parses a line in the format: '<ImpressionID> <json_list>'"""
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid line format: {line} (expected two parts)")
    impid, json_str = parts
    return impid, json.loads(json_str)

def compute_auc(y_true, ranks):
    """
    Computes AUC for one impression.

    Parameters:
      - y_true: list of 0/1 values indicating the clicked articles.
      - ranks: list of integers representing model ranking positions.
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_score = np.array([1.0 / rank for rank in ranks])

    if len(np.unique(y_true)) < 2:
        return 0.5  # AUC is undefined if there's only one class; return 0.5 as chance level
    return roc_auc_score(y_true, y_score)

def main():
    folder = "outputs_and_truth"
    truth_path = os.path.join(folder, "truth_val_sorted.txt")
    baseline_path = os.path.join(folder, "baseline.txt")
    bpr_path = os.path.join(folder, "bpr.txt")
    cbf_path = os.path.join(folder, "cbf.txt")

    # Read all files into lists of lines
    with open(truth_path, "r") as f_truth, \
            open(baseline_path, "r") as f_baseline, \
            open(bpr_path, "r") as f_bpr, \
            open(cbf_path, "r") as f_cbf:
        truth_lines = f_truth.readlines()
        baseline_lines = f_baseline.readlines()
        bpr_lines = f_bpr.readlines()
        cbf_lines = f_cbf.readlines()

    results = []
    all_true = []
    all_pred = []

    print("Processing impressions:")
    for idx, truth_line in tqdm(enumerate(truth_lines), total=len(truth_lines)):
        impid, truth_lst = parse_line(truth_line)

        baseline_impid, baseline_lst = parse_line(baseline_lines[idx]) if idx < len(baseline_lines) else (impid, [1] * len(truth_lst))
        bpr_impid, bpr_lst = parse_line(bpr_lines[idx]) if idx < len(bpr_lines) else (impid, [1] * len(truth_lst))
        cbf_impid, cbf_lst = parse_line(cbf_lines[idx]) if idx < len(cbf_lines) else (impid, [1] * len(truth_lst))

        if baseline_impid != impid or bpr_impid != impid or cbf_impid != impid:
            raise ValueError(f"Impression ID mismatch in line {idx+1}")

        auc_baseline = compute_auc(truth_lst, baseline_lst)
        auc_bpr = compute_auc(truth_lst, bpr_lst)
        auc_cbf = compute_auc(truth_lst, cbf_lst)

        model_scores = [(auc_baseline, 1, baseline_lst),
                        (auc_cbf, 2, cbf_lst),
                        (auc_bpr, 3, bpr_lst)]

        # Pick best model: prefer higher AUC, then higher model number (3 > 2 > 1)
        best_auc, best_model, best_ranks = max(model_scores, key=lambda x: (x[0], x[1]))

        results.append((impid, best_model))

        # Collect for final AUC
        y_true = np.array(truth_lst, dtype=np.float32)
        y_score = np.array([1.0 / rank for rank in best_ranks])
        all_true.extend(y_true)
        all_pred.extend(y_score)

    # Write results to CSV
    output_csv = "model_best.csv"
    print(f"Writing results to '{output_csv}'...")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Impression ID", "Best_Model"])
        for impid, best_model in tqdm(results, desc="Writing CSV"):
            writer.writerow([impid, best_model])

    # Compute final AUC from stitched predictions
    final_auc = roc_auc_score(all_true, all_pred)
    print(f"CSV file '{output_csv}' has been generated.")
    print(f"Final AUC using best model per impression: {final_auc:.4f}")

if __name__ == "__main__":
    main()