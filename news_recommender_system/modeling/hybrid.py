"""Hybrid recommender: a classifier selects which base model ranks each impression.

Pipeline:

1. ``best_model_labels`` finds, per validation impression, the base model (1 = popularity,
   2 = content-based, 3 = BPR) with the highest AUC. These labels are the classifier target.
2. ``selector_features`` describes every impression with a few cheap features (hour of day,
   number of candidates, history length, share of candidates seen in training).
3. ``train_selectors`` fits several classifiers on the features and predicts the best model.
4. ``evaluate_selectors`` ranks every impression with the model its selector chose and
   scores the result with AUC, MRR and nDCG.

The selectors are fit and scored on the same validation impressions, and the labels are
derived from the validation clicks. The resulting scores are therefore an optimistic upper
bound rather than a held-out estimate.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from news_recommender_system.evaluation import mrr_score, ndcg_score, parse_line

BASE_MODELS = {1: "popularity", 2: "content-based", 3: "bpr"}
FEATURES = ["time_imp", "imp_count", "hist_count", "perc_itm_in_train"]


def _read_lines(path: Path) -> list[str]:
    with open(path, "r") as f:
        return f.readlines()


def best_model_labels(truth_path: Path, prediction_paths: dict[int, Path]) -> pd.DataFrame:
    """Return ``ImpressionId, best_model``: the base model with the highest AUC per impression.

    Ties are resolved in favour of the higher model id. Impressions without both a click and
    a non-click get AUC 0.5.
    """
    truth_lines = _read_lines(truth_path)
    predictions = {model: _read_lines(path) for model, path in prediction_paths.items()}

    rows = []
    for idx, truth_line in enumerate(truth_lines):
        impression_id, labels = parse_line(truth_line)
        best = None
        for model, lines in predictions.items():
            pred_id, ranks = parse_line(lines[idx])
            if pred_id != impression_id:
                raise ValueError(f"Impression id mismatch in line {idx + 1}")
            if len(set(labels)) < 2:
                auc = 0.5
            else:
                auc = roc_auc_score(np.array(labels, dtype="float32"), [1.0 / r for r in ranks])
            if best is None or (auc, model) > best:
                best = (auc, model)
        rows.append((impression_id, best[1]))
    return pd.DataFrame(rows, columns=["ImpressionId", "best_model"])


def _all_news_ids(behaviors: pd.DataFrame) -> set[str]:
    """Collect every article id that appears in a history or an impression."""
    news_ids: set[str] = set()
    for history in behaviors["History"].dropna():
        news_ids.update(str(history).split())
    for impressions in behaviors["Impressions"].dropna():
        news_ids.update(item.rsplit("-", 1)[0] for item in str(impressions).split())
    return news_ids


def selector_features(
    behaviors_train: pd.DataFrame, behaviors_val: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Build the classifier dataset: one row per validation impression plus ``best_model``."""
    train_news = _all_news_ids(behaviors_train)
    times = pd.to_datetime(behaviors_val["Time"], errors="coerce")

    features = pd.DataFrame({"ImpressionId": behaviors_val["ImpressionId"].astype(str).values})
    features["time_imp"] = ((times.dt.hour + (times.dt.minute >= 30).astype(int)) % 24).values
    candidates = behaviors_val["Impressions"].apply(
        lambda x: [item.rsplit("-", 1)[0] for item in str(x).split()] if pd.notna(x) else []
    )
    features["imp_count"] = candidates.apply(len).values
    features["hist_count"] = (
        behaviors_val["History"].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0).values
    )
    features["perc_itm_in_train"] = candidates.apply(
        lambda items: 100 * sum(item in train_news for item in items) / len(items) if items else 0
    ).values
    return features.merge(labels, on="ImpressionId", how="inner")


def train_selectors(dataset: pd.DataFrame) -> pd.DataFrame:
    """Fit all selector classifiers and return their predicted best model per impression."""
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from tensorflow import keras

    X = StandardScaler().fit_transform(dataset[FEATURES])
    y = dataset["best_model"]
    predictions = pd.DataFrame({"True_Label": y.values})

    classifiers = {
        "DT_Prediction": DecisionTreeClassifier(random_state=42),
        "RF_Prediction": RandomForestClassifier(random_state=42),
        "LR_Prediction": LogisticRegression(solver="lbfgs", random_state=42, max_iter=1000),
        "KNN_Prediction": KNeighborsClassifier(n_neighbors=5),
        "LGB_Prediction": lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=42),
    }
    for name, classifier in classifiers.items():
        classifier.fit(X, y)
        predictions[name] = classifier.predict(X)
        print(f"{name}: accuracy {accuracy_score(y, predictions[name]):.4f}")

    y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y.values.reshape(-1, 1))
    network = keras.Sequential(
        [
            keras.Input(shape=(X.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )
    network.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    network.fit(X, y_onehot, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
    predictions["NN_Prediction"] = np.argmax(network.predict(X), axis=1) + 1
    print(f"NN_Prediction: accuracy {accuracy_score(y, predictions['NN_Prediction']):.4f}")
    return predictions


def evaluate_selectors(
    truth_path: Path, prediction_paths: dict[int, Path], selector_predictions: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Score every selector: each impression is ranked by the base model it picked."""
    truth_lines = _read_lines(truth_path)
    base_lines = {model: _read_lines(path) for model, path in prediction_paths.items()}
    selectors = [c for c in selector_predictions.columns if c != "True_Label"]

    results: dict[str, dict[str, float]] = {}
    for selector in selectors:
        chosen = selector_predictions[selector].tolist()
        if len(chosen) != len(truth_lines):
            raise ValueError(f"{selector}: {len(chosen)} predictions for {len(truth_lines)} lines")

        aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
        for idx, truth_line in enumerate(truth_lines):
            impression_id, labels = parse_line(truth_line)
            if not labels:
                continue
            pred_id, ranks = parse_line(base_lines[chosen[idx]][idx])
            if pred_id != impression_id:
                raise ValueError(f"Impression id mismatch in line {idx + 1}")
            y_true = np.array(labels, dtype="float32")
            y_score = np.array([1.0 / rank for rank in ranks], dtype="float32")
            aucs.append(roc_auc_score(y_true, y_score))
            mrrs.append(mrr_score(y_true, y_score))
            ndcg5s.append(ndcg_score(y_true, y_score, 5))
            ndcg10s.append(ndcg_score(y_true, y_score, 10))

        results[selector] = {
            "AUC": float(np.mean(aucs)),
            "MRR": float(np.mean(mrrs)),
            "nDCG@5": float(np.mean(ndcg5s)),
            "nDCG@10": float(np.mean(ndcg10s)),
        }
    return results
