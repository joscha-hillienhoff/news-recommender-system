"""Ground-truth files for evaluation.

A truth file has one line per impression, ``<impression_id> <json_list_of_labels>``, where
the labels (1 = clicked, 0 = not clicked) follow the order of the candidate articles in the
impression. Predictions must be written in the same line order as the truth file.
"""

import json
from pathlib import Path

import pandas as pd

from news_recommender_system.config import INTERIM_DATA_DIR, TRUTH_DIR


def generate_truth_file(behaviors: pd.DataFrame, output_path: Path) -> None:
    """Write the click labels of every impression in ``behaviors`` to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    impressions = behaviors.set_index("ImpressionId")["Impressions"].apply(lambda x: x.split())

    with open(output_path, "w", encoding="utf-8") as f:
        for impression_id, news_list in impressions.items():
            labels = [int(news.split("-")[1]) for news in news_list]
            f.write(f"{impression_id} {json.dumps(labels)}\n")

    print(f"Truth file saved at: {output_path}")


def make_validation_truth(
    behaviors_path: Path = INTERIM_DATA_DIR / "behaviors_val.parquet",
    output_path: Path = TRUTH_DIR / "truth_val.txt",
) -> None:
    """Create the validation truth file (time-ordered) from the interim behaviors."""
    generate_truth_file(pd.read_parquet(behaviors_path), output_path)


def sort_truth_file(input_file: Path, output_file: Path) -> None:
    """Sort a truth or prediction file by numeric impression id."""
    with open(input_file, "r") as f:
        parsed = [line.strip().split(" ", 1) for line in f if line.strip()]
    parsed.sort(key=lambda parts: int(parts[0]))

    with open(output_file, "w") as f:
        for impression_id, payload in parsed:
            f.write(f"{impression_id} {payload}\n")
