"""Convert the raw MIND-small TSV files into time-sorted Parquet files.

Expected raw layout (see the README for the download instructions)::

    data/raw/train/{behaviors,news}.tsv
    data/raw/validation/{behaviors,news}.tsv

The resulting files in ``data/interim`` are used by all models:

- ``behaviors_train``, ``behaviors_val`` and ``behaviors_train_val`` carry an extra
  ``Timestamp`` column (Unix seconds) and are sorted by it.
- ``news_train``, ``news_val`` and ``news_train_val`` hold the article metadata.
"""

from pathlib import Path
import time

import pandas as pd

from news_recommender_system.config import (
    COL_BEHAVIORS,
    COL_NEWS,
    DATA_DIR,
    TIME_FORMAT,
)


def _read_tsv(filepath: Path, file_type: str) -> pd.DataFrame:
    """Read a MIND TSV file (behaviors or news) into a DataFrame."""
    names = COL_BEHAVIORS if file_type == "behaviors" else COL_NEWS
    print(f"  -> Reading {file_type} data from: {filepath}")
    return pd.read_csv(filepath, sep="\t", header=None, names=names)


def load_raw_split(raw_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load behaviors and news for one split (``train`` or ``validation``)."""
    split_dir = raw_dir / split
    behaviors = _read_tsv(split_dir / "behaviors.tsv", "behaviors")
    news = _read_tsv(split_dir / "news.tsv", "news")
    return behaviors, news


def add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Unix ``Timestamp`` column from ``Time`` and sort by it."""
    df = df.copy()
    df["Timestamp"] = df["Time"].apply(lambda x: time.mktime(time.strptime(x, TIME_FORMAT)))
    return df.sort_values(by="Timestamp")


def make_interim_datasets(data_dir: Path = DATA_DIR) -> None:
    """Read the raw MIND-small files and write the interim Parquet files."""
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    behaviors_train, news_train = load_raw_split(raw_dir, "train")
    behaviors_val, news_val = load_raw_split(raw_dir, "validation")

    behaviors_train_val = pd.concat([behaviors_train, behaviors_val], ignore_index=True)
    news_train_val = pd.concat([news_train, news_val], ignore_index=True)

    datasets = {
        "behaviors_train": add_timestamp(behaviors_train),
        "news_train": news_train,
        "behaviors_val": add_timestamp(behaviors_val),
        "news_val": news_val,
        "behaviors_train_val": add_timestamp(behaviors_train_val),
        "news_train_val": news_train_val,
    }
    for name, df in datasets.items():
        filepath = interim_dir / f"{name}.parquet"
        print(f"  -> Saving {name} to {filepath}")
        df.to_parquet(filepath, index=False)
