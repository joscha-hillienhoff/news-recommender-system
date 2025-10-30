"""This module contains functions for the initial data processing pipeline of the news recommendation system.

It is responsible for:
1. Loading raw TSV data files (behaviors and news) from the 'data/raw' directory.
2. Concatenating training and validation sets.
3. Converting the time column to a Unix timestamp and sorting the behaviors data.
4. Saving the resulting, processed DataFrames as efficient Parquet files in the
   'data/interim' directory for use by subsequent analysis and modeling scripts.

The primary entry point for execution is the make_interim_datasets() function.
"""

from pathlib import Path
import time

import pandas as pd

from news_recommender_system.config import (
    COL_BEHAVIORS,
    COL_NEWS,
    DATA_DIR,
    PROJ_ROOT,
    TIME_FORMAT,
)

# --- Data Loading Helper ---


def _read_tsv_data(filepath: Path, file_type: str) -> pd.DataFrame:
    """Read a TSV file (behaviors or news) into a pandas DataFrame."""
    names = COL_BEHAVIORS if file_type == "behaviors" else COL_NEWS
    print(f"  -> Reading {file_type} data from: {filepath.name}")
    return pd.read_csv(filepath, sep="\t", header=None, names=names)


def _load_raw_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load behaviors and news data for a given split (train, validation, or test)."""
    split_dir = data_dir / split
    behaviors_df = _read_tsv_data(split_dir / "behaviors.tsv", "behaviors")
    news_df = _read_tsv_data(split_dir / "news.tsv", "news")
    return behaviors_df, news_df


# --- Data Processing and Cleaning ---


def _process_behaviors_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'Time' column to a Unix timestamp and sorts the DataFrame.

    Returns:
        The DataFrame with a new 'Timestamp' column, sorted by time.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    df["Timestamp"] = df["Time"].apply(lambda x: time.mktime(time.strptime(x, TIME_FORMAT)))
    df.sort_values(by="Timestamp", inplace=True)
    return df


def make_interim_datasets(data_dir: Path):
    """Make the interim datasets from raw data.

    Load raw data, combine train/val, process time, and save the
    interim DataFrames as Parquet files to the data/interim directory.
    """
    # Define raw data directory
    raw_data_dir = data_dir / "raw"

    # Ensure interim directory exists
    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Loading Raw Data (Same as before) ---
    print("--- 1. Loading Raw Data ---")
    behaviors_train, news_train = _load_raw_split(raw_data_dir, "train")
    behaviors_val, news_val = _load_raw_split(raw_data_dir, "validation")
    behaviors_test, news_test = _load_raw_split(raw_data_dir, "test")

    # --- 2. Combining and Processing Data (Same as before) ---
    print("\n--- 2. Combining and Processing Data ---")
    behaviors_train_val = pd.concat([behaviors_train, behaviors_val], ignore_index=True)
    news_train_val = pd.concat([news_train, news_val], ignore_index=True)
    behaviors_train = _process_behaviors_time(behaviors_train)
    behaviors_train_val = _process_behaviors_time(behaviors_train_val)
    behaviors_val = _process_behaviors_time(behaviors_val)

    print("\n--- 3. Saving Interim DataFrames as Parquet ---")

    # Define the datasets and save them using Parquet
    datasets_to_save = {
        "behaviors_train": behaviors_train,
        "news_train": news_train,
        "behaviors_train_val": behaviors_train_val,
        "news_train_val": news_train_val,
        "behaviors_val": behaviors_val,
        "news_val": news_val,
        "behaviors_test": behaviors_test,
        "news_test": news_test,
    }

    # Save the datasets
    for name, df in datasets_to_save.items():
        filepath = interim_dir / f"{name}.parquet"
        print(f"  -> Saving {name} to {filepath.name}")
        df.to_parquet(filepath, index=False)


if __name__ == "__main__":
    print(f"Starting data processing for project at: {PROJ_ROOT.name}")
    make_interim_datasets(DATA_DIR)
    print("Data processing complete. Interim files saved to data/interim.")
