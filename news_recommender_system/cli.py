"""Command line interface: ``python -m news_recommender_system --help``."""

from pathlib import Path
from typing import Callable

import pandas as pd
import typer

from news_recommender_system.config import (
    DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    TRUTH_DIR,
)

app = typer.Typer(help="News recommenders on MIND-small: run models and evaluate them.")

TRACK_CARBON = typer.Option(False, "--track-carbon", help="Measure CO2 emissions with CodeCarbon.")


def _read_interim(name: str) -> pd.DataFrame:
    return pd.read_parquet(INTERIM_DATA_DIR / f"{name}.parquet")


def _run(name: str, run: Callable[[], None], track_carbon: bool) -> None:
    """Run a model, optionally wrapped in a CodeCarbon measurement."""
    if not track_carbon:
        run()
        return
    from news_recommender_system.carbon_tracking import CarbonTracker

    tracker = CarbonTracker(project_name=name)
    tracker.start_tracking()
    run()
    tracker.end_report()


@app.command()
def prepare_data(data_dir: Path = DATA_DIR) -> None:
    """Convert data/raw TSV files to Parquet files in data/interim."""
    from news_recommender_system.data import make_interim_datasets

    make_interim_datasets(data_dir)


@app.command()
def truth() -> None:
    """Write data/truth/truth_val.txt with the click labels of the validation impressions."""
    from news_recommender_system.truth import make_validation_truth

    make_validation_truth()


@app.command()
def popularity(
    window_hours: int = 24, track_carbon: bool = TRACK_CARBON, name: str = "popularity"
) -> None:
    """Popularity baseline: rank by clicks within a rolling window."""
    from news_recommender_system.modeling.popularity_baseline import (
        generate_popularity_predictions,
    )

    behaviors_val = _read_interim("behaviors_val")
    _run(
        name,
        lambda: generate_popularity_predictions(
            behaviors_val, PREDICTIONS_DIR / f"{name}.txt", window_hours * 3600
        ),
        track_carbon,
    )


@app.command()
def ctr(window_hours: int = 24, track_carbon: bool = TRACK_CARBON) -> None:
    """CTR baseline: rank by click-through rate within a rolling window."""
    from news_recommender_system.modeling.ctr_baseline import generate_ctr_predictions

    name = f"ctr_{window_hours}h"
    behaviors_train_val = _read_interim("behaviors_train_val")
    behaviors_val = _read_interim("behaviors_val")
    _run(
        name,
        lambda: generate_ctr_predictions(
            behaviors_train_val,
            behaviors_val,
            PREDICTIONS_DIR / f"{name}.txt",
            window_hours * 3600,
        ),
        track_carbon,
    )


@app.command()
def content(method: str = "tfidf", track_carbon: bool = TRACK_CARBON) -> None:
    """Content-based model; METHOD is "tfidf" or "sbert" (sentence transformer)."""
    from news_recommender_system.modeling.content_based import generate_content_predictions

    behaviors_val = _read_interim("behaviors_val")
    news_val = _read_interim("news_val")
    _run(
        method,
        lambda: generate_content_predictions(
            behaviors_val, news_val, PREDICTIONS_DIR / f"{method}.txt", method
        ),
        track_carbon,
    )


@app.command()
def bpr(track_carbon: bool = TRACK_CARBON) -> None:
    """Collaborative filtering with BPR matrix factorization."""
    from news_recommender_system.modeling.bpr import generate_bpr_predictions

    behaviors_train = _read_interim("behaviors_train")
    behaviors_val = _read_interim("behaviors_val")
    _run(
        "bpr",
        lambda: generate_bpr_predictions(
            behaviors_train, behaviors_val, PREDICTIONS_DIR / "bpr.txt"
        ),
        track_carbon,
    )


@app.command()
def hybrid(
    popularity_file: str = "popularity",
    content_file: str = "sbert",
    bpr_file: str = "bpr",
    track_carbon: bool = TRACK_CARBON,
) -> None:
    """Hybrid model: classifiers pick the best of popularity, content-based and BPR."""
    from news_recommender_system.modeling import hybrid as hybrid_model

    truth_path = TRUTH_DIR / "truth_val.txt"
    paths = {
        1: PREDICTIONS_DIR / f"{popularity_file}.txt",
        2: PREDICTIONS_DIR / f"{content_file}.txt",
        3: PREDICTIONS_DIR / f"{bpr_file}.txt",
    }

    def run() -> None:
        labels = hybrid_model.best_model_labels(truth_path, paths)
        dataset = hybrid_model.selector_features(
            _read_interim("behaviors_train"), _read_interim("behaviors_val"), labels
        )
        selectors = hybrid_model.train_selectors(dataset)
        results = hybrid_model.evaluate_selectors(truth_path, paths, selectors)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        for selector, metrics in results.items():
            short = selector.split("_")[0].lower()
            lines = "\n".join(f"{k}:{v:.4f}" for k, v in metrics.items())
            (METRICS_DIR / f"scores_val_hybrid_{short}.txt").write_text(lines)
            print(selector, {k: round(v, 4) for k, v in metrics.items()})

    _run("hybrid", run, track_carbon)


@app.command()
def evaluate(name: str, truth_file: Path = TRUTH_DIR / "truth_val.txt") -> None:
    """Score predictions/NAME.txt and write reports/metrics/scores_val_NAME.txt."""
    from news_recommender_system.evaluation import evaluate_model

    evaluate_model(
        truth_file, PREDICTIONS_DIR / f"{name}.txt", METRICS_DIR / f"scores_val_{name}.txt"
    )


@app.command()
def plot(
    results: Path = REPORTS_DIR / "results.csv", output: Path = FIGURES_DIR / "results.png"
) -> None:
    """Draw the results figure from reports/results.csv."""
    from news_recommender_system.plots import plot_results

    plot_results(results, output)
