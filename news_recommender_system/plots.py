"""Results figure: ranking quality and CO2 cost per model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FAMILY_COLORS = {
    "Baseline": "#8c8c8c",
    "Content-based": "#2a9d8f",
    "Collaborative": "#e9a23b",
    "Hybrid": "#5b6fd6",
}


def plot_results(results_path: Path, output_path: Path) -> None:
    """Draw nDCG@10 and CO2 per model from ``results_path`` and save a PNG."""
    results = pd.read_csv(results_path)
    labels = results["model"].tolist()[::-1]
    colors = [FAMILY_COLORS[f] for f in results["family"]][::-1]
    ypos = range(len(labels))

    fig, (ax_quality, ax_co2) = plt.subplots(
        1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]}
    )

    ax_quality.barh(ypos, results["ndcg10"][::-1], color=colors)
    for y, value in zip(ypos, results["ndcg10"][::-1]):
        ax_quality.text(value + 0.005, y, f"{value:.3f}", va="center", fontsize=9)
    ax_quality.set_yticks(list(ypos), labels)
    ax_quality.set_xlim(0, 0.62)
    ax_quality.set_xlabel("nDCG@10 (higher is better)")

    ax_co2.barh(ypos, results["co2_kg"][::-1] * 1e6, color=colors)
    for y, value in zip(ypos, results["co2_kg"][::-1] * 1e6):
        ax_co2.text(value + 5, y, f"{value:.0f}", va="center", fontsize=9)
    ax_co2.set_xlim(0, 380)
    ax_co2.set_xlabel("CO$_2$ per run (mg CO$_2$eq, lower is better)")

    for ax in (ax_quality, ax_co2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("News recommenders on MIND-small (validation set)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(
        0.01,
        0.015,
        "* Selector fit and scored on the same validation impressions: optimistic upper bound.",
        fontsize=8,
        color="#555555",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
