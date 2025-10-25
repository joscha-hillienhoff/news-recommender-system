"""Carbon emissions tracking utilities built on top of CodeCarbon.

This module provides a small convenience wrapper around CodeCarbon's
`EmissionsTracker` to:
- ensure a consistent output directory inside the project
- start/stop tracking for a run
- print a concise console summary
- write a human-readable text report next to CodeCarbon's CSV

Usage:
    tracker = CarbonTracker(project_name="baseline")
    tracker.start_tracking()
    # ... your code ...
    tracker.end_report()
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from codecarbon import EmissionsTracker
import pandas as pd


class CarbonTracker:
    """
    Thin wrapper around CodeCarbon's `EmissionsTracker`.

    It standardizes where artifacts are written (``reports/emissions``),
    ensures directories exist, and emits a short, readable text report
    after the run.

    Parameters
    ----------
    project_name : str
        A short label used by CodeCarbon and for the saved text report.
        Example: "baseline", "xgboost_tuned", etc.
    """

    def __init__(self, project_name: str) -> None:
        """Initialize the CarbonTracker.

        Sets the project root, ensures the reports/emissions directory exists,
        and configures the CodeCarbon `EmissionsTracker` for this project.

        Parameters
        ----------
        project_name : str
            Name used for labeling the tracking session and report.
        """
        self.project_name = project_name

        # Project root = two levels above this file:
        # .../news_recommender_system/
        self.project_dir: Path = Path(__file__).resolve().parents[1]
        print(self.project_dir)

        # Ensure output directory exists: .../reports/emissions
        self.out_dir: Path = self.project_dir / "reports" / "emissions"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracker AFTER creating the directory so CodeCarbon
        # can open its CSV file without errors.
        self.tracker: EmissionsTracker = EmissionsTracker(
            project_name=self.project_name,
            output_dir=str(self.out_dir),
            log_level="critical",
        )

    def start_tracking(self) -> None:
        """Start the CodeCarbon tracker."""
        self.tracker.start()

    def end_report(self):
        """Stop tracking, print a console summary, and write a text report.

        The method:
        1) Stops CodeCarbon and retrieves total emissions for the run.
        2) Loads the latest row from CodeCarbon's ``emissions.csv`` to
           extract additional details (duration, energy, power, country).
        3) Prints a readable console summary.
        4) Saves a ``<project_name>.txt`` report next to the CSV.
        """
        # Stop tracking and get the emissions data
        emissions = self.tracker.stop()

        # Print the total emissions to the console
        print(f"💡 Carbon emissions from this run: {emissions:.6f} kg CO2eq")

        # Resolve the project root directory (two levels up from this file)
        project_dir = Path(__file__).resolve().parents[1]

        # Load latest emissions entry
        df = pd.read_csv(project_dir / "reports/emissions/emissions.csv")
        emissions_data = df.iloc[-1]

        # Extract and calculate relevant metrics for reporting
        duration_hr = emissions_data["duration"] / 3600
        energy_kwh = emissions_data["energy_consumed"]
        cpu_power = emissions_data["cpu_power"]
        gpu_power = (
            f"{emissions_data['gpu_power']:.2f} W"
            if "gpu_power" in emissions_data and not pd.isna(emissions_data["gpu_power"])
            else "Not available"
        )
        country = (
            emissions_data["country_name"] if "country_name" in emissions_data else "Not available"
        )

        # Generate timestamp for report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print to console
        print("\nDetailed emissions data:")
        print(f"- Duration: {duration_hr:.2f} hours")
        print(f"- Energy consumed: {energy_kwh:.4f} kWh")
        print(f"- CPU Power: {cpu_power:.2f} W")
        print(f"- GPU Power: {gpu_power}")
        print(f"- Country: {country}")

        # Create structured report text
        report = f"""\
        📄 Emissions Report – {timestamp}
        ====================================
        🌱 Total Emissions:     {emissions:.6f} kg CO2eq

        🕒 Duration:            {duration_hr:.2f} hours
        ⚡ Energy Consumed:     {energy_kwh:.4f} kWh
        🧠 CPU Power:           {cpu_power:.2f} W
        🎮 GPU Power:           {gpu_power}

        🌍 Country:             {country}
        ====================================
        """

        # Save to .txt file
        with open(project_dir / f"reports/emissions/{self.project_name}.txt", "w") as f:
            f.write(report)
