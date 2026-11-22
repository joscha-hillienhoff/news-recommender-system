"""Carbon emissions tracking built on top of CodeCarbon.

A small wrapper around CodeCarbon's ``EmissionsTracker`` that writes its CSV and a short
text report to ``reports/emissions``.

Usage::

    tracker = CarbonTracker(project_name="baseline")
    tracker.start_tracking()
    # ... run a model ...
    tracker.end_report()
"""

from __future__ import annotations

from datetime import datetime

from codecarbon import EmissionsTracker
import pandas as pd

from news_recommender_system.config import EMISSIONS_DIR


class CarbonTracker:
    """Track the emissions of one run and write a readable report.

    Parameters
    ----------
    project_name : str
        Label used by CodeCarbon and as the name of the text report, e.g. ``"baseline"``.
    """

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.tracker = EmissionsTracker(
            project_name=project_name, output_dir=str(EMISSIONS_DIR), log_level="critical"
        )

    def start_tracking(self) -> None:
        """Start the CodeCarbon tracker."""
        self.tracker.start()

    def end_report(self) -> float:
        """Stop tracking, print a summary, and write ``<project_name>.txt``.

        Returns the total emissions of the run in kg CO2eq.
        """
        emissions = self.tracker.stop()
        latest = pd.read_csv(EMISSIONS_DIR / "emissions.csv").iloc[-1]

        duration_hr = latest["duration"] / 3600
        energy_kwh = latest["energy_consumed"]
        cpu_power = latest["cpu_power"]
        gpu_power = (
            f"{latest['gpu_power']:.2f} W"
            if "gpu_power" in latest and not pd.isna(latest["gpu_power"])
            else "Not available"
        )
        country = latest["country_name"] if "country_name" in latest else "Not available"

        report = (
            f"Emissions report - {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"Total emissions: {emissions:.6f} kg CO2eq\n"
            f"Duration:        {duration_hr:.2f} hours\n"
            f"Energy consumed: {energy_kwh:.4f} kWh\n"
            f"CPU power:       {cpu_power:.2f} W\n"
            f"GPU power:       {gpu_power}\n"
            f"Country:         {country}\n"
        )
        print(report)
        (EMISSIONS_DIR / f"{self.project_name}.txt").write_text(report)
        return emissions
