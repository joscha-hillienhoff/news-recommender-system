"""Project paths and MIND file format constants."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
TRUTH_DIR = DATA_DIR / "truth"

PREDICTIONS_DIR = PROJ_ROOT / "predictions"

REPORTS_DIR = PROJ_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
EMISSIONS_DIR = REPORTS_DIR / "emissions"
FIGURES_DIR = REPORTS_DIR / "figures"

# Column names for the MIND behaviors files
COL_BEHAVIORS = ["ImpressionId", "User", "Time", "History", "Impressions"]

# Column names for the MIND news files
COL_NEWS = ["NewsId", "Category", "SubCat", "Title", "Abstract", "url", "TitleEnt", "AbstractEnt"]

# Time format used in the MIND behaviors files
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
