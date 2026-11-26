"""Project paths, resolved from this file's location (not the cwd)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "Korea Income and Welfare.csv"
JOB_XLSX = ROOT / "data" / "raw" / "job_code_translated.xlsx"
CPI_CSV = ROOT / "data" / "external" / "cpi_kor.csv"
GEOJSON = ROOT / "data" / "external" / "kr_provinces.geojson"
# KWINCOME_FIG_DIR redirects figure output (used by the notebook test).
FIG_DIR = Path(os.environ.get("KWINCOME_FIG_DIR") or ROOT / "reports" / "figures")
