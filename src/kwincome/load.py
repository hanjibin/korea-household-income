"""Raw data loading."""
from pathlib import Path

import pandas as pd

from kwincome.paths import JOB_XLSX, RAW_CSV

STR_COLS = ["occupation", "company_size", "reason_none_worker"]


def _blank_to_nan(s: pd.Series) -> pd.Series:
    s = s.str.strip()
    return s.mask(s == "")


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the CSV (UTF-8 BOM). Blank-string cells in the text columns become NaN."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={c: str for c in STR_COLS})
    for c in STR_COLS:
        df[c] = _blank_to_nan(df[c])
    return df


def load_job_codes(path: Path = JOB_XLSX) -> pd.DataFrame:
    """Occupation code table with columns job_code, job_title (both str).

    The sheet has a garbled leading header column(s); only `job_code` and
    `job_title` are used. Codes are read as text so they keep their form.
    """
    t = pd.read_excel(path, dtype=str)[["job_code", "job_title"]]
    t = t.dropna(subset=["job_code"])
    t = t.assign(job_code=t["job_code"].str.strip(), job_title=t["job_title"].str.strip())
    return t.drop_duplicates("job_code").reset_index(drop=True)
