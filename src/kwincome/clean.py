"""Cleaning: blanks and non-response to NaN, code labels, occupation mapping."""
import numpy as np
import pandas as pd

from kwincome import codes


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _nonresponse_to_nan(s: pd.Series, col: str) -> pd.Series:
    return s.mask(s.isin(codes.NONRESPONSE[col]))


def occupation_major(code: pd.Series) -> pd.Series:
    """Major occupation group: the code's first digit, with two exceptions.

    4-digit codes starting with "10" (military, e.g. 1011, 1012) -> "A"
    (armed forces) instead of "1" (managers); 9999 (unknown) -> missing
    instead of "9" (elementary).
    """
    c = code.astype("string").str.strip()
    major = c.str[0].mask((c.str.len() == 4) & c.str.startswith("10"), "A")
    major = major.mask(c == "9999")
    return major.astype(object).where(major.notna(), np.nan)


def clean(df: pd.DataFrame, job_codes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    for c in ("occupation", "company_size", "reason_none_worker"):
        s = out[c].astype("string").str.strip()
        out[c] = s.mask(s == "").astype(object).where(s.notna(), np.nan)

    out["gender_label"] = out["gender"].map(codes.GENDER)
    edu = pd.CategoricalDtype(list(codes.EDUCATION.values()), ordered=True)
    out["education_label"] = out["education_level"].map(codes.EDUCATION).astype(edu)
    out["region_label"] = out["region"].map(codes.REGION)
    out["marriage_label"] = _nonresponse_to_nan(out["marriage"], "marriage").map(codes.MARRIAGE)

    reason = _nonresponse_to_nan(_numeric(out["reason_none_worker"]), "reason_none_worker")
    out["reason_label"] = reason.map(codes.REASON)

    code = out["occupation"].astype("string").str.strip()
    out["occupation_code"] = code.astype(object).where(code.notna(), np.nan)
    titles = dict(zip(job_codes["job_code"], job_codes["job_title"]))
    label = out["occupation_code"].map(titles)
    out["occupation_label"] = label.where(label.notna() | code.isna(), "Unmapped")
    out["occupation_major"] = occupation_major(out["occupation_code"])

    present = out["occupation_code"].dropna()
    matched = present.isin(titles.keys())
    report = {
        "n_rows": len(out),
        "occupation_mapping_rate": float(matched.mean()) if len(present) else float("nan"),
        "unmapped_codes": sorted(present[~matched].unique().tolist()),
    }
    return out, report
