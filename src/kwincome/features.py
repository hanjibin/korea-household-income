"""Derived features: age, real (2018 prices) and equivalized income."""
import numpy as np
import pandas as pd

from kwincome.codes import NONRESPONSE
from kwincome.paths import CPI_CSV

AGE_BINS = [-np.inf, 34, 44, 54, 64, 74, np.inf]
AGE_LABELS = ["<35", "35-44", "45-54", "55-64", "65-74", "75+"]
OCC_YEARS = (2008, 2017)  # data years (Kaggle `year`) with comparable occupation codes


def load_cpi(path=CPI_CSV, base_year: int = 2018) -> pd.Series:
    """CPI indexed by year, rebased so cpi[base_year] == 100."""
    raw = pd.read_csv(path).set_index("year")["cpi"].astype(float)
    return raw / raw[base_year] * 100


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def work_status(occupation_code: pd.Series, reason_code: pd.Series) -> pd.Series:
    """Label-free work status: "Employed" when an occupation code is present,
    otherwise "Not employed: code k" from the raw non-work reason code.

    The inherited reason labels conflict with the data's profile, so the raw
    code is kept as is. Nonresponse or missing reason -> "code unknown".
    Every row lands in one of these two branches, so there is no "Unknown".
    """
    code = pd.to_numeric(reason_code, errors="coerce")
    code = code.mask(code.isin(NONRESPONSE["reason_none_worker"]))
    label = "Not employed: code " + code.astype("Int64").astype("string")
    label = label.fillna("Not employed: code unknown").astype(object)
    return label.where(occupation_code.isna(), "Employed")


def add_features(df: pd.DataFrame, cpi: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["age"] = out["year"] - out["year_born"]
    out["age_group"] = pd.cut(out["age"], AGE_BINS, labels=AGE_LABELS).astype(
        pd.CategoricalDtype(AGE_LABELS, ordered=True))
    # Kaggle `year` appears to be the income reference year already (fielded the
    # following year); see docs/analysis_decisions.md. No shift.
    out["income_year"] = out["year"]
    missing = sorted(set(out["income_year"].dropna().astype(int)) - set(cpi.index))
    if missing:
        raise ValueError(f"CPI series has no value for income years {missing}")
    deflator = out["income_year"].map(cpi)
    out["real_income"] = out["income"] / deflator * 100
    out["eq_income"] = out["real_income"] / np.sqrt(out["family_member"])
    out["income_pp"] = out["real_income"] / out["family_member"]
    out["log_eq_income"] = np.log(out["eq_income"].where(out["real_income"] > 0))
    out["female_head"] = (out["gender_label"] == "Female").astype(int)
    out["occ_consistent"] = out["year"].between(*OCC_YEARS)
    out["work_status"] = work_status(out["occupation_code"], out["reason_none_worker"])
    return out
