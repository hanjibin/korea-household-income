import numpy as np
import pandas as pd
import pytest

from kwincome.features import add_features, load_cpi, winsorize


def _frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"year": [2010] * n, "year_born": [1960] * n, "income": [100.0] * n,
            "family_member": [1] * n, "gender_label": ["Male"] * n,
            "occupation_code": [np.nan] * n, "reason_none_worker": [np.nan] * n}
    base.update(cols)
    return pd.DataFrame(base)


FLAT = pd.Series({y: 100.0 for y in range(2004, 2019)})


def test_cpi_rebased():
    cpi = load_cpi()
    assert cpi[2018] == pytest.approx(100)
    assert set(range(2004, 2019)) <= set(cpi.index)


def test_real_income_uses_income_year():
    # Kaggle `year` is read as the income reference year itself (no shift)
    df = add_features(_frame(year=[2006], income=[1000.0]), pd.Series({2006: 50.0, 2018: 100.0}))
    assert df["income_year"].iloc[0] == 2006
    assert df["real_income"].iloc[0] == pytest.approx(2000)


def test_missing_cpi_year_raises():
    with pytest.raises(ValueError, match="2003"):
        add_features(_frame(year=[2003, 2010]), FLAT)


def test_eq_income():
    df = add_features(_frame(income=[4000.0], family_member=[4]), FLAT)
    assert df["eq_income"].iloc[0] == pytest.approx(2000)
    assert df["income_pp"].iloc[0] == pytest.approx(1000)


def test_log_nan_for_nonpositive():
    df = add_features(_frame(income=[-5.0, 0.0, 10.0]), FLAT)
    v = df["log_eq_income"]
    assert v.iloc[:2].isna().all() and np.isfinite(v.iloc[2])


def test_age_group_bounds():
    df = add_features(_frame(year=[2010, 2010], year_born=[1976, 1935]), FLAT)
    assert list(df["age"]) == [34, 75]
    assert list(df["age_group"].astype(str)) == ["<35", "75+"]
    assert df["age_group"].cat.ordered


def test_work_status_is_label_free():
    # raw reason code drives the category; inherited reason labels are ignored
    df = add_features(_frame(occupation_code=["611", np.nan, np.nan, np.nan, np.nan],
                             reason_none_worker=["8", "8", "99", np.nan, "0"]), FLAT)
    assert list(df["work_status"]) == ["Employed", "Not employed: code 8",
                                       "Not employed: code unknown",
                                       "Not employed: code unknown",
                                       "Not employed: code unknown"]


def test_female_head():
    df = add_features(_frame(gender_label=["Female", "Male"]), FLAT)
    assert list(df["female_head"]) == [1, 0]


def test_occ_consistent_window():
    df = add_features(_frame(year=[2007, 2008, 2017, 2018]), FLAT)
    assert list(df["occ_consistent"]) == [False, True, True, False]


def test_winsorize():
    s = pd.Series(range(101), dtype=float)
    w = winsorize(s)
    assert w.min() == pytest.approx(1) and w.max() == pytest.approx(99)
