import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from kwincome.models import (STEPS, VAR_GROUPS, _design, _formula, model_frame, oaxaca,
                             quantile_gap, staged_gap)

EDU_LEVELS = ["No schooling (<7)", "No schooling (7+)", "Elementary", "Middle school",
              "High school", "Junior college", "University", "Master's", "Doctorate"]


@pytest.fixture(scope="module")
def synth():
    """4,000 rows, 1,000 households; education differs by group.

    log_y = 0.3*edu_high - 0.1*female + noise; other controls are unrelated noise.
    """
    rng = np.random.default_rng(0)
    n_id, per = 1000, 4
    ids = np.repeat(np.arange(n_id), per)
    female = np.repeat(rng.random(n_id) < 0.3, per).astype(int)
    edu_high = (rng.random(n_id * per) < np.where(female == 1, 0.25, 0.6)).astype(int)
    year = np.tile(np.arange(2010, 2010 + per), n_id)
    employed = rng.random(n_id * per) < 0.65
    occ = np.where(employed, rng.choice(["1", "2", "3", "4"], n_id * per), None)
    status = np.where(employed, "Employed",
                      rng.choice(["Care and housework", "Discouraged / no intention"], n_id * per))
    log_y = 7 + 0.3 * edu_high - 0.1 * female + rng.normal(0, 0.3, n_id * per)
    fam = rng.integers(1, 5, n_id * per)
    df = pd.DataFrame({
        "id": ids, "year": year, "income_year": year,
        "age": rng.integers(30, 80, n_id * per), "family_member": fam,
        "education_label": pd.Categorical(np.where(edu_high == 1, "University", "High school"),
                                          categories=EDU_LEVELS, ordered=True),
        "marriage_label": rng.choice(["code 1", "code 2", "code 3"], n_id * per).astype(object),
        "region_label": rng.choice(["Seoul", "Gyeonggi", "Gyeongnam"], n_id * per),
        "work_status": status.astype(object), "occupation_major": occ.astype(object),
        "female_head": female, "occ_consistent": True,
    })
    df["eq_income"] = np.exp(log_y)
    df["real_income"] = df["eq_income"] * np.sqrt(fam)
    df.loc[[5, 17, 300], "real_income"] = [0.0, -10.0, 0.0]
    df.loc[[0, 1], "marriage_label"] = np.nan
    df["log_eq_income"] = np.log(df["eq_income"].where(df["real_income"] > 0))
    return df


def test_steps_and_groups():
    assert [s for s, _ in STEPS] == ["Raw", "+ Age", "+ Education", "+ Household size",
                                     "+ Marital status", "+ Work / occupation",
                                     "+ Region & year"]
    assert list(VAR_GROUPS) == [s.removeprefix("+ ") for s, _ in STEPS[1:]]


def test_model_frame_drops_nonpositive(synth):
    mf, dropped = model_frame(synth)
    assert dropped == 3 and len(mf) == len(synth) - 3
    assert np.isfinite(mf["log_eq_income"]).all()
    # NaN marital status becomes an explicit level, not a dropped row
    assert "Missing" in set(mf["marriage_label"].astype(str))


def test_model_frame_drops_minus_inf(synth):
    d = synth.copy()
    d.loc[10, "log_eq_income"] = -np.inf
    _, dropped = model_frame(d)
    assert dropped == 4


def test_model_frame_occupation_restriction(synth):
    d = synth.copy()
    d.loc[d["year"] == 2010, "occ_consistent"] = False
    mf, _ = model_frame(d, include_occupation=True)
    assert mf["occ_consistent"].all()
    mf_all, _ = model_frame(d, include_occupation=False)
    assert len(mf_all) == len(d) - 3


def test_staged_gap_shrinks(synth):
    out = staged_gap(synth)
    assert list(out.columns) == ["step", "coef", "se", "pct_gap", "ci_low_pct",
                                 "ci_high_pct", "n"]
    assert list(out["step"]) == [s for s, _ in STEPS]
    raw = out.loc[out["step"] == "Raw", "coef"].item()
    edu = out.loc[out["step"] == "+ Education", "coef"].item()
    assert raw < edu
    assert out["coef"].iloc[-1] == pytest.approx(-0.1, abs=0.03)
    assert (out["n"] == len(synth) - 3).all()
    np.testing.assert_allclose(out["pct_gap"], 100 * (np.exp(out["coef"]) - 1))
    assert (out["ci_low_pct"] < out["pct_gap"]).all() and (out["pct_gap"] < out["ci_high_pct"]).all()


def test_staged_gap_without_occupation_and_winsor(synth):
    out = staged_gap(synth, include_occupation=False, winsor=True)
    assert "+ Work status" in set(out["step"])
    assert "+ Work / occupation" not in set(out["step"])
    assert out["coef"].iloc[-1] == pytest.approx(-0.1, abs=0.03)


def test_oaxaca_adds_up(synth):
    res = oaxaca(synth, n_boot=20)
    assert res["explained"] + res["unexplained"] == pytest.approx(res["total"])
    assert res["contrib"]["explained"].sum() == pytest.approx(res["explained"])
    assert res["unexplained"] == pytest.approx(0.1, abs=0.03)
    assert res["total"] > res["unexplained"] > 0  # male-headed higher; education explains part
    assert list(res["contrib"].columns) == ["group", "explained", "ci_low", "ci_high"]
    assert list(res["contrib"]["group"]) == list(VAR_GROUPS)
    edu = res["contrib"].set_index("group").loc["Education", "explained"]
    assert edu == pytest.approx(0.3 * 0.35, abs=0.03)
    for k in ("total", "explained", "unexplained"):
        lo, hi = res["ci"][k]
        assert lo <= res[k] <= hi
    assert res["n"] == len(synth) - 3


@pytest.mark.parametrize("reference", ["pooled", "male", "female"])
def test_oaxaca_reference_adds_up(synth, reference):
    res = oaxaca(synth, n_boot=5, reference=reference)
    assert res["explained"] + res["unexplained"] == pytest.approx(res["total"])
    assert res["contrib"]["explained"].sum() == pytest.approx(res["explained"])
    assert res["reference"] == reference


def test_oaxaca_references_agree_on_synthetic(synth):
    # same true coefficients in both groups, so every reference gives a similar split
    shares = {r: oaxaca(synth, n_boot=0, reference=r) for r in ("pooled", "male", "female")}
    total = shares["pooled"]["total"]
    for r, res in shares.items():
        assert res["total"] == pytest.approx(total)
        assert res["explained"] / res["total"] == pytest.approx(
            shares["pooled"]["explained"] / total, abs=0.15), r


# A "Missing" marital level occurs only among male-headed rows, so the
# female-only design is rank-deficient; both sides then use min-norm solutions.
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.SingularMatrixWarning")
@pytest.mark.parametrize("reference", ["male", "female"])
def test_oaxaca_group_reference_matches_subgroup_ols(synth, reference):
    # beta from a separate statsmodels OLS on that group only (no group dummy)
    res = oaxaca(synth, n_boot=0, reference=reference)
    mf, _ = model_frame(synth)
    yv, X = _design(_formula("log_eq_income", [t for ts in VAR_GROUPS.values() for t in ts]), mf)
    female = mf["female_head"].to_numpy()
    rows = female == (0 if reference == "male" else 1)
    Xg = X.drop(columns="female_head")
    beta = sm.OLS(yv[rows], Xg[rows]).fit().params
    dx = Xg[female == 0].mean() - Xg[female == 1].mean()
    assert res["explained"] == pytest.approx(float(dx @ beta), abs=1e-10)
    edu_cols = [c for c in Xg.columns if c.startswith("C(education_label)")]
    edu = res["contrib"].set_index("group").loc["Education", "explained"]
    assert edu == pytest.approx(float(dx[edu_cols] @ beta[edu_cols]), abs=1e-10)


def test_oaxaca_no_bootstrap_has_nan_ci(synth):
    res = oaxaca(synth, n_boot=0, reference="male")
    assert all(np.isnan(v) for v in res["ci"]["explained"])
    assert res["contrib"]["ci_low"].isna().all()


def test_oaxaca_pooled_default_unchanged(synth):
    a = oaxaca(synth, n_boot=5, seed=1)
    b = oaxaca(synth, n_boot=5, seed=1, reference="pooled")
    assert a["explained"] == b["explained"] and a["ci"] == b["ci"]


def test_oaxaca_invalid_reference(synth):
    with pytest.raises(ValueError):
        oaxaca(synth, n_boot=0, reference="both")


def test_oaxaca_deterministic(synth):
    a = oaxaca(synth, n_boot=10, seed=3)
    b = oaxaca(synth, n_boot=10, seed=3)
    assert a["ci"] == b["ci"]


def test_oaxaca_unexplained_equals_minus_dummy(synth):
    # pooled-with-dummy identity: unexplained = -(female_head coef) in the full OLS
    res = oaxaca(synth, n_boot=2)
    final = staged_gap(synth)["coef"].iloc[-1]
    assert res["unexplained"] == pytest.approx(-final, abs=1e-10)


# Synthetic controls are all discrete, so the LP optimum is not unique and IRLS
# can cycle until max_iter; the estimate is still accurate (checked below).
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.IterationLimitWarning")
def test_quantile_gap_shape(synth):
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    out = quantile_gap(synth, qs=qs)
    assert len(out) == 5 and list(out["q"]) == list(qs)
    assert list(out.columns) == ["q", "coef", "ci_low", "ci_high", "pct_gap", "n"]
    assert (out["coef"] - (-0.1)).abs().max() < 0.06
    assert (out["ci_low"] < out["coef"]).all() and (out["coef"] < out["ci_high"]).all()
