"""Gap models: staged OLS, pooled Oaxaca-Blinder decomposition, quantile regression.

The outcome is log real equivalized household income. The "gap" is the
coefficient on `female_head` (1 = female-headed household). All estimates are
associations, not causal effects.

Occupation codes are only comparable for data years (Kaggle `year`) 2008-2017
(`occ_consistent`). With `include_occupation=True` (default) every model
restricts to those rows and controls for occupation; with False it uses all
years and controls for work status only.

Work and occupation enter as ONE combined categorical, `work_occ`: the major
occupation group for employed heads, the raw non-work reason code otherwise
(label-free, see `features.work_status`).
Entering `C(work_status)` and `C(occupation_major)` side by side would be
singular, because "no occupation" is exactly the sum of the non-employed
work-status dummies. The combined term spans the same space as the full-rank
nested version, so it loses nothing.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from kwincome.features import winsorize
from kwincome.stats import cluster_bootstrap_sample

MISSING = "Missing"
WORK_OCC_TERM = "C(work_occ)"
WORK_ONLY_TERM = "C(work_status)"

STEPS: list[tuple[str, list[str]]] = [
    ("Raw", []),
    ("+ Age", ["age", "I(age**2)"]),
    ("+ Education", ["C(education_label)"]),
    ("+ Household size", ["family_member"]),
    ("+ Marital status", ["C(marriage_label)"]),
    ("+ Work / occupation", [WORK_OCC_TERM]),
    ("+ Region & year", ["C(region_label)", "C(income_year)"]),
]

VAR_GROUPS: dict[str, list[str]] = {name.removeprefix("+ "): terms for name, terms in STEPS[1:]}

CAT_COLS = ["education_label", "marriage_label", "region_label", "work_status",
            "occupation_major"]
NUM_COLS = ["id", "age", "family_member", "female_head", "income_year"]


def _steps(include_occupation: bool) -> list[tuple[str, list[str]]]:
    if include_occupation:
        return STEPS
    return [("+ Work status", [WORK_ONLY_TERM]) if terms == [WORK_OCC_TERM] else (name, terms)
            for name, terms in STEPS]


def _groups(include_occupation: bool) -> dict[str, list[str]]:
    return {name.removeprefix("+ "): terms for name, terms in _steps(include_occupation)[1:]}


def _as_category(s: pd.Series) -> pd.Series:
    """Categorical with NaN as an explicit "Missing" level and no unused levels."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        cat = s if MISSING in s.cat.categories else s.cat.add_categories(MISSING)
    else:
        s = s.astype(object)
        cats = sorted(s.dropna().astype(str).unique())
        cat = pd.Categorical(s.where(s.isna(), s.astype(str)), categories=[*cats, MISSING])
        cat = pd.Series(cat, index=s.index)
    return cat.fillna(MISSING).cat.remove_unused_categories()


def model_frame(df: pd.DataFrame, y: str = "log_eq_income",
                include_occupation: bool = True) -> tuple[pd.DataFrame, int]:
    """Rows usable for models, and the number dropped for a non-finite `y`.

    Non-finite `y` covers income <= 0 (log is NaN there) and any stray -inf.
    With `include_occupation`, rows outside 2008-2017 are removed first (not
    counted as dropped). Categorical controls get an explicit "Missing" level.
    """
    d = df[df["occ_consistent"]] if include_occupation else df
    ok = np.isfinite(d[y].to_numpy(dtype=float))
    dropped = int((~ok).sum())
    d = d[ok].copy()
    bad = [c for c in NUM_COLS if d[c].isna().any()]
    if bad:
        raise ValueError(f"numeric model columns contain NaN: {bad}")
    for c in CAT_COLS:
        d[c] = _as_category(d[c])
    occ = "Occ " + d["occupation_major"].astype(str)
    d["work_occ"] = _as_category(
        pd.Series(np.where(d["work_status"].astype(str) == "Employed", occ,
                           d["work_status"].astype(str)), index=d.index))
    return d.reset_index(drop=True), dropped


def _design(formula: str, mf: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Patsy design that refuses to drop rows and checks full column rank."""
    yv, X = patsy.dmatrices(formula, mf, return_type="dataframe", NA_action="raise")
    if len(X) != len(mf):
        raise RuntimeError("patsy changed the number of rows")
    rank = np.linalg.matrix_rank(X.to_numpy())
    if rank < X.shape[1]:
        raise RuntimeError(f"design is singular: rank {rank} < {X.shape[1]} columns")
    return yv.iloc[:, 0].to_numpy(), X


def _formula(y: str, terms: list[str]) -> str:
    return " + ".join([f"{y} ~ female_head", *terms])


def _pct(x):
    return 100 * (np.exp(x) - 1)


def staged_gap(df: pd.DataFrame, y: str = "log_eq_income", winsor: bool = False,
               include_occupation: bool = True) -> pd.DataFrame:
    """Female-head coefficient as control blocks are added cumulatively.

    Every step uses the same rows, so coefficients are comparable. Standard
    errors are clustered by household (`id`). `winsor=True` clips `y` at its
    1st/99th percentiles within the model sample.
    """
    mf, _ = model_frame(df, y, include_occupation)
    if winsor:
        mf[y] = winsorize(mf[y])
    groups = mf["id"].to_numpy()
    rows, terms = [], []
    for step, add in _steps(include_occupation):
        terms = [*terms, *add]
        yv, X = _design(_formula(y, terms), mf)
        fit = sm.OLS(yv, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        coef, se = fit.params["female_head"], fit.bse["female_head"]
        rows.append({"step": step, "coef": coef, "se": se, "pct_gap": _pct(coef),
                     "ci_low_pct": _pct(coef - 1.96 * se), "ci_high_pct": _pct(coef + 1.96 * se),
                     "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def _column_groups(X: pd.DataFrame, groups: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Design-matrix column positions belonging to each variable group."""
    term_to_group = {}
    for g, terms in groups.items():
        for t in patsy.ModelDesc.from_formula("~ 0 + " + " + ".join(terms)).rhs_termlist:
            term_to_group[t.name()] = g
    cols = {g: [] for g in groups}
    for name, sl in X.design_info.term_name_slices.items():
        if name in term_to_group:
            cols[term_to_group[name]].extend(range(sl.start, sl.stop))
    return {g: np.array(c, dtype=int) for g, c in cols.items()}


REFERENCES = ("pooled", "male", "female")


def _decompose(X: np.ndarray, yv: np.ndarray, female: np.ndarray,
               cols: dict[str, np.ndarray], reference: str = "pooled",
               dummy: int | None = None) -> tuple[float, dict[str, float]]:
    """Two-fold decomposition; returns total and explained by group.

    reference="pooled": beta from one OLS on both groups (with the group dummy).
    reference="male"/"female": beta from an OLS on that group only, with the
    group-dummy column `dummy` removed (it is constant within a group).
    """
    m, f = female == 0, female == 1
    if reference == "pooled":
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    else:
        rows = m if reference == "male" else f
        keep = np.array([j for j in range(X.shape[1]) if j != dummy])
        beta = np.zeros(X.shape[1])
        beta[keep] = np.linalg.lstsq(X[rows][:, keep], yv[rows], rcond=None)[0]
    dx = X[m].mean(axis=0) - X[f].mean(axis=0)
    total = yv[m].mean() - yv[f].mean()
    return total, {g: float(dx[c] @ beta[c]) for g, c in cols.items()}


def oaxaca(df: pd.DataFrame, y: str = "log_eq_income", n_boot: int = 500, seed: int = 42,
           include_occupation: bool = True, reference: str = "pooled") -> dict:
    """Two-fold Oaxaca-Blinder decomposition, pooled reference by default (Neumark / Jann).

    β* comes from one OLS on both groups with all VAR_GROUPS terms plus the
    `female_head` dummy; the dummy is excluded from the explained part.
    total = mean y(male-headed) - mean y(female-headed), so positive means
    male-headed households are higher. Group contributions sum to `explained`
    and are invariant to the reference level chosen inside each categorical.
    CIs: 95% percentile household-cluster bootstrap.

    `reference="male"` or `"female"` takes beta from that group's own OLS
    instead (classic two-fold; unexplained = total - explained). A category
    level absent from that group gets coefficient 0 (min-norm least squares).
    `n_boot=0` skips the bootstrap; CIs are then NaN.
    """
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}")
    groups = _groups(include_occupation)
    mf, dropped = model_frame(df, y, include_occupation)
    yv, Xdf = _design(_formula(y, [t for ts in groups.values() for t in ts]), mf)
    cols = _column_groups(Xdf, groups)
    X = Xdf.to_numpy()
    female = mf["female_head"].to_numpy()

    dummy = Xdf.columns.get_loc("female_head")
    total, contrib = _decompose(X, yv, female, cols, reference, dummy)
    explained = sum(contrib.values())

    rng = np.random.default_rng(seed)
    pos = pd.DataFrame({"id": mf["id"].to_numpy(), "pos": np.arange(len(mf))})
    boot = np.empty((n_boot, 3 + len(groups)))
    for b in range(n_boot):
        idx = cluster_bootstrap_sample(pos, "id", rng)["pos"].to_numpy()
        t, c = _decompose(X[idx], yv[idx], female[idx], cols, reference, dummy)
        e = sum(c.values())
        boot[b] = [t, e, t - e, *c.values()]
    if n_boot > 0:
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    else:
        lo = hi = np.full(3 + len(groups), np.nan)

    names = ["total", "explained", "unexplained"]
    return {
        "total": float(total), "explained": float(explained),
        "unexplained": float(total - explained),
        "contrib": pd.DataFrame({"group": list(groups), "explained": list(contrib.values()),
                                 "ci_low": lo[3:], "ci_high": hi[3:]}),
        "ci": {k: (float(lo[i]), float(hi[i])) for i, k in enumerate(names)},
        "n": len(mf), "n_dropped": dropped, "reference": reference,
    }


def quantile_gap(df: pd.DataFrame, y: str = "log_eq_income",
                 qs=(0.1, 0.25, 0.5, 0.75, 0.9), include_occupation: bool = True,
                 max_iter: int = 1000) -> pd.DataFrame:
    """Female-head coefficient at several quantiles, with all controls.

    CIs are statsmodels QuantReg defaults (sandwich with Hall-Sheather
    bandwidth). They are NOT clustered by household, so with repeated
    households they are likely too narrow.
    """
    groups = _groups(include_occupation)
    mf, _ = model_frame(df, y, include_occupation)
    yv, X = _design(_formula(y, [t for ts in groups.values() for t in ts]), mf)
    rows = []
    for q in qs:
        fit = sm.QuantReg(yv, X).fit(q=q, max_iter=max_iter)
        lo, hi = fit.conf_int().loc["female_head"]
        coef = fit.params["female_head"]
        rows.append({"q": q, "coef": coef, "ci_low": lo, "ci_high": hi,
                     "pct_gap": _pct(coef), "n": len(mf)})
    return pd.DataFrame(rows)
