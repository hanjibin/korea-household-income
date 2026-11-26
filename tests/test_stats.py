import numpy as np
import pandas as pd

from kwincome.stats import cluster_bootstrap_sample, median_ci_by_group


def _frame(n_a=101, n_b=150, seed=0):
    rng = np.random.default_rng(seed)
    a = pd.DataFrame({"id": np.arange(n_a), "g": "A", "v": np.arange(1, n_a + 1, dtype=float)})
    b = pd.DataFrame({"id": np.arange(1000, 1000 + n_b), "g": "B", "v": rng.normal(50, 10, n_b)})
    return pd.concat([a, b], ignore_index=True)


def test_median_exact():
    out = median_ci_by_group(_frame(), "g", "v", n_boot=50)
    assert list(out.columns) == ["g", "n", "median", "ci_low", "ci_high"]
    row = out[out["g"] == "A"].iloc[0]
    assert row["median"] == 51 and row["n"] == 101


def test_min_n_suppresses():
    df = pd.concat([_frame(), pd.DataFrame({"id": np.arange(5000, 5099), "g": "C", "v": 1.0})],
                   ignore_index=True)
    out = median_ci_by_group(df, "g", "v", n_boot=20)
    assert "C" not in set(out["g"]) and {"A", "B"} == set(out["g"])


def test_ci_contains_median_and_is_deterministic():
    df = _frame()
    o1 = median_ci_by_group(df, "g", "v", n_boot=100, seed=7)
    o2 = median_ci_by_group(df, "g", "v", n_boot=100, seed=7)
    pd.testing.assert_frame_equal(o1, o2)
    assert ((o1["ci_low"] <= o1["median"]) & (o1["median"] <= o1["ci_high"])).all()
    assert (o1["ci_low"] < o1["ci_high"]).all()


def test_cluster_resampling_keeps_households_whole():
    df = pd.DataFrame({"id": [1, 1, 1, 2, 3, 3], "x": range(6)})
    for seed in range(20):
        s = cluster_bootstrap_sample(df, "id", np.random.default_rng(seed))
        counts = s["id"].value_counts()
        assert counts.get(1, 0) % 3 == 0 and counts.get(3, 0) % 2 == 0
        # each draw is a whole household: boot_cluster sizes equal that household's row count
        sizes = s.groupby("boot_cluster")["id"].agg(["size", "nunique"])
        assert (sizes["nunique"] == 1).all()
        orig = df["id"].value_counts()
        by_draw = s.groupby("boot_cluster")["id"].first()
        assert (sizes["size"].to_numpy() == orig[by_draw].to_numpy()).all()
        assert s["boot_cluster"].nunique() == df["id"].nunique()


def test_multi_group_nan_value_and_nan_key():
    df = _frame()
    df["h"] = np.where(df["id"] % 2 == 0, "x", "y")
    df.loc[0, "v"] = np.nan
    df.loc[1, "g"] = np.nan
    out = median_ci_by_group(df, ["g", "h"], "v", n_boot=20, min_n=10)
    assert list(out.columns) == ["g", "h", "n", "median", "ci_low", "ci_high"]
    assert out["g"].notna().all()
    assert out["n"].sum() == len(df) - 2
