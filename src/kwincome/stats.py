"""Group medians with household-cluster bootstrap confidence intervals."""
import numpy as np
import pandas as pd


class _Clusters:
    """Row indices grouped by cluster, for fast whole-cluster resampling."""

    def __init__(self, labels: pd.Series):
        codes, uniques = pd.factorize(labels)
        self.n_clusters = len(uniques)
        self.order = np.argsort(codes, kind="stable")
        self.counts = np.bincount(codes, minlength=self.n_clusters)
        self.starts = np.cumsum(self.counts) - self.counts

    def draw(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Return (row positions, boot_cluster ids) for one bootstrap draw."""
        picked = rng.integers(0, self.n_clusters, self.n_clusters)
        lens = self.counts[picked]
        draw_id = np.repeat(np.arange(self.n_clusters), lens)
        offset = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        return self.order[np.repeat(self.starts[picked], lens) + offset], draw_id


def cluster_bootstrap_sample(df: pd.DataFrame, cluster: str,
                             rng: np.random.Generator) -> pd.DataFrame:
    """Resample whole clusters with replacement.

    Draws as many cluster ids as there are unique clusters; every row of each
    drawn cluster is kept. A cluster drawn twice appears twice, told apart by
    the new column `boot_cluster` (unique per draw).
    """
    pos, draw_id = _Clusters(df[cluster]).draw(rng)
    out = df.iloc[pos].reset_index(drop=True)
    out["boot_cluster"] = draw_id
    return out


def median_ci_by_group(df: pd.DataFrame, group: str | list[str], value: str,
                       cluster: str = "id", n_boot: int = 1000, min_n: int = 100,
                       seed: int = 42) -> pd.DataFrame:
    """Median of `value` per group with a 95% percentile cluster-bootstrap CI.

    Rows with NaN `value` or NaN group keys are dropped. Groups with fewer than
    `min_n` rows are suppressed.
    """
    keys = [group] if isinstance(group, str) else list(group)
    d = df[[*keys, value, cluster]].dropna(subset=[*keys, value]).reset_index(drop=True)
    grouped = d.groupby(keys, sort=True, observed=True)
    summary = grouped[value].agg(n="size", median="median").reset_index()
    d["_gid"] = grouped.ngroup().to_numpy()
    keep = summary["n"].to_numpy() >= min_n
    summary = summary[keep].reset_index(drop=True)
    if summary.empty:
        return summary.assign(ci_low=np.nan, ci_high=np.nan)

    n_groups = int(d["_gid"].max()) + 1
    kept_ids = np.flatnonzero(keep)
    slim = d[[cluster, "_gid", value]]
    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, n_groups), np.nan)
    for b in range(n_boot):
        s = cluster_bootstrap_sample(slim, cluster, rng)
        boot[b] = s.groupby("_gid")[value].median().reindex(range(n_groups)).to_numpy()
    lo, hi = np.nanpercentile(boot[:, kept_ids], [2.5, 97.5], axis=0)
    summary["ci_low"], summary["ci_high"] = lo, hi
    return summary
