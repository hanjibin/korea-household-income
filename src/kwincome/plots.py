"""Figures for the README: one palette, fixed Male/Female colors, findings in titles."""
from pathlib import Path

import branca.colormap as cm
import folium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .paths import FIG_DIR

# Okabe-Ito colorblind-safe palette.
BLUE, VERMILLION, GREEN, PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
GREY, INK, GRID = "#6B6B6B", "#222222", "#DDDDDD"
HEAD_COLORS = {"Male": BLUE, "Female": VERMILLION}
# Blue and vermillion are reserved for male/female heads; other series start at CYCLE[2:].
CYCLE = [GREEN, PINK, "#E69F00", "#56B4E9", "#000000"]
NEUTRAL = "#4D4D4D"
EXPLAINED_COLOR, UNEXPLAINED_COLOR = GREEN, PINK
SEQ_CMAP = "YlGnBu"
INCOME_LABEL = "10k KRW per year, 2018 prices, estimated"


def pretty_label(s: str) -> str:
    """Sentence-case a raw category label: "no capable" -> "No capable"."""
    s = str(s)
    return s[:1].upper() + s[1:]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK, "axes.edgecolor": GREY, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "legend.frameon": False, "figure.facecolor": "white",
    })


def _color(level, i: int) -> str:
    return NEUTRAL if level is None else HEAD_COLORS.get(str(level), CYCLE[i % len(CYCLE)])


def _pct(x):
    return (np.exp(x) - 1) * 100


def _caption(ax, text: str, dy: float = 0.16) -> None:
    ax.text(0, -dy, text, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=GREY)


def plot_ci_bars(stats_df, x: str, title: str, xlabel: str) -> Figure:
    """Horizontal medians with 95% CI whiskers; `x` is the group column, `xlabel` the value axis."""
    d = stats_df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(d) + 1.8))
    ypos = np.arange(len(d))
    colors = [HEAD_COLORS.get(str(v), NEUTRAL) for v in d[x]]
    ax.barh(ypos, d["median"], color=colors, height=0.6, alpha=0.9)
    err = np.vstack([d["median"] - d["ci_low"], d["ci_high"] - d["median"]])
    ax.errorbar(d["median"], ypos, xerr=err, fmt="none", ecolor=INK, capsize=3, lw=1.2)
    ax.set_yticks(ypos, [pretty_label(v) for v in d[x]])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for yi, (m, h, n) in enumerate(zip(d["median"], d["ci_high"], d["n"])):
        ax.text(h, yi, f"  {m:,.0f} (n={n:,})", va="center", fontsize=8, color=GREY)
    ax.set_xlim(0, d["ci_high"].max() * 1.3)
    _caption(ax, "Bar: median. Whisker: 95% household-cluster bootstrap CI.", 0.13)
    fig.tight_layout()
    return fig


def plot_trend(stats_df, x: str, hue: str | None, title: str, ylabel: str) -> Figure:
    """Median line with markers and a shaded 95% CI band, one line per `hue` level."""
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    groups = [(None, stats_df)] if hue is None else list(stats_df.groupby(hue, sort=False))
    for i, (level, g) in enumerate(groups):
        g = g.sort_values(x)
        c = _color(level, i)
        ax.fill_between(g[x], g["ci_low"], g["ci_high"], color=c, alpha=0.18, lw=0)
        ax.plot(g[x], g["median"], marker="o", ms=4, lw=2, color=c,
                label=None if level is None else pretty_label(level))
    ax.set_xlabel(x.replace("_", " ").capitalize())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hue is not None:
        ax.legend(loc="best")
    _caption(ax, "Line: median. Band: 95% household-cluster bootstrap CI.")
    fig.tight_layout()
    return fig


def plot_staged(staged_df, title: str = "Female-headed household gap as controls are added") -> Figure:
    """Female-head gap (%) at each cumulative control step with 95% CI whiskers."""
    d = staged_df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(d) + 2.0))
    ypos = np.arange(len(d))
    ax.barh(ypos, d["pct_gap"], color=VERMILLION, height=0.6, alpha=0.9)
    err = np.vstack([d["pct_gap"] - d["ci_low_pct"], d["ci_high_pct"] - d["pct_gap"]])
    ax.errorbar(d["pct_gap"], ypos, xerr=err, fmt="none", ecolor=INK, capsize=3, lw=1.2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(ypos, d["step"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, (v, lo) in enumerate(zip(d["pct_gap"], d["ci_low_pct"])):
        ax.text(lo - 0.5, yi, f"{v:.0f}%", ha="right", va="center", fontsize=8, color=INK)
    ax.margins(x=0.12)
    ax.set_xlabel("Income gap, female- vs male-headed households (%, estimated)")
    ax.set_title(title)
    _caption(ax, "Whisker: 95% CI, household-clustered standard errors.", 0.14)
    fig.tight_layout()
    return fig


def plot_oaxaca(result: dict,
                title: str = "How much of the gap is explained by observed characteristics") -> Figure:
    """Explained contributions by group (with CIs) and the unexplained part, in log points and %."""
    c = result["contrib"].reset_index(drop=True)
    labels = list(c["group"]) + ["Unexplained"]
    vals = np.array(list(c["explained"]) + [result["unexplained"]])
    lo = np.array(list(c["ci_low"]) + [result["ci"]["unexplained"][0]])
    hi = np.array(list(c["ci_high"]) + [result["ci"]["unexplained"][1]])
    colors = [EXPLAINED_COLOR] * len(c) + [UNEXPLAINED_COLOR]
    fig, ax = plt.subplots(figsize=(7.8, 0.45 * len(labels) + 2.6))
    ypos = np.arange(len(labels))
    ax.barh(ypos, vals, color=colors, height=0.6, alpha=0.9)
    ax.errorbar(vals, ypos, xerr=np.vstack([vals - lo, hi - vals]), fmt="none",
                ecolor=INK, capsize=3, lw=1.2)
    ax.axvline(0, color=INK, lw=1)
    ax.axhline(len(c) - 0.5, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribution to the gap (log points; positive = male-headed households higher)")
    t, e, u = result["total"], result["explained"], result["unexplained"]
    for yi, (v, h) in enumerate(zip(vals, hi)):
        ax.text(max(h, v) + 0.004, yi, f"{v / t:.0%}".replace("-0%", "0%"), va="center", fontsize=8.5,
                fontweight="bold", color=INK)
    ax.margins(x=0.1)
    ax.set_title(title)
    _caption(ax, f"Total gap: {t:.2f} log points (male-headed households' income about "
                 f"{_pct(t):.0f}% higher, exp({t:.2f})-1).\n"
                 f"Explained {e:.2f} ({e / t:.0%} of the log gap), unexplained {u:.2f} "
                 f"({u / t:.0%} of the log gap).\n"
                 f"Bar-end labels: share of the log gap, not % income difference. "
                 f"Whisker: 95% CI.\nn = {result['n']:,}; {result['n_dropped']:,} rows dropped "
                 f"(income <= 0).", 0.2)
    fig.tight_layout()
    return fig


def plot_quantiles(q_df,
                   title: str = "Female-headed household gap across the income distribution") -> Figure:
    """Gap (%) at each conditional quantile (quantile regression) with a 95% CI band."""
    d = q_df.sort_values("q")
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.fill_between(d["q"], _pct(d["ci_low"]), _pct(d["ci_high"]), color=VERMILLION,
                    alpha=0.2, lw=0)
    ax.plot(d["q"], d["pct_gap"], marker="o", lw=2, color=VERMILLION)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(d["q"], [f"{q:.0%}" for q in d["q"]])
    ax.set_xlabel("Conditional quantile of log real equivalized income, given controls (low to high)")
    ax.set_ylabel("Gap, female- vs male-headed (%, estimated)")
    ax.set_title(title)
    _caption(ax, "Gap at each conditional quantile (given all controls). Band: 95% CI, not clustered\n"
                 "by household, so likely too narrow.", 0.2)
    fig.tight_layout()
    return fig


def plot_choropleth_static(gdf, value_col: str, title: str, label: str = INCOME_LABEL) -> Figure:
    """Region map shaded by `value_col`, each region annotated with its value."""
    fig, ax = plt.subplots(figsize=(6.4, 7))
    gdf.plot(column=value_col, cmap=SEQ_CMAP, edgecolor="white", linewidth=1, ax=ax, legend=True,
             legend_kwds={"label": label, "shrink": 0.6}, missing_kwds={"color": GRID})
    # Seoul and Gyeonggi are adjacent and small: label them in the sea with leader lines.
    outside = {"Seoul": (125.2, 38.0), "Gyeonggi": (125.2, 36.9)}
    for _, r in gdf.iterrows():
        if np.isfinite(r[value_col]):
            name = r["region_label"]
            p = r.geometry.representative_point()
            xy = (127.2, 37.25) if name == "Gyeonggi" else (p.x, p.y)
            kw = dict(arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8)) if name in outside else {}
            ax.annotate(f"{name}\n{r[value_col]:,.0f}", xy, xytext=outside.get(name),
                        ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), **kw)
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def choropleth_folium(gdf, value_col: str, label: str | None = None) -> folium.Map:
    """Interactive map; tooltip shows region and value."""
    vals = gdf[value_col].dropna()
    cmap = cm.linear.YlGnBu_09.scale(vals.min(), vals.max())
    label = label or value_col
    cmap.caption = label
    m = folium.Map(location=[36.3, 127.8], zoom_start=7, tiles="OpenStreetMap")

    def style(f):
        v = f["properties"][value_col]
        return {"fillColor": cmap(v) if v is not None and np.isfinite(v) else "#DDDDDD",
                "color": "white", "weight": 1, "fillOpacity": 0.8}

    folium.GeoJson(
        gdf[["region_label", value_col, "geometry"]], style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["region_label", value_col],
                                      aliases=["Region", label], localize=True),
    ).add_to(m)
    cmap.add_to(m)
    return m


def save(fig: Figure, name: str) -> Path:
    """Write the figure as PNG (200 dpi) into FIG_DIR."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return out
