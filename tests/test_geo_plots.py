import json

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from kwincome import codes, geo, paths, plots


def test_all_provinces_mapped():
    names = [f["properties"]["shapeName"]
             for f in json.loads(paths.GEOJSON.read_text(encoding="utf-8"))["features"]]
    assert len(names) == 17
    assert set(names) == set(geo.PROVINCE_TO_REGION)
    assert set(geo.PROVINCE_TO_REGION.values()) == set(codes.REGION.values())


def test_region_geodata_dissolves_to_seven_regions():
    vals = pd.DataFrame({"region_label": list(codes.REGION.values()),
                         "median": [1.0, 2, 3, 4, 5, 6, 7]})
    gdf = geo.region_geodata(vals, "median")
    assert len(gdf) == 7
    assert set(gdf["region_label"]) == set(codes.REGION.values())
    assert gdf["median"].notna().all()
    assert gdf.geometry.is_valid.all()


def test_pretty_label():
    assert plots.pretty_label("no capable") == "No capable"
    assert plots.pretty_label("Seoul") == "Seoul"
    assert plots.pretty_label("in military service") == "In military service"


def test_head_colors_fixed():
    assert set(plots.HEAD_COLORS) == {"Male", "Female"}
    assert plots.HEAD_COLORS["Male"] != plots.HEAD_COLORS["Female"]


@pytest.fixture
def fixtures():
    stats = pd.DataFrame({"gender_label": ["Male", "Female", "Male", "Female"],
                          "income_year": [2010, 2010, 2011, 2011],
                          "n": [200, 150, 210, 160], "median": [2000., 1500, 2100, 1550],
                          "ci_low": [1900., 1400, 2000, 1450], "ci_high": [2100., 1600, 2200, 1650]})
    staged = pd.DataFrame({"step": ["Raw", "+ Age", "+ Education"], "coef": [-.5, -.3, -.2],
                           "se": [.05] * 3, "pct_gap": [-39., -26, -18],
                           "ci_low_pct": [-43., -30, -23], "ci_high_pct": [-35., -22, -13],
                           "n": [900] * 3})
    oax = {"total": .4, "explained": .25, "unexplained": .15,
           "contrib": pd.DataFrame({"group": ["Age", "Education", "Region & year"],
                                    "explained": [.1, .12, .03],
                                    "ci_low": [.05, .07, 0.], "ci_high": [.15, .17, .06]}),
           "ci": {"total": (.35, .45), "explained": (.2, .3), "unexplained": (.1, .2)},
           "n": 900, "n_dropped": 3}
    quant = pd.DataFrame({"q": [.1, .25, .5, .75, .9], "coef": [-.4, -.3, -.25, -.2, -.1],
                          "ci_low": [-.5, -.35, -.3, -.25, -.2], "ci_high": [-.3, -.25, -.2, -.15, 0.],
                          "pct_gap": [-33., -26, -22, -18, -10]})
    return stats, staged, oax, quant


def test_plots_return_figures_and_save(fixtures, monkeypatch, tmp_path):
    stats, staged, oax, quant = fixtures
    plots.set_style()
    cat = stats.iloc[:2]
    vals = pd.DataFrame({"region_label": list(codes.REGION.values()), "median": range(1, 8)})
    gdf = geo.region_geodata(vals, "median")
    figs = [
        plots.plot_ci_bars(cat, "gender_label", "Title", "10k KRW"),
        plots.plot_trend(stats, "income_year", "gender_label", "Title", "10k KRW"),
        plots.plot_trend(stats[stats.gender_label == "Male"], "income_year", None, "T", "y"),
        plots.plot_staged(staged),
        plots.plot_oaxaca(oax),
        plots.plot_quantiles(quant),
        plots.plot_choropleth_static(gdf, "median", "Title"),
    ]
    assert all(isinstance(f, Figure) for f in figs)
    monkeypatch.setattr(plots, "FIG_DIR", tmp_path / "figs")
    out = plots.save(figs[0], "demo")
    assert out == tmp_path / "figs" / "demo.png" and out.stat().st_size > 1000
    for f in figs:
        plots.plt.close(f)


def test_folium_map_has_tooltip_values():
    import folium
    vals = pd.DataFrame({"region_label": list(codes.REGION.values()), "median": range(1, 8)})
    m = plots.choropleth_folium(geo.region_geodata(vals, "median"), "median")
    assert isinstance(m, folium.Map)
    html = m.get_root().render()
    assert "Jeolla" in html and "region_label" in html


def test_oaxaca_caption_and_no_secondary_axis(fixtures):
    _, _, oax, _ = fixtures
    fig = plots.plot_oaxaca(oax)
    ax = fig.axes[0]
    text = " ".join(t.get_text() for t in ax.texts)
    assert "of the log gap" in text
    assert f"{.25 / .4:.0%}" in text and f"{.15 / .4:.0%}" in text  # computed from the dict
    assert len(fig.axes) == 1 and not ax.child_axes
    plots.plt.close(fig)


def test_folium_label_and_nan():
    vals = pd.DataFrame({"region_label": list(codes.REGION.values()),
                         "median": [1.0, 2, 3, 4, 5, 6, float("nan")]})
    m = plots.choropleth_folium(geo.region_geodata(vals, "median"), "median", label="Median (10k KRW)")
    assert "Median (10k KRW)" in m.get_root().render()
