"""End-to-end check: the story notebook executes cleanly and writes the README figures.

Figures go to a temporary directory (KWINCOME_FIG_DIR), so the tracked PNGs in
reports/figures are never deleted or overwritten by the test.
"""
import nbformat
import pytest
from nbclient import NotebookClient

from kwincome.paths import ROOT

NOTEBOOK = ROOT / "notebooks" / "analysis.ipynb"
FIGURES = ["trend.png", "staged_gap.png", "oaxaca.png", "quantile_gap.png", "region_map.png"]


@pytest.mark.slow
def test_notebook_executes(tmp_path, monkeypatch):
    assert NOTEBOOK.exists(), f"missing {NOTEBOOK}"
    fig_dir = tmp_path / "figures"
    monkeypatch.setenv("KWINCOME_FIG_DIR", str(fig_dir))  # inherited by the kernel
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors[0].get("ename") if errors else ""
    missing = [n for n in FIGURES if not (fig_dir / n).exists()]
    assert not missing, f"figures not written: {missing}"
