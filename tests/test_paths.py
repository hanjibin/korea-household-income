import importlib

from kwincome import paths
from kwincome.paths import RAW_CSV, JOB_XLSX


def test_raw_files_exist():
    assert RAW_CSV.exists()
    assert JOB_XLSX.exists()
    assert RAW_CSV.name == "Korea Income and Welfare.csv"


def test_fig_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KWINCOME_FIG_DIR", str(tmp_path))
    assert importlib.reload(paths).FIG_DIR == tmp_path
    monkeypatch.delenv("KWINCOME_FIG_DIR")
    assert importlib.reload(paths).FIG_DIR == paths.ROOT / "reports" / "figures"
