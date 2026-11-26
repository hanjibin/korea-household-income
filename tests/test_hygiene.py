"""Repository hygiene: no personal or school traces, writing rules in user-facing text, provenance."""
import json
import os
import subprocess

import pytest

from kwincome.paths import ROOT

# Banned strings are built from parts so this file does not match its own scan.
BANNED = ["syr" + ".edu", "jhan" + "70", "cass" + "tools"]
EM_DASH = "—"
EM_DASH_FILES = ["README.md", "docs/analysis_decisions.md"]
PROVENANCE = ["Extended from my 2025 IST256", "AI assistance"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def _tracked_text_files():
    for rel in _git("ls-files").splitlines():
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            yield rel, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary file (png etc.)


def test_no_banned_strings_in_tracked_files():
    hits = [(rel, word) for rel, text in _tracked_text_files()
            for word in BANNED if word in text.lower()]
    assert not hits, hits


# Identity checks depend on the local git config and full history, so they run
# only on request before publishing: set KWINCOME_PUBLISH_CHECK=1.
publish_check = pytest.mark.skipif(
    os.environ.get("KWINCOME_PUBLISH_CHECK") != "1",
    reason="set KWINCOME_PUBLISH_CHECK=1 to check git identity before publishing")


@publish_check
def test_git_email_is_noreply():
    assert _git("config", "user.email").strip().endswith("@users.noreply.github.com")


@publish_check
def test_all_commit_emails_are_noreply():
    emails = set(_git("log", "--format=%ae%n%ce").split())
    bad = sorted(e for e in emails if not e.endswith("@users.noreply.github.com"))
    assert emails and not bad, bad


@pytest.mark.parametrize("rel", EM_DASH_FILES)
def test_no_em_dash_in_user_facing_docs(rel):
    path = ROOT / rel
    assert path.exists(), f"missing {rel}"
    assert EM_DASH not in path.read_text(encoding="utf-8")


def test_no_em_dash_in_notebook_markdown():
    nb = json.loads((ROOT / "notebooks" / "analysis.ipynb").read_text(encoding="utf-8"))
    bad = [i for i, c in enumerate(nb["cells"])
           if c["cell_type"] == "markdown" and EM_DASH in "".join(c["source"])]
    assert not bad, f"markdown cells with an em dash: {bad}"


def test_provenance_present():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [p for p in PROVENANCE if p not in text]
    assert not missing, missing
