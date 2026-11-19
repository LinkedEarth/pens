"""Smoke test run INSIDE the freshly built container image by repo2docker-action
(see .github/workflows/build-container.yml): every notebook in docsrc/notebooks
must execute without error, otherwise the image is not pushed."""
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parents[1] / "docsrc" / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("eg25-*.ipynb"))


def test_notebooks_found():
    assert NOTEBOOKS, f"no notebooks found in {NB_DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(NB_DIR)}})
    client.execute()
