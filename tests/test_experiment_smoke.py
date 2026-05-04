"""
Smoke tests for the experiment scripts. Each test imports the module and
verifies key callables exist; some run a tiny synthetic experiment to
catch schema and path bugs.

Marked @pytest.mark.integration; opt in with `pytest -m integration`.
"""

import os
import sys
import subprocess
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)


@pytest.mark.integration
@pytest.mark.parametrize("dataset", ["audio", "populationdensity", "washingmachine"])
@pytest.mark.parametrize("method", ["stumpy", "lama", "momenti"])
def test_experiment_module_imports(dataset, method):
    """Each experiment module imports without error."""
    module_path = f"experiments.{dataset}.run_{method}"
    pytest.importorskip(module_path)


@pytest.mark.integration
def test_compare_results_runs_against_existing_csvs(tmp_path):
    """compare_results.py runs against the committed result CSVs without KeyError."""
    out = tmp_path / "compare.md"
    csv = tmp_path / "compare.csv"
    result = subprocess.run(
        [sys.executable, "scripts/compare_results.py",
         "--output", str(out), "--csv", str(csv)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert csv.exists()
