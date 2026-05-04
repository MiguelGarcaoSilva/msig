"""
Regression tests that compare regenerated experiment summary CSVs against
checked-in golden snapshots in tests/golden/. The snapshots are populated
by the validation step of plan task 44 (Section 8 of the spec).

Marked @pytest.mark.slow; opt in with `pytest -m slow`.
"""

import os
import pytest
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


CASES = [
    ("audio", "stumpy", "stumpy_relaxed", "summary_motifs_stumpy_relaxed.csv"),
    ("audio", "stumpy", "stumpy_moderate", "summary_motifs_stumpy_moderate.csv"),
    ("audio", "stumpy", "stumpy_conservative", "summary_motifs_stumpy_conservative.csv"),
    ("audio", "lama_iterative", None, "summary_motifs_lama_iterative.csv"),
    ("audio", "momenti", None, "summary_motifs_momenti.csv"),
    ("populationdensity", "stumpy", "stumpy_relaxed", "summary_motifs_stumpy_relaxed.csv"),
    ("populationdensity", "stumpy", "stumpy_moderate", "summary_motifs_stumpy_moderate.csv"),
    ("populationdensity", "stumpy", "stumpy_conservative", "summary_motifs_stumpy_conservative.csv"),
    ("populationdensity", "lama_iterative", None, "summary_motifs_lama_iterative.csv"),
    ("populationdensity", "momenti", None, "summary_motifs_momenti.csv"),
    ("washingmachine", "stumpy", "stumpy_relaxed", "summary_motifs_stumpy_relaxed.csv"),
    ("washingmachine", "stumpy", "stumpy_moderate", "summary_motifs_stumpy_moderate.csv"),
    ("washingmachine", "stumpy", "stumpy_conservative", "summary_motifs_stumpy_conservative.csv"),
    ("washingmachine", "lama_iterative", None, "summary_motifs_lama_iterative.csv"),
    ("washingmachine", "momenti", None, "summary_motifs_momenti.csv"),
]


@pytest.mark.slow
@pytest.mark.parametrize("dataset,method_dir,sub_dir,filename", CASES)
def test_summary_csv_matches_golden(dataset, method_dir, sub_dir, filename):
    """Each summary_motifs_*.csv must match its golden snapshot."""
    if sub_dir:
        result_path = os.path.join(REPO_ROOT, "results", dataset, method_dir, sub_dir, filename)
    else:
        result_path = os.path.join(REPO_ROOT, "results", dataset, method_dir, filename)

    golden_name = f"{dataset}_{method_dir}_{sub_dir or 'main'}_{filename}"
    golden_path = os.path.join(REPO_ROOT, "tests", "golden", golden_name)

    if not os.path.exists(golden_path):
        pytest.skip(f"Golden missing: {golden_path}. Regenerate via the validation step.")
    if not os.path.exists(result_path):
        pytest.skip(f"Result missing: {result_path}. Run the experiment first.")

    actual = pd.read_csv(result_path)
    expected = pd.read_csv(golden_path)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=False,
        rtol=1e-6,
        atol=1e-12,
    )
