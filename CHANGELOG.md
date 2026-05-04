# Changelog

All notable changes to MSig are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres
to [Semantic Versioning](https://semver.org/).

## [0.2.1] — 2026-05-04

### Changed

- **README polish for PyPI presentation.** Added PyPI / Python-versions /
  License / Downloads badges at the top, lifted the citation link
  above-the-fold, surfaced `CHANGELOG.md`, `REPRODUCING_EXPERIMENTS.md`,
  and `CONTRIBUTING.md` in the project description, and tightened the
  one-line summary to mention the *Pattern Recognition Letters* (2026)
  paper. No code changes; same wheel functionality as 0.2.0.

## [0.2.0] — 2026-05-04

### Fixed

- **2D Gaussian rectangle CDF** — `vars_indep_time_markov` in the
  `gaussian_theoretical` branch now uses the inclusion-exclusion formula
  `F(b1,b2) − F(a1,b2) − F(b1,a2) + F(a1,a2)`. Affects every `gaussian_theoretical`
  result. Empirical and KDE branches are unchanged. None of the published
  case studies use `gaussian_theoretical`, so paper tables are not affected.
- **Empirical conditional consistent normalization** — numerator and
  denominator both use the lag-1 marginal over `n−1` transition pairs.
  Removes a `n/(n−1)` bias that previously triggered the `cond_p > 1` clamp
  on small series.
- **KDE conditional uses lag-1 marginal** — denominator now integrates
  `gaussian_kde(y[:-1])` for theoretical consistency with the bivariate
  KDE numerator.
- **`set_significance` always sets `self.pvalue`** — every return path
  (including `p_Q ∈ {0, 1}` and `n_matches ≥ max_possible_matches`) now
  populates the attribute, eliminating discrepancy between return value
  and stored attribute.
- **Three momenti scripts** — `NameError` on undefined `m` (now `s`) and
  wrong relative data paths (`../data/` → `../../data/`). The MOMENTI
  experiments now run.
- **`scripts/compare_results.py` schema** — reads the actual CSV columns
  produced by experiment scripts (`median_pvalue`, `median_probability`,
  `#sig_motifs(≤0.01)`, `#sig_hochberg`, `significant` as percentage)
  rather than the never-implemented `pvalue`/`significant`/`pattern_probability` columns.
- **Population-density MOMENTI** — adds `s = 48` to motif lengths,
  matching paper Table 4 and the LAMA script.
- **Logger consistency** — `logging.info` → `logger.info` in `MSig.py`.
- **`__init__.py` version drift** — `__version__` reads from
  `importlib.metadata.version("msig")`. No more manual sync with
  `pyproject.toml`.
- **`vars_dep_time_markov` docstring** — cites paper Eq. (4) for the
  unimplemented variable-dependent null-model formula.

### Added

- **`pattern_prob_floor` kwarg** on `Motif.set_significance` — opt-in
  Laplace floor for `p_Q = 0` cases (zero-frequency problem). Default
  `None` preserves 0.1.x behaviour.
- **`EXCLUSION_ZONE_FACTOR` constant** in every experiment script and
  `experiments/common_utils.py` — surfaces the trivial-match factor
  (default `0.5`, matching published tables; paper §3.2 default is `0.25`).
- **`AVERAGE_DELTA` module constant** — replaces the per-script
  `average_delta = 0.3` hardcode.
- **`experiments.common_utils.get_dataset_paths`** — single-source helper
  for resolving data and results paths regardless of cwd; adopted by all
  nine experiment scripts.
- **`tests/golden/`** — paper-reproducibility regression suite locks in
  the CSV outputs so future drift is caught.
- **`CITATION.cff`**, **`CONTRIBUTING.md`**, **`REPRODUCING_EXPERIMENTS.md`** — accessibility documentation.
- **`msig/py.typed`** — PEP 561 marker enabling downstream type checking.
- **Validation that `δ = 0` is rejected** for `kde` and `gaussian_theoretical`
  null models (silently produced `p_Q = 0` before).
- **Private helpers `_rect_prob_1d` and `_rect_prob_2d`** in `msig/MSig.py`
  centralise the rectangle-probability computation across model branches.

### Changed

- **`bonferroni_correction(n_tests, alpha=0.05)`** — first parameter renamed.
  Backward-compatible: still accepts iterables for 0.1.x callers.
- **`idd_correction` documentation** — corrected to "identically-distributed
  dimensions" (was wrongly described as "Independent Dimension Discovery"
  applying BH FDR).
- **mSTUMP "conservative" regime comment** clarifies that the formula is
  `D_max = √s · δ · 0.5` (which produced the published tables), not the
  paper's `D_max = (1/q) Σ δ_j` (no `√s` scaling).
- **Python support range** aligned to `>=3.11,<3.14` across `pyproject.toml`,
  `environment.yml`, `requirements.txt`, classifiers, mypy, and black.
- **`bonferroni_correction` and `benjamini_hochberg_fdr` test coverage** —
  new canonical Benjamini & Hochberg (1995) §3 example replaces a
  non-standard "returns α when nothing significant" assertion.

### Removed

- **`PYPI_DESCRIPTION.md`** — duplicate of `README.md`; never reached PyPI
  (`pyproject.toml` uses `readme = "README.md"`). Removed to eliminate
  drift between the two.
- **Dead `OverflowError` fallback** in `set_significance` — `scipy.stats.binom.sf`
  underflows silently and never raises this exception.

### Migration from 0.1.x

- `bonferroni_correction([0.1, 0.2, 0.3])` continues to work; new code
  should pass `bonferroni_correction(3)` directly.
- If any user code relied on `gaussian_theoretical` p-values, expect
  numerically different results after the inclusion-exclusion fix.
- If any user code relied on inspecting `motif.pvalue` after the
  `p_Q ∈ {0, 1}` edge cases, the attribute is now always correct.

## [0.1.3] — 2025-10-27
- Hotfix release.

## [0.1.2] — 2025-10-26
- Maintenance release.

## [0.1.1] — 2024-07-03
- Initial PyPI publication; corresponds to paper-submission code.

## [0.1.0] — 2024-07-03
- First release.
