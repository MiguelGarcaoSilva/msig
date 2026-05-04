# MSig 0.2.0 — Deep Revision Design

**Date:** 2026-05-04
**Author:** Miguel G. Silva (with Claude as collaborator)
**Status:** Draft for review
**Target release:** `msig 0.2.0`

## Background

MSig is a Python library for statistical-significance testing of multivariate time series motifs, accompanying the paper *On Why and How Statistical Significance Criteria Can Guide Multivariate Time Series Motif Analysis* (Silva, Madeira, Henriques; *Pattern Recognition Letters*, 2026, Elsevier). The paper has just been published; the repository is currently at `0.1.3` on PyPI (Oct 2025) and needs a comprehensive revision pass to ensure correctness, accessibility, and reproducibility.

This design captures findings from a multi-agent review (one initial reviewer plus six independent verification agents) and the path to addressing them in a single coordinated `0.2.0` release.

## Goals

1. **Correctness.** Fix the bugs in the library that affect numeric output (notably the 2D-Gaussian rectangle CDF and the empirical conditional's inconsistent normalisation). Fix bugs in the experiment scripts that prevent execution (`NameError`, wrong relative paths).
2. **Accessibility.** Ensure that a researcher cloning the repo can install, run the examples, run the experiments end-to-end, and read documentation that matches the code.
3. **Reproducibility.** Document precisely which code revision produced the paper's published tables, and document expected divergence between the regenerated `results/` and the paper after correctness fixes are applied.
4. **API hygiene.** One backward-compatible signature change to `bonferroni_correction`; one opt-in feature for `set_significance` (`pattern_prob_floor`); no other breaking changes.

## Non-goals

- Filing a paper erratum with PRL. (One paper-text vs. paper-tables inconsistency around `l = ⌈s/4⌉` vs `l = ⌈s/2⌉` is documented in `REPRODUCING_EXPERIMENTS.md` rather than corrected upstream.)
- Implementing variable-dependent null models (`vars_dep_time_markov` remains a stub citing paper Eq. (4)).
- Adding higher-order Markov support.
- Re-running experiments to chase numerical exactness with the published tables (tolerance is acceptable; `tests/golden/` snapshots fix forward).

## Versioning and migration

- Bump to `0.2.0`. Single source of truth: `pyproject.toml`. `msig/__init__.py` reads from `importlib.metadata.version("msig")` to eliminate version drift.
- Backward-compatible signature changes only:
  - `bonferroni_correction(n_tests: int | Iterable[float], alpha=0.05)`. Old iterable callers still work.
  - `set_significance(..., pattern_prob_floor: float | None = None)`. Default preserves 0.1.x numerics.
- `CHANGELOG.md` documents every behaviour-affecting change (the 2D-Gaussian fix, the empirical/KDE conditional fixes, edge-case `self.pvalue`, etc.) so 0.1.x users have a migration note.

## Architecture changes

### Library (`msig/MSig.py`)

| Change | Bug ref | Description |
|---|---|---|
| 2D Gaussian rectangle CDF inclusion-exclusion | #1 | Extract `_rect_prob_2d(dist, lo, hi) = max(0, F(b1,b2) − F(a1,b2) − F(b1,a2) + F(a1,a2))`. Use everywhere `dist_bivar.cdf(...)` currently appears. |
| Extract rectangle helpers | structural | `_rect_prob_1d(dist, lo, hi, model)` and `_rect_prob_2d(dist_bivar, lo, hi, model)` private helpers, dispatching on `empirical | kde | gaussian_theoretical`. Removes the four duplicated if/elif trees in `vars_indep_time_markov`. |
| Empirical conditional consistent normalisation | #4 | Both numerator and denominator use the lag-1 marginal: `count_pair / (n−1)` and `count_marginal_lag1 / (n−1)`. The `cond_p > 1.0` clamp at line 358 becomes a defensive `assert` rather than a numerical bandage. |
| KDE conditional uses lag-1 marginal | #24 | Replace univariate KDE in denominator with the bivariate KDE marginalised onto the first coordinate (`gaussian_kde(pairs[0])` cached in `pre_computed_lag1_marginal`). |
| `gaussian_theoretical` and `kde` reject `δ = 0` | #13 | Raise `ValueError("delta must be > 0 for continuous null models; use model='empirical' for exact matching")` on first call into `vars_indep_time_markov` if `delta == 0` for any variable in the motif. |
| Edge-case returns set `self.pvalue` | #14 | Refactor `set_significance` so every path goes through one final `self.pvalue = pvalue; return pvalue` block. |
| Laplace smoothing of zero pattern probability | #37 | New kwarg `pattern_prob_floor: float | None = None`. When `p_Q == 0` and floor set, replace with floor before binomial-tail computation. Default `None` preserves 0.1.x behaviour. |
| `bonferroni_correction` signature | #35 | First arg renamed to `n_tests`; accepts `int | Iterable[float]` with `if not isinstance(n_tests, int): n_tests = sum(1 for _ in n_tests)`. Old call-sites continue to work. |
| `idd_correction` docstring | #5 | Rewrite — expand "IDD" as "identically-distributed dimensions correction"; remove false BH-FDR claim; cite paper §3.2. |
| Logger consistency | #36 | `logging.info` → `logger.info` at line 627. |
| Dead-code removal | #25 | Remove `OverflowError` fallback in `set_significance` (`scipy.stats.binom.sf` underflows to 0, never raises). |
| Public API stability | — | `__all__` and signatures of `Motif`, `NullModel`, `benjamini_hochberg_fdr`, `bonferroni_correction` preserved (modulo the `n_tests` rename which is BC). |
| `vars_dep_time_markov` honest stub | minor | Keep `NotImplementedError` but cite paper Eq. (4) in the docstring. |

### Experiment scripts (`experiments/`, `scripts/`, `run_experiments.py`)

| Change | Bug ref | Description |
|---|---|---|
| Three momenti scripts have `NameError` on undefined `m` | #9 | Replace `"s": m,` with `"s": s,` in `audio/run_momenti.py:187`, `washingmachine/:179`, `populationdensity/:177`. |
| Three momenti scripts have wrong relative paths | #10 | `"../data/..."` → `"../../data/..."` in all three momenti `main()`s. |
| Path derivation centralised | #10 follow-up | Move per-dataset path derivation into `experiments/common_utils.py:get_dataset_paths(dataset)` so the three method-families don't drift again. |
| `compare_results.py` schema mismatch | #12 | Rewrite the loader and report sections to match the columns the scripts actually save (`median_pvalue`, `median_probability`, `#sig_motifs(≤0.01)`, `#sig_hochberg`, `significant` as percentage). Add a docstring at the top stating the expected schema and version it. |
| Trivial-match factor configurable (Q1.ii) | paper-vs-code | Expose `EXCLUSION_ZONE_FACTOR` (default `0.5`) at the top of each experiment script and in `experiments/common_utils.py:test_motifs_significance`. Comment cites paper §3.2 and explicitly notes that published tables used `0.5`. Numeric default unchanged. |
| Population-density MOMENTI s=48 row | #16 | Add `48` to `experiments/populationdensity/run_momenti.py:204` so its motif lengths match LAMA and the paper's Table 4. |
| mSTUMP "conservative" regime preserved + clarified (Q3.a) | #15 | Keep the `√s · δ · 0.5` formula as-is. Update the in-code comment so the regime name does not lie about the formula: `# 'conservative' here means stricter distance-factor, not the paper's D_max = (1/q)Σδ formula.` |
| `idd_correction=False` annotated | Agent H verdict | One-line comment per script citing why (`# Variables not identically distributed; see REPRODUCING_EXPERIMENTS.md`). |
| `average_delta = 0.3` documented | #34 | Lift to `AVERAGE_DELTA` module-level constant with comment. No numeric change. |

### Packaging & metadata

| Change | Bug ref | Description |
|---|---|---|
| Bump version to `0.2.0` | #3 | `pyproject.toml`. `msig/__init__.py:__version__` switches to `importlib.metadata.version("msig")` (no manual sync). |
| Python version alignment | #7 | `requires-python = ">=3.11,<3.14"`; classifiers list 3.11/3.12/3.13; `[tool.mypy] python_version = "3.11"`; `[tool.black] target-version = ["py311","py312","py313"]`; `environment.yml` pins `python>=3.11,<3.14`; PYPI/README/INSTALLATION all say "3.11–3.13". |
| `msig/py.typed` marker file | #17 | Create empty file so PEP 561 type-checking activates downstream. |
| `requirements.txt` rewritten | #18 | Drop the self-pin (`msig==0.1.3`); align deps to `pyproject.toml`'s `experiments` + `dev` extras; add a comment pointing to `pyproject.toml` as the source of truth. |
| `environment.yml` rewritten | #19 | Uncomment the full `experiments` extra in the `pip:` block (pandas/matplotlib/librosa/statsmodels/jinja2/leitmotif/psutil). |
| Classifiers expanded | #28 | `License :: OSI Approved :: MIT License`, `Operating System :: OS Independent`, `Topic :: Scientific/Engineering`, `Topic :: Scientific/Engineering :: Information Analysis`, per-minor Python rows. |
| `.gitignore` cleaned | #29 | Add `*.npz`, `*.parquet`, `.mypy_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`. Remove the committed `msig.egg-info/` from the working tree. |
| `validate_reproducibility.py` core-install-safe | #30 | Move `pandas`/heavy imports inside their respective check functions so a `pip install msig` (no extras) install can run it. |
| `test_setup.sh` aligned | #30 | Import every dep declared in `pyproject.toml`; no more drift. |
| Delete `PYPI_DESCRIPTION.md` | new (PyPI audit) | Dead file — `pyproject.toml` uses `readme = "README.md"` for long-description. The duplicate has already drifted (year 2025 vs README 2024). README becomes the single source. |
| Add `black`/`isort` to `dev` extra | #31 | Lift `CLAUDE.md`'s recommendation into actual declared deps so `uv pip install -e ".[dev]"` ships them. |

### Documentation & examples

| Change | Bug ref | Description |
|---|---|---|
| Citation block updated everywhere | #6 | New BibTeX (`silva2026and`, *Pattern Recognition Letters*, 2026, Elsevier) in `README.md`. `PYPI_DESCRIPTION.md` deleted (above). |
| `CITATION.cff` at repo root | new | Canonical metadata: title, authors in paper order, year 2026, journal, repository URL, DOI when assigned. GitHub renders this. |
| Public docstring on `Motif` fixed | #11 | `set_significance(data_length=1000)` → `set_significance(max_possible_matches=N, data_n_variables=m)`. Doctest must be runnable. |
| `simple_example.py` final-print refs | #8 | Remove dangling `REPRODUCING_EXPERIMENTS.md`/`CONTRIBUTING.md` lines (or link to the new files created below). |
| `README.md` quick-start fixed | #33 | `motif_pattern = data[motif_vars, 5:15]` instead of `data[:, 5:15]`. |
| `MSig.py` docstrings rewritten | #5 | `idd_correction` description corrected; `vars_dep_time_markov` cites paper Eq. (4). |
| `CONTRIBUTING.md` (new) | gap | Minimal: clone, `uv sync`, `uv run pytest`, `uv run pytest -m integration`, `uv run black/isort`, conventional commits, sign-off. |
| `REPRODUCING_EXPERIMENTS.md` (new) | gap | (i) Environment; (ii) Data sources (audio: provide your own MP3; population density: Kaggle synthetic from manuscript; washing machine: LARCO Zenodo `10.5281/zenodo.17081452`); (iii) Running experiments; (iv) Expected outputs and runtimes; (v) **Paper-vs-code reconciliation**: explicit note that paper §3.2 says `l = ⌈s/4⌉` but tables match `⌈s/2⌉` (`EXCLUSION_ZONE_FACTOR = 0.5`); the "conservative" regime preserves the published `√s·δ·0.5` formula; `idd_correction=False` everywhere because variables are not identically distributed. (vi) **Pinning to the paper-revision code:** "to reproduce the published tables exactly, `git checkout v0.1.1` (or whichever tag corresponds to paper-submission code); 0.2.0 fixes affect `gaussian_theoretical` and KDE branches that the experiments don't use, so empirical-only experiments should match within the noise of regime parameter changes." |
| `validate_reproducibility.py` data pointers | #32 | "Where to get the data" pointers to the URLs above. |
| `examples/example.ipynb` cleanup | #D-notebook | Clear stale outputs; replace `sys.path.insert` cwd hack with first markdown cell instructing `pip install -e .`. Verify it executes top-to-bottom. |
| `CLAUDE.md` aligned | #31 | Remove black/isort lines that pre-dated declared deps; they're now in `dev` extra so the existing instructions just work. |

## Testing strategy

### Unit tests (`tests/test_basic.py`, `tests/test_statistical_methods.py`)

| Area | Test |
|---|---|
| `_rect_prob_2d` helper | Independent standard normals, rectangle `[-1,1]²` vs `≈ 0.4661²`; correlated case via `multivariate_normal.cdf` 4-corner formula |
| `vars_indep_time_markov` numerical | Hand-computed `p_Q` for a small empirical series with `δ=0` and `δ≠0`; exact equality (`pytest.approx` rel=1e-12) |
| KDE branch executed | Build `gaussian_kde` on 200-sample standard-normal series; assert KDE `p_Q` and empirical `p_Q` agree within ±20% |
| Gaussian-theoretical branch executed | Hand-computed `norm.cdf` rectangle for one chosen subsequence; would have caught #1 |
| `gaussian_theoretical` rejects δ=0 | `pytest.raises(ValueError, match="delta")` |
| `set_significance` numerical | Fixed `(p_Q, n_matches, max_possible_matches)` triple vs hand-computed `binom.sf(k-1, N, p)`; rel=1e-12 |
| `idd_correction=True` numerical | `idd=True, q=m` ⇒ uncorrected pvalue; `q<m` ⇒ exactly `p × C(m,q)`, capped at 1 |
| `pattern_prob_floor` smoothing | `p_Q=0` and `floor=None` ⇒ 0.0; `floor=1/(N+1)` ⇒ binomial tail at floor |
| Edge-case `self.pvalue` set | After every `set_significance` return path, `self.pvalue == returned value` |
| `bonferroni_correction` signature | `bonferroni_correction(5, 0.05) == 0.01`; `bonferroni_correction([0.1]*5, 0.05) == 0.01` |
| BH FDR canonical case | Benjamini & Hochberg (1995) §3 example; assert critical value matches the published value |
| Conditional probability invariant | Across all model branches, on a 50-sample series, every conditional probability is in `[0, 1]` (no clamp triggered) |

### Integration smoke tests (new file `tests/test_experiment_smoke.py`, `@pytest.mark.integration`)

| Test | What it verifies |
|---|---|
| All 9 experiment modules import without error | Catches `NameError`-class bugs (#9) |
| Each experiment runs end-to-end on a 200-sample synthetic fixture | Catches data-path bugs (#10), schema mismatches, ImportError. ~30s total. |
| `compare_results.py` runs against the smoke-test outputs | Catches #12 |

`pyproject.toml` already declares `slow` and `integration` markers; no config change needed.

### Regression tests for paper claims (`tests/test_paper_reproducibility.py`, `@pytest.mark.slow`)

Parametrised test per (dataset, method, regime): runs the experiment, loads the saved CSV, asserts it matches a checked-in `tests/golden/` CSV. Goldens are regenerated by step 6 of the validation plan (full experiment runs), then committed in step 8.

## Validation plan

After all implementation is complete, in order:

1. **Unit tests** — `uv run pytest tests/ -m "not slow and not integration" -v`. All green.
2. **Integration smokes** — `uv run pytest -m integration -v`. All green. ~1 min.
3. **Type check** — `uv run mypy msig/`. No errors on `msig/` itself; experiments may have warnings.
4. **Lint** — `uv run black --check`; `uv run isort --check-only`. Clean.
5. **Validation script** — `uv run python validate_reproducibility.py`. All checks pass.
6. **Full experiment runs** — `uv run python run_experiments.py --all`. Acceptance: every script exits 0 and writes a non-empty `summary_motifs_*.csv`. Numeric divergence from prior CSVs is expected and not a failure.
7. **`compare_results.py`** regenerates the cross-method comparison report; must succeed.
8. **Commit** the new CSVs and golden snapshots.

### Caveats

- **Data files** — `data/{audio,populationdensity,washingmachine,synthetic}/` are not in git but are present locally on the dev machine (synced 2026-05-04 from `miguel@10.10.4.60:~/raid_backup/`): `data/audio/imblue.mp3` (3.3 MB), `data/populationdensity/hourly_saodomingosbenfica.csv` (16.8 MB, NDA-protected — never commit), `data/washingmachine/main_readings.csv` (3.0 MB), `data/synthetic/multivar_time_series.csv` (2.4 MB). The existing `.gitignore` already protects all four subdirs from accidental commit. Step 6 can therefore run end-to-end for STUMPY and LAMA on this machine; only MOMENTI requires a separate pass.
- **MOMENTI on macOS** — MOMENTI is Linux/Windows only. On macOS, `run_experiments.py` auto-skips MOMENTI and step 6's MOMENTI portion must be performed on a Linux/Windows box separately before declaring step 6 complete.
- **Total runtime** — `run_experiments.py` estimates ~142 minutes for all 9 (per its own `ESTIMATED_RUNTIME` table). Plan a half-day for step 6.

## PyPI release procedure

1. Verify all validation steps above are green.
2. Run `git status` and confirm there are no uncommitted changes; `git diff main...HEAD` reflects only the design's intended changes.
3. Tag: `git tag -a v0.2.0 -m "MSig 0.2.0 — comprehensive revision"`. Push the tag.
4. Build artefacts: `uv build` (produces `dist/msig-0.2.0-*.whl` and `dist/msig-0.2.0.tar.gz`).
5. **Inspect wheel METADATA** before upload — `unzip -p dist/msig-0.2.0-*.whl '*/METADATA' | grep -i 'requires-python\|classifier'`. Confirm `Requires-Python: >=3.11,<3.14` and the expanded classifier list show up. (PyPI 0.1.3 currently misadvertises "Requires Python >=3.12" because of an earlier `pyproject.toml` state; this is the chance to fix it.)
6. Upload: `uv publish` (or `twine upload dist/*`).
7. Verify in a fresh venv: `pip install msig==0.2.0 && python -c "import msig; print(msig.__version__)"` — must print `0.2.0`. Then run `python validate_reproducibility.py`.
8. Update GitHub Release page with the `CHANGELOG.md` 0.2.0 section.
9. Do **not** yank `0.1.0`/`0.1.1`/`0.1.2`/`0.1.3` — backwards compat is preserved; users on 0.1.x continue to work.

## Out of scope

- Implementing `vars_dep_time_markov` (paper Eq. (4)).
- Higher-order Markov.
- A formal corrigendum to PRL for the `l = ⌈s/4⌉` vs `⌈s/2⌉` text mismatch.
- Adding new motif-discovery integrations beyond STUMPY/LAMA/MOMENTI.
- A web demo or interactive dashboard.
- Migrating tests to property-based (Hypothesis) — useful but separate effort.

## Acceptance criteria

- All unit and integration tests pass.
- `validate_reproducibility.py` reports all green.
- `run_experiments.py --all` exits 0 on a machine with data files (with documented MOMENTI carve-out).
- `compare_results.py` runs without schema errors against regenerated CSVs.
- `pip install msig==0.2.0` in a fresh venv works and `msig.__version__ == "0.2.0"`.
- README's quick-start runs without modification.
- Public docstring example for `Motif` is a runnable doctest.
- `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`, `REPRODUCING_EXPERIMENTS.md` exist with the content described above.
- No outstanding "TODO" or "fixme" markers introduced by the revision pass.

## Risks

| Risk | Mitigation |
|---|---|
| Regenerating `results/*.csv` produces numbers that look "wrong" relative to paper expectations | `REPRODUCING_EXPERIMENTS.md` documents the divergence sources; `tests/golden/diff-vs-paper.md` records per-row deltas |
| MOMENTI step 6 cannot run on the dev machine | Documented carve-out; require explicit Linux/Windows validation before release-tagging |
| `bonferroni_correction` signature change confuses `pip install`-existing-code users | Backward-compat `isinstance(int)` shim; CHANGELOG migration note |
| 2D-Gaussian fix changes a downstream user's results | None of the experiments use `gaussian_theoretical`; user-facing impact is minimal. CHANGELOG flags the fix prominently |
| Data files unavailable for full-experiment validation | Steps 1–5 still run; step 6 partial; release blocked only by lack of green smoke for the partial run |

## Open questions resolved during brainstorming

- **Q1 (trivial-match factor):** option (ii) — make `EXCLUSION_ZONE_FACTOR` configurable, default `0.5`, document paper-vs-code mismatch.
- **Q2 (smoothing default):** `pattern_prob_floor: float | None = None` — opt-in, preserves 0.1.x numerics.
- **Q3 (mSTUMP conservative):** option (a) — preserve `√s·δ·0.5` formula, rename comment to clarify.
- **Q4 (data files on dev machine):** synced from remote `miguel@10.10.4.60` on 2026-05-04. All four are present locally; only MOMENTI requires a Linux/Windows machine.
- **Q5 (macOS dev machine):** confirmed; MOMENTI carve-out documented.
- **Q6 (commit design doc):** yes, this file lives at `docs/superpowers/specs/2026-05-04-msig-revision-design.md` in the repo.

## Implementation plan

To be produced by the `writing-plans` skill in the next session, structured as work-stream sub-tasks suitable for `executing-plans` or `subagent-driven-development`.
