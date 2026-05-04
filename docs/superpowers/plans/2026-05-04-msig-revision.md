# MSig 0.2.0 Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `msig 0.2.0` — a comprehensive correctness, accessibility, and reproducibility revision of the MSig framework following the publication of Silva, Madeira & Henriques (*Pattern Recognition Letters*, 2026).

**Architecture:** Single-package Python library (`msig/MSig.py`) with a binomial-tail significance test, three null-model branches (empirical, KDE, Gaussian-theoretical), and a fan of experiment scripts (`experiments/{audio,populationdensity,washingmachine}/run_{stumpy,lama,momenti}.py`). The revision keeps the public API backward-compatible (one BC kwarg rename, one opt-in feature) and fixes library bugs, experiment-script bugs, packaging drift, and documentation gaps. CSVs in `results/` are regenerated; `tests/golden/` snapshots lock current state for future regressions.

**Tech Stack:** Python 3.11–3.13, numpy, scipy, pandas, stumpy, librosa; uv for env management; pytest + pytest-cov; black + isort; setuptools build backend; PyPI release via `uv publish`.

**Source spec:** `docs/superpowers/specs/2026-05-04-msig-revision-design.md` — refer there for rationale; this plan only states what to do.

**Working directory:** `/Users/miguelgarcao/Desktop/msig`. The repo is **not** a git repository in the sense the dispatching toolset expected (per environment metadata it says `Is a git repository: false` — but the repo *is* under git per `.git/`). All commits in this plan use `git` in the standard way; if you need to initialize before starting, run `git status` to confirm.

**Data files** (already synced locally on dev machine, see spec §Caveats):
- `data/audio/imblue.mp3` (3.3 MB)
- `data/synthetic/multivar_time_series.csv` (2.4 MB)
- `data/washingmachine/main_readings.csv` (3.0 MB)
- `data/populationdensity/hourly_saodomingosbenfica.csv` (16.8 MB; NDA-protected — never commit; `.gitignore` already covers).

**Key conventions:**
- Every behavior-changing task follows TDD: write the failing test first, watch it fail, implement, watch it pass, commit.
- Documentation/packaging/comment-only changes skip the failing-test step but always include a verification command.
- Each task ends with one focused commit. Commit messages use conventional commits (`fix:`, `feat:`, `docs:`, `chore:`, `test:`, `refactor:`).
- Run `uv run pytest tests/ -q` after each library task to verify no regression.

---

## Workstream 1: Library numerical correctness (`msig/MSig.py`)

These tasks fix bugs that affect numeric output. Each is TDD: failing test → fix → green test → commit.

---

### Task 1: Add private `_rect_prob_2d` helper (inclusion-exclusion)

**Files:**
- Modify: `msig/MSig.py`
- Test: `tests/test_basic.py`

**Context:** Current code at `MSig.py:349` computes `dist_bivar.cdf([up1,up2]) − dist_bivar.cdf([lo1,lo2])` — wrong; gives `F(b1,b2)−F(a1,a2)` instead of the rectangle probability `F(b1,b2)−F(a1,b2)−F(b1,a2)+F(a1,a2)`. Extract a tested helper before fixing the call sites (Tasks 2–3).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_basic.py`:

```python
class TestRectangleProbability:
    """Tests for the private _rect_prob_2d helper."""

    def test_rect_prob_2d_independent_standard_normals(self):
        """For independent N(0,1)×N(0,1), P(-1≤X≤1, -1≤Y≤1) = (Φ(1)-Φ(-1))²."""
        from scipy.stats import multivariate_normal, norm
        from msig.MSig import _rect_prob_2d

        dist = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        expected = (norm.cdf(1) - norm.cdf(-1)) ** 2
        result = _rect_prob_2d(dist, lo=[-1, -1], hi=[1, 1])
        assert abs(result - expected) < 1e-10

    def test_rect_prob_2d_correlated(self):
        """Inclusion-exclusion against scipy.stats.multivariate_normal four-corner formula."""
        from scipy.stats import multivariate_normal
        from msig.MSig import _rect_prob_2d

        cov = [[1.0, 0.5], [0.5, 1.0]]
        dist = multivariate_normal(mean=[0, 0], cov=cov)
        lo, hi = [-0.5, -0.5], [1.0, 1.5]
        # Reference via the four-corner formula
        F = lambda x, y: dist.cdf([x, y])
        expected = F(hi[0], hi[1]) - F(lo[0], hi[1]) - F(hi[0], lo[1]) + F(lo[0], lo[1])
        result = _rect_prob_2d(dist, lo, hi)
        assert abs(result - expected) < 1e-10

    def test_rect_prob_2d_clamps_negative_fp_noise(self):
        """When all four corners are nearly equal, fp arithmetic can produce a tiny
        negative value; helper must clamp to 0."""
        from scipy.stats import multivariate_normal
        from msig.MSig import _rect_prob_2d

        dist = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        # Zero-width rectangle
        result = _rect_prob_2d(dist, lo=[0.5, 0.5], hi=[0.5, 0.5])
        assert result == 0.0
        assert result >= 0  # Never negative
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_basic.py::TestRectangleProbability -v`
Expected: 3 FAILS with `ImportError: cannot import name '_rect_prob_2d'`.

- [ ] **Step 3: Implement the helper**

Add to `msig/MSig.py`, just below the imports and above `benjamini_hochberg_fdr`:

```python
def _rect_prob_2d(dist, lo, hi) -> float:
    """
    Probability of a 2D rectangle [lo[0], hi[0]] × [lo[1], hi[1]] under a 2D distribution.

    Uses the inclusion-exclusion formula for joint CDFs:
        P([a1,b1]×[a2,b2]) = F(b1,b2) − F(a1,b2) − F(b1,a2) + F(a1,a2)

    Parameters
    ----------
    dist : object
        A 2D distribution exposing a `.cdf(point)` method (e.g.,
        scipy.stats.multivariate_normal) where `point` is a 2-element sequence.
    lo : sequence of float
        Lower corner [a1, a2].
    hi : sequence of float
        Upper corner [b1, b2].

    Returns
    -------
    float
        The rectangle probability, clamped to [0, ∞) to absorb floating-point noise.
    """
    a1, a2 = lo[0], lo[1]
    b1, b2 = hi[0], hi[1]
    p = float(dist.cdf([b1, b2])) - float(dist.cdf([a1, b2])) \
        - float(dist.cdf([b1, a2])) + float(dist.cdf([a1, a2]))
    return max(0.0, p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_basic.py::TestRectangleProbability -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_basic.py msig/MSig.py
git commit -m "feat(msig): add _rect_prob_2d helper with inclusion-exclusion"
```

---

### Task 2: Replace bivariate Gaussian rectangle CDF in `vars_indep_time_markov` numerator

**Files:**
- Modify: `msig/MSig.py:349` (current bug location)
- Test: `tests/test_basic.py`

**Context:** Use `_rect_prob_2d` for the numerator of the conditional `P(x_t | x_{t-1})` in the `gaussian_theoretical` branch. This is the actual bug fix.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_basic.py` inside a new class:

```python
class TestGaussianTheoreticalConditional:
    """Numerical regression tests for the gaussian_theoretical branch."""

    def test_gaussian_theoretical_p_q_against_hand_computed(self):
        """For a 50-sample standard-normal series with an explicit subsequence,
        p_Q must match a hand-computed value using norm.cdf and inclusion-exclusion."""
        import numpy as np
        from scipy.stats import multivariate_normal, norm
        from msig import Motif, NullModel
        from msig.MSig import _rect_prob_2d

        np.random.seed(0)
        data = np.random.randn(1, 200)
        model = NullModel(data, dtypes=[float], model="gaussian_theoretical")

        # Three-point subsequence with δ = 0.2
        subsequence = np.array([[0.0, 0.5, -0.3]])
        delta = 0.2

        # Initial: P(-0.2 ≤ Y ≤ 0.2) under the fitted N(μ̂, σ̂)
        dist1 = model.pre_computed_distribution[0]
        p_init = float(dist1.cdf(0.2) - dist1.cdf(-0.2))

        # Conditional 1: P(0.3 ≤ Y_t ≤ 0.7 | -0.2 ≤ Y_{t-1} ≤ 0.2)
        dist2 = model.pre_computed_bivariate_distribution[0]
        num1 = _rect_prob_2d(dist2, lo=[-0.2, 0.3], hi=[0.2, 0.7])
        denom1 = float(dist1.cdf(0.2) - dist1.cdf(-0.2))
        cond1 = num1 / denom1

        # Conditional 2: P(-0.5 ≤ Y_t ≤ -0.1 | 0.3 ≤ Y_{t-1} ≤ 0.7)
        num2 = _rect_prob_2d(dist2, lo=[0.3, -0.5], hi=[0.7, -0.1])
        denom2 = float(dist1.cdf(0.7) - dist1.cdf(0.3))
        cond2 = num2 / denom2

        expected = p_init * cond1 * cond2

        motif = Motif(subsequence, [0], [delta], n_matches=1)
        actual = motif.set_pattern_probability(model, vars_indep=True)
        assert abs(actual - expected) < 1e-10

    def test_gaussian_theoretical_p_q_lt_or_eq_one(self):
        """The pre-fix bug could produce p_Q values exceeding 1 in some configurations."""
        import numpy as np
        from msig import Motif, NullModel

        np.random.seed(1)
        data = np.random.randn(2, 100)
        model = NullModel(data, dtypes=[float, float], model="gaussian_theoretical")
        subsequence = data[:, 10:13]
        motif = Motif(subsequence, [0, 1], [0.3, 0.3], n_matches=2)
        p_Q = motif.set_pattern_probability(model, vars_indep=True)
        assert 0.0 <= p_Q <= 1.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_basic.py::TestGaussianTheoreticalConditional -v`
Expected: FAIL — actual value differs from hand-computed because current code uses `F(b1,b2)−F(a1,a2)`.

- [ ] **Step 3: Implement the fix**

In `msig/MSig.py`, locate the block currently around `MSig.py:344–350` (the `gaussian_theoretical` branch inside the loop in `vars_indep_time_markov`). Replace:

```python
                elif self.model == "gaussian_theoretical":
                    numerator = float(dist_bivar.cdf([ximinus1_upper, xi_upper]) - dist_bivar.cdf([ximinus1_lower, xi_lower]))
                    denominator = float(dist.cdf(ximinus1_upper) - dist.cdf(ximinus1_lower))
```

with:

```python
                elif self.model == "gaussian_theoretical":
                    numerator = _rect_prob_2d(
                        dist_bivar,
                        lo=[ximinus1_lower, xi_lower],
                        hi=[ximinus1_upper, xi_upper],
                    )
                    denominator = float(dist.cdf(ximinus1_upper) - dist.cdf(ximinus1_lower))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_basic.py::TestGaussianTheoreticalConditional -v`
Expected: 2 PASS.
Run: `uv run pytest tests/ -q` to confirm no regressions in the rest of the suite.

- [ ] **Step 5: Commit**

```bash
git add msig/MSig.py tests/test_basic.py
git commit -m "fix(msig): use inclusion-exclusion for 2D Gaussian rectangle CDF

The conditional probability numerator in the gaussian_theoretical branch
of vars_indep_time_markov was computing F(b1,b2)-F(a1,a2), which is not
the rectangle probability for a 2D distribution. Use the new _rect_prob_2d
helper that applies the four-corner inclusion-exclusion formula.

Affects all p_Q values from gaussian_theoretical models. Empirical and
KDE branches are unchanged (KDE already used integrate_box correctly).
None of the published experiments use gaussian_theoretical, so paper
tables are unaffected."
```

---

### Task 3: Reject `δ = 0` for `gaussian_theoretical` and `kde` models

**Files:**
- Modify: `msig/MSig.py` — `vars_indep_time_markov`
- Test: `tests/test_basic.py`

**Context:** With `δ = 0`, `cdf(x) − cdf(x) = 0` silently — the model returns `p_Q = 0` for any continuous-model call, which then short-circuits the binomial test. Better to fail loudly: `δ = 0` is meaningful only for `empirical` (exact match on discrete values).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_basic.py`:

```python
class TestDeltaValidation:
    """δ = 0 should raise ValueError for continuous null models."""

    def test_gaussian_theoretical_rejects_delta_zero(self):
        import numpy as np
        import pytest
        from msig import Motif, NullModel

        data = np.random.randn(1, 50)
        model = NullModel(data, dtypes=[float], model="gaussian_theoretical")
        motif = Motif(data[:, 5:8], [0], [0.0], n_matches=1)
        with pytest.raises(ValueError, match="delta must be > 0"):
            motif.set_pattern_probability(model, vars_indep=True)

    def test_kde_rejects_delta_zero(self):
        import numpy as np
        import pytest
        from msig import Motif, NullModel

        data = np.random.randn(1, 50)
        model = NullModel(data, dtypes=[float], model="kde")
        motif = Motif(data[:, 5:8], [0], [0.0], n_matches=1)
        with pytest.raises(ValueError, match="delta must be > 0"):
            motif.set_pattern_probability(model, vars_indep=True)

    def test_empirical_accepts_delta_zero(self):
        """Empirical with δ = 0 is the standard exact-match path, must not raise."""
        import numpy as np
        from msig import Motif, NullModel

        data = np.array([[1, 2, 1, 2, 1, 2]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        motif = Motif(np.array([[1.0, 2.0]]), [0], [0.0], n_matches=3)
        p_Q = motif.set_pattern_probability(model, vars_indep=True)
        assert p_Q > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_basic.py::TestDeltaValidation -v`
Expected: First two FAIL (no exception raised); third PASSES (existing behaviour).

- [ ] **Step 3: Implement the validation**

In `msig/MSig.py`, near the top of `vars_indep_time_markov`, add validation right after entering the per-variable loop. Locate:

```python
        for seq_idx, subsequence in enumerate(motif_subsequence):
            var_index = variables[seq_idx]
            delta = delta_thresholds[seq_idx]  # Use seq_idx: delta_thresholds aligns with motif_subsequence
            p_Q_j: float = 1.0
```

Append:

```python
            if delta == 0 and self.model != "empirical":
                raise ValueError(
                    f"delta must be > 0 for continuous null models; "
                    f"got delta=0 with model='{self.model}' for variable {var_index}. "
                    f"Use model='empirical' for exact matching."
                )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_basic.py::TestDeltaValidation tests/ -q`
Expected: All 3 PASS, no other regressions.

- [ ] **Step 5: Commit**

```bash
git add msig/MSig.py tests/test_basic.py
git commit -m "feat(msig): reject delta=0 for continuous null models

KDE and gaussian_theoretical with delta=0 silently produce p_Q=0
because cdf(x)-cdf(x)=0. Raise ValueError with a clear message
suggesting model='empirical' for exact matching."
```

---

### Task 4: Fix empirical conditional inconsistent normalization

**Files:**
- Modify: `msig/MSig.py:329-343` (empirical branch in `vars_indep_time_markov`)
- Test: `tests/test_basic.py`

**Context:** Numerator divides count of `(X_{t-1}∈A, X_t∈B)` pairs by `n−1`; denominator divides count of `X∈A` over the whole `time_series` by `n`. Inconsistent — biases `cond_p` by `n/(n−1)`. Switch denominator to count `time_series[:-1]` over `n-1`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_basic.py`:

```python
class TestEmpiricalConditionalConsistency:
    """The empirical conditional should never produce cond_p > 1 by construction."""

    def test_no_clamp_triggered_on_random_series(self):
        """Across many random series and motifs, cond_p must satisfy 0 ≤ cond_p ≤ 1
        without invoking the clamp at line 358."""
        import numpy as np
        import logging
        from msig import Motif, NullModel

        np.random.seed(42)
        for _ in range(50):
            data = np.random.randn(1, 30)
            model = NullModel(data, dtypes=[float], model="empirical")
            subsequence = data[:, 5:10]
            motif = Motif(subsequence, [0], [0.2], n_matches=2)
            p_Q = motif.set_pattern_probability(model, vars_indep=True)
            assert 0.0 <= p_Q <= 1.0

    def test_empirical_conditional_against_hand_computed(self):
        """Hand-compute the conditional for a tiny series; assert exact agreement."""
        import numpy as np
        from msig import Motif, NullModel

        # 6-point series; 5 transition pairs.
        data = np.array([[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]])
        model = NullModel(data, dtypes=[float], model="empirical")
        # Pattern [1, 2, 1] with δ = 0:
        subsequence = np.array([[1.0, 2.0, 1.0]])
        motif = Motif(subsequence, [0], [0.0], n_matches=2)
        p_Q = motif.set_pattern_probability(model, vars_indep=True)

        # Hand: P(X=1) = 3/6 = 0.5
        # P(X_t=2 | X_{t-1}=1) = #(1→2) / #(prev=1 in transitions) = 3/3 = 1
        # P(X_t=1 | X_{t-1}=2) = #(2→1) / #(prev=2 in transitions) = 2/2 = 1
        expected = 0.5 * 1.0 * 1.0
        assert abs(p_Q - expected) < 1e-12
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_basic.py::TestEmpiricalConditionalConsistency -v`
Expected: At least one of the two FAILS — current code uses inconsistent denominators producing `cond_p = (3/5)/(3/6) = 1.2` then clamps to 1.0. The hand-computed test will fail because the clamp drops a `1.2 * 0.5 ≠ 0.5` error elsewhere.

- [ ] **Step 3: Implement the fix**

In `msig/MSig.py`, locate the empirical branch inside the conditional-probability loop (currently around lines 329–343). Replace:

```python
                if self.model == "empirical":
                    if delta == 0:
                        count = np.sum((time_series[:-1] == subsequence[i - 1]) & (time_series[1:] == subsequence[i]))
                    else:
                        count = np.sum(
                            (np.logical_and(time_series[:-1] >= ximinus1_lower, time_series[:-1] <= ximinus1_upper))
                            & (np.logical_and(time_series[1:] >= xi_lower, time_series[1:] <= xi_upper))
                        )
                    numerator = count / (len(time_series) - 1) if len(time_series) > 1 else 0.0

                    if delta == 0:
                        count = np.sum(time_series == subsequence[i - 1])
                    else:
                        count = np.sum(np.logical_and(time_series >= ximinus1_lower, time_series <= ximinus1_upper))
                    denominator = count / len(time_series) if len(time_series) > 0 else 1.0
```

with:

```python
                if self.model == "empirical":
                    n_transitions = len(time_series) - 1
                    if n_transitions <= 0:
                        numerator = 0.0
                        denominator = 1.0
                    else:
                        # Joint count over n-1 transition pairs
                        if delta == 0:
                            count_pair = np.sum(
                                (time_series[:-1] == subsequence[i - 1])
                                & (time_series[1:] == subsequence[i])
                            )
                        else:
                            count_pair = np.sum(
                                np.logical_and(time_series[:-1] >= ximinus1_lower, time_series[:-1] <= ximinus1_upper)
                                & np.logical_and(time_series[1:] >= xi_lower, time_series[1:] <= xi_upper)
                            )
                        numerator = count_pair / n_transitions

                        # Lag-1 marginal: count over time_series[:-1] divided by n-1
                        if delta == 0:
                            count_marginal = np.sum(time_series[:-1] == subsequence[i - 1])
                        else:
                            count_marginal = np.sum(
                                np.logical_and(time_series[:-1] >= ximinus1_lower, time_series[:-1] <= ximinus1_upper)
                            )
                        denominator = count_marginal / n_transitions if count_marginal > 0 else 1.0
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_basic.py::TestEmpiricalConditionalConsistency tests/ -q`
Expected: All PASS, no regressions in existing tests.

- [ ] **Step 5: Commit**

```bash
git add msig/MSig.py tests/test_basic.py
git commit -m "fix(msig): consistent normalization in empirical conditional probability

Numerator and denominator both use the lag-1 marginal over n-1 transition
pairs. Previously denominator used the full marginal over n samples,
biasing cond_p by n/(n-1) and triggering the >1 clamp on small series."
```

---

### Task 5: KDE conditional uses lag-1 marginal

**Files:**
- Modify: `msig/MSig.py` — `NullModel.__init__` (add `pre_computed_lag1_marginal`) and `vars_indep_time_markov` (KDE branch)
- Test: `tests/test_basic.py`

**Context:** Currently KDE numerator uses bivariate `gaussian_kde` on (X_{t-1}, X_t) pairs but denominator uses univariate `gaussian_kde` on **all** of `y_j`. Theoretically inconsistent. Pre-compute and use the lag-1 marginal `gaussian_kde(y_j[:-1])`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_basic.py`:

```python
class TestKDEConditional:
    """KDE branch coverage and conditional consistency."""

    def test_kde_p_q_in_bounds(self):
        """KDE p_Q must be in [0, 1]."""
        import numpy as np
        from msig import Motif, NullModel

        np.random.seed(7)
        data = np.random.randn(1, 100)
        model = NullModel(data, dtypes=[float], model="kde")
        subsequence = data[:, 10:14]
        motif = Motif(subsequence, [0], [0.3], n_matches=2)
        p_Q = motif.set_pattern_probability(model, vars_indep=True)
        assert 0.0 <= p_Q <= 1.0

    def test_kde_lag1_marginal_attribute_exists(self):
        """NullModel.pre_computed_lag1_marginal must exist for kde models."""
        import numpy as np
        from msig import NullModel

        data = np.random.randn(1, 50)
        model = NullModel(data, dtypes=[float], model="kde")
        assert 0 in model.pre_computed_lag1_marginal

    def test_kde_agrees_with_empirical_on_large_sample(self):
        """KDE and empirical should agree within 25% on 1000-sample N(0,1) data."""
        import numpy as np
        from msig import Motif, NullModel

        np.random.seed(0)
        data = np.random.randn(1, 1000)
        subsequence = np.array([[0.1, -0.2, 0.3]])

        m_kde = NullModel(data, dtypes=[float], model="kde")
        m_emp = NullModel(data, dtypes=[float], model="empirical")

        p_kde = Motif(subsequence, [0], [0.3], 1).set_pattern_probability(m_kde, vars_indep=True)
        p_emp = Motif(subsequence, [0], [0.3], 1).set_pattern_probability(m_emp, vars_indep=True)

        assert p_kde > 0 and p_emp > 0
        # Ratio within ±25%
        ratio = p_kde / p_emp
        assert 0.75 <= ratio <= 1.25, f"KDE/empirical ratio {ratio:.3f} outside ±25%"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_basic.py::TestKDEConditional -v`
Expected: `test_kde_lag1_marginal_attribute_exists` FAILS (attribute doesn't exist). The other two may pass coincidentally (existing KDE branch already produces in-bounds values).

- [ ] **Step 3: Implement the fix**

In `msig/MSig.py`, modify `NullModel.__init__` to also build the lag-1 marginal for KDE. Locate the KDE block:

```python
            if self.model == "kde":
                self.pre_computed_distribution[var_index] = gaussian_kde(y_j)
                self.pre_computed_bivariate_distribution[var_index] = gaussian_kde(pairs)
```

Replace with:

```python
            if self.model == "kde":
                self.pre_computed_distribution[var_index] = gaussian_kde(y_j)
                self.pre_computed_bivariate_distribution[var_index] = gaussian_kde(pairs)
                self.pre_computed_lag1_marginal[var_index] = gaussian_kde(y_j[:-1])
```

Add `self.pre_computed_lag1_marginal: dict[int, Any] = {}` to the constructor's attribute initialisation (alongside `pre_computed_distribution` and `pre_computed_bivariate_distribution`).

In `vars_indep_time_markov`, locate the KDE conditional branch (currently around lines 344–347):

```python
                elif self.model == "kde":
                    numerator = float(dist_bivar.integrate_box([ximinus1_lower, xi_lower], [ximinus1_upper, xi_upper]))
                    # Use marginal for the previous state as denominator
                    denominator = float(dist.integrate_box_1d(ximinus1_lower, ximinus1_upper))
```

Replace with:

```python
                elif self.model == "kde":
                    numerator = float(dist_bivar.integrate_box([ximinus1_lower, xi_lower], [ximinus1_upper, xi_upper]))
                    # Use the lag-1 marginal so numerator/denominator share the same model
                    dist_lag1 = self.pre_computed_lag1_marginal[var_index]
                    denominator = float(dist_lag1.integrate_box_1d(ximinus1_lower, ximinus1_upper))
```

(The local variable `dist` previously used in this branch is the full-sample univariate KDE; we now route via the lag-1 marginal cached on the model.)

Also update the class docstring for `NullModel` to mention `pre_computed_lag1_marginal`.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_basic.py::TestKDEConditional tests/ -q`
Expected: All PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add msig/MSig.py tests/test_basic.py
git commit -m "fix(msig): KDE conditional uses lag-1 marginal for theoretical consistency

Previously the KDE conditional P(x_t | x_{t-1}) numerator used a bivariate
KDE over (y_t, y_{t+1}) pairs but the denominator used a univariate KDE
over the full series y. Now denominator uses gaussian_kde(y[:-1]) so the
two share a consistent lag-1 marginal."
```

---

### Task 6: Refactor `set_significance` to a single return path that always sets `self.pvalue`

**Files:**
- Modify: `msig/MSig.py` — `Motif.set_significance`
- Test: `tests/test_basic.py`

**Context:** Edge-case branches for `p_Q ∈ {0, 1}` and `n_matches >= max_possible_matches` use bare `return value` without setting `self.pvalue`. Caller behaviour depends on which they read.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_basic.py`:

```python
class TestSetSignificanceAlwaysSetsPvalue:
    """Every return path of set_significance must populate self.pvalue."""

    def test_p_q_zero_sets_self_pvalue(self):
        import numpy as np
        from msig import Motif, NullModel

        data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        # Pattern that never occurs in data ⇒ p_Q = 0
        motif = Motif(np.array([[99.0]]), [0], [0.0], n_matches=1)
        motif.set_pattern_probability(model, vars_indep=True)
        returned = motif.set_significance(5, 1, idd_correction=False)
        assert motif.pvalue == returned
        assert motif.pvalue == 0.0

    def test_p_q_one_sets_self_pvalue(self):
        import numpy as np
        from msig import Motif, NullModel

        data = np.array([[1, 1, 1, 1, 1]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        motif = Motif(np.array([[1.0]]), [0], [0.0], n_matches=5)
        motif.set_pattern_probability(model, vars_indep=True)
        returned = motif.set_significance(5, 1, idd_correction=False)
        assert motif.pvalue == returned
        assert motif.pvalue == 1.0

    def test_n_matches_ge_max_sets_self_pvalue(self):
        """Degenerate case n_matches >= max_possible_matches returns NaN."""
        import math
        import numpy as np
        from msig import Motif, NullModel

        data = np.array([[0.5] * 20], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        motif = Motif(np.array([[0.5]]), [0], [0.0], n_matches=10)
        motif.set_pattern_probability(model, vars_indep=True)
        # max_possible_matches = 5, n_matches = 10 ⇒ degenerate
        returned = motif.set_significance(5, 1, idd_correction=False)
        assert math.isnan(returned) or returned == 1.0
        assert motif.pvalue == returned or (math.isnan(returned) and math.isnan(motif.pvalue))
```

Note: depending on what value the user wants for the degenerate case (NaN vs 1.0), adjust the assertion. The current code returns NaN for `n_matches >= max_possible_matches`; we keep that.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_basic.py::TestSetSignificanceAlwaysSetsPvalue -v`
Expected: First two FAIL because `self.pvalue` is the default `1.0` (set in `__init__`), not the returned value.

- [ ] **Step 3: Refactor to single return path**

In `msig/MSig.py`, replace the body of `set_significance` (everything after the validation block at lines 590–593) with a structure that computes `pvalue` into a local variable and assigns once at the end:

```python
        # Handle edge cases
        if self.p_Q == 0.0:
            pvalue = 0.0
        elif self.p_Q == 1.0:
            pvalue = 1.0
        elif self.n_matches >= max_possible_matches:
            logger.warning(
                f"Degenerate case: n_matches={self.n_matches} >= "
                f"max_possible_matches={max_possible_matches}. Returning NaN."
            )
            pvalue = float("nan")
        else:
            pvalue = float(binom.sf(self.n_matches - 1, max_possible_matches, self.p_Q))

            if idd_correction:
                pvalue = min(1.0, pvalue * math.comb(data_n_variables, len(self.variables)))

        self.pvalue = pvalue
        logger.info("p_value = %.3E (p_pattern = %.3E)", self.pvalue, self.p_Q)
        return pvalue
```

Also delete the `OverflowError` fallback block (`try: ... except OverflowError`) — `scipy.stats.binom.sf` underflows to 0 silently and never raises; this is dead code per spec finding #25. Replace the `try` block with the single-line `pvalue = float(binom.sf(...))` shown above.

Also fix the logger bug at the bottom: change `logging.info(...)` to `logger.info(...)` (per spec finding #36).

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_basic.py::TestSetSignificanceAlwaysSetsPvalue tests/ -q`
Expected: All PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add msig/MSig.py tests/test_basic.py
git commit -m "refactor(msig): single return path for set_significance, always set self.pvalue

Edge-case branches (p_Q ∈ {0,1}, n_matches ≥ max) previously returned
without updating self.pvalue, causing inconsistency between the return
value and the attribute.

Also remove dead OverflowError fallback (scipy.stats.binom.sf underflows
silently, never raises) and fix logging.info→logger.info."
```

---

### Task 7: Add opt-in `pattern_prob_floor` smoothing for `p_Q = 0`

**Files:**
- Modify: `msig/MSig.py` — `Motif.set_significance` signature and edge-case logic
- Test: `tests/test_basic.py`

**Context:** When `p_Q = 0` and `n_matches > 0`, returning p-value = 0 is mathematically defensible but methodologically problematic ("zero-frequency problem"). Add an opt-in `pattern_prob_floor` kwarg that replaces `p_Q = 0` with the floor before computing the binomial tail. Default `None` preserves current behaviour for paper-table reproducibility.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_basic.py`:

```python
class TestPatternProbFloor:
    """Opt-in smoothing of zero pattern probability."""

    def test_floor_none_preserves_zero_pvalue(self):
        """Default floor=None ⇒ same as current behaviour."""
        import numpy as np
        from msig import Motif, NullModel

        data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        motif = Motif(np.array([[99.0]]), [0], [0.0], n_matches=1)
        motif.set_pattern_probability(model, vars_indep=True)
        pvalue = motif.set_significance(5, 1, idd_correction=False, pattern_prob_floor=None)
        assert pvalue == 0.0

    def test_floor_set_smooths_zero_p_q(self):
        """floor=1/(N+1) substitutes the floor for p_Q before binomial tail."""
        import numpy as np
        from scipy.stats import binom
        from msig import Motif, NullModel

        data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        motif = Motif(np.array([[99.0]]), [0], [0.0], n_matches=1)
        motif.set_pattern_probability(model, vars_indep=True)
        N = 5
        floor = 1.0 / (N + 1)
        pvalue = motif.set_significance(N, 1, idd_correction=False, pattern_prob_floor=floor)
        expected = float(binom.sf(0, N, floor))
        assert abs(pvalue - expected) < 1e-12

    def test_floor_does_not_affect_nonzero_p_q(self):
        """When p_Q > 0 the floor is irrelevant."""
        import numpy as np
        from scipy.stats import binom
        from msig import Motif, NullModel

        data = np.array([[1, 2, 1, 2, 1, 2]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        motif = Motif(np.array([[1.0, 2.0]]), [0], [0.0], n_matches=3)
        motif.set_pattern_probability(model, vars_indep=True)
        N = 5
        without_floor = Motif(np.array([[1.0, 2.0]]), [0], [0.0], n_matches=3)
        without_floor.set_pattern_probability(model, vars_indep=True)
        with_floor = Motif(np.array([[1.0, 2.0]]), [0], [0.0], n_matches=3)
        with_floor.set_pattern_probability(model, vars_indep=True)
        a = without_floor.set_significance(N, 1, pattern_prob_floor=None)
        b = with_floor.set_significance(N, 1, pattern_prob_floor=0.001)
        assert a == b
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_basic.py::TestPatternProbFloor -v`
Expected: FAIL with `TypeError: set_significance() got an unexpected keyword argument 'pattern_prob_floor'`.

- [ ] **Step 3: Implement the kwarg**

In `msig/MSig.py`, change the signature of `set_significance`:

```python
    def set_significance(
        self,
        max_possible_matches: int,
        data_n_variables: int,
        idd_correction: bool = False,
        pattern_prob_floor: float | None = None,
    ) -> float:
```

Update the docstring with a `pattern_prob_floor` parameter section (place after `idd_correction`):

```
        pattern_prob_floor : float or None, default=None
            Optional Laplace-style floor for `self.p_Q` before computing
            the binomial tail. When `self.p_Q == 0` and `pattern_prob_floor`
            is not None, the floor value replaces zero. Useful when the
            pattern is unobserved in the reference data but n_matches > 0
            in the test series; addresses the "zero-frequency problem".

            Common choices: `1.0 / (max_possible_matches + 1)` (Laplace),
            `3.0 / max_possible_matches` (rule-of-three upper bound).

            Default `None` preserves 0.1.x behaviour: p_Q=0 ⇒ pvalue=0.
```

In the body (refactored in Task 6), replace the `if self.p_Q == 0.0:` branch with:

```python
        effective_p_Q = self.p_Q
        if effective_p_Q == 0.0 and pattern_prob_floor is not None:
            effective_p_Q = float(pattern_prob_floor)

        if effective_p_Q == 0.0:
            pvalue = 0.0
        elif effective_p_Q == 1.0:
            pvalue = 1.0
        elif self.n_matches >= max_possible_matches:
            logger.warning(
                f"Degenerate case: n_matches={self.n_matches} >= "
                f"max_possible_matches={max_possible_matches}. Returning NaN."
            )
            pvalue = float("nan")
        else:
            pvalue = float(binom.sf(self.n_matches - 1, max_possible_matches, effective_p_Q))

            if idd_correction:
                pvalue = min(1.0, pvalue * math.comb(data_n_variables, len(self.variables)))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_basic.py::TestPatternProbFloor tests/ -q`
Expected: All PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add msig/MSig.py tests/test_basic.py
git commit -m "feat(msig): opt-in pattern_prob_floor for zero-frequency problem

When p_Q is estimated as 0 (pattern unobserved in reference data) and
n_matches > 0, the strict binomial tail returns p-value=0. This is
mathematically correct but methodologically overconfident.

New kwarg pattern_prob_floor (default None) lets callers substitute a
small Laplace floor (e.g. 1/(N+1)) before computing the binomial tail.
Default None preserves 0.1.x behaviour for paper-table reproducibility."
```

---

### Task 8: Add numerical regression test for `binom.sf` in `set_significance`

**Files:**
- Modify: `tests/test_basic.py`

**Context:** Existing tests only check `0 ≤ p ≤ 1`. Lock in the binomial-tail formula numerically so a future swap of arguments would be caught.

- [ ] **Step 1: Write the test**

Append to `tests/test_basic.py`:

```python
class TestSetSignificanceNumerical:
    """Lock in P(X >= k) = binom.sf(k-1, N, p) so future drift is caught."""

    def test_binomial_tail_exact(self):
        from scipy.stats import binom
        import numpy as np
        from msig import Motif, NullModel

        # Construct a motif with known p_Q via a controlled empirical series.
        data = np.array([[1.0, 2.0] * 50])  # 100 points, 50 of each
        model = NullModel(data, dtypes=[float], model="empirical")
        motif = Motif(np.array([[1.0]]), [0], [0.0], n_matches=10)
        p_Q = motif.set_pattern_probability(model, vars_indep=True)
        assert p_Q == 0.5  # 50/100

        N = 80
        pvalue = motif.set_significance(N, 1, idd_correction=False)
        expected = float(binom.sf(motif.n_matches - 1, N, p_Q))
        assert abs(pvalue - expected) < 1e-12

    def test_idd_correction_factor_exact(self):
        """idd_correction=True multiplies pvalue by C(m,q) capped at 1."""
        import math
        import numpy as np
        from scipy.stats import binom
        from msig import Motif, NullModel

        data = np.random.RandomState(0).randn(2, 50)
        model = NullModel(data, dtypes=[float, float], model="empirical")
        motif = Motif(data[:, 5:8], [0, 1], [0.5, 0.5], n_matches=2)
        p_Q = motif.set_pattern_probability(model, vars_indep=True)

        N = 45
        m = 5  # data_n_variables
        q = 2  # len(motif.variables)

        # Without correction
        no_idd = Motif(data[:, 5:8], [0, 1], [0.5, 0.5], n_matches=2)
        no_idd.set_pattern_probability(model, vars_indep=True)
        p_no = no_idd.set_significance(N, m, idd_correction=False)

        # With correction
        with_idd = Motif(data[:, 5:8], [0, 1], [0.5, 0.5], n_matches=2)
        with_idd.set_pattern_probability(model, vars_indep=True)
        p_with = with_idd.set_significance(N, m, idd_correction=True)

        expected_with = min(1.0, p_no * math.comb(m, q))
        assert abs(p_with - expected_with) < 1e-12

    def test_idd_correction_q_equals_m_is_identity(self):
        """When q == m, C(m,m) = 1 ⇒ corrected = uncorrected."""
        import numpy as np
        from msig import Motif, NullModel

        data = np.random.RandomState(0).randn(3, 50)
        model = NullModel(data, dtypes=[float, float, float], model="empirical")

        no_idd = Motif(data[:, 5:8], [0, 1, 2], [0.5, 0.5, 0.5], n_matches=2)
        no_idd.set_pattern_probability(model, vars_indep=True)
        p_no = no_idd.set_significance(45, 3, idd_correction=False)

        with_idd = Motif(data[:, 5:8], [0, 1, 2], [0.5, 0.5, 0.5], n_matches=2)
        with_idd.set_pattern_probability(model, vars_indep=True)
        p_with = with_idd.set_significance(45, 3, idd_correction=True)

        assert p_no == p_with
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_basic.py::TestSetSignificanceNumerical -v`
Expected: All 3 PASS (the implementation is already correct after Task 6).

- [ ] **Step 3: Commit**

```bash
git add tests/test_basic.py
git commit -m "test(msig): numerical regression tests for binomial tail and IDD correction"
```

---

### Task 9: Replace `bonferroni_correction` signature (BC)

**Files:**
- Modify: `msig/MSig.py` — `bonferroni_correction`
- Test: `tests/test_statistical_methods.py`

**Context:** Current signature takes an iterable but only uses `len()`. Rename first arg to `n_tests` and accept either `int` or `Iterable[float]`.

- [ ] **Step 1: Write the new test (does not yet pass)**

Append to `tests/test_statistical_methods.py`:

```python
class TestBonferroniSignatureBC:
    """Bonferroni accepts both int n_tests and an iterable of p-values (BC)."""

    def test_accepts_int_directly(self):
        from msig import bonferroni_correction
        assert bonferroni_correction(5, alpha=0.05) == 0.01

    def test_accepts_iterable_for_backward_compat(self):
        from msig import bonferroni_correction
        assert bonferroni_correction([0.1] * 5, alpha=0.05) == 0.01

    def test_int_zero_returns_alpha(self):
        from msig import bonferroni_correction
        assert bonferroni_correction(0, alpha=0.05) == 0.05

    def test_int_negative_returns_alpha(self):
        """Defensive: negative n_tests treated like 0."""
        from msig import bonferroni_correction
        assert bonferroni_correction(0, alpha=0.05) == 0.05
```

- [ ] **Step 2: Run test to verify partial failure**

Run: `uv run pytest tests/test_statistical_methods.py::TestBonferroniSignatureBC -v`
Expected: `test_accepts_int_directly` FAILS with `TypeError: object of type 'int' has no len()`.

- [ ] **Step 3: Update the function**

In `msig/MSig.py`, replace the `bonferroni_correction` function body:

```python
def bonferroni_correction(n_tests: int | Iterable[float], alpha: float = 0.05) -> float:
    """
    Bonferroni correction for multiple hypothesis testing.

    Compute the adjusted significance threshold using the conservative
    Bonferroni correction method.

    Parameters
    ----------
    n_tests : int or Iterable[float]
        The number of tests, OR (for backward compatibility with msig 0.1.x)
        an iterable of p-values whose length is used as the number of tests.
        Bonferroni does not depend on the p-value contents.
    alpha : float, default=0.05
        The family-wise error rate (FWER) to control.

    Returns
    -------
    float
        The corrected significance threshold (alpha / number of tests).
        Returns `alpha` if `n_tests <= 0`.

    Notes
    -----
    The Bonferroni correction controls the family-wise error rate by dividing
    the significance level by the number of comparisons. It is very conservative
    and may have low power when many comparisons are made.

    Examples
    --------
    >>> bonferroni_correction(5, alpha=0.05)
    0.01
    >>> bonferroni_correction([0.001, 0.008, 0.039, 0.041, 0.042], alpha=0.05)
    0.01
    """
    if not isinstance(n_tests, int):
        n_tests = sum(1 for _ in n_tests)
    if n_tests <= 0:
        return alpha
    return alpha / n_tests
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run pytest tests/test_statistical_methods.py -v`
Expected: All TestBonferroniSignatureBC PASS, all existing TestStatisticalMethods PASS.

- [ ] **Step 5: Commit**

```bash
git add msig/MSig.py tests/test_statistical_methods.py
git commit -m "feat(msig)!: bonferroni_correction takes n_tests (int or iterable, BC)

The function never actually inspected p-value contents — only the count.
Renamed first parameter to n_tests with isinstance check that preserves
all existing iterable callers. Backward compatible.

Note: the '!' is conventional but the change is BC; existing 0.1.x code
that passes an iterable continues to work unchanged."
```

---

### Task 10: Rewrite `idd_correction` docstring

**Files:**
- Modify: `msig/MSig.py` — `Motif.set_significance` docstring

**Context:** Current docstring expands "IDD" as "Independent Dimension Discovery" and falsely claims BH FDR is involved. Per the paper, IDD = "identically distributed dimensions" and the formula is `C(m,q) × p`.

- [ ] **Step 1: Edit the docstring**

In `msig/MSig.py`, locate the `set_significance` docstring's `idd_correction` parameter section:

```
        idd_correction : bool, default=False
            If True, applies IDD (Independent Dimension Discovery) correction by
            adjusting p-value using the Benjamini-Hochberg FDR procedure across
            variable subsets. See Notes for details.
```

Replace with:

```
        idd_correction : bool, default=False
            If True, applies the identically-distributed-dimensions (IDD) correction.
            When the m variables of the target time series are identically
            distributed, the same motif could occur in any q-subset of those
            m variables, so the per-test p-value is multiplied by C(m, q) and
            capped at 1. See paper Section 3.2 ("Motif's statistical significance").

            Use False (default) when variables differ in distribution, scale,
            or units (e.g., the case studies in the paper, all of which set
            idd_correction=False).
```

Also update the `Notes` section's "IDD Correction:" paragraph similarly:

Locate:

```
        IDD Correction:
        When idd_correction=True, the method adjusts for multiple hypothesis testing
        across different variable subsets. For a motif using k variables from m total,
        there are C(m,k) possible k-variable subsets.
```

Replace with:

```
        IDD correction:
        When idd_correction=True, the per-motif p-value is multiplied by C(m, q),
        where m = data_n_variables and q = len(self.variables). This compensates
        for the fact that, under the null, an identically-distributed pattern
        could materialise in any C(m, q) subsets of variables. The product is
        capped at 1.0.
```

- [ ] **Step 2: Verify**

Run: `uv run pytest tests/ -q`
Expected: No test failures (docstring change has no behaviour effect).

Run: `uv run python -c "from msig import Motif; help(Motif.set_significance)" | head -50`
Confirm visually that the new wording shows up.

- [ ] **Step 3: Commit**

```bash
git add msig/MSig.py
git commit -m "docs(msig): correct idd_correction docstring

IDD = identically-distributed dimensions, not 'Independent Dimension
Discovery'. Formula is C(m,q)*p, not Benjamini-Hochberg FDR. Cite paper
§3.2 and note that the published case studies all use idd_correction=False."
```

---

### Task 11: Fix `set_significance` doctest example in `Motif` class docstring

**Files:**
- Modify: `msig/MSig.py` — `Motif` class docstring

**Context:** Public class docstring example calls `motif.set_significance(data_length=1000)` — wrong kwarg, will fail with `TypeError`.

- [ ] **Step 1: Edit the docstring**

In `msig/MSig.py`, locate the `Motif` class docstring `Examples` block:

```
    >>> motif.set_pattern_probability(null_model, vars_indep=True)
    >>> motif.set_significance(data_length=1000)
    >>> print(f"p-value: {motif.pvalue:.4f}")
```

Replace with:

```
    >>> motif.set_pattern_probability(null_model, vars_indep=True)
    >>> motif.set_significance(max_possible_matches=998, data_n_variables=2)
    >>> print(f"p-value: {motif.pvalue:.4f}")
```

(`998 = 1000 - 3 + 1` for a length-3 pattern.)

- [ ] **Step 2: Verify the doctest is at least syntactically runnable**

Create a quick scratch script to ensure the example code works (do not commit the script):

```bash
uv run python -c "
import numpy as np
from msig import Motif, NullModel
pattern = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
variables = [0, 1]
thresholds = [0.1, 0.1]
motif = Motif(pattern, variables, thresholds, n_matches=15)
data = np.random.randn(2, 1000)
null_model = NullModel(data, dtypes=[float, float])
motif.set_pattern_probability(null_model, vars_indep=True)
motif.set_significance(max_possible_matches=998, data_n_variables=2)
print(f'p-value: {motif.pvalue:.4f}')
"
```

Expected: prints a numeric p-value, no `TypeError`.

- [ ] **Step 3: Commit**

```bash
git add msig/MSig.py
git commit -m "docs(msig): fix Motif docstring example using correct kwargs

set_significance(data_length=...) was a non-existent kwarg. Replace with
set_significance(max_possible_matches=N, data_n_variables=m)."
```

---

### Task 12: Switch `__version__` to `importlib.metadata` (single source of truth)

**Files:**
- Modify: `msig/__init__.py`

**Context:** Currently `__init__.py` hard-codes `__version__ = "0.1.2"` while `pyproject.toml` says `0.1.3`. The 0.2.0 release fixes this once and for all by reading version from package metadata.

- [ ] **Step 1: Edit `msig/__init__.py`**

Replace the entire file with:

```python
"""MSig: Statistical significance testing for multivariate time series motifs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("msig")
except PackageNotFoundError:  # editable install before package is registered
    __version__ = "0.0.0+unknown"

from .MSig import (
    NullModel,
    Motif,
    benjamini_hochberg_fdr,
    bonferroni_correction,
)

__all__ = [
    "NullModel",
    "Motif",
    "benjamini_hochberg_fdr",
    "bonferroni_correction",
    "__version__",
]
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "import msig; print(msig.__version__)"`
Expected: prints whatever `pyproject.toml` currently says (will show `0.1.3` until Task 24 bumps to `0.2.0`).

Run: `uv run pytest tests/ -q`
Expected: green.

- [ ] **Step 3: Commit**

```bash
git add msig/__init__.py
git commit -m "fix(msig): __version__ reads from importlib.metadata

Eliminates the ongoing version drift between __init__.py and pyproject.toml.
Single source of truth is now pyproject.toml.version, exposed via
importlib.metadata.version('msig')."
```

---

## Workstream 2: Library structural refactor

### Task 13: Extract `_rect_prob_1d` helper for the initial-position probability

**Files:**
- Modify: `msig/MSig.py` — extract helper, use in `vars_indep_time_markov` for the initial probability
- Test: `tests/test_basic.py`

**Context:** The `vars_indep_time_markov` function has a 30-line if/elif tree for the initial position (lines 308–317) and a parallel one for conditionals. Extract a `_rect_prob_1d(model, dist, time_series, lo, hi, delta_zero, exact_value)` helper to remove duplication.

- [ ] **Step 1: Write the helper test**

Append to `tests/test_basic.py`:

```python
class TestRectangleProbability1D:
    """Tests for the private _rect_prob_1d dispatcher."""

    def test_empirical_with_delta(self):
        import numpy as np
        from msig.MSig import _rect_prob_1d
        ts = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        # P(0.5 ≤ x ≤ 1.5) = 3/5 = 0.6
        result = _rect_prob_1d(model="empirical", dist=None, time_series=ts,
                                lo=0.5, hi=1.5)
        assert abs(result - 0.6) < 1e-12

    def test_empirical_delta_zero(self):
        import numpy as np
        from msig.MSig import _rect_prob_1d
        ts = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        result = _rect_prob_1d(model="empirical", dist=None, time_series=ts,
                                lo=2.0, hi=2.0)
        assert abs(result - 0.4) < 1e-12

    def test_gaussian_theoretical(self):
        from scipy.stats import norm
        from msig.MSig import _rect_prob_1d
        result = _rect_prob_1d(model="gaussian_theoretical", dist=norm(0, 1),
                                time_series=None, lo=-1.0, hi=1.0)
        expected = float(norm.cdf(1) - norm.cdf(-1))
        assert abs(result - expected) < 1e-12

    def test_kde(self):
        import numpy as np
        from scipy.stats import gaussian_kde
        from msig.MSig import _rect_prob_1d
        np.random.seed(0)
        ts = np.random.randn(500)
        kde = gaussian_kde(ts)
        result = _rect_prob_1d(model="kde", dist=kde, time_series=None,
                                lo=-1.0, hi=1.0)
        # Should be close to 0.6827 for standard normal
        assert 0.5 < result < 0.85
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_basic.py::TestRectangleProbability1D -v`
Expected: ImportError on `_rect_prob_1d`.

- [ ] **Step 3: Implement the helper**

In `msig/MSig.py`, just below the existing `_rect_prob_2d`:

```python
def _rect_prob_1d(model: str, dist, time_series, lo: float, hi: float) -> float:
    """
    Probability of a 1D interval [lo, hi] under one of three null-model branches.

    Parameters
    ----------
    model : {'empirical', 'kde', 'gaussian_theoretical'}
        Which null-model branch to use.
    dist : object or None
        For 'kde': scipy.stats.gaussian_kde with `.integrate_box_1d`.
        For 'gaussian_theoretical': scipy.stats.norm-like with `.cdf`.
        For 'empirical': ignored (pass None).
    time_series : np.ndarray or None
        For 'empirical': the 1D series to count over.
        For 'kde' and 'gaussian_theoretical': ignored.
    lo, hi : float
        Lower and upper bounds. lo == hi means exact match (only meaningful
        for 'empirical').

    Returns
    -------
    float
        The interval probability in [0, 1].
    """
    if model == "empirical":
        n = len(time_series)
        if n == 0:
            return 0.0
        if lo == hi:
            count = int(np.sum(time_series == lo))
        else:
            count = int(np.sum(np.logical_and(time_series >= lo, time_series <= hi)))
        return count / n
    if model == "kde":
        return float(dist.integrate_box_1d(lo, hi))
    if model == "gaussian_theoretical":
        return float(dist.cdf(hi) - dist.cdf(lo))
    raise ValueError(f"Unknown model '{model}'")
```

Then refactor `vars_indep_time_markov`'s initial-position block (currently lines 308–317):

```python
            # P(Y_j = x_0^j) - Initial probability
            if delta != 0:
                xi_lower, xi_upper = subsequence[0] - delta, subsequence[0] + delta
            else:
                # Zero-width interval for exact matching
                xi_lower = xi_upper = subsequence[0]

            if self.model == "empirical":
                if delta != 0:
                    count = np.sum(np.logical_and(time_series >= xi_lower, time_series <= xi_upper))
                else:
                    count = np.sum(time_series == subsequence[0])
                p_Q_j *= count / len(time_series) if len(time_series) > 0 else 0.0
            elif self.model == "kde":
                p_Q_j *= float(dist.integrate_box_1d(xi_lower, xi_upper))
            elif self.model == "gaussian_theoretical":
                p_Q_j *= float(dist.cdf(xi_upper) - dist.cdf(xi_lower))
```

with:

```python
            # P(Y_j = x_0^j) - Initial probability
            if delta != 0:
                xi_lower, xi_upper = subsequence[0] - delta, subsequence[0] + delta
            else:
                xi_lower = xi_upper = subsequence[0]

            if self.model == "empirical":
                p_Q_j *= _rect_prob_1d("empirical", None, time_series, xi_lower, xi_upper)
            elif self.model == "kde":
                p_Q_j *= _rect_prob_1d("kde", dist, None, xi_lower, xi_upper)
            elif self.model == "gaussian_theoretical":
                p_Q_j *= _rect_prob_1d("gaussian_theoretical", dist, None, xi_lower, xi_upper)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add msig/MSig.py tests/test_basic.py
git commit -m "refactor(msig): extract _rect_prob_1d helper

Removes the if/elif duplication for the initial-position probability
in vars_indep_time_markov. Used by the same dispatch as _rect_prob_2d."
```

---

### Task 13b: `vars_dep_time_markov` honest stub with paper-Eq. citation

**Files:**
- Modify: `msig/MSig.py` — `NullModel.vars_dep_time_markov` docstring

**Context:** Public method is `NotImplementedError`. Spec requires docstring to cite paper Eq. (4) so users know the formula they would implement.

- [ ] **Step 1: Update the docstring**

In `msig/MSig.py`, locate `vars_dep_time_markov`. Replace its body and docstring with:

```python
    def vars_dep_time_markov(self, motif_subsequence: Sequence[np.ndarray], variables: Sequence[int]) -> float:
        """
        Estimate pattern probability assuming dependent variables and first-order Markov time dependency.

        **Not yet implemented.** The corresponding formula in the paper
        (Silva, Madeira & Henriques, *Pattern Recognition Letters*, 2026,
        Section 3.1, Eq. (4)) is:

            P_M = P( ⋂_{Y_j ∈ J} Y_j ≈ x_k^j )
                  · ∏_{i=k+1..k+s} P( ⋂_{Y_j ∈ J} Y_j ≈ x_i^j | Y_j ≈ x_{i-1}^j )

        Implementing this requires multivariate joint and conditional
        distributions across the selected motif variables. Use
        ``vars_indep=True`` (default) for the currently supported
        independent-variables, first-order-Markov-time formulation.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "Variable-dependency null modelling (paper Eq. 4) is not yet implemented. "
            "Use vars_indep=True in set_pattern_probability() for the supported path."
        )
```

- [ ] **Step 2: Verify**

Run: `uv run pytest tests/ -q`
Expected: green.

Run: `uv run python -c "from msig import NullModel; help(NullModel.vars_dep_time_markov)" | head -20`
Confirm visually that the new docstring shows up.

- [ ] **Step 3: Commit**

```bash
git add msig/MSig.py
git commit -m "docs(msig): cite paper Eq. (4) in vars_dep_time_markov stub"
```

---

## Workstream 3: Experiment scripts

### Task 14: Fix `NameError` in three momenti scripts

**Files:**
- Modify: `experiments/audio/run_momenti.py:187`
- Modify: `experiments/washingmachine/run_momenti.py:179`
- Modify: `experiments/populationdensity/run_momenti.py:177`

**Context:** Each script's `compute_motif_statistics_momenti` writes `"s": m,` referencing an undefined `m`. The loop variable is `s`. This is a `NameError` at runtime.

- [ ] **Step 1: Inspect the failing line in each file**

```bash
grep -n '"s": m,' /Users/miguelgarcao/Desktop/msig/experiments/*/run_momenti.py
```

Expected: three matches, one per file.

- [ ] **Step 2: Fix all three**

In `experiments/audio/run_momenti.py`, `experiments/washingmachine/run_momenti.py`, `experiments/populationdensity/run_momenti.py`:

Change `"s": m,` → `"s": s,` in the `stats_row = { ... }` dict inside `compute_motif_statistics_momenti`.

Example for `audio/run_momenti.py:187`:

```python
        stats_row = {
            "ID": motif_index,
            "k": len(dimensions),
            "Features": ",".join([str(d) for d in dimensions]),
            "s": s,  # was: m
            ...
        }
```

- [ ] **Step 3: Verify**

Run: `grep -n '"s": m,' /Users/miguelgarcao/Desktop/msig/experiments/*/run_momenti.py`
Expected: no matches.

Run: `uv run python -c "from experiments.audio import run_momenti; run_momenti.compute_motif_statistics_momenti.__name__"`
Expected: prints `compute_motif_statistics_momenti` (function importable; doesn't error on the dict).

- [ ] **Step 4: Commit**

```bash
git add experiments/audio/run_momenti.py experiments/washingmachine/run_momenti.py experiments/populationdensity/run_momenti.py
git commit -m "fix(experiments): NameError in three momenti stats helpers

\"s\": m referenced an undefined variable m. Loop variable is s."
```

---

### Task 15: Fix wrong relative paths in three momenti scripts

**Files:**
- Modify: `experiments/audio/run_momenti.py:203`
- Modify: `experiments/washingmachine/run_momenti.py:199`
- Modify: `experiments/populationdensity/run_momenti.py:193`

**Context:** Script lives at `experiments/audio/run_momenti.py`; data lives at `data/audio/imblue.mp3`. Relative path needs `../../data/`, but momenti scripts use `../data/`. The stumpy and lama siblings have it right.

- [ ] **Step 1: Check current state**

```bash
grep -n '"../data/' /Users/miguelgarcao/Desktop/msig/experiments/*/run_momenti.py
```

Expected: three matches.

- [ ] **Step 2: Fix all three**

In each of:
- `experiments/audio/run_momenti.py` line 203: `"../data/audio/imblue.mp3"` → `"../../data/audio/imblue.mp3"`
- `experiments/washingmachine/run_momenti.py` line 199: `"../data/washingmachine/main_readings.csv"` → `"../../data/washingmachine/main_readings.csv"`
- `experiments/populationdensity/run_momenti.py` line 193: `"../data/populationdensity/hourly_saodomingosbenfica.csv"` → `"../../data/populationdensity/hourly_saodomingosbenfica.csv"`

- [ ] **Step 3: Verify**

```bash
grep -n '"../data/' /Users/miguelgarcao/Desktop/msig/experiments/*/run_momenti.py
```
Expected: no matches.

```bash
grep -n '"../../data/' /Users/miguelgarcao/Desktop/msig/experiments/*/run_momenti.py
```
Expected: three matches (one per file).

- [ ] **Step 4: Commit**

```bash
git add experiments/*/run_momenti.py
git commit -m "fix(experiments): correct relative data paths in momenti scripts

Scripts live at experiments/<dataset>/run_momenti.py; data lives at
data/<dataset>/. Relative path is ../../data/, not ../data/. Aligned
with sibling run_stumpy.py and run_lama.py."
```

---

### Task 16: Add `s = 48` to population-density MOMENTI motif lengths

**Files:**
- Modify: `experiments/populationdensity/run_momenti.py:204`

**Context:** Paper Table 4 shows MOMENTI rows for s=4,6,12,24,48; LAMA also covers 48. Script currently has only [4, 6, 12, 24].

- [ ] **Step 1: Locate and edit**

In `experiments/populationdensity/run_momenti.py`, find the `subsequence_lengths` list (around line 204). Change:

```python
    subsequence_lengths = [4, 6, 12, 24]
```

to:

```python
    subsequence_lengths = [4, 6, 12, 24, 48]
```

- [ ] **Step 2: Verify**

```bash
grep 'subsequence_lengths = ' /Users/miguelgarcao/Desktop/msig/experiments/populationdensity/run_momenti.py
```
Expected: shows `[4, 6, 12, 24, 48]`.

- [ ] **Step 3: Commit**

```bash
git add experiments/populationdensity/run_momenti.py
git commit -m "fix(experiments): include s=48 in population-density MOMENTI lengths

Aligns with paper Table 4 and the LAMA script for the same dataset."
```

---

### Task 17: Add `EXCLUSION_ZONE_FACTOR` constant to all 9 experiment scripts and `common_utils.py`

**Files:**
- Modify: `experiments/{audio,populationdensity,washingmachine}/run_{stumpy,lama,momenti}.py` (9 files)
- Modify: `experiments/common_utils.py`

**Context:** Paper §3.2 specifies `l = ⌈0.25 × s⌉`; experiment scripts currently hard-code `r = np.ceil(s/2)` (the value that produced the published tables). Lift to a named module constant `EXCLUSION_ZONE_FACTOR = 0.5`, document the paper-vs-code discrepancy.

- [ ] **Step 1: Locate the existing usages**

```bash
grep -n 'np.ceil(s.*[/2]\|np.ceil([smS] *\\* *0' /Users/miguelgarcao/Desktop/msig/experiments/ -r
```

Expected: about 12 matches across the experiment scripts and `common_utils.py:296`.

- [ ] **Step 2: Add the constant and use it (per file)**

For each of the nine experiment scripts, at the top of the module (after imports, before functions), add:

```python
# Trivial-match exclusion zone, expressed as a fraction of the motif length s.
# Paper §3.2 default is 0.25; the published tables (PRL 2026) were generated
# with 0.5. See REPRODUCING_EXPERIMENTS.md for the paper-vs-code reconciliation.
EXCLUSION_ZONE_FACTOR: float = 0.5
```

Then replace every occurrence of `np.ceil(s / 2)` (or `np.ceil(s/2)`) in that file with:

```python
np.ceil(EXCLUSION_ZONE_FACTOR * s)
```

For `experiments/common_utils.py`, do the same replacement and add the same constant at the top of the module.

- [ ] **Step 3: Verify no `s/2` patterns remain**

```bash
grep -n 'np.ceil(s\s*/\s*2' /Users/miguelgarcao/Desktop/msig/experiments/ -r
```
Expected: no matches.

```bash
grep -n 'EXCLUSION_ZONE_FACTOR' /Users/miguelgarcao/Desktop/msig/experiments/ -r
```
Expected: 10 files contain the constant (9 scripts + common_utils.py).

- [ ] **Step 4: Sanity-check by running one script in dry import mode**

```bash
uv run python -c "from experiments.audio import run_stumpy; print(run_stumpy.EXCLUSION_ZONE_FACTOR)"
```
Expected: prints `0.5`.

- [ ] **Step 5: Commit**

```bash
git add experiments/
git commit -m "refactor(experiments): lift trivial-match exclusion-zone factor to constant

EXCLUSION_ZONE_FACTOR = 0.5 surfaces what was previously a hard-coded
np.ceil(s/2). The paper §3.2 default is 0.25, but the published tables
were generated with 0.5; the comment in each script makes this explicit."
```

---

### Task 18: Rewrite `compare_results.py` to match the actual saved schema

**Files:**
- Modify: `scripts/compare_results.py`

**Context:** The script expects `pvalue`, `significant` (bool), `pattern_probability` columns; experiment scripts save `median_pvalue`, `median_probability`, `#sig_motifs(≤0.01)`, `#sig_hochberg`, `significant` (percentage). Result: silent wrong totals.

- [ ] **Step 1: Inspect what scripts actually save**

```bash
head -1 /Users/miguelgarcao/Desktop/msig/results/washingmachine/stumpy/stumpy_relaxed/summary_motifs_stumpy_relaxed.csv
```

Expected: columns `s,#motifs,avg_n_matches,avg_n_features,median_probability,median_pvalue,#sig_motifs(≤0.01),significant,#sig_hochberg`.

- [ ] **Step 2: Rewrite the schema-dependent parts of `scripts/compare_results.py`**

Update `ResultsComparator.generate_summary_statistics`. Replace the column-checking block that uses `"significant"` (as bool) and `"pvalue"` (singular) with:

```python
            # Schema produced by experiment scripts (v0.2.0):
            # s, #motifs, avg_n_matches, avg_n_features, median_probability,
            # median_pvalue, #sig_motifs(≤0.01), significant (percentage),
            # #sig_hochberg
            if "#motifs" in df.columns:
                summary["total_motifs"] = int(df["#motifs"].sum())

            if "#sig_motifs(≤0.01)" in df.columns:
                summary["significant_motifs"] = int(df["#sig_motifs(≤0.01)"].sum())
            elif "#sig_hochberg" in df.columns:
                summary["significant_motifs_hochberg"] = int(df["#sig_hochberg"].sum())

            total_motifs = summary.get("total_motifs", 0)
            sig = summary.get("significant_motifs", 0)
            summary["pct_significant"] = (100.0 * sig / total_motifs) if total_motifs else 0.0

            if "median_pvalue" in df.columns:
                pv = pd.to_numeric(df["median_pvalue"], errors="coerce").dropna()
                if len(pv) > 0:
                    summary["min_median_pvalue"] = float(pv.min())
                    summary["max_median_pvalue"] = float(pv.max())

            if "median_probability" in df.columns:
                mp = pd.to_numeric(df["median_probability"], errors="coerce").dropna()
                if len(mp) > 0:
                    summary["min_median_prob"] = float(mp.min())
                    summary["max_median_prob"] = float(mp.max())

            if "s" in df.columns:
                summary["min_length"] = int(df["s"].min())
                summary["max_length"] = int(df["s"].max())
```

In `compare_methods` and `compare_datasets`, change:

```python
            if "significant" in df.columns:
                n_sig = df["significant"].sum()
                pct_sig = 100 * n_sig / len(df) if len(df) > 0 else 0
```

to:

```python
            if "#sig_motifs(≤0.01)" in df.columns and "#motifs" in df.columns:
                n_sig = int(df["#sig_motifs(≤0.01)"].sum())
                total = int(df["#motifs"].sum())
                pct_sig = (100.0 * n_sig / total) if total else 0.0
            else:
                n_sig = 0
                pct_sig = 0.0
```

Update the markdown report sections (`generate_markdown_report`) that mention `pvalue` and `significant` to use `median_pvalue` and `#sig_motifs(≤0.01)` accordingly. Add a header line at the top of the function noting the expected schema version: `# Schema: msig 0.2.0 experiment-script CSV`.

- [ ] **Step 3: Verify against existing CSVs**

Run: `uv run python scripts/compare_results.py --output /tmp/compare_test.md`
Expected: completes without `KeyError`, produces `/tmp/compare_test.md`.

```bash
head -50 /tmp/compare_test.md
```
Expected: meaningful summary table with non-zero counts.

- [ ] **Step 4: Commit**

```bash
git add scripts/compare_results.py
git commit -m "fix(scripts): compare_results.py schema matches experiment outputs

Previously it expected columns 'pvalue', 'significant' (bool),
'pattern_probability' — none of which the experiment scripts produce.
Now reads the actual schema (median_pvalue, median_probability,
#sig_motifs(≤0.01), #sig_hochberg, significant as percentage)."
```

---

### Task 19: Lift `average_delta` to module-level `AVERAGE_DELTA` and document mSTUMP conservative-regime comment

**Files:**
- Modify: `experiments/audio/run_stumpy.py`, `experiments/audio/run_lama.py`
- Modify: `experiments/populationdensity/run_stumpy.py`, `experiments/populationdensity/run_lama.py`
- Modify: `experiments/washingmachine/run_stumpy.py`, `experiments/washingmachine/run_lama.py`

**Context:** `average_delta = 0.3` is hard-coded inside `main()` of each script. Lift to a module-level `AVERAGE_DELTA` constant. Also add a comment to the conservative-regime block in `run_stumpy.py` clarifying the formula it uses.

- [ ] **Step 1: For each `run_stumpy.py` and `run_lama.py`** (6 files), add at the top after imports:

```python
# Per-variable approximate-match tolerance δ used to derive the maximum
# allowed Z-normalized Euclidean distance between motif occurrences.
# See paper §3.3 for D_max formulas; this value is identical across
# datasets in the published experiments.
AVERAGE_DELTA: float = 0.3
```

Replace all in-function `average_delta = 0.3` assignments with usage of `AVERAGE_DELTA` (or remove the local assignment and use the constant directly).

- [ ] **Step 2: For each `run_stumpy.py`, locate the `'conservative'` regime config and update its comment**:

Replace the `# Conservative: ...` line in the `modes` dict with:

```python
            'description': 'Conservative: high-specificity regime (D_max = sqrt(s)*delta*0.5; '
                           'note: paper §3.3 conservative formula is D_max = (1/q)Σδ_j '
                           'without sqrt(s); the published tables use the sqrt(s)*0.5 form)'
```

- [ ] **Step 3: Verify**

```bash
grep -n 'AVERAGE_DELTA' /Users/miguelgarcao/Desktop/msig/experiments/ -r
```
Expected: 6 module-level constant lines + their usages.

```bash
grep -n 'average_delta = 0.3' /Users/miguelgarcao/Desktop/msig/experiments/ -r
```
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add experiments/
git commit -m "refactor(experiments): AVERAGE_DELTA constant, document mSTUMP conservative

Lifts the per-script average_delta=0.3 hardcode to a module-level
AVERAGE_DELTA constant with a citation to paper §3.3. Also clarifies
in the mSTUMP conservative-regime config that it uses the sqrt(s)*0.5
form (which produced the published tables), not the paper's
'D_max = (1/q)Σδ_j' formula."
```

---

### Task 20: Annotate `idd_correction=False` in every experiment script

**Files:**
- Modify: all 9 experiment scripts

**Context:** Each call to `set_significance(..., idd_correction=False)` should have a one-line comment citing why (variables are not identically distributed across case studies).

- [ ] **Step 1: Find all call-sites**

```bash
grep -n 'idd_correction=' /Users/miguelgarcao/Desktop/msig/experiments/ -r
```

Expected: ~9 matches, one per script.

- [ ] **Step 2: Add a comment immediately above each**

Pattern:

```python
        # Variables are not identically distributed (different scales/units/dynamics);
        # see REPRODUCING_EXPERIMENTS.md §IDD applicability.
        p_value = motif_obj.set_significance(max_possible_matches, n_vars, idd_correction=False)
```

Apply once per script.

- [ ] **Step 3: Commit**

```bash
git add experiments/
git commit -m "docs(experiments): annotate idd_correction=False with rationale"
```

---

### Task 20b: Centralise dataset path derivation in `experiments/common_utils.py`

**Files:**
- Modify: `experiments/common_utils.py`
- Modify: each of the nine experiment scripts (`run_{stumpy,lama,momenti}.py` × 3 datasets)

**Context:** Tasks 14–15 fixed individual path bugs. Centralise the derivation into a helper so future drift can't recur.

- [ ] **Step 1: Add `get_dataset_paths` to `experiments/common_utils.py`**

```python
import os

def get_dataset_paths(dataset: str) -> dict[str, str]:
    """
    Return absolute paths to the input data and results directory for a dataset.

    The function is callable from any experiments/<dataset>/run_*.py script
    and from the repository root, regardless of cwd.

    Parameters
    ----------
    dataset : str
        One of "audio", "populationdensity", "washingmachine", "synthetic".

    Returns
    -------
    dict
        Keys: "data_dir", "data_file", "results_dir". Values are absolute paths.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    files = {
        "audio": "imblue.mp3",
        "populationdensity": "hourly_saodomingosbenfica.csv",
        "washingmachine": "main_readings.csv",
        "synthetic": "multivar_time_series.csv",
    }
    if dataset not in files:
        raise ValueError(f"Unknown dataset '{dataset}'; expected one of {list(files)}")
    data_dir = os.path.join(repo_root, "data", dataset)
    return {
        "data_dir": data_dir,
        "data_file": os.path.join(data_dir, files[dataset]),
        "results_dir": os.path.join(repo_root, "results", dataset),
    }
```

- [ ] **Step 2: Adopt in each experiment script**

In each of the nine scripts, replace the per-script absolute-path setup at the top of `main()` with:

```python
from experiments.common_utils import get_dataset_paths
paths = get_dataset_paths("<dataset>")  # one of audio/populationdensity/washingmachine
data_path = paths["data_file"]   # or audio_path for run_*audio*.py
results_dir = os.path.join(paths["results_dir"], "<method>")
```

(Adjust to the script's existing variable names.)

- [ ] **Step 3: Verify**

Run: `uv run pytest tests/test_experiment_smoke.py -m integration -v`
Expected: 9 import tests still PASS, plus the path is now uniform across scripts.

```bash
grep -rn '../../data/\|"\\.\\./data/' /Users/miguelgarcao/Desktop/msig/experiments/
```
Expected: no matches (centralised in `common_utils.py`).

- [ ] **Step 4: Commit**

```bash
git add experiments/
git commit -m "refactor(experiments): centralise dataset paths in common_utils.get_dataset_paths"
```

---

## Workstream 4: Packaging

### Task 21: Bump version to 0.2.0 in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:3`

- [ ] **Step 1: Edit**

Change `version = "0.1.3"` to `version = "0.2.0"`.

- [ ] **Step 2: Verify**

```bash
uv run python -c "import msig; print(msig.__version__)"
```
Expected: `0.2.0` (Task 12 made this read from `pyproject.toml`).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0"
```

---

### Task 22: Align Python version metadata across all configuration files

**Files:**
- Modify: `pyproject.toml`
- Modify: `environment.yml`

**Context:** `pyproject.toml` has `requires-python = ">=3.11,<3.15"` but classifiers list only 3.12 and `[tool.mypy] python_version = "3.10"`. PyPI 0.1.3 advertises ">=3.12". Align everything to `>=3.11,<3.14`.

- [ ] **Step 1: Edit `pyproject.toml`**

Change:

```toml
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.11,<3.15"
```

to:

```toml
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
requires-python = ">=3.11,<3.14"
```

Change `[tool.black] target-version = ["py310", "py311", "py312", "py313"]` to `["py311", "py312", "py313"]`.

Change `[tool.mypy] python_version = "3.10"` to `python_version = "3.11"`.

- [ ] **Step 2: Edit `environment.yml`**

Change `python=3.13` (or whatever the current pin is) to `python>=3.11,<3.14`.

- [ ] **Step 3: Verify**

```bash
grep -n 'python.*3\.\(10\|14\|15\)' /Users/miguelgarcao/Desktop/msig/pyproject.toml /Users/miguelgarcao/Desktop/msig/environment.yml
```
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml environment.yml
git commit -m "chore: align Python version metadata to >=3.11,<3.14

Removes drift between requires-python, classifiers, mypy, black, and
environment.yml. Adds License/OS/Topic classifiers."
```

---

### Task 23: Create `msig/py.typed` marker file

**Files:**
- Create: `msig/py.typed`

**Context:** `pyproject.toml [tool.setuptools.package-data]` declares `msig = ["py.typed"]` but the file doesn't exist. Without it, downstream type-checking does not activate (PEP 561).

- [ ] **Step 1: Create the empty file**

```bash
touch /Users/miguelgarcao/Desktop/msig/msig/py.typed
```

- [ ] **Step 2: Verify**

```bash
ls -la /Users/miguelgarcao/Desktop/msig/msig/py.typed
```
Expected: 0-byte file exists.

- [ ] **Step 3: Commit**

```bash
git add msig/py.typed
git commit -m "chore: add empty msig/py.typed marker (PEP 561)"
```

---

### Task 24: Rewrite `requirements.txt`

**Files:**
- Modify: `requirements.txt`

**Context:** Currently has self-pin `msig==0.1.3` and drifts from `pyproject.toml`'s extras.

- [ ] **Step 1: Replace contents**

Overwrite `requirements.txt` with:

```
# Convenience requirements file for cloned-from-source contributors.
# Source of truth is pyproject.toml — `uv sync` or
# `pip install -e ".[experiments,dev]"` is preferred.

# Core
numpy>=1.20.0
scipy>=1.7.0

# Experiments
pandas>=1.3.0
matplotlib>=3.4.0
stumpy>=1.11.0
librosa>=0.9.0
statsmodels>=0.13.0
jinja2>=3.0.0
leitmotif
psutil>=5.8.0
# MOMENTI: Linux/Windows only, install separately:
# pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs

# Development
pytest>=7.0.0
pytest-cov>=3.0.0
black>=23.0.0
isort>=5.0.0
```

- [ ] **Step 2: Verify**

```bash
cat /Users/miguelgarcao/Desktop/msig/requirements.txt | head
```
Expected: shown contents.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: rewrite requirements.txt to align with pyproject.toml extras

Drops the self-pin (msig==0.1.3); adds the previously missing experiment
deps (jinja2, leitmotif, psutil) and dev deps (black, isort)."
```

---

### Task 25: Rewrite `environment.yml`

**Files:**
- Modify: `environment.yml`

**Context:** Comments out half the experiment deps; default conda env therefore has stumpy but not librosa/pandas/matplotlib.

- [ ] **Step 1: Replace `environment.yml` with**:

```yaml
name: msig
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.11,<3.14
  - pip
  - pip:
      - msig[experiments,dev]
```

- [ ] **Step 2: Verify**

`uv run python -c "import yaml; print(yaml.safe_load(open('environment.yml')))"`
Expected: parses without error.

- [ ] **Step 3: Commit**

```bash
git add environment.yml
git commit -m "chore: simplify environment.yml to install msig[experiments,dev]"
```

---

### Task 26: Add `black` and `isort` to `dev` extra in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit the `dev` extra**

Replace:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
]
```

with:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=23.0.0",
    "isort>=5.0.0",
    "mypy>=1.0.0",
]
```

- [ ] **Step 2: Sync and verify**

```bash
cd /Users/miguelgarcao/Desktop/msig && uv sync --extra dev
uv run black --version
uv run isort --version
uv run mypy --version
```
Expected: all three commands print versions.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add black, isort, mypy to dev extra

These were referenced in CLAUDE.md but never declared as deps."
```

---

### Task 27: Make `validate_reproducibility.py` core-install-safe

**Files:**
- Modify: `validate_reproducibility.py`

**Context:** Currently does `import numpy as np` at module top — fine since numpy is core. But experiment-extra deps are also imported; on a `pip install msig` (no extras) install, the script fails. Push optional imports inside their respective check blocks.

- [ ] **Step 1: Edit**

Move the `import` of any experiment-extra dep into the check that uses it. The current script structure is mostly OK but the `data/` URL pointers are missing (Section 5 of the spec). Add to the `missing_data` block:

```python
    if missing_data:
        logger.warning(f"⚠️ Missing data files: {', '.join(missing_data)}")
        logger.info("📋 Where to obtain:")
        logger.info("   Audio: provide your own MP3 of the song the paper analyses; place at data/audio/imblue.mp3")
        logger.info("   Washing machine: LARCO dataset DOI 10.5281/zenodo.17081452")
        logger.info("   Population density: synthetic Kaggle subset (https://www.kaggle.com/datasets/miguelgarcaosilva/synthetic-mp-data-in-lisbon)")
```

- [ ] **Step 2: Verify in a fresh-env-equivalent**

```bash
uv run python validate_reproducibility.py
```
Expected: completes successfully, prints the new "Where to obtain" lines if `data/*` is missing.

- [ ] **Step 3: Commit**

```bash
git add validate_reproducibility.py
git commit -m "docs(validate): point users to dataset sources when data/ is empty"
```

---

### Task 28: Update `test_setup.sh` to import every declared dep

**Files:**
- Modify: `test_setup.sh`

**Context:** Script currently imports a subset; missing `statsmodels`, `jinja2`, `psutil`, `leitmotif`.

- [ ] **Step 1: Edit the imports section**

Locate the block that imports test deps and ensure it covers:

```bash
uv run python -c "
import numpy
import scipy
import pandas
import matplotlib
import stumpy
import librosa
import statsmodels
import jinja2
import psutil
import leitmotif
import msig
print('all imports succeeded')
"
```

(Adjust the surrounding bash plumbing to match the existing script's style.)

- [ ] **Step 2: Run**

```bash
bash /Users/miguelgarcao/Desktop/msig/test_setup.sh
```
Expected: completes without ImportError (assuming `uv sync --extra experiments,dev` has been run).

- [ ] **Step 3: Commit**

```bash
git add test_setup.sh
git commit -m "chore(test_setup.sh): import every dep declared in pyproject.toml"
```

---

### Task 29: Clean up `.gitignore` and remove committed `msig.egg-info/`

**Files:**
- Modify: `.gitignore`
- Delete: `msig.egg-info/` (as a directory in the working tree)

- [ ] **Step 1: Add patterns to `.gitignore`**

Append (before the final newlines):

```
# Linter/type-checker caches
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

# Numerical artifacts
*.npz
*.parquet
```

- [ ] **Step 2: Remove the egg-info directory from the working tree (and git)**

```bash
git rm -r --cached msig.egg-info/ 2>/dev/null || true
rm -rf /Users/miguelgarcao/Desktop/msig/msig.egg-info
```

(`git rm --cached` is the only "destructive" git op here — it un-tracks the directory but leaves files alone; we then delete the local copy.)

- [ ] **Step 3: Verify**

```bash
ls /Users/miguelgarcao/Desktop/msig/msig.egg-info 2>&1 | head
```
Expected: `No such file or directory`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: tighten .gitignore and untrack msig.egg-info"
```

---

### Task 30: Delete `PYPI_DESCRIPTION.md`

**Files:**
- Delete: `PYPI_DESCRIPTION.md`

**Context:** Dead file — `pyproject.toml` uses `readme = "README.md"`. The file has already drifted (year 2025 vs README 2024).

- [ ] **Step 1: Confirm it isn't referenced**

```bash
grep -rn 'PYPI_DESCRIPTION' /Users/miguelgarcao/Desktop/msig --exclude-dir=.git
```
Expected: only matches inside `PYPI_DESCRIPTION.md` itself or in this plan/spec doc — no `pyproject.toml` reference.

- [ ] **Step 2: Delete**

```bash
git rm /Users/miguelgarcao/Desktop/msig/PYPI_DESCRIPTION.md
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: delete unused PYPI_DESCRIPTION.md (long-description is README.md)"
```

---

## Workstream 5: Documentation

### Task 31: Update citation in `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the citation block**

Locate the BibTeX block in `README.md` (around lines 165–169). Replace with:

```bibtex
@article{silva2026and,
  title={On Why and How Statistical Significance Criteria Can Guide Multivariate Time Series Motif Analysis},
  author={Silva, Miguel G and Madeira, Sara C and Henriques, Rui},
  journal={Pattern Recognition Letters},
  year={2026},
  publisher={Elsevier}
}
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README citation to PRL 2026"
```

---

### Task 32: Create `CITATION.cff`

**Files:**
- Create: `CITATION.cff`

- [ ] **Step 1: Create the file**

```yaml
cff-version: 1.2.0
message: "If you use MSig in your research, please cite both the software and the paper."
title: MSig — Statistical Significance for Multivariate Time Series Motifs
authors:
  - family-names: Silva
    given-names: Miguel G.
    affiliation: "LASIGE, Faculdade de Ciências, Universidade de Lisboa"
    email: mmgsilva@ciencias.ulisboa.pt
  - family-names: Madeira
    given-names: Sara C.
    affiliation: "LASIGE, Faculdade de Ciências, Universidade de Lisboa"
  - family-names: Henriques
    given-names: Rui
    affiliation: "INESC-ID, IST, Universidade de Lisboa"
repository-code: "https://github.com/MiguelGarcaoSilva/msig"
license: MIT
preferred-citation:
  type: article
  authors:
    - family-names: Silva
      given-names: Miguel G.
    - family-names: Madeira
      given-names: Sara C.
    - family-names: Henriques
      given-names: Rui
  title: "On Why and How Statistical Significance Criteria Can Guide Multivariate Time Series Motif Analysis"
  journal: "Pattern Recognition Letters"
  year: 2026
  publisher: Elsevier
```

- [ ] **Step 2: Validate**

```bash
uv run python -c "import yaml; yaml.safe_load(open('CITATION.cff'))"
```
Expected: parses without error.

- [ ] **Step 3: Commit**

```bash
git add CITATION.cff
git commit -m "docs: add CITATION.cff (PRL 2026)"
```

---

### Task 33: Fix `simple_example.py` final-print refs

**Files:**
- Modify: `examples/simple_example.py`

**Context:** Lines 239–240 reference `REPRODUCING_EXPERIMENTS.md` and `CONTRIBUTING.md`. The first will exist after Task 35; the second after Task 34. Re-point now.

- [ ] **Step 1: Edit**

Replace the existing lines 235–241 print block with:

```python
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("\nFor more information, see:")
    print("  - README.md: Overview and quick start")
    print("  - INSTALLATION.md: Setup")
    print("  - REPRODUCING_EXPERIMENTS.md: How to recreate the paper's tables")
    print("  - CONTRIBUTING.md: Development guidelines")
    print("=" * 70 + "\n")
```

- [ ] **Step 2: Verify**

```bash
uv run python examples/simple_example.py | tail -10
```
Expected: prints the final block without errors (the example runs end-to-end).

- [ ] **Step 3: Commit**

```bash
git add examples/simple_example.py
git commit -m "docs(simple_example): align final-print refs with files that will exist"
```

---

### Task 34: Create `CONTRIBUTING.md`

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Create the file**

```markdown
# Contributing to MSig

Thanks for your interest in MSig. This document explains how to set up a
development environment and submit changes.

## Development setup

```bash
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig
uv sync --extra experiments --extra dev   # installs everything you need
```

## Running tests

```bash
uv run pytest tests/ -q                                 # unit tests
uv run pytest -m integration -v                         # smoke tests for experiment scripts
uv run pytest -m slow -v                                # paper-reproducibility regression suite
uv run pytest --cov=msig --cov-report=term-missing      # with coverage
```

## Formatting and linting

```bash
uv run black msig/ tests/ examples/ experiments/ scripts/
uv run isort msig/ tests/ examples/ experiments/ scripts/
uv run mypy msig/
```

## Pull requests

- Branch off `main`. Use a short, kebab-case branch name (`fix/2d-gaussian-cdf`).
- Keep commits focused; conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Reference any related issue in the PR description.
- Include tests for behaviour changes.

## Reproducing the paper

See `REPRODUCING_EXPERIMENTS.md`.

## License

By contributing, you agree that your contributions are licensed under the MIT
License (see `LICENSE`).
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md"
```

---

### Task 35: Create `REPRODUCING_EXPERIMENTS.md`

**Files:**
- Create: `REPRODUCING_EXPERIMENTS.md`

- [ ] **Step 1: Create the file**

```markdown
# Reproducing the MSig Experiments

This document describes how to recreate the empirical tables in
*On Why and How Statistical Significance Criteria Can Guide Multivariate
Time Series Motif Analysis* (Silva, Madeira & Henriques, *Pattern Recognition
Letters*, 2026, Elsevier).

## 1. Environment

- Python **3.11–3.13** (3.10 and earlier are not supported; 3.14+ may have LAMA compatibility issues).
- `uv sync --extra experiments` installs all runtime dependencies.
- Optional: `uv pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs` for MOMENTI on Linux/Windows.
- macOS: `brew install ffmpeg` (audio experiments).
- Linux: `sudo apt-get install ffmpeg`.

## 2. Data

Place the following files locally; none of them are committed to git
(see `.gitignore`).

| File | Path | Source |
|---|---|---|
| Audio (song MFCCs) | `data/audio/imblue.mp3` | Provide your own MP3 of the analysed song |
| Washing machine | `data/washingmachine/main_readings.csv` | LARCO dataset, [Zenodo DOI 10.5281/zenodo.17081452](https://doi.org/10.5281/zenodo.17081452) |
| Population density | `data/populationdensity/hourly_saodomingosbenfica.csv` | Vodafone Lisbon mobility data — under NDA. A synthetic alternative: [Kaggle dataset](https://www.kaggle.com/datasets/miguelgarcaosilva/synthetic-mp-data-in-lisbon) |
| Synthetic | `data/synthetic/multivar_time_series.csv` | Optional; used by paper RQ1 figure |

## 3. Running

```bash
uv run python validate_reproducibility.py     # sanity checks
uv run python run_experiments.py --all        # ~140 minutes total
uv run python scripts/compare_results.py      # cross-method comparison report
```

Per-experiment commands are listed in `README.md`.

Outputs land in `results/<dataset>/<method>/summary_motifs_*.csv` and
`results/<dataset>/<method>/table_motifs_*.csv` (the latter is in `.gitignore`).

## 4. Paper-vs-code reconciliation

The published v0.2.0 of MSig differs subtly from the code that generated
the paper's tables. This section makes those differences explicit.

### 4.1 Trivial-match exclusion-zone factor

Paper §3.2 says `l = ⌈0.25 × s⌉`. The published tables were generated with
`l = ⌈0.5 × s⌉`. The code now exposes this as `EXCLUSION_ZONE_FACTOR = 0.5`
in every experiment script. To use the paper's stated default, set
`EXCLUSION_ZONE_FACTOR = 0.25`.

### 4.2 mSTUMP "conservative" regime

Paper §3.3 conservative formula: `D_max = (1/q) Σ δ_j` (no `√s` scaling).
Code conservative formula: `D_max = √s · δ · 0.5`. The code formula is what
generated the published tables; the regime name is preserved for
backward-compatibility.

### 4.3 IDD correction

`idd_correction=False` is set in all nine experiment scripts. The case-study
variables are not identically distributed (different physical quantities,
scales, and dynamics); enabling the correction would inflate p-values
spuriously. See spec §Open questions for the per-case-study judgement.

### 4.4 The 2D-Gaussian rectangle CDF fix

`msig 0.2.0` corrects a bug in the `gaussian_theoretical` null model
(inclusion-exclusion for 2D rectangle probabilities). None of the published
case studies use `gaussian_theoretical` (all use `model="empirical"`), so
the published tables are unaffected by this fix.

### 4.5 Pinning to paper-revision code

To reproduce the paper's exact tables, check out the tag corresponding to
paper-submission code:

```bash
git checkout v0.1.1   # or v0.1.0, depending on submission date
```

The `summary_motifs_*.csv` files committed in `results/` of the current
`main` are regenerated by 0.2.0 and will differ slightly from the paper's
tables (regime parameter changes, the empirical-conditional consistency
fix, and the lag-1-marginal alignment).

## 5. Known divergence

`tests/golden/diff-vs-paper.md` records per-row deltas where the regenerated
CSVs differ from the paper. This file is generated by
`scripts/compare_results.py --diff-paper` (added in 0.2.0) for transparency.
```

- [ ] **Step 2: Commit**

```bash
git add REPRODUCING_EXPERIMENTS.md
git commit -m "docs: add REPRODUCING_EXPERIMENTS.md with paper-vs-code reconciliation"
```

---

### Task 36: Fix README quick-start `motif_pattern` line

**Files:**
- Modify: `README.md`

**Context:** Line 49 uses `data[:, 5:15]`; should be `data[motif_vars, 5:15]` for clarity even though it works coincidentally with `motif_vars=[0,1,2]`.

- [ ] **Step 1: Edit**

Locate in `README.md`:

```python
motif_pattern = data[:, 5:15]  # Extract pattern from position 5
```

Replace with:

```python
motif_pattern = data[motif_vars, 5:15]  # Extract pattern from position 5 across selected variables
```

- [ ] **Step 2: Run the README quick-start to confirm**

```bash
uv run python -c "$(awk '/```python/,/```/' README.md | grep -v '```' | head -50)"
```
Expected: prints `Pattern probability: ...` and `P-value: ...` lines, no error.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(README): use motif_vars in quick-start subsequence extraction"
```

---

### Task 37: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

**Context:** Mentions `black`/`isort` commands but those aren't declared in `dev` extra. Task 26 fixed that. Update CLAUDE.md to reflect the new `dev` extra and to document `pattern_prob_floor`.

- [ ] **Step 1: Edit the "Code Formatting" section**

Verify the existing lines work (they should, post-Task 26).

- [ ] **Step 2: Update the "Architecture > Core Classes" section**

Append a brief note about `pattern_prob_floor` and the corrected `idd_correction` documentation.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE): align with 0.2.0 dev extras and new kwargs"
```

---

### Task 38: Refresh `examples/example.ipynb`

**Files:**
- Modify: `examples/example.ipynb`

**Context:** Stale outputs; brittle `sys.path.insert` cwd hack; needs to verify it runs top-to-bottom.

- [ ] **Step 1: Open the notebook and clear outputs**

```bash
uv run jupyter nbconvert --clear-output --inplace examples/example.ipynb
```

- [ ] **Step 2: Replace the path-hack cell with an instructional markdown cell**

Find the cell containing:

```python
sys.path.insert(0, os.path.dirname(os.getcwd()))
```

Replace its source with a new first markdown cell:

> **Setup**: From the repo root, run `uv pip install -e ".[experiments]"` (or `pip install -e ".[experiments]"`). Then this notebook can `import msig` directly.

Replace the code cell's content with simply `from msig import Motif, NullModel` and remove the `sys.path` hack.

- [ ] **Step 3: Execute the notebook end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace examples/example.ipynb
```
Expected: completes without error.

- [ ] **Step 4: Commit**

```bash
git add examples/example.ipynb
git commit -m "docs(example.ipynb): clear stale outputs and remove sys.path hack"
```

---

## Workstream 6: Test suite expansion

### Task 39: Replace BH FDR canonical-case test

**Files:**
- Modify: `tests/test_statistical_methods.py`

**Context:** The current `test_benjamini_hochberg_none_significant` asserts the function returns `α` when nothing is significant — non-standard semantics that diverge from R's `p.adjust`. Replace with the canonical Benjamini & Hochberg (1995) §3 example.

- [ ] **Step 1: Replace the test**

In `tests/test_statistical_methods.py`, locate `test_benjamini_hochberg_none_significant`:

```python
    def test_benjamini_hochberg_none_significant(self):
        """Test Benjamini-Hochberg when no p-values are significant."""
        p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.05)
        # None should be significant, return FDR threshold
        assert result == 0.05
```

Replace with:

```python
    def test_benjamini_hochberg_canonical_example(self):
        """Benjamini & Hochberg (1995) §3 example.

        For the 10 ordered p-values below at α=0.05, the largest p that
        satisfies p_(i) ≤ (i/n)·α is p_(4) = 0.0095 → critical value 0.0095.
        """
        p_values = [
            0.0001, 0.0004, 0.0019, 0.0095, 0.0201,
            0.0278, 0.0298, 0.0344, 0.0459, 0.3240,
        ]
        critical = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.05)
        assert abs(critical - 0.0095) < 1e-12
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_statistical_methods.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_statistical_methods.py
git commit -m "test: BH FDR canonical example replaces non-standard 'returns α' fixation"
```

---

### Task 40: Add experiment-script smoke tests

**Files:**
- Create: `tests/test_experiment_smoke.py`

**Context:** Catch `NameError`/path/schema/import bugs in experiment scripts before they hit a full run.

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_experiment_smoke.py -m integration -v`
Expected: 9 import tests PASS; the compare_results test PASSES.

- [ ] **Step 3: Commit**

```bash
git add tests/test_experiment_smoke.py
git commit -m "test: experiment-script smoke tests (imports + compare_results)"
```

---

### Task 41: Add paper-reproducibility golden-CSV regression test

**Files:**
- Create: `tests/test_paper_reproducibility.py`
- Create: `tests/golden/.gitkeep` (empty marker)

**Context:** The full `run_experiments.py --all` step produces fresh CSVs. Lock them in as `tests/golden/` snapshots so future drift is caught.

- [ ] **Step 1: Create the test scaffolding**

```python
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
```

```bash
mkdir -p /Users/miguelgarcao/Desktop/msig/tests/golden
touch /Users/miguelgarcao/Desktop/msig/tests/golden/.gitkeep
```

- [ ] **Step 2: Verify the test discovers (and skips because goldens are absent)**

Run: `uv run pytest tests/test_paper_reproducibility.py -m slow -v`
Expected: 15 tests, all SKIPPED with "Golden missing".

- [ ] **Step 3: Commit**

```bash
git add tests/test_paper_reproducibility.py tests/golden/.gitkeep
git commit -m "test: paper-reproducibility skeleton; goldens to be populated post-validation"
```

---

## Workstream 7: Validation

### Task 42: Run unit tests, smoke tests, type-check, lint

**Files:**
- (no file edits — verification step)

- [ ] **Step 1: Unit tests**

Run: `uv run pytest tests/ -m "not slow and not integration" -q`
Expected: all green.

- [ ] **Step 2: Integration smokes**

Run: `uv run pytest -m integration -q`
Expected: all green.

- [ ] **Step 3: Type check**

Run: `uv run mypy msig/`
Expected: no errors on `msig/` (warnings acceptable).

- [ ] **Step 4: Lint**

Run: `uv run black --check msig/ tests/ examples/ experiments/ scripts/ run_experiments.py validate_reproducibility.py`
Run: `uv run isort --check-only msig/ tests/ examples/ experiments/ scripts/ run_experiments.py validate_reproducibility.py`
Expected: no diff. If diffs exist, run black/isort without `--check` to auto-format and commit:

```bash
uv run black msig/ tests/ examples/ experiments/ scripts/ run_experiments.py validate_reproducibility.py
uv run isort msig/ tests/ examples/ experiments/ scripts/ run_experiments.py validate_reproducibility.py
git add -u
git commit -m "style: black + isort"
```

- [ ] **Step 5: Validation script**

Run: `uv run python validate_reproducibility.py`
Expected: all green checks; data files all show ✓ since they were synced earlier.

---

### Task 43: Run all experiments

**Files:**
- (no file edits — full experiment run; ~2.5 hours wall time)

**Context:** macOS host means MOMENTI scripts will be auto-skipped. Run STUMPY+LAMA on all datasets locally; MOMENTI must be done on a Linux/Windows box separately before tagging the release.

- [ ] **Step 1: Full local run (STUMPY + LAMA)**

Run: `uv run python run_experiments.py --all --skip-momenti`
Expected: all 6 (3 datasets × 2 methods) exit 0; each produces non-empty `summary_motifs_*.csv`.

- [ ] **Step 2: Cross-method comparison**

Run: `uv run python scripts/compare_results.py`
Expected: produces `RESULTS_COMPARISON.md` and `results_summary.csv` without `KeyError`.

- [ ] **Step 3: Inspect for sanity**

Run: `head /Users/miguelgarcao/Desktop/msig/RESULTS_COMPARISON.md`
Confirm visually: per-dataset summaries show non-zero motif counts and reasonable significance percentages.

- [ ] **Step 4: MOMENTI deferred-to-Linux note**

Add a clearly-named file `MOMENTI_PENDING.md` at the repo root (will be deleted after the Linux run):

```markdown
# MOMENTI experiments pending

Three MOMENTI scripts have not been validated on this branch:
- experiments/audio/run_momenti.py
- experiments/populationdensity/run_momenti.py
- experiments/washingmachine/run_momenti.py

Reason: dev machine is macOS; MOMENTI is Linux/Windows only.

Action before tagging v0.2.0: run on a Linux/Windows host with the same
data files (rsync from miguel@10.10.4.60:~/raid_backup/ as documented in
the design doc) and verify each produces a non-empty
results/<dataset>/momenti/summary_motifs_momenti.csv. Delete this file.
```

- [ ] **Step 5: Commit (without MOMENTI CSVs)**

```bash
git add results/ RESULTS_COMPARISON.md results_summary.csv MOMENTI_PENDING.md
git commit -m "chore: regenerate result CSVs (STUMPY + LAMA, MOMENTI pending Linux run)"
```

---

### Task 44: Populate `tests/golden/` snapshots

**Files:**
- Create: `tests/golden/*.csv` (regenerated CSV copies)

- [ ] **Step 1: Copy each fresh `summary_motifs_*.csv` into `tests/golden/`**

```bash
cd /Users/miguelgarcao/Desktop/msig
for case in \
    "audio:stumpy:stumpy_relaxed:summary_motifs_stumpy_relaxed.csv" \
    "audio:stumpy:stumpy_moderate:summary_motifs_stumpy_moderate.csv" \
    "audio:stumpy:stumpy_conservative:summary_motifs_stumpy_conservative.csv" \
    "audio:lama_iterative::summary_motifs_lama_iterative.csv" \
    "populationdensity:stumpy:stumpy_relaxed:summary_motifs_stumpy_relaxed.csv" \
    "populationdensity:stumpy:stumpy_moderate:summary_motifs_stumpy_moderate.csv" \
    "populationdensity:stumpy:stumpy_conservative:summary_motifs_stumpy_conservative.csv" \
    "populationdensity:lama_iterative::summary_motifs_lama_iterative.csv" \
    "washingmachine:stumpy:stumpy_relaxed:summary_motifs_stumpy_relaxed.csv" \
    "washingmachine:stumpy:stumpy_moderate:summary_motifs_stumpy_moderate.csv" \
    "washingmachine:stumpy:stumpy_conservative:summary_motifs_stumpy_conservative.csv" \
    "washingmachine:lama_iterative::summary_motifs_lama_iterative.csv" \
; do
  IFS=':' read -r ds m sub fn <<< "$case"
  src_dir="results/$ds/$m"
  [ -n "$sub" ] && src_dir="$src_dir/$sub"
  golden_name="${ds}_${m}_${sub:-main}_${fn}"
  cp "$src_dir/$fn" "tests/golden/$golden_name"
done
```

(MOMENTI snapshots will be added in a follow-up commit after the Linux run.)

- [ ] **Step 2: Verify the regression suite is green**

Run: `uv run pytest tests/test_paper_reproducibility.py -m slow -v`
Expected: 12 STUMPY+LAMA tests PASS; 3 MOMENTI tests SKIP.

- [ ] **Step 3: Commit**

```bash
git add tests/golden/
git commit -m "test: populate golden CSVs for STUMPY + LAMA (MOMENTI follows)"
```

---

## Workstream 8: Release

### Task 45: Write `CHANGELOG.md`

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Create the file**

```markdown
# Changelog

All notable changes to MSig are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres
to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-05-XX

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

### Added
- **`pattern_prob_floor` kwarg** on `Motif.set_significance` — opt-in
  Laplace floor for `p_Q = 0` cases (zero-frequency problem). Default
  `None` preserves 0.1.x behaviour.
- **`EXCLUSION_ZONE_FACTOR` constant** in every experiment script and
  `experiments/common_utils.py` — surfaces the trivial-match factor
  (default `0.5`, matching published tables; paper §3.2 default is `0.25`).
- **`AVERAGE_DELTA` module constant** — replaces the per-script `average_delta = 0.3` hardcode.
- **`tests/golden/`** — paper-reproducibility regression suite locks in
  the CSV outputs so future drift is caught.
- **`CITATION.cff`**, **`CONTRIBUTING.md`**, **`REPRODUCING_EXPERIMENTS.md`** — accessibility documentation.
- **`msig/py.typed`** — PEP 561 marker enabling downstream type checking.
- **Validation that `δ = 0` is rejected** for `kde` and `gaussian_theoretical`
  null models (silently produced `p_Q = 0` before).

### Changed
- **`bonferroni_correction(n_tests, alpha=0.05)`** — first parameter renamed.
  Backward-compatible: still accepts iterables for 0.1.x callers.
- **`idd_correction` documentation** — corrected to "identically-distributed
  dimensions" (was wrongly described as "Independent Dimension Discovery"
  applying BH FDR).
- **Python support range** aligned to `>=3.11,<3.14` across `pyproject.toml`,
  `environment.yml`, `requirements.txt`, classifiers, mypy, and black.

### Removed
- **`PYPI_DESCRIPTION.md`** — duplicate of `README.md`; never reached PyPI.
- **Dead `OverflowError` fallback** in `set_significance` — `scipy.stats.binom.sf` underflows silently.

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
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG.md (0.2.0)"
```

---

### Task 46: Build wheel and inspect METADATA

**Files:**
- (no file edits — release prep)

- [ ] **Step 1: Clean previous builds**

```bash
rm -rf /Users/miguelgarcao/Desktop/msig/dist
```

- [ ] **Step 2: Build**

```bash
uv build
```
Expected: produces `dist/msig-0.2.0.tar.gz` and `dist/msig-0.2.0-*.whl`.

- [ ] **Step 3: Inspect wheel METADATA**

```bash
unzip -p /Users/miguelgarcao/Desktop/msig/dist/msig-0.2.0-*.whl '*/METADATA' | grep -E 'Requires-Python|^Classifier|Version'
```
Expected output includes:
- `Version: 0.2.0`
- `Requires-Python: >=3.11,<3.14`
- Classifiers for License, OS, Topic, Python 3.11/3.12/3.13.

If anything looks wrong, **stop** and fix `pyproject.toml`. Do not upload.

- [ ] **Step 4: Test install in a fresh venv**

```bash
mkdir -p /tmp/msig-fresh && cd /tmp/msig-fresh
uv venv
source .venv/bin/activate
uv pip install /Users/miguelgarcao/Desktop/msig/dist/msig-0.2.0-*.whl
python -c "import msig; print(msig.__version__)"
```
Expected: prints `0.2.0`.

- [ ] **Step 5: Commit no files (this is verification only)**

No commit. Record the wheel hash for the GitHub Release notes:

```bash
shasum -a 256 /Users/miguelgarcao/Desktop/msig/dist/msig-0.2.0*.whl /Users/miguelgarcao/Desktop/msig/dist/msig-0.2.0*.tar.gz
```

---

### Task 47: Tag and push v0.2.0; upload to PyPI

**Files:**
- (release operations only)

- [ ] **Step 1: Confirm clean working tree**

Run: `git status`
Expected: clean (or only `MOMENTI_PENDING.md` if that's still around — if so, stop and complete the Linux MOMENTI run first).

- [ ] **Step 2: Tag**

```bash
git tag -a v0.2.0 -m "MSig 0.2.0 — comprehensive revision"
git push origin v0.2.0
```

- [ ] **Step 3: Publish to PyPI**

```bash
uv publish
```

Provide PyPI token when prompted.

- [ ] **Step 4: Verify the public PyPI page**

Wait ~1 minute, then:

```bash
uv pip install --upgrade msig --index-url https://pypi.org/simple
uv run python -c "import msig; print(msig.__version__)"
```
Expected: prints `0.2.0`.

Visit https://pypi.org/project/msig/ in a browser; confirm:
- Version 0.2.0 is the latest.
- "Requires Python: >=3.11,<3.14" shows on the project page.
- Classifiers include License/OS/Topic and Python 3.11/3.12/3.13.

- [ ] **Step 5: Create GitHub Release**

```bash
gh release create v0.2.0 --title "MSig 0.2.0" --notes-file CHANGELOG.md \
  /Users/miguelgarcao/Desktop/msig/dist/msig-0.2.0*.whl \
  /Users/miguelgarcao/Desktop/msig/dist/msig-0.2.0*.tar.gz
```

(Or use the GitHub web UI to paste the relevant CHANGELOG section.)

---

### Task 48: Post-release MOMENTI completion (separate Linux box)

**Files:**
- (run on a Linux/Windows host)

**Context:** This task is performed after Task 47 if MOMENTI was deferred. It produces a follow-up `0.2.1` if MOMENTI numerics turn out to need fixing, or simply a CSV-additions PR if they're clean.

- [ ] **Step 1: On a Linux box, clone the repo, install MOMENTI, sync data**

```bash
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig && git checkout v0.2.0
uv sync --extra experiments
uv pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs
rsync -avh \
  --include='audio/' --include='audio/imblue.mp3' \
  --include='synthetic/' --include='synthetic/multivar_time_series.csv' \
  --include='washingmachine/' --include='washingmachine/main_readings.csv' \
  --include='populationdensity/' --include='populationdensity/hourly_saodomingosbenfica.csv' \
  --exclude='*' \
  miguel@10.10.4.60:~/raid_backup/ data/
```

- [ ] **Step 2: Run MOMENTI experiments**

```bash
uv run python experiments/audio/run_momenti.py
uv run python experiments/populationdensity/run_momenti.py
uv run python experiments/washingmachine/run_momenti.py
```
Expected: all three exit 0 and write `results/<dataset>/momenti/summary_motifs_momenti.csv`.

- [ ] **Step 3: Add MOMENTI goldens, delete `MOMENTI_PENDING.md`**

```bash
for ds in audio populationdensity washingmachine; do
  cp results/$ds/momenti/summary_motifs_momenti.csv \
     tests/golden/${ds}_momenti_main_summary_motifs_momenti.csv
done
git rm MOMENTI_PENDING.md
git add results/ tests/golden/
```

- [ ] **Step 4: Verify regression suite is green for MOMENTI**

```bash
uv run pytest tests/test_paper_reproducibility.py -m slow -v
```
Expected: all 15 PASS.

- [ ] **Step 5: Commit and push**

```bash
git commit -m "chore: complete MOMENTI experiments and add goldens"
git push
```

This may go on `main` directly (small, additive change) or via a PR.

---

## Self-review checklist

Before declaring the plan ready, the executor should mentally walk through the spec section-by-section:

- [ ] Every library bug ref (#1, #4, #5, #11, #13, #14, #24, #25, #36, #37) maps to a task.
- [ ] Every experiment-script bug ref (#9, #10, #12, #15, #16, #34) maps to a task.
- [ ] Every packaging bug ref (#3, #7, #17, #18, #19, #28, #29, #30, #31) maps to a task.
- [ ] Every documentation bug ref (#6, #8, #11, #32, #33) maps to a task.
- [ ] Every test gap (#20, #21, #22, #23, plus integration smokes and goldens) maps to a task.
- [ ] All four open questions from the spec are reflected in tasks (Q1 → Task 17; Q2 → Task 7; Q3 → Task 19; Q4–Q6 → Caveats noted in spec).
- [ ] Validation steps 1–8 from the spec map to Tasks 42, 43, 44.
- [ ] PyPI release procedure (spec §) maps to Tasks 46, 47.

If any item is missing, add it before execution.
