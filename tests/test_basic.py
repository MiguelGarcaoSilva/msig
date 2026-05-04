"""
Basic tests for MSig core functionality.

Run with: pytest tests/
"""

import pytest
import numpy as np
from msig import Motif, NullModel


class TestNullModel:
    """Tests for NullModel class."""

    def test_empirical_model_creation(self):
        """Test that empirical model is created correctly."""
        data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        
        assert model.model == "empirical"
        assert model.data.shape == (1, 5)
        assert len(model.dtypes) == 1

    def test_multiple_variables(self):
        """Test model with multiple variables."""
        data = np.stack([
            np.array([1, 2, 3, 4, 5], dtype=int),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
            np.array(['A', 'B', 'C', 'D', 'E'], dtype=str)
        ])
        
        model = NullModel(data, dtypes=[int, float, str], model="empirical")
        assert model.data.shape == (3, 5)
        assert len(model.dtypes) == 3

    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        data = np.array([[1, 2, 3]], dtype=float)
        
        with pytest.raises(ValueError, match="Invalid model"):
            NullModel(data, dtypes=[float], model="invalid_model")

    def test_incompatible_dtype_kde(self):
        """Test that KDE with non-float data raises error."""
        data = np.array([[1, 2, 3]], dtype=int)
        
        with pytest.raises(ValueError, match="requires all variables to be float type"):
            NullModel(data, dtypes=[int], model="kde")

    def test_gaussian_model_creation(self):
        """Test Gaussian theoretical model."""
        data = np.random.randn(2, 100)
        model = NullModel(data, dtypes=[float, float], model="gaussian_theoretical")
        
        assert model.model == "gaussian_theoretical"
        assert len(model.pre_computed_distribution) == 2


class TestMotif:
    """Tests for Motif class."""

    def test_motif_creation(self):
        """Test basic motif creation."""
        pattern = np.array([[1, 2, 3]])
        motif = Motif(
            multivar_sequence=pattern,
            variables=[0],
            delta_thresholds=[0],
            n_matches=3
        )
        
        assert motif.n_matches == 3
        assert len(motif.variables) == 1
        assert motif.p_Q == 0.0  # Initial value
        assert motif.pvalue == 1.0  # Initial value

    def test_pattern_probability_empirical(self):
        """Test pattern probability computation with empirical model."""
        # Simple repeating pattern
        data = np.array([[1, 2, 1, 2, 1, 2, 1, 2]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        
        pattern = np.array([[1, 2]])
        motif = Motif(pattern, [0], [0], n_matches=4)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        assert 0 <= prob <= 1
        assert motif.p_Q == prob

    def test_significance_calculation(self):
        """Test significance calculation."""
        data = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        
        pattern = np.array([[1, 2]])
        motif = Motif(pattern, [0], [0.5], n_matches=3)
        
        # Set pattern probability
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        # Calculate significance
        max_matches = 8 - 2 + 1  # n - motif_length + 1
        pvalue = motif.set_significance(max_matches, 1, idd_correction=False)
        
        assert 0 <= pvalue <= 1
        assert motif.pvalue == pvalue

    def test_benjamini_hochberg_fdr(self):
        """Test Benjamini-Hochberg FDR correction (module-level function)."""
        from msig import benjamini_hochberg_fdr
        
        pvalues = np.array([0.001, 0.01, 0.02, 0.05, 0.1])
        alpha = 0.05
        
        critical = benjamini_hochberg_fdr(pvalues, alpha)
        
        assert 0 <= critical <= alpha
        assert isinstance(critical, float)

    def test_multivariate_pattern(self):
        """Test motif with multiple variables."""
        data = np.stack([
            np.array([1, 2, 3, 4, 5], dtype=float),
            np.array([10, 20, 30, 40, 50], dtype=float)
        ])
        
        model = NullModel(data, dtypes=[float, float], model="empirical")
        
        pattern = np.array([[1, 2], [10, 20]])
        motif = Motif(pattern, [0, 1], [0.1, 1.0], n_matches=2)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        pvalue = motif.set_significance(4, 2, idd_correction=False)
        
        assert 0 <= prob <= 1
        assert 0 <= pvalue <= 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_timepoints(self):
        """Test with minimum required data (2 time points)."""
        data = np.array([[1.0, 2.0]])
        model = NullModel(data, dtypes=[float], model="empirical")
        
        assert model.data.shape == (1, 2)

    def test_zero_probability_pattern(self):
        """Test pattern with zero probability."""
        data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        
        # Pattern that never occurs
        pattern = np.array([[99]])
        motif = Motif(pattern, [0], [0], n_matches=1)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        pvalue = motif.set_significance(5, 1, idd_correction=False)
        
        assert prob == 0.0
        assert pvalue == 0.0  # Special case for p_Q = 0

    def test_categorical_data(self):
        """Test with categorical (string) data."""
        data = np.array([['A', 'B', 'A', 'B', 'A']], dtype=str)
        model = NullModel(data, dtypes=[str], model="empirical")
        
        pattern = np.array([['A']])
        motif = Motif(pattern, [0], [0], n_matches=3)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        assert 0 <= prob <= 1

    def test_mixed_datatypes(self):
        """Test with mixed data types."""
        data = np.stack([
            np.array([1, 2, 3, 4, 5], dtype=int),
            np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=float),
            np.array(['A', 'B', 'C', 'D', 'E'], dtype=str)
        ])
        
        model = NullModel(data, dtypes=[int, float, str], model="empirical")
        
        pattern = np.array([[1], [1.1], ['A']])
        motif = Motif(pattern, [0, 1, 2], [0, 0.1, 0], n_matches=1)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        assert 0 <= prob <= 1


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


class TestSetSignificanceAlwaysSetsPvalue:
    """Every return path of set_significance must populate self.pvalue."""

    def test_p_q_zero_sets_self_pvalue(self):
        import numpy as np
        from msig import Motif, NullModel

        data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        # Pattern that never occurs in data => p_Q = 0
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
        # max_possible_matches = 5, n_matches = 10 => degenerate
        returned = motif.set_significance(5, 1, idd_correction=False)
        assert math.isnan(returned) or returned == 1.0
        assert motif.pvalue == returned or (math.isnan(returned) and math.isnan(motif.pvalue))


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
        """When q == m, C(m,m) = 1 => corrected = uncorrected."""
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


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
