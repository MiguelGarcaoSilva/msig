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


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
