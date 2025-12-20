"""
Unit tests for MSig statistical methods.
Tests statistical functions in isolation.
"""

import pytest
import numpy as np
from msig import benjamini_hochberg_fdr, bonferroni_correction


class TestStatisticalMethods:
    """Tests for statistical correction methods."""

    def test_benjamini_hochberg_empty_input(self):
        """Test Benjamini-Hochberg with empty input."""
        p_values = []
        result = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.05)
        assert result == 0.0

    def test_benjamini_hochberg_single_value(self):
        """Test Benjamini-Hochberg with single p-value."""
        p_values = [0.03]
        result = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.05)
        assert result == 0.03

    def test_benjamini_hochberg_multiple_values(self):
        """Test Benjamini-Hochberg with multiple p-values."""
        p_values = [0.001, 0.01, 0.03, 0.04, 0.05]
        result = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.05)
        # Should return the largest p-value that satisfies p_i <= (i/m)*alpha
        assert 0.0 <= result <= 0.05
        assert isinstance(result, float)

    def test_benjamini_hochberg_all_significant(self):
        """Test Benjamini-Hochberg when all p-values are significant."""
        p_values = [0.001, 0.002, 0.003, 0.004, 0.005]
        result = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.05)
        # All should be significant, return the largest
        assert result == 0.005

    def test_benjamini_hochberg_none_significant(self):
        """Test Benjamini-Hochberg when no p-values are significant."""
        p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.05)
        # None should be significant, return FDR threshold
        assert result == 0.05

    def test_benjamini_hochberg_different_alpha(self):
        """Test Benjamini-Hochberg with different alpha values."""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.04]
        
        result_01 = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.01)
        result_05 = benjamini_hochberg_fdr(p_values, false_discovery_rate=0.05)
        
        assert result_01 <= result_05
        assert result_01 <= 0.01
        assert result_05 <= 0.05

    def test_bonferroni_correction_empty(self):
        """Test Bonferroni correction with empty input."""
        p_values = []
        result = bonferroni_correction(p_values, alpha=0.05)
        assert result == 0.05

    def test_bonferroni_correction_single(self):
        """Test Bonferroni correction with single p-value."""
        p_values = [0.03]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert result == 0.05  # alpha / 1 = alpha

    def test_bonferroni_correction_multiple(self):
        """Test Bonferroni correction with multiple p-values."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = bonferroni_correction(p_values, alpha=0.05)
        assert result == 0.01  # alpha / 5 = 0.01

    def test_bonferroni_correction_different_alpha(self):
        """Test Bonferroni correction with different alpha values."""
        p_values = [0.01, 0.02, 0.03]
        
        result_01 = bonferroni_correction(p_values, alpha=0.01)
        result_05 = bonferroni_correction(p_values, alpha=0.05)
        
        assert result_01 == 0.01 / 3
        assert result_05 == 0.05 / 3
        assert result_01 < result_05


class TestProbabilityCalculations:
    """Tests for probability calculation methods."""

    def test_pattern_probability_bounds(self):
        """Test that pattern probabilities are within valid bounds."""
        from msig import Motif, NullModel
        
        np.random.seed(42)
        data = np.random.randn(2, 50)
        model = NullModel(data, dtypes=[float, float], model="empirical")
        
        pattern = data[:, 5:10]
        motif = Motif(list(pattern), [0, 1], [0.1, 0.1], n_matches=3)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        # Probability should be between 0 and 1
        assert 0.0 <= prob <= 1.0
        assert isinstance(prob, float)

    def test_zero_probability_pattern(self):
        """Test pattern with zero probability."""
        from msig import Motif, NullModel
        
        data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        
        # Pattern that never occurs in data
        pattern = np.array([[99.0]])
        motif = Motif(list(pattern), [0], [0.0], n_matches=1)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        assert prob == 0.0

    def test_certain_probability_pattern(self):
        """Test pattern with certain probability."""
        from msig import Motif, NullModel
        
        # Data where pattern is certain
        data = np.array([[1, 1, 1, 1, 1]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        
        pattern = np.array([[1.0]])
        motif = Motif(list(pattern), [0], [0.0], n_matches=5)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        assert prob == 1.0


class TestSignificanceTesting:
    """Tests for significance testing methods."""

    def test_significance_calculation_bounds(self):
        """Test that p-values are within valid bounds."""
        from msig import Motif, NullModel
        
        np.random.seed(42)
        data = np.random.randn(2, 50)
        model = NullModel(data, dtypes=[float, float], model="empirical")
        
        pattern = data[:, 5:10]
        motif = Motif(list(pattern), [0, 1], [0.1, 0.1], n_matches=3)
        
        # Set pattern probability
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        # Calculate significance
        max_matches = 45  # 50 - 5 + 1
        pvalue = motif.set_significance(max_matches, 2, idd_correction=False)
        
        # P-value should be between 0 and 1
        assert 0.0 <= pvalue <= 1.0
        assert isinstance(pvalue, float)

    def test_significance_with_zero_probability(self):
        """Test significance calculation with zero pattern probability."""
        from msig import Motif, NullModel
        
        data = np.array([[1, 2, 3, 4, 5]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        
        pattern = np.array([[99.0]])
        motif = Motif(list(pattern), [0], [0.0], n_matches=1)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        pvalue = motif.set_significance(5, 1, idd_correction=False)
        
        # With zero probability, p-value should be zero
        assert prob == 0.0
        assert pvalue == 0.0

    def test_significance_with_certain_probability(self):
        """Test significance calculation with certain pattern probability."""
        from msig import Motif, NullModel
        
        data = np.array([[1, 1, 1, 1, 1]], dtype=float)
        model = NullModel(data, dtypes=[float], model="empirical")
        
        pattern = np.array([[1.0]])
        motif = Motif(list(pattern), [0], [0.0], n_matches=5)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        pvalue = motif.set_significance(5, 1, idd_correction=False)
        
        # With certain probability, p-value should be 1.0
        assert prob == 1.0
        assert pvalue == 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])