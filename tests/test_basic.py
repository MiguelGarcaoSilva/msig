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


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
