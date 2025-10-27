#!/usr/bin/env python3
"""
Simple Example: MSig Statistical Significance Testing

This example demonstrates the basic usage of MSig without requiring
any external motif discovery libraries. It shows how to:
1. Create a null model from multivariate time series
2. Define a motif pattern
3. Compute pattern probability
4. Calculate statistical significance
"""

from msig import Motif, NullModel
import numpy as np


def example_multivariate_time_series():
    """Example 1: Multivariate time series with mixed data types."""
    
    print("=" * 70)
    print("EXAMPLE 1: Mixed Data Types (int, float, categorical)")
    print("=" * 70)
    
    # Create sample time series (4 variables × 15 time points)
    ts1 = [1, 3, 3, 5, 5, 2, 3, 3, 5, 5, 3, 3, 5, 4, 4]  # Integer
    ts2 = [4.3, 4.5, 2.6, 3.0, 3.0, 1.7, 4.9, 2.9, 3.3, 1.9, 4.9, 2.5, 3.1, 1.8, 0.3]  # Float
    ts3 = ["A", "D", "B", "D", "A", "A", "A", "C", "C", "B", "D", "D", "C", "A", "A"]  # Categorical
    ts4 = ["T", "L", "T", "Z", "Z", "T", "L", "T", "Z", "T", "L", "T", "Z", "L", "L"]  # Categorical
    
    # Stack into array (4 variables × 15 time points)
    data = np.stack([
        np.asarray(ts1, dtype=int),
        np.asarray(ts2, dtype=float),
        np.asarray(ts3, dtype=str),
        np.asarray(ts4, dtype=str)
    ])
    
    m, n = data.shape
    print(f"\nData shape: {m} variables × {n} time points")
    print(f"Data types: int, float, str, str")
    
    # Create empirical null model
    model = NullModel(data, dtypes=[int, float, str, str], model="empirical")
    print(f"\nNull model: empirical (based on observed frequencies)")
    
    # Define motif: length 3, variables [0, 1, 3], occurring at indices 1, 6, 10
    motif_length = 3
    motif_vars = np.array([0, 1, 3])  # Use variables 0, 1, and 3
    motif_pattern = data[motif_vars, 1:4]  # Extract pattern from position 1
    
    print(f"\nMotif pattern (variables {motif_vars}, length {motif_length}):")
    for i, var_idx in enumerate(motif_vars):
        print(f"  Variable {var_idx}: {motif_pattern[i]}")
    
    # Tolerance: exact match for discrete (0), δ=0.5 for continuous
    # Need one threshold per variable in the motif: [var0, var1, var3]
    delta_thresholds = np.array([0, 0.5, 0])  # int, float, str
    
    # Create motif object (3 observed matches)
    motif = Motif(
        multivar_sequence=motif_pattern,
        variables=motif_vars,
        delta_thresholds=delta_thresholds,
        n_matches=3
    )
    
    # Compute pattern probability
    prob = motif.set_pattern_probability(model, vars_indep=True)
    print(f"\nPattern probability P(Q) = {prob:.6f}")
    print(f"  (Probability that a random subsequence matches this pattern)")
    
    # Compute statistical significance
    max_possible_matches = n - motif_length + 1
    pvalue = motif.set_significance(
        max_possible_matches=max_possible_matches,
        data_n_variables=m,
        idd_correction=False
    )
    
    print(f"\nStatistical significance:")
    print(f"  Observed matches: 3")
    print(f"  Max possible matches: {max_possible_matches}")
    print(f"  P-value: {pvalue:.6e}")
    print(f"  Significant at α=0.01? {pvalue <= 0.01}")
    print(f"  Significant at α=0.05? {pvalue <= 0.05}")
    

def example_continuous_data():
    """Example 2: Continuous multivariate time series."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Continuous Data (sensor readings)")
    print("=" * 70)
    
    # Simulate 3 sensor readings over 100 time points
    np.random.seed(42)
    
    # Generate correlated sensor data with periodic pattern
    t = np.linspace(0, 10, 100)
    sensor1 = 10 + 2 * np.sin(2 * np.pi * t) + np.random.randn(100) * 0.5
    sensor2 = 5 + 1.5 * np.cos(2 * np.pi * t) + np.random.randn(100) * 0.3
    sensor3 = 15 + 3 * np.sin(2 * np.pi * t + np.pi/4) + np.random.randn(100) * 0.7
    
    data = np.stack([sensor1, sensor2, sensor3])
    m, n = data.shape
    
    print(f"\nData shape: {m} sensors × {n} time points")
    print(f"Data types: all float")
    
    # Create Gaussian null model (appropriate for continuous data)
    model = NullModel(data, dtypes=[float, float, float], model="gaussian_theoretical")
    print(f"\nNull model: gaussian_theoretical (assumes Gaussian distributions)")
    
    # Define a motif pattern (length 10, all 3 sensors, 8 occurrences)
    motif_length = 10
    motif_vars = np.array([0, 1, 2])
    motif_pattern = data[motif_vars, 5:15]
    n_matches = 8
    
    print(f"\nMotif: length {motif_length}, all {len(motif_vars)} sensors, {n_matches} occurrences")
    
    # Use tolerance δ = 0.3 for all sensors
    delta_thresholds = np.array([0.3, 0.3, 0.3])
    
    motif = Motif(
        multivar_sequence=motif_pattern,
        variables=motif_vars,
        delta_thresholds=delta_thresholds,
        n_matches=n_matches
    )
    
    # Compute statistics
    prob = motif.set_pattern_probability(model, vars_indep=True)
    max_possible_matches = n - motif_length + 1
    pvalue = motif.set_significance(max_possible_matches, m, idd_correction=False)
    
    print(f"\nResults:")
    print(f"  Pattern probability: {prob:.6e}")
    print(f"  P-value: {pvalue:.6e}")
    print(f"  Significant at α=0.01? {pvalue <= 0.01}")


def example_hochberg_correction():
    """Example 3: Multiple testing with Benjamini-Hochberg FDR correction."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiple Testing Correction (Benjamini-Hochberg FDR)")
    print("=" * 70)
    
    # Simulate discovering multiple motifs with different p-values
    pvalues = np.array([0.001, 0.005, 0.015, 0.023, 0.034, 0.048, 0.062, 0.089, 0.112, 0.157])
    alpha = 0.05
    
    print(f"\nDiscovered 10 motifs with p-values:")
    for i, p in enumerate(pvalues, 1):
        print(f"  Motif {i}: p = {p:.4f}")
    
    # Compute Benjamini-Hochberg critical value
    from msig import benjamini_hochberg_fdr
    critical_value = benjamini_hochberg_fdr(pvalues, alpha)
    
    print(f"\nBenjamini-Hochberg FDR correction (α = {alpha}):")
    print(f"  Critical value: {critical_value:.6f}")
    
    # Determine which motifs are significant
    significant = pvalues <= critical_value
    n_significant = np.sum(significant)
    
    print(f"\nSignificant motifs after FDR correction:")
    for i, (p, sig) in enumerate(zip(pvalues, significant), 1):
        status = "✓ SIGNIFICANT" if sig else "  not significant"
        print(f"  Motif {i}: p = {p:.4f} {status}")
    
    print(f"\nSummary: {n_significant}/{len(pvalues)} motifs remain significant after correction")


def example_null_model_comparison():
    """Example 4: Comparing different null models."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Comparing Null Models")
    print("=" * 70)
    
    # Generate continuous data
    np.random.seed(123)
    data = np.random.randn(2, 50)
    
    motif_pattern = data[:, 10:15]
    motif_vars = np.array([0, 1])
    delta_thresholds = np.array([0.5, 0.5])
    n_matches = 5
    
    models = {
        "empirical": "empirical",
        "kde": "kde",
        "gaussian": "gaussian_theoretical"
    }
    
    print(f"\nComparing null models for the same motif:")
    print(f"  Pattern length: 5")
    print(f"  Variables: [0, 1]")
    print(f"  Tolerance: δ = 0.5")
    print(f"  Matches: {n_matches}")
    
    print(f"\n{'Model':<20} {'P(pattern)':<15} {'p-value':<15}")
    print("-" * 50)
    
    for name, model_type in models.items():
        model = NullModel(data, dtypes=[float, float], model=model_type)
        motif = Motif(motif_pattern, motif_vars, delta_thresholds, n_matches)
        
        prob = motif.set_pattern_probability(model, vars_indep=True)
        pvalue = motif.set_significance(50 - 5 + 1, 2, idd_correction=False)
        
        print(f"{name:<20} {prob:<15.6e} {pvalue:<15.6e}")
    
    print("\nNote: Different null models can give different probability estimates")
    print("Choose based on your data characteristics and assumptions.")


def main():
    """Run all examples."""
    
    print("\n" + "=" * 70)
    print("MSig: Statistical Significance for Time Series Motifs")
    print("Simple Examples")
    print("=" * 70)
    
    # Run examples
    example_multivariate_time_series()
    example_continuous_data()
    example_hochberg_correction()
    example_null_model_comparison()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("\nFor more information, see:")
    print("  - README.md: Overview and installation")
    print("  - REPRODUCING_EXPERIMENTS.md: Case studies with real data")
    print("  - CONTRIBUTING.md: Development guidelines")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
