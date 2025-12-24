# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSig is a Python library for statistical significance testing of multivariate time series motifs. It evaluates whether discovered motifs occur more frequently than expected by chance using binomial tests against null models.

## Commands

### Installation
```bash
uv sync                              # Install with uv (recommended)
pip install -e ".[experiments]"      # Install with pip
```

### Testing
```bash
uv run python -m pytest tests/ -v    # Run all tests
uv run python -m pytest tests/test_basic.py -v  # Run single test file
uv run python -m pytest tests/ -v -k "test_name"  # Run specific test
```

### Validation
```bash
python validate_reproducibility.py   # Check environment and dependencies
```

### Running Experiments
```bash
uv run python experiments/audio/run_stumpy.py
uv run python experiments/populationdensity/run_lama.py
uv run python experiments/washingmachine/run_momenti.py
```

### Code Formatting
```bash
black msig/ tests/ --line-length 100
isort msig/ tests/
```

## Architecture

### Core Classes (msig/MSig.py)

**NullModel**: Builds statistical null models from observed data to compute pattern probabilities.
- Three model types: `empirical` (frequency-based), `kde` (kernel density), `gaussian_theoretical`
- Pre-computes marginal and bivariate distributions for first-order Markov modeling
- `vars_indep_time_markov()`: Computes P(Q) assuming independent variables with Markov time dependency

**Motif**: Represents a multivariate pattern with significance testing.
- Stores pattern definition, variable indices, delta thresholds, and match count
- `set_pattern_probability(model, vars_indep)`: Computes pattern probability against null model
- `set_significance(max_possible_matches, data_n_variables, idd_correction)`: Computes p-value using binomial survival function

### Multiple Testing Corrections (msig/MSig.py)

- `benjamini_hochberg_fdr()`: FDR correction for multiple hypothesis testing
- `bonferroni_correction()`: Conservative family-wise error rate control

### Data Flow

1. Create time series data as numpy array (m variables x n time points)
2. Build `NullModel` with data and variable dtypes
3. Create `Motif` with pattern, variables, thresholds, and match count
4. Call `motif.set_pattern_probability(model)` to compute P(Q)
5. Call `motif.set_significance()` to compute p-value
6. Apply multiple testing correction if comparing multiple motifs

## Key Conventions

- Data shape is always (m_variables, n_timepoints)
- `delta_thresholds` must align with `motif_vars` (one threshold per variable in the motif)
- Use `delta=0` for exact matching (categorical/discrete data)
- Use `model="empirical"` for mixed dtypes; `kde` and `gaussian_theoretical` require all floats
- Pattern probability uses first-order Markov assumption: P(x_t | x_{t-1})

## Dependencies

- Core: numpy, scipy
- Experiments: pandas, matplotlib, stumpy, librosa, leitmotif (LAMA)
- MOMENTI: Install separately from GitHub (Linux/Windows only)
- Audio experiments require ffmpeg
