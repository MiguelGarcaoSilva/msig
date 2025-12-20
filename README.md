# MSig

**Statistical Significance Testing for Multivariate Time Series Motifs**

MSig evaluates whether discovered motifs occur more frequently than expected by chance, using rigorous statistical methods.

## Installation

### From PyPI (recommended)

```bash
# Core package only
pip install msig

# With experiment dependencies (includes STUMPY and LAMA)
pip install "msig[experiments]"
```

### From source with uv

```bash
# Clone the repository
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig

# Sync dependencies (includes STUMPY and LAMA)
uv sync

# Optional: Install MOMENTI (for MOMENTI experiments - Linux/Windows only)
uv pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs
```

### From source with pip

```bash
# Clone repository
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig

# Install with experiment dependencies
pip install -e ".[experiments]"
```

**Notes**: 
- MOMENTI has platform-specific dependencies and may not install on macOS.
- Audio experiments require **ffmpeg** for MP3 processing: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Linux)

## Quick Start

```python
from msig import Motif, NullModel
import numpy as np

# Create sample multivariate time series (3 sensors × 100 time points)
np.random.seed(42)
t = np.linspace(0, 10, 100)
sensor1 = 10 + 2 * np.sin(2 * np.pi * t) + np.random.randn(100) * 0.5
sensor2 = 5 + 1.5 * np.cos(2 * np.pi * t) + np.random.randn(100) * 0.3
sensor3 = 15 + 3 * np.sin(2 * np.pi * t + np.pi/4) + np.random.randn(100) * 0.7
data = np.stack([sensor1, sensor2, sensor3])

# Create null model (assumes Gaussian distributions)
model = NullModel(data, dtypes=[float, float, float], model="gaussian_theoretical")

# Define a motif: length 10, all 3 sensors, 8 occurrences
motif_length = 10
motif_pattern = data[:, 5:15]  # Extract pattern from position 5
motif_vars = np.array([0, 1, 2])  # Use all sensors
delta_thresholds = np.array([0.3, 0.3, 0.3])  # Tolerance for matching

# Create motif and test significance
motif = Motif(motif_pattern, motif_vars, delta_thresholds, n_matches=8)
prob = motif.set_pattern_probability(model, vars_indep=True)
pvalue = motif.set_significance(
    max_possible_matches=100 - motif_length + 1,
    data_n_variables=3,
    idd_correction=False
)

print(f"Pattern probability: {prob:.6e}")
print(f"P-value: {pvalue:.6e}")
print(f"Significant at α=0.01? {pvalue <= 0.01}")
```

See the `examples/` folder for more examples (`simple_example.py` and `example.ipynb`).

## Running Experiments

The repository includes case studies on three datasets with three discovery methods (STUMPY, LAMA, MOMENTI):

```bash
# Run individual experiments
uv run python experiments/audio/run_stumpy.py
uv run python experiments/audio/run_lama.py
uv run python experiments/audio/run_momenti.py

uv run python experiments/populationdensity/run_stumpy.py
uv run python experiments/populationdensity/run_lama.py
uv run python experiments/populationdensity/run_momenti.py

uv run python experiments/washingmachine/run_stumpy.py
uv run python experiments/washingmachine/run_lama.py
uv run python experiments/washingmachine/run_momenti.py
```

Results are saved to `results/<dataset>/<method>/`.

## Reproducibility Validation

MSig includes validation scripts to ensure experiments can be reproduced on your system:

### Validation Scripts

1. **Environment Validation**
   ```bash
   python validate_environment.py
   ```
   - Checks Python version and all dependencies
   - Validates MSig core functionality
   - Verifies data file availability
   - Tests system tool requirements

2. **Dataset Validation**
   ```bash
   python validate_all_datasets.py
   ```
   - Tests audio, population density, and washing machine datasets
   - Validates data loading and preprocessing
   - Checks data consistency and format compatibility

3. **Priority 1 Validation** (Quick Check)
   ```bash
   python run_priority1_validation.py
   ```
   - Validates core reproducibility components
   - Tests environment, datasets, and basic experiment execution
   - Fast execution (~30 seconds)

4. **Comprehensive Validation** (Full Check)
   ```bash
   python run_all_tests.py
   ```
   - Runs unit tests and all validation scripts
   - Provides complete reproducibility verification
   - Includes detailed reporting and diagnostics

### Quick Validation

For a quick check before running experiments:
```bash
# Validate environment and data
python validate_environment.py
python validate_all_datasets.py

# Or run comprehensive validation
python run_priority1_validation.py
```

### Unit Tests

For developers, run the unit test suite:
```bash
# Run all unit tests
uv run python -m pytest tests/ -v

# Run specific test files
uv run python -m pytest tests/test_basic.py -v
uv run python -m pytest tests/test_statistical_methods.py -v
```

## Citation

```bibtex
@article{silva2024msig,
  title={On Why and How Statistical Significance Criteria Can Guide Multivariate Time Series Motif Analysis},
  author={Silva, Miguel G. and Henriques, Rui and Madeira, Sara C.},
  year={2024}
}
```

## License

MIT License - see LICENSE file.


