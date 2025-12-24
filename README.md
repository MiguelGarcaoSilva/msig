# MSig

**Statistical Significance Testing for Multivariate Time Series Motifs**

MSig evaluates whether discovered motifs occur more frequently than expected by chance, using rigorous statistical methods.

## Installation

### Quick Install from PyPI

```bash
pip install msig                    # Core package only
pip install "msig[experiments]"     # With experiment dependencies
```

### Development Install with uv (Recommended)

```bash
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig
uv sync                             # Installs all dependencies

# Optional: MOMENTI (Linux x86_64/Windows only)
uv pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs
```

**Requirements**:
- Python 3.11-3.13 (3.14+ may have LAMA compatibility issues)
- ffmpeg (for audio experiments): `brew install ffmpeg` (macOS) or `sudo apt-get install ffmpeg` (Linux)

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

The repository includes case studies on three datasets (audio, population density, washing machine) with three discovery methods (STUMPY, LAMA, MOMENTI).

### Validate Setup

```bash
uv run python validate_reproducibility.py
```

### Run All Experiments

```bash
uv run python run_experiments.py --all
uv run python scripts/compare_results.py  # Compare results
```

**Note**: MOMENTI requires `uv pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs` (Linux x86_64/Windows only)

### Individual Experiments

```bash
# Audio experiments
uv run python experiments/audio/run_stumpy.py
uv run python experiments/audio/run_lama.py
uv run python experiments/audio/run_momenti.py

# Population density
uv run python experiments/populationdensity/run_stumpy.py
uv run python experiments/populationdensity/run_lama.py
uv run python experiments/populationdensity/run_momenti.py

# Washing machine
uv run python experiments/washingmachine/run_stumpy.py
uv run python experiments/washingmachine/run_lama.py
uv run python experiments/washingmachine/run_momenti.py
```

### Experiment Runner Options

```bash
# Run specific combinations
uv run python run_experiments.py --dataset audio           # All methods on audio
uv run python run_experiments.py --method stumpy           # STUMPY on all datasets
uv run python run_experiments.py --dataset audio --method lama

# Preview what would run
uv run python run_experiments.py --all --dry-run
```

### Understanding Results

Results are saved to `results/<dataset>/<method>/`:

- **`summary_motifs_{method}.csv`** (tracked in git)
  - Aggregated statistics per motif length
  - Columns: `s` (length), `k` (dimensionality), `#Matches`, `P` (probability), `p-value`, `significant`

- **`metadata.json`** (tracked in git)
  - Experiment parameters and environment info
  - Python/library versions, timestamps

- **`table_motifs_{method}.csv`** (not tracked - regenerate if needed)
  - Detailed per-motif information with indices

Key result interpretations:
- **Low p-value** (< 0.05): Motif is statistically significant
- **Low P (pattern probability)**: Motif is rare under null hypothesis
- **High #Matches**: Motif occurs frequently
- **significant=True**: Occurs more often than expected by chance (after FDR correction)

### Comparing Results

```bash
uv run python scripts/compare_results.py                    # Full comparison
uv run python scripts/compare_results.py --dataset audio    # Methods on audio
uv run python scripts/compare_results.py --method stumpy    # STUMPY across datasets
```

### Datasets

1. **Audio**: 12 MFCC features from `imblue.mp3` - musical patterns (beats, measures, phrases)
2. **Population Density**: 3 variables (Terminals, Roaming, Calls) - daily urban mobility patterns
3. **Washing Machine**: 7 sensor variables - operating mode patterns

## Testing

```bash
uv run python validate_reproducibility.py  # Validate environment
uv run python -m pytest tests/ -v          # Run unit tests
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


