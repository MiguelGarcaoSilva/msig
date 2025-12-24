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

### From source with uv (recommended)

```bash
# Clone the repository
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig

# Sync dependencies (includes STUMPY and LAMA)
uv sync

# Optional: Install MOMENTI (for MOMENTI experiments - Linux x86_64/Windows only)
# Note: MOMENTI requires Intel compiler runtime, not available on macOS
uv pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs
```

### From source with pip

```bash
# Clone repository
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig

# Install with experiment dependencies
pip install -e ".[experiments]"
# or
pip install -r requirements.txt
```

**For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md)**

**Notes**:
- **Python Version**: Python 3.11-3.13 recommended (3.14+ may have LAMA issues)
- **MOMENTI**: Requires Intel compiler runtime libraries (Linux x86_64/Windows only, not available on macOS)
- **Audio experiments**: Require **ffmpeg** for MP3 processing
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`
  - Windows: Download from https://ffmpeg.org/

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

### Quick Start - Run All Experiments

**First, test your setup:**
```bash
./test_setup.sh  # Verify dependencies, data files, and environment
```

**Then run experiments:**
```bash
# Run all experiments
uv run python run_experiments.py --all

# Or run in tmux (for long runs - can detach/reattach)
# From inside tmux:
uv run python run_experiments.py --all
# Then: Ctrl-b + d to detach, experiments continue running

# Compare results when done
uv run python scripts/compare_results.py
```

This will:
- Run all 6-9 experiments (depending on platform/dependencies)
- Track progress with estimated runtimes
- Save metadata for reproducibility
- Generate run logs

**Estimated total time**: ~2-3 hours

**Note**: MOMENTI requires separate installation: `pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs` (Linux x86_64/Windows only - not macOS)

### Individual Experiments

Run specific experiments manually:

```bash
# Audio experiments (~15-25 min each)
python experiments/audio/run_stumpy.py
python experiments/audio/run_lama.py
python experiments/audio/run_momenti.py

# Population density (~10-15 min each)
python experiments/populationdensity/run_stumpy.py
python experiments/populationdensity/run_lama.py
python experiments/populationdensity/run_momenti.py

# Washing machine (~12-18 min each)
python experiments/washingmachine/run_stumpy.py
python experiments/washingmachine/run_lama.py
python experiments/washingmachine/run_momenti.py
```

### Experiment Runner Options

```bash
# Run specific combinations
python run_experiments.py --dataset audio           # All methods on audio
python run_experiments.py --method stumpy           # STUMPY on all datasets
python run_experiments.py --dataset audio --method lama

# Preview what would run (dry-run mode)
python run_experiments.py --all --dry-run

# Show detailed output
python run_experiments.py --all --verbose
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

Generate comprehensive comparison reports:

```bash
# Full comparison across all methods and datasets
python scripts/compare_results.py
# Creates: RESULTS_COMPARISON.md and results_summary.csv

# Compare specific aspects
python scripts/compare_results.py --dataset audio    # Compare methods on audio
python scripts/compare_results.py --method stumpy    # Compare STUMPY across datasets
```

### Datasets

1. **Audio** (`data/audio/imblue.mp3` - 8.4 MB)
   - 12 MFCC features, subsequence lengths: [21, 43, 130, 217] frames (0.5s-5s)
   - Discovery goal: Musical patterns (beats, measures, phrases)

2. **Population Density** (`data/populationdensity/hourly_saodomingosbenfica.csv` - 17 MB)
   - 3 variables (Terminals, Roaming, Calls), lengths: [4, 6, 12, 24] hours
   - Discovery goal: Daily patterns in urban mobility

3. **Washing Machine** (`data/washingmachine/main_readings.csv` - 3.0 MB)
   - 7 sensor variables (Current, Power Factor, Water levels/temps)
   - Discovery goal: Operating mode patterns

## Reproducibility Validation

Validate your setup before running experiments:

```bash
python validate_reproducibility.py
```

This single script checks:
- Python version compatibility
- Core and experiment dependencies
- ffmpeg availability
- Data file availability
- Core MSig functionality

### Unit Tests

For developers:
```bash
uv run python -m pytest tests/ -v
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


