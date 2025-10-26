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

**Note**: MOMENTI has platform-specific dependencies and may not install on macOS.

## Quick Start

```python
from msig import Motif, NullModel
import numpy as np

# Your multivariate time series (m variables × n time points)
data = np.array([...])

# Create null model
model = NullModel(data, dtypes=[float, float], model="empirical")

# Test motif significance
motif = Motif(pattern, variables, thresholds, n_matches)
probability = motif.set_pattern_probability(model, vars_indep=True)
pvalue = motif.set_significance(max_possible_matches, n_variables)

print(f"p-value: {pvalue:.2e}")
```

See the `examples/` folder for complete examples (`simple_example.py` and `example.ipynb`).

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


