# MSig - Statistical Significance for Time Series Motifs

MSig is a statistical framework for evaluating the significance of motifs in multivariate time series data, accommodating different variable types (continuous, discrete, categorical).

## Key Features

- **Pattern probability**: Estimates the probability of a motif occurring by chance using a null model
- **Significance testing**: Calculates motif significance based on binomial tails to assess motif recurrence
- **Flexible null models**: Supports empirical, KDE, and Gaussian null models
- **Mixed data types**: Handles continuous (float), discrete (int), and categorical (str) variables
- **Multiple testing correction**: Includes Benjamini-Hochberg FDR correction for controlling false discovery rate

## Installation

```bash
pip install msig                    # Core package
pip install "msig[experiments]"     # With experiment dependencies
```

**Development install with uv:**

```bash
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig
uv sync
```

## Quick Start

```python
from msig import Motif, NullModel
import numpy as np

# Create multivariate time series data (4 variables × 15 time points)
ts1 = [1, 3, 3, 5, 5, 2, 3, 3, 5, 5, 3, 3, 5, 4, 4]
ts2 = [4.3, 4.5, 2.6, 3.0, 3.0, 1.7, 4.9, 2.9, 3.3, 1.9, 4.9, 2.5, 3.1, 1.8, 0.3]
ts3 = ["A", "D", "B", "D", "A", "A", "A", "C", "C", "B", "D", "D", "C", "A", "A"]
ts4 = ["T", "L", "T", "Z", "Z", "T", "L", "T", "Z", "T", "L", "T", "Z", "L", "L"]

data = np.stack([
    np.asarray(ts1, dtype=int),
    np.asarray(ts2, dtype=float),
    np.asarray(ts3, dtype=str),
    np.asarray(ts4, dtype=str)
])
m, n = data.shape  # (4 variables, 15 time points)

# Create null model based on observed data
model = NullModel(data, dtypes=[int, float, str, str], model="empirical")

# Define motif pattern (length 3, spanning variables 0, 1, and 3)
vars = np.array([0, 1, 3])
multivar_sequence = data[vars, 1:4]

# Create motif object (3 observed matches)
# Tolerance: exact match for discrete/categorical (δ=0), δ=0.5 for continuous
# Need 3 thresholds (one per variable in motif): [var0=int, var1=float, var3=str]
motif = Motif(
    multivar_sequence=multivar_sequence,
    variables=vars,
    delta_thresholds=np.array([0, 0.5, 0]),
    n_matches=3
)

# Calculate pattern probability under null model
probability = motif.set_pattern_probability(model, vars_indep=True)
print(f"Pattern probability: {probability:.6f}")

# Calculate statistical significance
p = len(multivar_sequence[0])  # motif length
max_possible_matches = n - p + 1
pvalue = motif.set_significance(
    max_possible_matches=max_possible_matches,
    data_n_variables=m,
    idd_correction=False
)
print(f"P-value: {pvalue:.2e}")
print(f"Significant at α=0.05? {pvalue <= 0.05}")
```

## Examples and Experiments

The repository includes comprehensive examples and case studies:

- **Simple examples**: See `examples/simple_example.py` and `examples/example.ipynb`
- **Real-world experiments**: Audio (MFCC features), population density, and washing machine sensor data
- **Motif discovery methods**: Integration with STUMPY, LAMA, and MOMENTI

See the [GitHub repository](https://github.com/MiguelGarcaoSilva/msig) for complete documentation and experiment code.

## Requirements

- Python 3.11-3.13
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0

## Citation

If you use MSig in your research, please cite:

```
Silva, M.G., Henriques, R., and Madeira, S.C. (2025).
On Why and How Statistical Significance Criteria Can Guide 
Multivariate Time Series Motif Analysis.
```

## Authors

- Miguel G. Silva - [GitHub](https://github.com/MiguelGarcaoSilva)
- Rui Henriques - [Homepage](https://web.ist.utl.pt/rmch)
- Sara C. Madeira - [Homepage](https://saracmadeira.wordpress.com)

## Acknowledgements

This work was partially funded by Fundação para a Ciência e a Tecnologia (FCT) through:
- LASIGE Research Unit: UIDB/00408/2020, UIDP/00408/2020
- INESC-ID Pluriannual: UIDB/50021/2020
- PhD scholarship UIBD/153086/2022 to Miguel G. Silva

## License

MIT License - see [LICENSE](https://github.com/MiguelGarcaoSilva/msig/blob/main/LICENSE) file for details.
