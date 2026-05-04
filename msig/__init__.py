"""MSig: Statistical significance testing for multivariate time series motifs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("msig")
except PackageNotFoundError:  # editable install before package is registered
    __version__ = "0.0.0+unknown"

from .MSig import (
    NullModel,
    Motif,
    benjamini_hochberg_fdr,
    bonferroni_correction,
)

__all__ = [
    "NullModel",
    "Motif",
    "benjamini_hochberg_fdr",
    "bonferroni_correction",
    "__version__",
]
