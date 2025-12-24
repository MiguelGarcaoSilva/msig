"""
Common utilities for MSig experiments.

This module provides shared functionality across all experiment scripts:
- Metadata tracking (Python version, library versions, timestamps)
- Result saving with standardized formats
- Environment validation
- Common statistical testing workflows
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_environment_info() -> Dict[str, Any]:
    """
    Collect environment information for reproducibility.

    Returns
    -------
    dict
        Environment metadata including Python version, library versions,
        platform info, and timestamp.
    """
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "python_version_short": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    # Collect library versions
    libraries = [
        "numpy", "scipy", "pandas", "matplotlib",
        "stumpy", "librosa", "msig"
    ]

    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, "__version__", "unknown")
            env_info[f"{lib}_version"] = version
        except ImportError:
            env_info[f"{lib}_version"] = "not installed"

    # Platform info
    import platform
    env_info["platform"] = platform.system()
    env_info["platform_version"] = platform.version()
    env_info["platform_machine"] = platform.machine()

    return env_info


def save_experiment_metadata(
    output_dir: str,
    dataset_name: str,
    method_name: str,
    parameters: Dict[str, Any],
    **extra_info
) -> Path:
    """
    Save experiment metadata to JSON file.

    Parameters
    ----------
    output_dir : str
        Directory where metadata will be saved.
    dataset_name : str
        Name of the dataset (e.g., 'audio', 'populationdensity').
    method_name : str
        Discovery method used (e.g., 'stumpy', 'lama', 'momenti').
    parameters : dict
        Experiment parameters (subsequence lengths, thresholds, etc.).
    **extra_info
        Additional metadata fields to include.

    Returns
    -------
    Path
        Path to the saved metadata file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": dataset_name,
        "method": method_name,
        "parameters": parameters,
        "environment": get_environment_info(),
        **extra_info
    }

    metadata_file = output_path / "metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Metadata saved to {metadata_file}")
    return metadata_file


def validate_data_file(file_path: str, required: bool = True) -> bool:
    """
    Validate that a data file exists and is readable.

    Parameters
    ----------
    file_path : str
        Path to the data file.
    required : bool, default=True
        If True, raise FileNotFoundError if file doesn't exist.
        If False, just log a warning and return False.

    Returns
    -------
    bool
        True if file exists and is readable, False otherwise.

    Raises
    ------
    FileNotFoundError
        If required=True and file doesn't exist.
    """
    path = Path(file_path)

    if not path.exists():
        msg = f"Data file not found: {file_path}"
        if required:
            logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            logger.warning(msg)
            return False

    if not path.is_file():
        msg = f"Path exists but is not a file: {file_path}"
        if required:
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    logger.info(f"✓ Data file validated: {file_path}")
    return True


def save_results_with_metadata(
    results_df: pd.DataFrame,
    output_dir: str,
    file_prefix: str,
    dataset_name: str,
    method_name: str,
    parameters: Dict[str, Any],
    **extra_metadata
) -> Tuple[Path, Path]:
    """
    Save results DataFrame and associated metadata.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results to save.
    output_dir : str
        Output directory.
    file_prefix : str
        Prefix for the CSV filename (e.g., 'summary_motifs', 'table_motifs').
    dataset_name : str
        Dataset name.
    method_name : str
        Method name.
    parameters : dict
        Experiment parameters.
    **extra_metadata
        Additional metadata to include.

    Returns
    -------
    tuple of Path
        (csv_file_path, metadata_file_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_file = output_path / f"{file_prefix}.csv"
    results_df.to_csv(csv_file, index=False)
    logger.info(f"Results saved to {csv_file}")

    # Save metadata
    metadata_file = save_experiment_metadata(
        output_dir=output_dir,
        dataset_name=dataset_name,
        method_name=method_name,
        parameters=parameters,
        results_file=str(csv_file),
        num_results=len(results_df),
        **extra_metadata
    )

    return csv_file, metadata_file


def test_motifs_significance(
    data: np.ndarray,
    motifs: List[Dict[str, Any]],
    dtypes: List[type],
    alpha: float = 0.05,
    null_model_type: str = "empirical",
    vars_indep: bool = True,
    idd_correction: bool = False
) -> pd.DataFrame:
    """
    Test statistical significance for a list of discovered motifs.

    This is a common workflow used across all experiment scripts.

    Parameters
    ----------
    data : np.ndarray
        Multivariate time series data (m variables × n time points).
    motifs : list of dict
        List of motifs, where each dict must contain:
        - 'pattern': np.ndarray - The motif pattern
        - 'variables': list/array - Variable indices
        - 'delta_thresholds': list/array - Tolerance thresholds
        - 'n_matches': int - Number of occurrences
        Additional fields are preserved in output.
    dtypes : list of type
        Data types for each variable.
    alpha : float, default=0.05
        Significance level.
    null_model_type : str, default='empirical'
        Type of null model ('empirical', 'kde', 'gaussian_theoretical').
    vars_indep : bool, default=True
        Assume variables are independent.
    idd_correction : bool, default=False
        Apply IDD correction for multiple testing.

    Returns
    -------
    pd.DataFrame
        DataFrame with motif information and statistical test results.
        Columns include: pattern_probability, pvalue, significant (at alpha level).
    """
    from msig import NullModel, Motif, benjamini_hochberg_fdr

    logger.info(f"Testing significance for {len(motifs)} motifs...")

    # Create null model
    logger.info(f"Building {null_model_type} null model...")
    null_model = NullModel(data, dtypes=dtypes, model=null_model_type)

    m, n = data.shape  # m variables, n time points

    results = []
    pvalues = []

    for i, motif_dict in enumerate(motifs):
        try:
            # Extract motif parameters
            pattern = motif_dict["pattern"]
            variables = motif_dict["variables"]
            delta_thresholds = motif_dict["delta_thresholds"]
            n_matches = motif_dict["n_matches"]

            # Determine motif length
            if isinstance(pattern, np.ndarray):
                if pattern.ndim == 1:
                    motif_length = len(pattern)
                else:
                    motif_length = pattern.shape[1]
            else:
                motif_length = len(pattern[0])

            # Create Motif object
            motif = Motif(
                multivar_sequence=pattern,
                variables=variables,
                delta_thresholds=delta_thresholds,
                n_matches=n_matches
            )

            # Compute pattern probability
            p_pattern = motif.set_pattern_probability(null_model, vars_indep=vars_indep)

            # Compute p-value
            max_possible_matches = n - motif_length + 1
            pvalue = motif.set_significance(
                max_possible_matches=max_possible_matches,
                data_n_variables=m,
                idd_correction=idd_correction
            )

            pvalues.append(pvalue)

            # Build result dict
            result = {
                "motif_id": i,
                "pattern_probability": p_pattern,
                "pvalue": pvalue,
                **motif_dict  # Include all original fields
            }

            results.append(result)

        except Exception as e:
            logger.warning(f"Failed to test motif {i}: {e}")
            # Add failed entry
            results.append({
                "motif_id": i,
                "pattern_probability": np.nan,
                "pvalue": np.nan,
                "error": str(e),
                **motif_dict
            })
            pvalues.append(1.0)  # Conservative p-value for failed tests

    # Apply FDR correction
    logger.info("Applying Benjamini-Hochberg FDR correction...")
    critical_value = benjamini_hochberg_fdr(pvalues, false_discovery_rate=alpha)

    # Mark significant motifs
    for result in results:
        result["fdr_critical_value"] = critical_value
        result["significant"] = result.get("pvalue", 1.0) <= critical_value

    results_df = pd.DataFrame(results)

    n_significant = results_df["significant"].sum()
    logger.info(f"Found {n_significant}/{len(motifs)} significant motifs (α={alpha})")

    return results_df


def format_runtime(seconds: float) -> str:
    """
    Format runtime in a human-readable way.

    Parameters
    ----------
    seconds : float
        Runtime in seconds.

    Returns
    -------
    str
        Formatted runtime string (e.g., "2m 30s", "1h 15m").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.0f}h {minutes:.0f}m"


def estimate_memory_requirements(
    n_variables: int,
    n_timepoints: int,
    method: str = "stumpy"
) -> Dict[str, float]:
    """
    Estimate peak memory requirements for an experiment.

    Parameters
    ----------
    n_variables : int
        Number of variables in the dataset.
    n_timepoints : int
        Number of time points.
    method : str, default='stumpy'
        Discovery method ('stumpy', 'lama', 'momenti').

    Returns
    -------
    dict
        Dictionary with memory estimates in MB for different components.
    """
    # Data size (float64)
    data_mb = (n_variables * n_timepoints * 8) / (1024**2)

    estimates = {"data": data_mb}

    if method == "stumpy":
        # Matrix profile: O(n * m^2)
        mp_mb = (n_timepoints * n_variables**2 * 8) / (1024**2)
        estimates["matrix_profile"] = mp_mb
        estimates["total"] = data_mb + mp_mb

    elif method in ["lama", "momenti"]:
        # Distance matrix: O(n^2 / 2)
        # This is an approximation
        dist_mb = (n_timepoints**2 * 8) / (2 * 1024**2)
        estimates["distance_matrix"] = dist_mb
        estimates["total"] = data_mb + dist_mb

    return estimates


# Logging setup helper
def setup_experiment_logging(
    log_file: str = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up consistent logging for experiment scripts.

    Parameters
    ----------
    log_file : str, optional
        Path to log file. If None, only logs to console.
    level : int, default=logging.INFO
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger
