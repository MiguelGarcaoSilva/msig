#!/usr/bin/env python3
"""
LAMA-based Case Study for Washing Machine Data
Iterative motif discovery using masking approach:
- Run LAMA to find best motif
- Mask found motif with exclusion zone
- Repeat to find multiple non-overlapping motifs
- Extract elbow points to determine natural k values
"""

import sys
import os

# Add parent directory to path for msig import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from msig import Motif, NullModel

# Add leitmotifs to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../leitmotifs')))
from leitmotifs import lama

import numpy as np
import pandas as pd
import math
import logging
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_washingmachine_data(data_path: str) -> tuple:
    """Load washing machine sensor data."""
    logger.info(f"Loading washing machine data from {data_path}")
    
    colnames = ["timestamp", "current", "power_factor", "water_monitoring_inlet", 
                "water_monitoring_outlet", "water_temperature_inlet", "water_temperature_outlet"]
    data = pd.read_csv(data_path, usecols=colnames)
    
    # Rename columns
    data.columns = ["timestamp", "Current (A)", "Power Factor", "Water Inlet (ml)", 
                   "Water Outlet (ml)", "Water Inlet Temperature(°C)", "Water Outlet Temperature(°C)"]
    
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data['timestamp'] = data['timestamp'].dt.floor('s')
    
    # Remove duplicates
    data = data.drop_duplicates(subset='timestamp', keep='first')
    
    # Normalize time to start at 00:00:00
    first_datetime = data['timestamp'].iloc[0]
    time_diff = data['timestamp'].apply(lambda x: x - first_datetime)
    first_datetime = first_datetime.replace(hour=0, minute=0, second=0)
    data['timestamp'] = first_datetime + time_diff
    data = data.set_index('timestamp')
    
    X = data.to_numpy().astype(np.float64).T
    
    logger.info(f"Loaded data: X.shape={X.shape}, duration={X.shape[1]} seconds")
    return X, data


def search_leitmotifs_iterative(
    data: np.ndarray,
    motif_length: int,
    k_max: int = 100,
    max_motifs: int = 50,
    min_separation_factor: float = 0.5,
    elbow_deviation: float = 1.0,
    slack: float = 0.5,
    n_dims: int = None,
    n_jobs: int = -1,
    backend: str = 'scalable'
) -> List[Dict]:
    """
    Search for multiple leitmotifs using iterative masking approach.
    
    LAMA finds ONE best motif per run, so we:
    1. Run LAMA to find best motif (use elbow points for natural k)
    2. Mask the found motif with exclusion zone
    3. Repeat until no more motifs found or max_motifs reached
    
    Parameters:
        n_dims: Number of dimensions to use. If None, uses all dimensions.
                If < d, LAMA adaptively selects the best n_dims dimensions per motif.
    
    This is similar to STUMPY's exhaustive discovery.
    """
    n_vars, n_time = data.shape
    min_sep = int(motif_length * min_separation_factor)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Searching leitmotifs iteratively for s={motif_length}, k_max={k_max}")
    logger.info(f"{'=' * 60}")
    
    # Normalize data
    data_norm = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    
    # Create a mask to track excluded regions
    mask = np.ones(n_time, dtype=bool)  # True = available, False = masked
    
    all_motifs = []
    iteration = 0
    
    while len(all_motifs) < max_motifs and np.sum(mask) > motif_length:
        iteration += 1
        logger.info(f"\n--- Iteration {iteration} ---")
        logger.info(f"Available time points: {np.sum(mask)}/{n_time}")
        
        # Create masked data (set masked regions to large random values to prevent matching)
        data_masked = data_norm.copy()
        for dim in range(n_vars):
            data_masked[dim, ~mask] = np.random.randn(np.sum(~mask)) * 100  # Large random values
        
        try:
            # Run LAMA on masked data
            # n_dims enables adaptive dimension selection: LAMA picks best n_dims dimensions per motif
            result = lama.search_leitmotifs_elbow(
                data=data_masked,
                motif_length=motif_length,
                k_max=k_max,
                n_dims=n_dims,  # Let LAMA adaptively select best dimensions
                elbow_deviation=elbow_deviation,
                filter=True,
                slack=slack,
                n_jobs=n_jobs,
                backend=backend
            )
            
            # Unpack results
            dists = result[0]
            candidates = result[1]
            dims = result[2]
            elbows = result[3] if len(result) > 3 else None
            
            if elbows is None or len(elbows) == 0:
                logger.info("No elbows found, stopping")
                break
            
            # Use the largest elbow point (most interesting motif size)
            best_elbow = int(elbows[-1])
            motif_indices = candidates[best_elbow]
            motif_dims = dims[best_elbow]
            motif_extent = float(dists[best_elbow])
            
            if motif_indices is None or len(motif_indices) == 0:
                logger.info("No motif indices found, stopping")
                break
            
            motif_indices = np.array(motif_indices, dtype=int)
            
            # Filter for non-trivial matches within this motif
            non_trivial = []
            for idx in motif_indices:
                # Check if index is in available (unmasked) region
                if idx >= n_time or not mask[idx]:
                    continue
                    
                # Check for trivial matches with already selected indices
                is_trivial = False
                for existing_idx in non_trivial:
                    if abs(idx - existing_idx) <= min_sep:
                        is_trivial = True
                        break
                
                if not is_trivial:
                    non_trivial.append(idx)
            
            if len(non_trivial) < 2:
                logger.info(f"Only {len(non_trivial)} non-trivial matches, stopping")
                break
            
            # Accept this motif
            motif_info = {
                'indices': np.array(non_trivial, dtype=int),
                'dims': motif_dims if motif_dims is not None else np.arange(n_vars),
                'extent': motif_extent,
                'k': best_elbow,
                'distance': motif_extent
            }
            all_motifs.append(motif_info)
            
            logger.info(f"Found motif #{len(all_motifs)}: k={best_elbow}, extent={motif_extent:.4f}, "
                       f"#matches={len(non_trivial)}")
            
            # Mask out the found motif (apply exclusion zone)
            for idx in non_trivial:
                start = max(0, idx - min_sep)
                end = min(n_time, idx + motif_length + min_sep)
                mask[start:end] = False
            
            logger.info(f"Masked {len(non_trivial)} regions")
            
        except Exception as e:
            logger.error(f"Error during LAMA search iteration {iteration}: {e}")
            break
    
    logger.info(f"\nTotal motifs found: {len(all_motifs)}")
    return all_motifs


def search_leitmotifs_all_elbows(
    data: np.ndarray,
    motif_length: int,
    k_max: int = 100,
    min_separation_factor: float = 0.5,
    elbow_deviation: float = 1.0,
    slack: float = 0.5,
    n_jobs: int = -1,
    backend: str = 'scalable'
) -> List[Dict]:
    """
    Search for ALL leitmotifs by extracting ALL elbow points.
    
    This matches STUMPY's behavior:
    1. Run LAMA once to get distance curve
    2. Find ALL elbow points
    3. Extract motifs for each elbow
    4. Filter for non-trivial matches in post-processing
    
    NO iterative masking!
    """
    n_vars, n_time = data.shape
    min_sep = int(motif_length * min_separation_factor)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Searching ALL leitmotifs for s={motif_length}, k_max={k_max}")
    logger.info(f"{'=' * 60}")
    
    # Normalize data
    data_norm = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    
    # Run LAMA ONCE
    try:
        result = lama.search_leitmotifs_elbow(
            data=data_norm,
            motif_length=motif_length,
            k_max=k_max,
            n_dims=None,
            elbow_deviation=elbow_deviation,
            filter=True,
            slack=slack,
            n_jobs=n_jobs,
            backend=backend
        )
        
        # Unpack results
        dists = result[0]
        candidates = result[1]
        dims = result[2]
        elbows = result[3] if len(result) > 3 else None
        
        if elbows is None or len(elbows) == 0:
            logger.warning("No elbows found!")
            return []
        
        logger.info(f"Found {len(elbows)} elbow points: {elbows}")
        
        # Extract motif for EACH elbow point
        all_motifs = []
        for elbow_idx in reversed(elbows):  # Process from largest k to smallest
            k_value = int(elbow_idx)
            motif_indices = candidates[k_value]
            motif_dims = dims[k_value]
            motif_extent = float(dists[k_value])
            
            if motif_indices is None or len(motif_indices) == 0:
                continue
            
            motif_indices = np.array(motif_indices, dtype=int)
            
            # Filter for non-trivial matches (like STUMPY does)
            non_trivial = []
            for idx in motif_indices:
                is_trivial = False
                for existing_idx in non_trivial:
                    if abs(idx - existing_idx) <= min_sep:
                        is_trivial = True
                        break
                if not is_trivial:
                    non_trivial.append(idx)
            
            if len(non_trivial) < 2:
                continue
            
            # Check if this motif overlaps with already accepted motifs
            overlaps_existing = False
            for existing_motif in all_motifs:
                # Check if ANY index from new motif is too close to ANY index in existing motif
                for new_idx in non_trivial:
                    for exist_idx in existing_motif['indices']:
                        if abs(new_idx - exist_idx) <= min_sep:
                            overlaps_existing = True
                            break
                    if overlaps_existing:
                        break
                if overlaps_existing:
                    break
            
            if overlaps_existing:
                logger.info(f"  Skipping elbow k={k_value} (overlaps with existing motif)")
                continue
            
            # Accept this motif
            motif_info = {
                'indices': np.array(non_trivial, dtype=int),
                'dims': motif_dims,
                'extent': motif_extent,
                'k': k_value,
                'distance': motif_extent
            }
            all_motifs.append(motif_info)
            
            logger.info(f"  Motif #{len(all_motifs)}: k={k_value}, extent={motif_extent:.4f}, "
                       f"#matches={len(non_trivial)}, indices={non_trivial[:3]}...")
        
        logger.info(f"\nTotal non-overlapping motifs found: {len(all_motifs)}")
        return all_motifs
        
    except Exception as e:
        logger.error(f"Error during LAMA search: {e}")
        import traceback
        traceback.print_exc()
        return []


def compute_motif_statistics_lama(
    motifs: List[Dict],
    data: np.ndarray,
    m: int,
    normalize: bool = True,
    average_delta: float = 0.3
) -> pd.DataFrame:
    """
    Compute statistical significance for LAMA-discovered motifs.
    """
    logger.info(f"\nComputing statistics for {len(motifs)} motifs")
    
    stats_table = pd.DataFrame(columns=[
        "ID", "k", "Features", "m", "#Matches", "Indices", 
        "extent", "P", "p-value"
    ])
    
    n_vars, n_time = data.shape
    
    # Z-normalize if needed
    if normalize:
        data_norm = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    else:
        data_norm = data.copy()
    
    # Create null model
    dtypes = [float] * n_vars
    model_empirical = NullModel(data_norm, dtypes=dtypes, model="empirical")
    
    # Calculate max possible matches
    r = np.ceil(m / 2)
    max_possible_matches = int(np.floor((n_time - m) / r) + 1)
    
    for motif_idx, motif_info in enumerate(motifs):
        indices = motif_info['indices']
        dimensions = motif_info['dims']
        
        if dimensions is None or len(dimensions) == 0:
            dimensions = np.arange(n_vars)
        
        dimensions = np.array(dimensions, dtype=int)
        n_matches = len(indices)
        
        if n_matches < 2:
            continue
        
        # Extract pattern from first occurrence
        pattern_pos = int(indices[0])
        multivar_subsequence = data_norm[dimensions, pattern_pos:pattern_pos + s]
        
        # Calculate delta threshold
        max_distance = math.sqrt(s) * average_delta
        max_delta = math.sqrt(max_distance**2 / s)
        delta_thresholds = [max_delta] * len(dimensions)
        
        # Compute significance
        motif_obj = Motif(multivar_subsequence, dimensions, delta_thresholds, n_matches)
        p_pattern = motif_obj.set_pattern_probability(model_empirical, vars_indep=True)
        p_value = motif_obj.set_significance(max_possible_matches, n_vars, idd_correction=False)
        
        logger.info(f"Motif {motif_idx}: k={len(dimensions)}, #matches={n_matches}, p-value={p_value:.2e}")
        
        # Store results
        stats_row = {
            "ID": motif_idx,
            "k": len(dimensions),
            "Features": ",".join([str(d) for d in dimensions]),
            "s": s,
            "#Matches": n_matches,
            "Indices": indices.tolist(),
            "extent": motif_info['extent'],
            "P": p_pattern,
            "p-value": p_value
        }
        
        stats_table = pd.concat([stats_table, pd.DataFrame([stats_row])], ignore_index=True)
    
    return stats_table


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../data/washingmachine/main_readings.csv")
    results_dir = os.path.join(script_dir, "../../results/washingmachine/lama_iterative")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    X, data_df = load_washingmachine_data(data_path)
    
    # Parameters (matching STUMPY case study)
    normalize = True
    subsequence_lengths = [30, 60, 300, 600]  # Seconds (30s, 1min, 5min, 10min)
    average_delta = 0.3
    k_max = 99
    n_jobs = -1
    
    # EXHAUSTIVE dimension search: Run LAMA for each dimensionality
    # Similar to MOMENTI's approach but using LAMA's adaptive selection
    # For d=6 sensors, try n_dims = 2, 3, 4, 5
    n_dims_range = [2, 3, 4, 5]  # Range of dimensionalities to explore
    
    logger.info(f"Subsequence lengths: {subsequence_lengths} (seconds)")
    logger.info(f"Dimensionality range: {n_dims_range}")
    
    # Storage for all results
    all_stats = pd.DataFrame()
    
    # Process each motif length
    for s in subsequence_lengths:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing motif length s={s} ({s} seconds = {s / 60:.1f} minutes)")
        logger.info(f"{'=' * 70}")
        
        # Adjust k_max for this motif length
        n_time = X.shape[1]
        r = np.ceil(s / 2)
        max_possible_k = int(np.floor((n_time - s) / r) + 1)
        k_max_adjusted = min(k_max, max_possible_k - 1)
        
        # Run for EACH dimensionality
        for n_dims_val in n_dims_range:
            logger.info(f"\n--- Running with n_dims={n_dims_val} ---")
            logger.info(f"Using k_max={k_max_adjusted}, n_dims={n_dims_val}")
            
            # Find ALL motifs for this length and dimensionality using iterative masking
            # LAMA adaptively picks best n_dims_val dimensions per motif
            motifs = search_leitmotifs_iterative(
                data=X,
                motif_length=s,
                k_max=k_max_adjusted,
                max_motifs=50,  # Match STUMPY's max_motifs
                min_separation_factor=0.5,  # Same as STUMPY (s / 2)
                elbow_deviation=1.0,
                slack=0.5,
                n_dims=n_dims_val,  # Run for this specific dimensionality
                n_jobs=n_jobs,
                backend='scalable'
            )
            
            if len(motifs) == 0:
                logger.warning(f"No motifs found for s={s}, n_dims={n_dims_val}")
                continue
            
            # Compute statistics
            stats = compute_motif_statistics_lama(
                motifs, X, s,                 normalize=normalize, 
                average_delta=average_delta
            )
            
            if stats.empty:
                continue
            
            # Add n_dims column to track which dimensionality was used
            stats["n_dims_param"] = n_dims_val
            
            # Add Hochberg correction (standard BH uses ≤)
            p_values = stats["p-value"].to_numpy()
            critical_value = NullModel.hochberg_critical_value(p_values, 0.05)
            sig_hochberg = stats["p-value"] <= critical_value
            stats["Sig_Hochberg"] = sig_hochberg
            
            # Print results
            n_significant = np.sum(stats["p-value"] <= 0.01)
            n_hochberg = np.sum(sig_hochberg)
            
            logger.info(f"Results for s={s}, n_dims={n_dims_val}:")
            logger.info(f"  Total motifs: {len(motifs)}")
            logger.info(f"  Significant (p<0.01): {n_significant}")
            logger.info(f"  Significant (Hochberg): {n_hochberg}, critical={critical_value:.4e}")
            
            # Accumulate results
            all_stats = pd.concat([all_stats, stats], ignore_index=True)
    
    # Save detailed results
    detailed_path = os.path.join(results_dir, "table_motifs_lama_iterative.csv")
    all_stats.to_csv(detailed_path, index=False)
    logger.info(f"\nSaved detailed results to {detailed_path}")
    
    # Create summary table
    summary_table = pd.DataFrame()
    for s in subsequence_lengths:
        s_stats = all_stats[all_stats["s"] == s]
        if s_stats.empty:
            continue
        
        n_motifs = len(s_stats)
        n_sig = np.sum(s_stats["p-value"] <= 0.01)
        n_hochberg = np.sum(s_stats["Sig_Hochberg"])
        
        avg_matches = s_stats["#Matches"].mean()
        std_matches = s_stats["#Matches"].std()
        avg_k = s_stats["k"].mean()
        std_k = s_stats["k"].std()
        median_p = s_stats["P"].median()
        median_pval = s_stats["p-value"].median()
        
        stats_df = {
            "s": s,
            "#motifs": n_motifs,
            "avg_n_matches": (avg_matches, std_matches),
            "avg_n_features": (avg_k, std_k),
            "median_probability": median_p,
            "median_pvalue": median_pval,
            "#sig_motifs(≤0.01)": n_sig,
            "significant": 100.0 * n_sig / n_motifs if n_motifs > 0 else 0,
            "#sig_hochberg": n_hochberg
        }
        
        summary_table = pd.concat([summary_table, pd.DataFrame([stats_df])], 
                                  ignore_index=True)
    
    # Save summary
    summary_path = os.path.join(results_dir, "summary_motifs_lama_iterative.csv")
    summary_table.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")
    
    # Print summary
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'=' * 70}")
    print(summary_table.to_string(index=False))
    
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPLETE")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
