#!/usr/bin/env python3
"""
STUMPY-based Case Study for Washing Machine Data
Exhaustive motif discovery using matrix profile approach
"""

import numpy as np
import pandas as pd
import stumpy
import math
import logging
import os
from typing import Tuple

# Add parent directory to path for msig import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from msig import Motif, NullModel

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_washingmachine_data(data_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
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


def compute_matrix_profiles(
    X: np.ndarray,
    subsequence_lengths: list,
    normalize: bool = True,
    include=None,
    results_dir: str = "../results/washingmachine"
) -> None:
    """Compute and save matrix profiles for all subsequence lengths."""
    logger.info(f"\nComputing matrix profiles for {len(subsequence_lengths)} lengths")
    
    os.makedirs(f'{results_dir}/mp', exist_ok=True)
    os.makedirs(f'{results_dir}/mp_indices', exist_ok=True)
    
    for s in subsequence_lengths:
        logger.info(f"  Computing MP for s={s}...")
        
        mp, mp_indices = stumpy.mstump(X, s, include=include, normalize=normalize)
        
        np.save(f'{results_dir}/mp/mp_normalized={normalize}_include={include}_s={s}.npy', mp)
        np.save(f'{results_dir}/mp_indices/mp_indices_normalized={normalize}_include={include}_s={s}.npy', mp_indices)
    
    logger.info("Matrix profiles saved")


def compute_motif_statistics_stumpy(
    motif_indices: list | np.ndarray,
    motif_distances: list | np.ndarray,
    motif_subspaces: list,
    data: np.ndarray,
    s: int,
    normalize: bool = True,
    max_allowed_dist: float | None = None
) -> pd.DataFrame:
    """Compute statistical significance for STUMPY-discovered motifs."""
    stats_table = pd.DataFrame(columns=[
        "ID", "k", "Features", "s", "#Matches", "Indices", 
        "max(dists)", "min(dists)", "med(dists)", "P", "p-value"
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
    r = np.ceil(s / 2)
    max_possible_matches = int(np.floor((n_time - s) / r) + 1)
    
    motif_index = 0
    
    for motif_idx, (indices, distances, dimensions) in enumerate(zip(motif_indices, motif_distances, motif_subspaces)):
        # Remove -1 padding and NaNs
        indices = indices[indices != -1]
        distances = distances[~np.isnan(distances)]
        
        if len(indices) == 0:
            continue
        
        # Filter for non-trivial matches (s/2 separation)
        non_trivial = []
        for idx in indices:
            is_trivial = False
            for existing_idx in non_trivial:
                if abs(idx - existing_idx) <= s / 2:
                    is_trivial = True
                    break
            if not is_trivial:
                non_trivial.append(idx)
        
        if len(non_trivial) == 0:
            continue
        
        indices = non_trivial
        
        # Extract pattern from first occurrence
        pattern_pos = int(indices[0])
        multivar_subsequence = data_norm[dimensions, pattern_pos:pattern_pos + s]
        
        # Calculate distance statistics
        max_dist = np.max(distances)
        min_dist = np.min(distances[1:]) if len(distances) > 1 else np.min(distances)
        med_dist = np.median(distances[1:]) if len(distances) > 1 else np.median(distances)
        
        # Calculate delta threshold
        if max_allowed_dist is None:
            D = np.empty((n_time - s + 1, len(dimensions)))
            for i, dim in enumerate(dimensions):
                D[:, i] = stumpy.mass(multivar_subsequence[i], data_norm[dim], normalize=True)
            D = np.nanmean(D, axis=1)
            max_allowed_dist = np.nanmax([np.nanmean(D) - 2.0 * np.nanstd(D), np.nanmin(D)])
        
        assert max_allowed_dist is not None  # Type narrowing for Pylance
        max_delta = math.sqrt(max_allowed_dist**2 / s)
        delta_thresholds = [max_delta] * n_vars
        
        # Compute significance
        motif_obj = Motif(list(multivar_subsequence), dimensions, delta_thresholds, len(indices))
        p_pattern = motif_obj.set_pattern_probability(model_empirical, vars_indep=True)
        p_value = motif_obj.set_significance(max_possible_matches, n_vars, idd_correction=False)
        
        # Store results
        stats_row = {
            "ID": motif_index,
            "k": len(dimensions),
            "Features": ",".join([str(d) for d in dimensions]),
            "s": s,
            "#Matches": len(indices),  # Total occurrences (consistent with LAMA and statistical test)
            "Indices": [int(i) for i in indices],
            "max(dists)": round(max_dist, 3),
            "min(dists)": round(min_dist, 3),
            "med(dists)": round(med_dist, 3),
            "P": p_pattern,
            "p-value": p_value
        }
        
        stats_table = pd.concat([stats_table, pd.DataFrame([stats_row])], ignore_index=True)
        motif_index += 1
    
    return stats_table


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../data/washingmachine/main_readings.csv")
    results_dir = os.path.abspath(os.path.join(script_dir, "../../results/washingmachine/stumpy"))
    
    # Load data
    X, data_df = load_washingmachine_data(data_path)
    
    # Parameters
    normalize = True
    include = None
    subsequence_lengths = [30, 60, 300, 600]  # Seconds (30s, 1, 5, 10 minutes)
    
    logger.info(f"Subsequence lengths: {subsequence_lengths} (seconds)")
    
    # Configure STUMPY exclusion zone (newer versions handle this automatically)
    # stumpy.config.STUMPY_EXCL_ZONE_DENOM = 2  # r = np.ceil(m/2)
    
    # Compute matrix profiles
    compute_matrix_profiles(X, subsequence_lengths, normalize, include, results_dir)
    
    # Define discovery modes: liberal, moderate, and conservative
    # These modes systematically explore the trade-off between sensitivity and specificity
    # Liberal: High sensitivity - discover all potential patterns (upper bound)
    # Moderate: Balanced - practical recommendation for real use cases
    # Conservative: High specificity - minimize false positives (lower bound)
    # Each mode controls quality vs quantity of discovered motifs through 5 parameters:
    # - min_neighbors: minimum matches required (occurrence filter)
    # - max_distance_factor: distance threshold scaling (quality filter)
    # - cutoffs: per-dimension distance threshold (quality filter for multidimensional)
    # - max_matches: max occurrences per motif (computational limit)
    # - max_motifs: max total motifs (computational limit)
    modes = {
        'liberal': {
            'min_neighbors': 1,           # Accept even single matches
            'max_distance_factor': 1.0,   # D_max = sqrt(s) * delta (most permissive)
            'cutoffs': np.inf,            # No per-dimension filtering
            'max_matches': 99999,         # No practical limit
            'max_motifs': 99999,          # No practical limit
            'description': 'Liberal: Maximum sensitivity, accept all potential patterns'
        },
        'moderate': {
            'min_neighbors': 2,           # Require at least 3 matches (excludes singletons)
            'max_distance_factor': 0.75,   # D_max = sqrt(s) * delta * 0.75 (middle ground)
            'cutoffs': np.inf,            # No per-dimension filtering
            'max_matches': 99999,         # No practical limit
            'max_motifs': 100,          # No practical limit
            'description': 'Moderate: Balanced sensitivity and specificity'
        },
        'conservative': {
            'min_neighbors': 3,           # Require at least 5 matches (strong evidence)
            'max_distance_factor': 0.5,   # D_max = delta (strictest distance)
            'cutoffs': np.inf,            # No per-dimension filtering
            'max_matches': 99999,         # No practical limit
            'max_motifs': 50,             # Focus on top 50 patterns
            'description': 'Conservative: High confidence, minimize false positives'
        }
    }
    
    # Run all modes
    for mode_name, mode_config in modes.items():
        logger.info(f"\n{'=' * 70}")
        logger.info(f"RUNNING MODE: {mode_name.upper()} - {mode_config['description']}")
        logger.info(f"{'=' * 70}")
        logger.info(f"Parameters: min_neighbors={mode_config['min_neighbors']}, "
                   f"max_matches={mode_config['max_matches']}, max_motifs={mode_config['max_motifs']}, "
                   f"max_distance_factor={mode_config['max_distance_factor']}, "
                   f"cutoffs={mode_config['cutoffs']}")
        
        # Create mode-specific output directory
        mode_results_dir = os.path.join(results_dir, f"stumpy_{mode_name}")
        os.makedirs(mode_results_dir, exist_ok=True)
        
        # Extract mode parameters
        min_neighbors = mode_config['min_neighbors']
        max_matches = mode_config['max_matches']
        max_motifs = mode_config['max_motifs']
        k = None  # Auto-compute dimensionality using MDL (unconstrained search)
        average_delta = 0.3
        
        all_stats = pd.DataFrame()
        max_dists = []
        cutoffs_list = []
        
        for s in subsequence_lengths:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Processing motif length s={s} ({s} seconds = {s / 60:.1f} minutes)")
            logger.info(f"={'=' * 70}")
            
            # Load matrix profile
            mp = np.load(f'{results_dir}/mp/mp_normalized={normalize}_include={include}_s={s}.npy')
            mp_indices = np.load(f'{results_dir}/mp_indices/mp_indices_normalized={normalize}_include={include}_s={s}.npy')
            
            # Calculate distance threshold with mode-specific factor
            if mode_config['max_distance_factor'] is None:
                # Let STUMPY auto-calculate: nanmax([nanmean(D) - 2*nanstd(D), nanmin(D)])
                max_distance = None
            elif np.isinf(mode_config['max_distance_factor']):
                max_distance = np.inf
            elif mode_config['max_distance_factor'] == 0.0:
                # Restrictive mode: D_max = delta (no sqrt(s) scaling)
                max_distance = average_delta
            else:
                # Comprehensive mode: D_max = sqrt(s) * delta * factor
                max_distance = math.sqrt(s) * average_delta * mode_config['max_distance_factor']
            max_dists.append(max_distance)
            
            # Calculate cutoffs (per-dimension distance thresholds)
            if mode_config['cutoffs'] is None:
                # Auto-calculate based on matrix profile statistics per dimension
                cutoffs = np.array([np.nanmean(mp[:, i]) + 1.0 * np.nanstd(mp[:, i]) 
                                   for i in range(mp.shape[1])])
                cutoffs = np.maximum(cutoffs, np.nanmin(mp, axis=0))  # At least the minimum
            elif np.isinf(mode_config['cutoffs']):
                cutoffs = np.inf
            else:
                cutoffs = mode_config['cutoffs']
            
            cutoffs_list.append(cutoffs)
            logger.info(f"max_distance={max_distance if max_distance is not None and not np.isinf(max_distance) else ('inf' if max_distance is not None else 'auto')}, cutoffs={cutoffs if isinstance(cutoffs, (float, int)) else 'per-dimension'}")
            
            # Find motifs
            motif_distances, motif_indices, motif_subspaces, motif_mdls = stumpy.mmotifs(
                X, mp, mp_indices,
                max_distance=max_distance,
                max_matches=max_matches,
                cutoffs=cutoffs,
                min_neighbors=min_neighbors,
                max_motifs=max_motifs,
                k=k,
                include=include,
                normalize=normalize
            )
            
            if len(motif_indices[0]) == 0:
                logger.warning(f"No motifs found for s={s}")
                continue
            
            logger.info(f"Found {len(motif_indices)} motif candidates")
            
            # Compute statistics
            stats = compute_motif_statistics_stumpy(
                motif_indices, motif_distances, motif_subspaces,
                X, s, normalize, max_distance
            )
            
            if stats.empty:
                continue
            
            # Count significant motifs
            n_significant = np.sum(stats["p-value"] <= 0.01)
            logger.info(f"Significant (p≤0.01): {n_significant}/{len(stats)}")
            
            # Add Hochberg correction (standard BH uses ≤)
            p_values = stats["p-value"].to_numpy()
            critical_value = NullModel.hochberg_critical_value(p_values, 0.05)
            sig_hochberg = stats["p-value"] <= critical_value
            stats["Sig_Hochberg"] = sig_hochberg
            
            n_hochberg = np.sum(sig_hochberg)
            logger.info(f"Significant (Hochberg): {n_hochberg}/{len(stats)}, critical={critical_value:.4e}")
            
            # Accumulate results
            all_stats = pd.concat([all_stats, stats], ignore_index=True)
        
        # Save detailed results for this mode (simplified filename)
        filename = f"table_motifs_{mode_name}.csv"
        detailed_path = os.path.join(mode_results_dir, filename)
        all_stats.to_csv(detailed_path, index=False)
        logger.info(f"\nSaved detailed results to {detailed_path}")
        
        # Create summary table
        summary_table = pd.DataFrame()
        
        # Check if any motifs were found
        if all_stats.empty:
            logger.warning(f"No motifs found for any subsequence length in {mode_name} mode")
        else:
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
        
        # Save summary for this mode
        summary_path = os.path.join(mode_results_dir, f"summary_motifs_stumpy_{mode_name}.csv")
        summary_table.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")
        
        # Print summary
        logger.info(f"\n{'=' * 70}")
        logger.info(f"SUMMARY TABLE - {mode_name.upper()} MODE")
        logger.info(f"{'=' * 70}")
        print(summary_table.to_string(index=False))
    
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPLETE - ALL MODES FINISHED")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
