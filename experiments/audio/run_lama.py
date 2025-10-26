#!/usr/bin/env python3
"""
Audio Motif Discovery using LAMA
Iterative motif discovery using masking approach
"""

import sys
import os
# Add parent directory to path for msig imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import logging
from typing import List, Dict, Tuple
from msig import Motif, NullModel

# Add leitmotifs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../leitmotifs'))
import leitmotifs.lama as lama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio_data(audio_path: str) -> Tuple[np.ndarray, pd.DataFrame, int, int]:
    """
    Load audio file and extract MFCC features.
    
    Returns:
        X: MFCC features (n_mfcc, n_frames)
        data: DataFrame with datetime index
        sr: sample rate
        hop_length: hop length in samples
    """
    logger.info(f"Loading audio from {audio_path}")
    y, sr = librosa.load(audio_path)
    
    # Extract MFCCs
    n_mfcc = 12
    n_fft = int(sr * 0.046)  # 46 milliseconds STFT window
    hop_length = int(sr * 0.023)  # 23 milliseconds STFT hop
    X = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, 
                             n_fft=n_fft, n_mels=32)
    X = X.astype(np.float64)
    
    # Create DataFrame
    data = pd.DataFrame(X.T)
    data['datetime'] = data.index * hop_length / sr
    data['datetime'] = pd.to_datetime(data['datetime'], unit='s')
    data = data.set_index('datetime')
    data.columns = ['Coefficient ' + str(i) for i in range(n_mfcc)]
    
    logger.info(f"Loaded audio: X.shape={X.shape}, duration={len(y)/sr:.2f}s")
    return X, data, sr, hop_length


def check_overlap(indices1: np.ndarray, indices2: np.ndarray, min_separation: int) -> bool:
    """
    Check if two motif sets overlap (have trivial matches).
    
    Returns True if they overlap significantly.
    """
    for idx1 in indices1:
        for idx2 in indices2:
            if abs(idx1 - idx2) < min_separation:
                return True
    return False


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
    
    Parameters:
        motifs: List of motif dictionaries from search_leitmotifs_multiple
        data: Original time series data (n_vars, n_time)
        m: Motif length
        normalize: Whether to z-normalize the data
        average_delta: Parameter for delta threshold calculation
    
    Returns:
        DataFrame with motif statistics and p-values
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
        delta_thresholds = [max_delta] * n_vars
        
        # Compute significance
        motif_obj = Motif(multivar_subsequence, dimensions, delta_thresholds, n_matches)
        p_pattern = motif_obj.set_pattern_probability(model_empirical, vars_indep=True)
        p_value = motif_obj.set_significance(max_possible_matches, n_vars, idd_correction=False)
        
        # Create row
        stats_row = {
            "ID": f"lama_{motif_idx}",
            "k": len(dimensions),
            "Features": ",".join(map(str, dimensions)),
            "s": s,
            "#Matches": n_matches,
            "Indices": list(map(int, indices)),
            "extent": float(motif_info['extent']),
            "P": p_pattern,
            "p-value": p_value
        }
        
        stats_table = pd.concat([stats_table, pd.DataFrame([stats_row])], ignore_index=True)
        
        logger.info(f"Motif {motif_idx}: k={len(dimensions)}, "
                   f"#matches={n_matches}, p-value={p_value:.2e}")
    
    return stats_table


def create_summary_table(stats_table: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics grouped by motif length.
    """
    subsequence_lengths = stats_table["s"].unique()
    
    summary_cols = [
        "m", "#motifs", "avg_n_matches", "avg_n_features", 
        "median_probability", "median_pvalue", "#sig_motifs(≤0.01)", 
        "significant", "#sig_hochberg"
    ]
    summary_table = pd.DataFrame(columns=summary_cols)
    
    for s in subsequence_lengths:
        table = stats_table[stats_table["s"] == m]
        n_motifs = table.shape[0]
        
        if n_motifs == 0:
            continue
        
        n_sig_motifs_001 = table[table["p-value"] <= 0.01].shape[0]
        
        # Hochberg correction (standard BH uses ≤)
        p_values = table["p-value"].to_numpy()
        critical_value = NullModel.hochberg_critical_value(p_values, 0.05)
        sig = table["p-value"] <= critical_value
        n_sig_motifs_hochberg = sig.sum()
        
        avg_n_matches = (round(table["#Matches"].mean(), 2), 
                        round(table["#Matches"].std(), 3))
        avg_n_features = (round(table["k"].mean(), 2), 
                         round(table["k"].std(), 3))
        median_probability = table["P"].median()
        median_pvalue = table["p-value"].median()
        
        stats_df = {
            "s": s,
            "#motifs": n_motifs,
            "#sig_motifs(≤0.01)": n_sig_motifs_001,
            "significant": (n_sig_motifs_001 * 100) / n_motifs if n_motifs > 0 else 0,
            "#sig_hochberg": n_sig_motifs_hochberg,
            "avg_n_matches": avg_n_matches,
            "avg_n_features": avg_n_features,
            "median_probability": median_probability,
            "median_pvalue": median_pvalue
        }
        
        summary_table = pd.concat([summary_table, pd.DataFrame([stats_df])], 
                                 ignore_index=True)
    
    return summary_table


def main():
    """
    Main execution: Run LAMA-based motif discovery on audio data.
    """
    # Configuration
    audio_path = '../../data/audio/imblue.mp3'
    output_dir = '../../results/audio/lama_iterative'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X, data_df, sr, hop_length = load_audio_data(audio_path)
    
    # Define subsequence lengths (same as STUMPY version)
    subsequence_lengths = [int(secs * sr / hop_length) for secs in [0.5, 1, 3, 5]]
    logger.info(f"Subsequence lengths: {subsequence_lengths}")
    
    # Parameters
    normalize = True
    average_delta = 0.3
    max_motifs_per_length = 20  # Try to find up to 20 motifs per length
    n_jobs = -1  # Use all available CPUs
    
    # EXHAUSTIVE dimension search: Run LAMA for each dimensionality
    # Similar to MOMENTI's approach but using LAMA's adaptive selection
    # For d=12 MFCCs, try n_dims = 2, 3, 4, 5, 6
    n_dims_range = [2, 3, 4, 5, 6]  # Range of dimensionalities to explore
    
    # Run LAMA for each motif length AND each n_dims
    all_stats = pd.DataFrame()
    
    for s in subsequence_lengths:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing motif length s={s} ({m*hop_length/sr:.2f} seconds)")
        logger.info(f"{'='*70}")
        
        # Determine k_max (same logic as in leitmotifs_multiple notebook)
        N = X.shape[1]
        theo_max = int(np.floor((N - m) / (s / 2)) + 1)
        k_max_adjusted = max(2, min(99, theo_max))
        
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
            
            # Compute statistics and significance
            stats_table = compute_motif_statistics_lama(
                motifs=motifs,
                data=X,
                s={s},
                normalize=normalize,
                average_delta=average_delta
            )
            
            # Add n_dims column to track which dimensionality was used
            if len(stats_table) > 0:
                stats_table["n_dims_param"] = n_dims_val
            
            # Add Hochberg significance column
            if len(stats_table) > 0:
                p_values = stats_table["p-value"].to_numpy()
                critical_value = NullModel.hochberg_critical_value(p_values, 0.05)
                sig = stats_table["p-value"] <= critical_value  # Standard BH uses ≤
                stats_table["Sig_Hochber"] = sig
                
                logger.info(f"Results for s={s}, n_dims={n_dims_val}:")
                logger.info(f"  Total motifs: {len(stats_table)}")
                logger.info(f"  Significant (p<0.01): {np.sum(stats_table['p-value'] < 0.01)}")
                logger.info(f"  Significant (Hochberg): {np.sum(sig)}, critical={critical_value:.4e}")
            
            all_stats = pd.concat([all_stats, stats_table], ignore_index=True)
    
    # Save detailed results
    output_file = os.path.join(output_dir, 'table_motifs_lama_iterative.csv')
    all_stats.to_csv(output_file, index=False)
    logger.info(f"\nSaved detailed results to {output_file}")
    
    # Create and save summary table
    if len(all_stats) > 0:
        summary_table = create_summary_table(all_stats)
        summary_file = os.path.join(output_dir, 'summary_motifs_lama_iterative.csv')
        summary_table.to_csv(summary_file, index=False)
        logger.info(f"Saved summary to {summary_file}")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("SUMMARY TABLE")
        logger.info("="*70)
        print(summary_table.to_string(index=False))
        
        # Print comparison with STUMPY (if available)
        stumpy_file = '../results/audio/table_motifs_min_neighbors=1_max_distance=[0.6708203932499369, 0.9486832980505138, 1.643167672515498, 2.1213203435596424]_cutoffs=inf_max_matches=99999_max_motifs=99999.csv'
        if os.path.exists(stumpy_file):
            stumpy_stats = pd.read_csv(stumpy_file)
            logger.info("\n" + "="*70)
            logger.info("COMPARISON: LAMA vs STUMPY")
            logger.info("="*70)
            for s in subsequence_lengths:
                if s in all_stats["s"].values and s in stumpy_stats["s"].values:
                    lama_count = len(all_stats[all_stats["s"] == s])
                    stumpy_count = len(stumpy_stats[stumpy_stats["s"] == s])
                    lama_sig = np.sum(all_stats[all_stats["s"] == s]["p-value"] <= 0.01)
                    stumpy_sig = np.sum(stumpy_stats[stumpy_stats["s"] == s]["p-value"] <= 0.01)
                    
                    logger.info(f"\ns={s}:")
                    logger.info(f"  LAMA:   {lama_count:3d} motifs, {lama_sig:3d} significant ({lama_sig/lama_count*100:.1f}%)")
                    logger.info(f"  STUMPY: {stumpy_count:3d} motifs, {stumpy_sig:3d} significant ({stumpy_sig/stumpy_count*100:.1f}%)")
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
