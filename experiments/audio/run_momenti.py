#!/usr/bin/env python3
"""
MOMENTI-based Case Study for Audio Data (MFCC Features)
Subdimensional motif discovery using LSH-based approach
"""

import sys
import os
import numpy as np
import pandas as pd
import librosa
import math
import logging
from typing import Tuple

# Add parent directory to path for msig import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from msig import Motif, NullModel

# Add MOMENTI to path
script_dir = os.path.dirname(os.path.abspath(__file__))
momenti_path = os.path.join(script_dir, "../../MOMENTI-motifs/source")
sys.path.insert(0, momenti_path)
from MOMENTI import MOMENTI

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_audio_data(audio_path: str) -> Tuple[np.ndarray, pd.DataFrame, int | float]:
    """
    Load audio file and extract MFCC features.
    
    Returns:
        X: MFCC features transposed to (n_frames, n_mfcc) for MOMENTI
        data_df: DataFrame with MFCC coefficients and timestamps
        sr: Sample rate
    """
    logger.info(f"Loading audio from {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path)
    
    # MFCC parameters
    n_mfcc = 12
    n_fft = int(sr * 0.046)
    hop_length = int(sr * 0.023)
    
    # Extract MFCCs
    X = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, 
        hop_length=hop_length, n_fft=n_fft, n_mels=32
    )
    
    # Transpose for MOMENTI: (n_frames, n_mfcc)
    X = X.T.astype(np.float32)
    
    # Create DataFrame
    data_df = pd.DataFrame(X)
    data_df['datetime'] = data_df.index * hop_length / sr
    data_df['datetime'] = pd.to_datetime(data_df['datetime'], unit='s')
    data_df = data_df.set_index('datetime')
    data_df.columns = [f'Coefficient {i}' for i in range(n_mfcc)]
    
    logger.info(f"Extracted MFCCs: X.shape={X.shape}, sr={sr}, duration={X.shape[0] * hop_length / sr:.2f}s")
    
    return X, data_df, sr


def compute_motif_statistics_momenti(
    motifs: list,
    data: np.ndarray,
    m: int,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute statistical significance for MOMENTI-discovered motifs.
    
    MOMENTI returns: [distance, [id, [indices], [dimensions], [dimensional_distances]]]
    """
    stats_table = pd.DataFrame(columns=[
        "ID", "k", "Features", "s", "#Matches", "Indices", 
        "Distance", "P", "p-value"
    ])
    
    n_time, n_vars = data.shape
    
    # Z-normalize if needed
    if normalize:
        data_norm = (data - np.mean(data, axis=0, keepdims=True)) / np.std(data, axis=0, keepdims=True)
    else:
        data_norm = data.copy()
    
    # Transpose back to (n_vars, n_time) for MSig
    data_norm_t = data_norm.T
    
    # Create null model
    dtypes = [float] * n_vars
    model_empirical = NullModel(data_norm_t, dtypes=dtypes, model="empirical")
    
    # Calculate max possible matches
    r = np.ceil(m / 2)
    max_possible_matches = int(np.floor((n_time - m) / r) + 1)
    
    for motif_data in motifs:
        # Robust unpacking: MOMENTI sometimes returns nested tuples/lists like
        # ((distance, info),) or (distance, info) or [[(distance, info)], ...]
        def _find_distance_info(obj):
            """Recursively search for a (distance, info) pair in nested sequences."""
            # Direct pair
            if isinstance(obj, (list, tuple)) and len(obj) == 2:
                a, b = obj[0], obj[1]
                # a numeric-ish and b is info list/tuple
                if (isinstance(a, (int, float, np.floating, np.integer)) or
                        (isinstance(a, (list, tuple, np.ndarray)) and len(a) == 1 and isinstance(a[0], (int, float, np.floating, np.integer)))):
                    return a, b

            # Single-element wrapper
            if isinstance(obj, (list, tuple)) and len(obj) == 1:
                return _find_distance_info(obj[0])

            # Search inside iterable for a candidate pair
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    res = _find_distance_info(item)
                    if res is not None:
                        return res

            return None

        try:
            res = _find_distance_info(motif_data)
            if res is None:
                logger.warning(f"Unexpected motif structure, could not unpack: {type(motif_data)}")
                continue

            distance_raw, motif_info = res
            # distance_raw may be an array-like of length 1
            if isinstance(distance_raw, (list, tuple, np.ndarray)):
                distance_raw = distance_raw[0]

            distance = abs(float(distance_raw))

            if not isinstance(motif_info, (list, tuple)) or len(motif_info) < 2:
                logger.warning(f"Unexpected motif info structure after unpacking: {motif_info}")
                continue

            motif_id = motif_info[0]
            indices = np.array(motif_info[1]) if len(motif_info) > 1 else np.array([])
            dimensions = np.array(motif_info[2]) if len(motif_info) > 2 else np.array([])
            dim_distances = motif_info[3] if len(motif_info) > 3 else []

        except Exception as e:
            logger.warning(f"Error unpacking motif data: {e}")
            continue
        
        if len(indices) == 0:
            continue
        
        # Extract pattern from first occurrence
        pattern_pos = int(indices[0])
        multivar_subsequence = data_norm_t[dimensions, pattern_pos:pattern_pos + m]
        
        # Calculate delta threshold from distance
        max_delta = math.sqrt(distance**2 / m) if distance > 0 else 0.1
        delta_thresholds = [max_delta] * n_vars  # Must be length n_vars, not len(dimensions)
        
        # Compute significance (with error handling for MSig issues)
        try:
            motif_obj = Motif(list(multivar_subsequence), list(dimensions), delta_thresholds, len(indices))
            p_pattern = motif_obj.set_pattern_probability(model_empirical, vars_indep=True)
            p_value = motif_obj.set_significance(max_possible_matches, n_vars, idd_correction=False)
        except Exception as e:
            # If significance computation fails (e.g., sampling issues), use NaN
            logger.warning(f"Could not compute significance for motif {motif_id}: {e}")
            p_pattern = np.nan
            p_value = np.nan
        
        # Store results
        stats_row = {
            "ID": motif_id,
            "k": len(dimensions),
            "Features": ",".join([str(d) for d in dimensions]),
            "s": m,
            "#Matches": len(indices),  # Total occurrences (consistent with LAMA and statistical test)
            "Indices": [int(i) for i in indices],
            "Distance": round(distance, 3),
            "P": p_pattern,
            "p-value": p_value
        }
        
        stats_table = pd.concat([stats_table, pd.DataFrame([stats_row])], ignore_index=True)
    
    return stats_table


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(script_dir, "../data/audio/imblue.mp3")
    results_dir = os.path.abspath(os.path.join(script_dir, "../results/audio/momenti"))
    os.makedirs(results_dir, exist_ok=True)
    
    # Load audio and extract MFCCs
    X, data_df, sr = load_audio_data(audio_path)
    hop_length = int(sr * 0.023)
    
    logger.info(f"Data shape for MOMENTI: {X.shape} (time, dims)")
    
    # Parameters
    normalize = True
    # Use all 4 window sizes to match LAMA and STUMPY
    subsequence_lengths = [int(secs * sr / hop_length) for secs in [0.5, 1.0, 3.0, 5.0]]  # 0.5s, 1s, 3s, 5s
    logger.info(f"Subsequence lengths: {subsequence_lengths} frames")
    logger.info(f"  Corresponding to: 0.5s, 1.0s, 3.0s, 5.0s")
    
    # MOMENTI parameters - OPTIMIZED FOR SPEED
    # L=50 provides good recall with reasonable speed
    # motif_dim_range reduced to (2,6) to focus on lower-dimensional patterns (faster)
    k_motifs = 100  # Number of top motifs to return (maximize discovery)
    L = 50   # LSH repetitions (balanced speed/recall)
    K = 8    # LSH concatenations (optimal precision/speed balance)
    motif_dim_range = (2, 6)  # Focus on 2-6 dimensions (vs full 12) for speed
    
    logger.info(f"\nMOMENTI Parameters (Speed-Optimized):")
    logger.info(f"  k={k_motifs} motifs, L={L}, K={K}")
    logger.info(f"  Motif dimensionality range: {motif_dim_range}")
    logger.info(f"  Note: L=30 for ~1.2 hours/m (vs 2 hours with L=50), ~75-80% recall")
    
    all_stats = pd.DataFrame()
    
    for s in subsequence_lengths:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing motif length s={s} ({m * hop_length / sr:.2f} seconds)")
        logger.info(f"{'=' * 70}")
        
        try:
            # Run MOMENTI
            logger.info("Running MOMENTI...")
            motifs, num_dist, exec_time = MOMENTI(
                time_series=X,
                window=s,
                k=k_motifs,
                motif_dimensionality_range=motif_dim_range,
                L=L,
                K=K,
                failure_probability=0.01
            )
            
            logger.info(f"MOMENTI completed in {exec_time:.2f}s")
            logger.info(f"Distance computations: {num_dist}")
            # MOMENTI returns a list of lists (one inner list per dimensionality).
            # Count total candidates and flatten for processing.
            total_candidates = sum(len(x) if isinstance(x, (list, tuple)) else 1 for x in motifs)
            logger.info(f"Found {total_candidates} motif candidates (across {len(motifs)} dimensionalities)")
            # Flatten motifs into a single list of candidates for downstream processing
            flat_motifs = []
            for el in motifs:
                if isinstance(el, (list, tuple)):
                    flat_motifs.extend(list(el))
                else:
                    flat_motifs.append(el)
            
            if len(flat_motifs) == 0:
                logger.warning(f"No motifs found for s={s}")
                continue
            
            # Compute statistics
            stats = compute_motif_statistics_momenti(flat_motifs, X, s, normalize)
            
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
            
        except Exception as e:
            logger.error(f"Error processing s={s}: {e}")
            continue
    
    # Save detailed results
    detailed_path = os.path.join(results_dir, "table_motifs_momenti.csv")
    all_stats.to_csv(detailed_path, index=False)
    logger.info(f"\nSaved detailed results to {detailed_path}")
    
    # Create summary table only if we have results
    if all_stats.empty:
        logger.warning("No motifs found across all subsequence lengths")
        logger.info("Creating empty summary table")
        summary_table = pd.DataFrame()
    else:
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
                                        ignore_index=True)    # Save summary
    summary_path = os.path.join(results_dir, "summary_motifs_momenti.csv")
    summary_table.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")
    
    # Print summary
    if not summary_table.empty:
        logger.info(f"\n{'=' * 70}")
        logger.info("SUMMARY TABLE - MOMENTI")
        logger.info(f"{'=' * 70}")
        print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
