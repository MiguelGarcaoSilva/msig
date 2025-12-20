#!/usr/bin/env python3
"""
Simple test script to validate audio experiment can run successfully.
This tests Priority 1 tasks: environment validation and basic experiment execution.
"""

import sys
import os
import numpy as np
import librosa
import stumpy
import logging
from msig import Motif, NullModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """Test that all required dependencies are available."""
    logger.info("Testing environment dependencies...")
    
    try:
        import msig
        import numpy
        import scipy
        import pandas
        import matplotlib
        import stumpy
        import librosa
        logger.info("✓ All core dependencies available")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False

def test_data_availability():
    """Test that required data files are available."""
    logger.info("Testing data availability...")
    
    audio_path = 'data/audio/imblue.mp3'
    if not os.path.exists(audio_path):
        logger.error(f"✗ Audio file not found: {audio_path}")
        return False
    
    try:
        y, sr = librosa.load(audio_path)
        logger.info(f"✓ Audio data loaded: {len(y)/sr:.2f}s @ {sr}Hz")
        return True
    except Exception as e:
        logger.error(f"✗ Error loading audio: {e}")
        return False

def test_basic_motif_discovery():
    """Test basic motif discovery and significance testing."""
    logger.info("Testing basic motif discovery...")
    
    try:
        # Load audio and extract MFCCs
        audio_path = 'data/audio/imblue.mp3'
        y, sr = librosa.load(audio_path)
        
        # Simple parameters for quick test
        n_mfcc = 6  # Reduced for faster testing
        hop_length = int(sr * 0.023)
        subsequence_length = int(1.0 * sr / hop_length)  # 1 second
        
        X = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        X = X.astype(np.float64)
        
        logger.info(f"✓ MFCC extracted: {X.shape}")
        
        # Test STUMPY matrix profile (small window for speed)
        mp, mp_indices = stumpy.mstump(X, subsequence_length, normalize=True)
        logger.info(f"✓ Matrix profile computed: mp.shape={mp.shape}")
        
        # Test motif discovery (very relaxed parameters for speed)
        motif_distances, motif_indices, motif_subspaces, _ = stumpy.mmotifs(
            X, mp, mp_indices,
            max_distance=1.0,  # Very permissive
            min_neighbors=1,   # Accept single matches
            max_matches=5,     # Limit for speed
            max_motifs=1,      # Just find one motif
            normalize=True
        )
        
        if len(motif_indices) == 0 or len(motif_indices[0]) == 0:
            logger.warning("⚠ No motifs found (expected with relaxed parameters)")
        else:
            logger.info(f"✓ Found {len(motif_indices)} motif candidates")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error in motif discovery: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_significance_testing():
    """Test MSig significance testing on discovered motifs."""
    logger.info("Testing significance testing...")
    
    try:
        # Create simple test data
        np.random.seed(42)
        data = np.random.randn(3, 100)  # 3 variables, 100 time points
        
        # Create null model
        model = NullModel(data, dtypes=[float, float, float], model="empirical")
        
        # Create a simple motif
        pattern = data[:, 5:15]  # 10-timepoint pattern
        motif = Motif(
            multivar_sequence=list(pattern),
            variables=[0, 1, 2],
            delta_thresholds=[0.3, 0.3, 0.3],
            n_matches=8
        )
        
        # Test probability calculation
        prob = motif.set_pattern_probability(model, vars_indep=True)
        logger.info(f"✓ Pattern probability: {prob:.6e}")
        
        # Test significance calculation
        max_matches = 100 - 10 + 1  # n - motif_length + 1
        pvalue = motif.set_significance(max_matches, 3, idd_correction=False)
        logger.info(f"✓ P-value: {pvalue:.6e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error in significance testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all priority 1 validation tests."""
    logger.info("=" * 60)
    logger.info("PRIORITY 1 REPRODUCIBILITY VALIDATION")
    logger.info("Testing environment, data, and basic experiment execution")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Environment
    results['environment'] = test_environment()
    
    # Test 2: Data availability
    results['data_availability'] = test_data_availability()
    
    # Test 3: Basic motif discovery
    results['motif_discovery'] = test_basic_motif_discovery()
    
    # Test 4: Significance testing
    results['significance_testing'] = test_significance_testing()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:20}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 ALL PRIORITY 1 TESTS PASSED!")
        logger.info("Basic reproducibility validated - experiments can run successfully")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED")
        logger.error("Priority 1 reproducibility issues need to be addressed")
        return 1

if __name__ == "__main__":
    sys.exit(main())