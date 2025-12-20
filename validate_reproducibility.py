#!/usr/bin/env python3
"""
Single validation script for MSig reproducibility.
Checks environment, dependencies, and provides clear guidance.
"""

import sys
import os
import subprocess
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("MSig Reproducibility Validation")
    logger.info("=" * 60)

    # 1. Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    logger.info(f"Python version: {python_version}")

    if sys.version_info < (3, 11):
        logger.error(f"✗ Python {python_version} too old - minimum 3.11 required")
        return False
    elif sys.version_info >= (3, 14):
        logger.warning(f"⚠️ Python {python_version} may have LAMA issues - use 3.11-3.13")
    else:
        logger.info("✓ Python version compatible")

    # 2. Check core dependencies
    try:
        import msig
        import numpy
        import scipy
        logger.info("✓ Core dependencies available")
    except ImportError as e:
        logger.error(f"✗ Missing core dependency: {e}")
        return False

    # 3. Check experiment dependencies
    experiment_deps = ['pandas', 'matplotlib', 'stumpy', 'librosa']
    missing = []
    for dep in experiment_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        logger.warning(f"⚠️ Missing experiment dependencies: {', '.join(missing)}")
        logger.info("📋 Install with: uv pip install pandas matplotlib stumpy librosa")
    else:
        logger.info("✓ All experiment dependencies available")

    # 4. Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, timeout=5)
        logger.info("✓ ffmpeg available")
    except:
        logger.warning("⚠️ ffmpeg not found - audio experiments need it")
        logger.info("📋 Install with: brew install ffmpeg (macOS)")

    # 5. Check data files
    data_checks = [
        ('data/audio/imblue.mp3', 'Audio data'),
        ('data/populationdensity/', 'Population density data'),
        ('data/washingmachine/', 'Washing machine data')
    ]

    missing_data = []
    for file_path, description in data_checks:
        if not os.path.exists(file_path):
            missing_data.append(description)

    if missing_data:
        logger.warning(f"⚠️ Missing data files: {', '.join(missing_data)}")
        logger.info("📋 Data files not in git - download separately")
    else:
        logger.info("✓ All data files available")

    # 6. Test core functionality
    try:
        from msig import Motif, NullModel
        data = np.random.randn(2, 50)
        model = NullModel(data, dtypes=[float, float], model="empirical")
        pattern = data[:, 5:10]
        motif = Motif(list(pattern), [0, 1], [0.1, 0.1], n_matches=3)
        prob = motif.set_pattern_probability(model, vars_indep=True)
        logger.info(f"✓ Core MSig functionality works (p={prob:.2e})")
    except Exception as e:
        logger.error(f"✗ Core functionality test failed: {e}")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("🎉 Reproducibility Validation Complete!")
    logger.info("=" * 60)
    logger.info("✅ Environment is ready for experiments")
    logger.info("📋 Follow INSTALLATION.md to run experiments")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)