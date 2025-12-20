#!/usr/bin/env python3
"""
Environment validation script for MSig reproducibility.
Tests that all required dependencies are available and functional.
"""

import sys
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_python_version():
    """Test Python version compatibility."""
    logger.info("Testing Python version...")
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    logger.info(f"Python version: {python_version}")
    
    # Check version requirements
    if sys.version_info < (3, 11):
        logger.error(f"✗ Python {python_version} is too old. Minimum required: 3.11")
        return False
    elif sys.version_info >= (3, 14):
        logger.warning(f"⚠️ Python {python_version} may have issues with LAMA dependencies")
        logger.info("✅ STUMPY experiments will work fine")
        logger.info("📋 Consider Python 3.11-3.13 for full LAMA compatibility")
    
    logger.info("✓ Python version compatible")
    return True

def test_core_dependencies():
    """Test that core MSig dependencies are available."""
    logger.info("Testing core dependencies...")
    
    required_packages = [
        'msig',
        'numpy',
        'scipy',
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.debug(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"  ✗ {package}")
    
    if missing_packages:
        logger.error(f"✗ Missing core packages: {', '.join(missing_packages)}")
        return False
    
    logger.info("✓ All core dependencies available")
    return True

def test_experiment_dependencies():
    """Test that experiment dependencies are available."""
    logger.info("Testing experiment dependencies...")
    
    # Test optional experiment packages
    experiment_packages = [
        ('pandas', 'Data manipulation'),
        ('matplotlib', 'Visualization'),
        ('stumpy', 'Matrix profile computation'),
        ('librosa', 'Audio processing'),
    ]
    
    missing_packages = []
    for package, description in experiment_packages:
        try:
            __import__(package)
            logger.debug(f"  ✓ {package} ({description})")
        except ImportError:
            missing_packages.append(f"{package} ({description})")
            logger.warning(f"  ⚠ {package} ({description}) - optional for experiments")
    
    if missing_packages:
        logger.warning(f"⚠ Missing experiment packages: {', '.join(missing_packages)}")
        logger.warning("  Experiments may not run without these packages")
    else:
        logger.info("✓ All experiment dependencies available")
    
    return True  # Don't fail for missing experiment packages

def test_system_tools():
    """Test that required system tools are available."""
    logger.info("Testing system tools...")
    
    # Check for ffmpeg (required for audio experiments)
    ffmpeg_available = False
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✓ ffmpeg available")
            ffmpeg_available = True
        else:
            logger.warning("⚠ ffmpeg not found - audio experiments will fail")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("⚠ ffmpeg not found - audio experiments will fail")
    except Exception as e:
        logger.warning(f"⚠ Error checking ffmpeg: {e}")
    
    if not ffmpeg_available:
        logger.info("📋 Install ffmpeg with:")
        logger.info("   macOS: brew install ffmpeg")
        logger.info("   Linux: sudo apt-get install ffmpeg")
        logger.info("   Windows: Download from https://ffmpeg.org/")
        logger.info("📋 Note: Population density and washing machine experiments don't require ffmpeg")
    
    return True  # Don't fail for missing system tools

def test_msig_functionality():
    """Test basic MSig functionality."""
    logger.info("Testing MSig functionality...")
    
    try:
        from msig import Motif, NullModel
        import numpy as np
        
        # Create test data
        np.random.seed(42)
        data = np.random.randn(2, 50)
        
        # Test NullModel
        model = NullModel(data, dtypes=[float, float], model="empirical")
        
        # Test Motif
        pattern = data[:, 5:10]
        motif = Motif(list(pattern), [0, 1], [0.1, 0.1], n_matches=3)
        
        # Test probability calculation
        prob = motif.set_pattern_probability(model, vars_indep=True)
        
        # Test significance calculation
        pvalue = motif.set_significance(45, 2, idd_correction=False)
        
        logger.info(f"✓ MSig functionality test passed (p={prob:.2e}, pvalue={pvalue:.2e})")
        return True
        
    except Exception as e:
        logger.error(f"✗ MSig functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_files():
    """Test that required data files are available."""
    logger.info("Testing data files...")
    
    import os
    
    data_checks = [
        ('data/audio/imblue.mp3', 'Audio data'),
        ('data/populationdensity/', 'Population density data'),
        ('data/washingmachine/', 'Washing machine data'),
    ]
    
    missing_files = []
    for file_path, description in data_checks:
        if os.path.exists(file_path):
            logger.debug(f"  ✓ {description}")
        else:
            missing_files.append(description)
            logger.error(f"  ✗ {description} not found: {file_path}")
    
    if missing_files:
        logger.error(f"✗ Missing data files: {', '.join(missing_files)}")
        return False
    
    logger.info("✓ All data files available")
    return True

def main():
    """Run comprehensive environment validation."""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT VALIDATION FOR MSIG REPRODUCIBILITY")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Python version
    results['python_version'] = test_python_version()
    
    # Test 2: Core dependencies
    results['core_dependencies'] = test_core_dependencies()
    
    # Test 3: Experiment dependencies
    results['experiment_dependencies'] = test_experiment_dependencies()
    
    # Test 4: System tools
    results['system_tools'] = test_system_tools()
    
    # Test 5: MSig functionality
    results['msig_functionality'] = test_msig_functionality()
    
    # Test 6: Data files
    results['data_files'] = test_data_files()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ENVIRONMENT VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    critical_tests = ['python_version', 'core_dependencies', 'msig_functionality', 'data_files']
    
    for test_name, passed in results.items():
        if test_name in critical_tests:
            status = "✓ PASS" if passed else "✗ FAIL (CRITICAL)"
        else:
            status = "✓ PASS" if passed else "⚠ FAIL (WARNING)"
        logger.info(f"{test_name:20}: {status}")
    
    # Check critical tests
    critical_passed = all(results[test] for test in critical_tests)
    
    if critical_passed:
        logger.info("\n🎉 ENVIRONMENT VALIDATION PASSED!")
        logger.info("All critical components are available for reproducibility")
        
        # Check for warnings
        non_critical_failed = any(not results[test] for test in results if test not in critical_tests)
        if non_critical_failed:
            logger.warning("⚠ Some optional components missing - experiments may have limited functionality")
        
        return 0
    else:
        logger.error("\n❌ ENVIRONMENT VALIDATION FAILED!")
        logger.error("Critical components missing - reproducibility cannot be guaranteed")
        return 1

if __name__ == "__main__":
    sys.exit(main())