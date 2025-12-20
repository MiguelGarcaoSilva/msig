#!/usr/bin/env python3
"""
Comprehensive test runner for MSig.
Runs both unit tests and reproducibility validation.
"""

import sys
import subprocess
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_unit_tests():
    """Run unit tests using pytest."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING UNIT TESTS")
    logger.info("="*60)
    
    try:
        # First try to run pytest
        result = subprocess.run(
            ['uv', 'run', 'python', '-m', 'pytest', 'tests/', '-v'],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        success = result.returncode == 0
        
        if success:
            logger.info("✓ Unit tests PASSED")
        else:
            logger.error(f"✗ Unit tests FAILED with return code {result.returncode}")
        
        return success
        
    except subprocess.TimeoutExpired:
        logger.error("✗ Unit tests TIMED OUT after 2 minutes")
        return False
    except Exception as e:
        logger.error(f"✗ Unit tests ERROR: {e}")
        logger.info("📋 Trying to install pytest and run tests...")
        
        # Try to install pytest and run again
        try:
            install_result = subprocess.run(
                ['uv', 'pip', 'install', 'pytest'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if install_result.returncode == 0:
                logger.info("✓ pytest installed successfully")
                
                # Try running tests again
                result = subprocess.run(
                    ['uv', 'run', 'python', '-m', 'pytest', 'tests/', '-v'],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                
                success = result.returncode == 0
                
                if success:
                    logger.info("✓ Unit tests PASSED (after installing pytest)")
                else:
                    logger.error(f"✗ Unit tests FAILED with return code {result.returncode}")
                
                return success
            else:
                logger.error("✗ Failed to install pytest")
                return False
        except Exception as e:
            logger.error(f"✗ Failed to install pytest: {e}")
            logger.info("📋 You can install pytest manually with: uv pip install pytest")
            return False

def run_reproducibility_validation():
    """Run reproducibility validation scripts."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING REPRODUCIBILITY VALIDATION")
    logger.info("="*60)
    
    validation_scripts = [
        ('validate_environment.py', 'Environment Validation'),
        ('validate_all_datasets.py', 'Dataset Validation'),
        ('test_audio_experiment.py', 'Audio Experiment Validation'),
        ('test_simple_stumpy_experiment.py', 'STUMPY Experiment Validation'),
    ]
    
    results = {}
    
    for script_name, description in validation_scripts:
        logger.info(f"\n  Running: {description}")
        
        try:
            result = subprocess.run(
                ['uv', 'run', 'python', script_name],
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout per script
            )
            
            # Print output for this script
            print(f"\n  {description} output:")
            print("  " + "-"*56)
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"  {line}", file=sys.stderr)
            print("  " + "-"*56)
            
            success = result.returncode == 0
            results[description] = success
            
            if success:
                logger.info(f"  ✓ {description} PASSED")
            else:
                logger.error(f"  ✗ {description} FAILED")
                
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ {description} TIMED OUT after 3 minutes")
            results[description] = False
        except Exception as e:
            logger.error(f"  ✗ {description} ERROR: {e}")
            results[description] = False
    
    # Summary of reproducibility validation
    logger.info("\n  Reproducibility Validation Summary:")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"    {test_name:30}: {status}")
    
    all_passed = all(results.values())
    return all_passed

def main():
    """Run comprehensive test suite."""
    logger.info("="*60)
    logger.info("COMPREHENSIVE MSIG TEST SUITE")
    logger.info("Testing unit functionality and reproducibility")
    logger.info("="*60)
    
    # Start timer
    start_time = datetime.now()
    logger.info(f"Started at: {start_time}")
    
    results = {}
    
    # Run unit tests
    results['unit_tests'] = run_unit_tests()
    
    # Run reproducibility validation
    results['reproducibility_validation'] = run_reproducibility_validation()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info("="*60)
    
    for test_category, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_category:25}: {status}")
    
    # End timer
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"\nCompleted at: {end_time}")
    logger.info(f"Total duration: {duration}")
    
    # Final result
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ Unit tests: All core functionality working")
        logger.info("✅ Reproducibility: Environment and experiments validated")
        logger.info("✅ Ready for full experiment execution")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
        
        if not results['unit_tests']:
            logger.error("❌ Unit tests failed - core functionality issues")
        
        if not results['reproducibility_validation']:
            logger.error("❌ Reproducibility validation failed - environment/experiment issues")
        
        logger.error("\n🔧 Troubleshooting:")
        logger.error("   - Check unit test failures for core functionality issues")
        logger.error("   - Check reproducibility validation for environment problems")
        logger.error("   - Run individual validation scripts for detailed diagnostics")
        return 1

if __name__ == "__main__":
    sys.exit(main())