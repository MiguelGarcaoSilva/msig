#!/usr/bin/env python3
"""
Master script to run all Priority 1 reproducibility validation tests.
This script validates that the environment is set up correctly, all datasets
are available, and basic experiment execution works.
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

def run_validation_script(script_name, description):
    """Run a validation script and return its success status."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"Script: {script_name}")
    logger.info('='*60)
    
    try:
        result = subprocess.run(
            ['uv', 'run', 'python', script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        success = result.returncode == 0
        
        if success:
            logger.info(f"✓ {description} PASSED")
        else:
            logger.error(f"✗ {description} FAILED with return code {result.returncode}")
        
        return success
        
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} TIMED OUT after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"✗ {description} ERROR: {e}")
        return False

def main():
    """Run all Priority 1 validation tests."""
    logger.info("="*60)
    logger.info("PRIORITY 1 REPRODUCIBILITY VALIDATION")
    logger.info("Validating environment, data, and basic experiment execution")
    logger.info("="*60)
    
    # Start timer
    start_time = datetime.now()
    logger.info(f"Started at: {start_time}")
    
    # Define validation scripts
    validation_scripts = [
        ('validate_environment.py', 'Environment Validation'),
        ('validate_all_datasets.py', 'Dataset Validation'),
        ('test_audio_experiment.py', 'Audio Experiment Validation'),
    ]
    
    results = {}
    
    # Run each validation script
    for script_name, description in validation_scripts:
        results[description] = run_validation_script(script_name, description)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PRIORITY 1 VALIDATION SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:30}: {status}")
    
    # End timer
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"\nCompleted at: {end_time}")
    logger.info(f"Total duration: {duration}")
    
    # Final result
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 ALL PRIORITY 1 VALIDATION TESTS PASSED!")
        logger.info("✅ Environment is properly configured")
        logger.info("✅ All datasets are available and accessible")
        logger.info("✅ Basic experiment execution works")
        logger.info("✅ Reproducibility foundation is established")
        logger.info("\n📋 Next steps:")
        logger.info("   - Run individual experiments (experiments/audio/run_stumpy.py, etc.)")
        logger.info("   - Validate experiment results consistency")
        logger.info("   - Implement Priority 2 tasks for enhanced reproducibility")
        return 0
    else:
        logger.error("\n❌ SOME PRIORITY 1 VALIDATION TESTS FAILED!")
        logger.error("❌ Reproducibility cannot be guaranteed until issues are resolved")
        logger.error("\n🔧 Troubleshooting steps:")
        logger.error("   - Check the error messages above for specific issues")
        logger.error("   - Ensure all dependencies are installed: uv sync")
        logger.error("   - Verify data files are in the correct locations")
        logger.error("   - Check system requirements (Python version, ffmpeg, etc.)")
        return 1

if __name__ == "__main__":
    sys.exit(main())