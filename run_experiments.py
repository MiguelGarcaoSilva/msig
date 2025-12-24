#!/usr/bin/env python3
"""
Orchestrate all MSig experiments across datasets and discovery methods.

This script provides a unified interface to run all experiments,
track their status, and generate comparison reports.

Usage:
    python run_experiments.py --all                    # Run all experiments
    python run_experiments.py --dataset audio          # Run all methods on audio
    python run_experiments.py --method stumpy          # Run STUMPY on all datasets
    python run_experiments.py --dataset audio --method stumpy  # Specific combination
    python run_experiments.py --dry-run                # Show what would run
    python run_experiments.py --skip-momenti           # Skip MOMENTI (macOS users)
"""

import argparse
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Experiment configuration
DATASETS = ["audio", "populationdensity", "washingmachine"]
METHODS = ["stumpy", "lama", "momenti"]

EXPERIMENT_SCRIPTS = {
    ("audio", "stumpy"): "experiments/audio/run_stumpy.py",
    ("audio", "lama"): "experiments/audio/run_lama.py",
    ("audio", "momenti"): "experiments/audio/run_momenti.py",
    ("populationdensity", "stumpy"): "experiments/populationdensity/run_stumpy.py",
    ("populationdensity", "lama"): "experiments/populationdensity/run_lama.py",
    ("populationdensity", "momenti"): "experiments/populationdensity/run_momenti.py",
    ("washingmachine", "stumpy"): "experiments/washingmachine/run_stumpy.py",
    ("washingmachine", "lama"): "experiments/washingmachine/run_lama.py",
    ("washingmachine", "momenti"): "experiments/washingmachine/run_momenti.py",
}

# Estimated runtimes (in minutes) - rough estimates
ESTIMATED_RUNTIME = {
    ("audio", "stumpy"): 15,
    ("audio", "lama"): 20,
    ("audio", "momenti"): 25,
    ("populationdensity", "stumpy"): 10,
    ("populationdensity", "lama"): 12,
    ("populationdensity", "momenti"): 15,
    ("washingmachine", "stumpy"): 12,
    ("washingmachine", "lama"): 15,
    ("washingmachine", "momenti"): 18,
}


class ExperimentRunner:
    """Orchestrate MSig experiments."""

    def __init__(
        self,
        dry_run: bool = False,
        skip_momenti: bool = False,
        verbose: bool = False
    ):
        self.dry_run = dry_run
        self.skip_momenti = skip_momenti
        self.verbose = verbose
        self.results = []
        self.start_time = datetime.now()

    def check_prerequisites(self) -> bool:
        """Check that dependencies are installed."""
        logger.info("Checking prerequisites...")

        # Check Python version
        if sys.version_info < (3, 11):
            logger.error(f"Python 3.11+ required, got {sys.version_info.major}.{sys.version_info.minor}")
            return False

        # Check core dependencies
        try:
            import msig
            import numpy
            import scipy
        except ImportError as e:
            logger.error(f"Missing core dependency: {e}")
            logger.info("Run: pip install -e .")
            return False

        # Check experiment dependencies
        missing = []
        for dep in ["pandas", "matplotlib", "stumpy", "librosa"]:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        if missing:
            logger.error(f"Missing experiment dependencies: {', '.join(missing)}")
            logger.info("Run: pip install 'msig[experiments]'")
            return False

        # Check MOMENTI if not skipping
        if not self.skip_momenti:
            try:
                from MOMENTI import MOMENTI
                logger.info("✓ MOMENTI available")
            except ImportError:
                logger.warning("MOMENTI not installed - will skip MOMENTI experiments")
                logger.info("Install with: pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs")
                self.skip_momenti = True

        logger.info("✓ Prerequisites check passed")
        return True

    def get_experiments_to_run(
        self,
        datasets: Optional[List[str]] = None,
        methods: Optional[List[str]] = None
    ) -> List[Tuple[str, str, str]]:
        """Get list of (dataset, method, script) tuples to run."""
        experiments = []

        target_datasets = datasets or DATASETS
        target_methods = methods or METHODS

        for dataset in target_datasets:
            for method in target_methods:
                # Skip MOMENTI if requested
                if method == "momenti" and self.skip_momenti:
                    logger.info(f"Skipping {dataset}/{method} (MOMENTI disabled)")
                    continue

                key = (dataset, method)
                if key in EXPERIMENT_SCRIPTS:
                    script = EXPERIMENT_SCRIPTS[key]
                    experiments.append((dataset, method, script))
                else:
                    logger.warning(f"No script found for {dataset}/{method}")

        return experiments

    def run_experiment(
        self,
        dataset: str,
        method: str,
        script: str
    ) -> Tuple[bool, float, str]:
        """Run a single experiment script."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {dataset}/{method}")
        logger.info(f"Script: {script}")

        # Estimate runtime
        estimated = ESTIMATED_RUNTIME.get((dataset, method), "unknown")
        if isinstance(estimated, int):
            logger.info(f"Estimated runtime: ~{estimated} minutes")

        if self.dry_run:
            logger.info("[DRY RUN] Would execute script")
            return True, 0.0, "dry-run"

        # Check if script exists
        script_path = Path(script)
        if not script_path.exists():
            logger.error(f"Script not found: {script}")
            return False, 0.0, "script-not-found"

        # Run the experiment
        start = datetime.now()
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=not self.verbose,
                text=True,
                timeout=3600 * 2  # 2 hour timeout
            )

            duration = (datetime.now() - start).total_seconds() / 60  # minutes

            if result.returncode == 0:
                logger.info(f"✓ Success! Duration: {duration:.1f} minutes")
                return True, duration, "success"
            else:
                logger.error(f"✗ Failed with exit code {result.returncode}")
                if not self.verbose and result.stderr:
                    logger.error(f"Error output:\n{result.stderr}")
                return False, duration, "failed"

        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start).total_seconds() / 60
            logger.error(f"✗ Timeout after {duration:.1f} minutes")
            return False, duration, "timeout"

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() / 60
            logger.error(f"✗ Exception: {e}")
            return False, duration, str(e)

    def run_all(
        self,
        datasets: Optional[List[str]] = None,
        methods: Optional[List[str]] = None
    ):
        """Run selected experiments and track results."""
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed - aborting")
            sys.exit(1)

        experiments = self.get_experiments_to_run(datasets, methods)

        if not experiments:
            logger.warning("No experiments to run")
            return

        total_estimated = sum(
            ESTIMATED_RUNTIME.get((d, m), 0)
            for d, m, _ in experiments
        )

        logger.info(f"\n{'='*70}")
        logger.info(f"Planning to run {len(experiments)} experiments")
        logger.info(f"Estimated total time: ~{total_estimated} minutes ({total_estimated/60:.1f} hours)")
        logger.info(f"{'='*70}\n")

        if self.dry_run:
            logger.info("DRY RUN MODE - showing what would run:\n")
            for i, (dataset, method, script) in enumerate(experiments, 1):
                est = ESTIMATED_RUNTIME.get((dataset, method), "?")
                logger.info(f"{i}. {dataset}/{method} (~{est} min) - {script}")
            logger.info(f"\nTotal: {len(experiments)} experiments")
            return

        # Run experiments
        for i, (dataset, method, script) in enumerate(experiments, 1):
            logger.info(f"\n[{i}/{len(experiments)}] Starting {dataset}/{method}")

            success, duration, status = self.run_experiment(dataset, method, script)

            self.results.append({
                "dataset": dataset,
                "method": method,
                "script": script,
                "success": success,
                "duration_minutes": duration,
                "status": status
            })

        # Summary
        self._print_summary()
        self._save_run_log()

    def _print_summary(self):
        """Print summary of experiment runs."""
        total_duration = (datetime.now() - self.start_time).total_seconds() / 60

        logger.info(f"\n{'='*70}")
        logger.info("EXPERIMENT RUN SUMMARY")
        logger.info(f"{'='*70}")

        successful = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - successful

        logger.info(f"Total experiments: {len(self.results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total time: {total_duration:.1f} minutes")

        if failed > 0:
            logger.info("\nFailed experiments:")
            for r in self.results:
                if not r["success"]:
                    logger.info(f"  - {r['dataset']}/{r['method']}: {r['status']}")

        logger.info(f"\n{'='*70}")

    def _save_run_log(self):
        """Save experiment run log to file."""
        log_dir = Path("results/experiment_runs")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"run_{timestamp}.json"

        log_data = {
            "timestamp": self.start_time.isoformat(),
            "platform": platform.system(),
            "python_version": sys.version,
            "total_duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "experiments": self.results
        }

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"\nRun log saved to: {log_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run MSig experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          # Run all experiments
  %(prog)s --dataset audio                # Run all methods on audio
  %(prog)s --method stumpy                # Run STUMPY on all datasets
  %(prog)s --dataset audio --method lama  # Run specific combination
  %(prog)s --dry-run                      # Show what would run
  %(prog)s --skip-momenti                 # Skip MOMENTI experiments
        """
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments (default if no filters specified)"
    )

    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        help="Run experiments for specific dataset"
    )

    parser.add_argument(
        "--method",
        choices=METHODS,
        help="Run specific discovery method"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing"
    )

    parser.add_argument(
        "--skip-momenti",
        action="store_true",
        help="Skip MOMENTI experiments (useful on macOS)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output from experiment scripts"
    )

    args = parser.parse_args()

    # Determine which experiments to run
    datasets = [args.dataset] if args.dataset else None
    methods = [args.method] if args.method else None

    # Auto-skip MOMENTI on macOS unless explicitly requested
    skip_momenti = args.skip_momenti or (
        platform.system() == "Darwin" and not args.method == "momenti"
    )

    # Create runner and execute
    runner = ExperimentRunner(
        dry_run=args.dry_run,
        skip_momenti=skip_momenti,
        verbose=args.verbose
    )

    try:
        runner.run_all(datasets=datasets, methods=methods)
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
