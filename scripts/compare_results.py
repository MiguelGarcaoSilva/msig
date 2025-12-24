#!/usr/bin/env python3
"""
Compare MSig experiment results across datasets and methods.

This script generates comprehensive comparison reports from experiment results,
including statistical summaries, cross-method comparisons, and visualizations.

Usage:
    python scripts/compare_results.py                    # Generate full report
    python scripts/compare_results.py --dataset audio    # Compare methods on audio
    python scripts/compare_results.py --method stumpy    # Compare STUMPY across datasets
    python scripts/compare_results.py --output report.md # Custom output file
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsComparator:
    """Compare and analyze experiment results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.datasets = ["audio", "populationdensity", "washingmachine"]
        self.methods = ["stumpy", "lama", "momenti"]
        self.results = {}

    def load_all_results(self) -> Dict[tuple, pd.DataFrame]:
        """
        Load all available experiment results.

        Returns
        -------
        dict
            Dictionary mapping (dataset, method) to DataFrame of results.
        """
        logger.info("Loading experiment results...")

        for dataset in self.datasets:
            for method in self.methods:
                # Try different result file patterns
                patterns = [
                    f"{dataset}/{method}/summary_motifs_{method}.csv",
                    f"{dataset}/{method}_*/summary_motifs_{method}_*.csv",
                    f"{dataset}/{method}/summary_*.csv",
                ]

                for pattern in patterns:
                    files = list(self.results_dir.glob(pattern))
                    if files:
                        # Use the most recent file if multiple exist
                        result_file = max(files, key=lambda p: p.stat().st_mtime)
                        try:
                            df = pd.read_csv(result_file)
                            self.results[(dataset, method)] = df
                            logger.info(f"✓ Loaded {dataset}/{method}: {len(df)} rows")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load {result_file}: {e}")

        logger.info(f"Loaded {len(self.results)} result files")
        return self.results

    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics across all experiments.

        Returns
        -------
        pd.DataFrame
            Summary statistics for each (dataset, method) combination.
        """
        summaries = []

        for (dataset, method), df in self.results.items():
            if df.empty:
                continue

            summary = {
                "dataset": dataset,
                "method": method,
                "total_motifs": len(df),
            }

            # Count significant motifs
            if "significant" in df.columns:
                summary["significant_motifs"] = df["significant"].sum()
                summary["pct_significant"] = (
                    100 * df["significant"].sum() / len(df) if len(df) > 0 else 0
                )

            # P-value statistics
            if "pvalue" in df.columns:
                pvalues = df["pvalue"].dropna()
                if len(pvalues) > 0:
                    summary["mean_pvalue"] = pvalues.mean()
                    summary["median_pvalue"] = pvalues.median()
                    summary["min_pvalue"] = pvalues.min()

            # Pattern probability statistics
            if "pattern_probability" in df.columns or "P" in df.columns:
                prob_col = "pattern_probability" if "pattern_probability" in df.columns else "P"
                probs = df[prob_col].dropna()
                if len(probs) > 0:
                    summary["mean_prob"] = probs.mean()
                    summary["median_prob"] = probs.median()

            # Motif length statistics
            if "s" in df.columns:  # s = subsequence length
                summary["mean_length"] = df["s"].mean()
                summary["median_length"] = df["s"].median()

            # Dimensionality statistics
            if "k" in df.columns:  # k = number of variables
                summary["mean_dims"] = df["k"].mean()
                summary["median_dims"] = df["k"].median()

            summaries.append(summary)

        return pd.DataFrame(summaries)

    def compare_methods(self, dataset: str) -> Optional[pd.DataFrame]:
        """
        Compare all methods for a specific dataset.

        Parameters
        ----------
        dataset : str
            Dataset name (e.g., 'audio', 'populationdensity').

        Returns
        -------
        pd.DataFrame or None
            Comparison DataFrame, or None if insufficient data.
        """
        logger.info(f"Comparing methods for {dataset}...")

        method_results = {}
        for method in self.methods:
            if (dataset, method) in self.results:
                method_results[method] = self.results[(dataset, method)]

        if len(method_results) < 2:
            logger.warning(f"Insufficient methods for {dataset} comparison")
            return None

        comparison = []
        for method, df in method_results.items():
            if "significant" in df.columns:
                n_sig = df["significant"].sum()
                pct_sig = 100 * n_sig / len(df) if len(df) > 0 else 0
            else:
                n_sig = np.nan
                pct_sig = np.nan

            comparison.append({
                "method": method,
                "total_motifs": len(df),
                "significant_motifs": n_sig,
                "pct_significant": pct_sig,
            })

        return pd.DataFrame(comparison)

    def compare_datasets(self, method: str) -> Optional[pd.DataFrame]:
        """
        Compare all datasets for a specific method.

        Parameters
        ----------
        method : str
            Method name (e.g., 'stumpy', 'lama', 'momenti').

        Returns
        -------
        pd.DataFrame or None
            Comparison DataFrame, or None if insufficient data.
        """
        logger.info(f"Comparing datasets for {method}...")

        dataset_results = {}
        for dataset in self.datasets:
            if (dataset, method) in self.results:
                dataset_results[dataset] = self.results[(dataset, method)]

        if len(dataset_results) < 2:
            logger.warning(f"Insufficient datasets for {method} comparison")
            return None

        comparison = []
        for dataset, df in dataset_results.items():
            if "significant" in df.columns:
                n_sig = df["significant"].sum()
                pct_sig = 100 * n_sig / len(df) if len(df) > 0 else 0
            else:
                n_sig = np.nan
                pct_sig = np.nan

            comparison.append({
                "dataset": dataset,
                "total_motifs": len(df),
                "significant_motifs": n_sig,
                "pct_significant": pct_sig,
            })

        return pd.DataFrame(comparison)

    def generate_markdown_report(self, output_file: str = "RESULTS_COMPARISON.md"):
        """
        Generate a comprehensive markdown report.

        Parameters
        ----------
        output_file : str
            Path to output markdown file.
        """
        logger.info(f"Generating markdown report: {output_file}")

        with open(output_file, "w") as f:
            # Header
            f.write("# MSig Experiment Results Comparison\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Overall summary
            f.write("## Overall Summary\n\n")
            summary_df = self.generate_summary_statistics()
            if not summary_df.empty:
                f.write(summary_df.to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("*No results available*\n\n")

            # Method comparisons by dataset
            f.write("## Method Comparisons by Dataset\n\n")
            for dataset in self.datasets:
                comparison = self.compare_methods(dataset)
                if comparison is not None and not comparison.empty:
                    f.write(f"### {dataset.title()}\n\n")
                    f.write(comparison.to_markdown(index=False))
                    f.write("\n\n")

            # Dataset comparisons by method
            f.write("## Dataset Comparisons by Method\n\n")
            for method in self.methods:
                comparison = self.compare_datasets(method)
                if comparison is not None and not comparison.empty:
                    f.write(f"### {method.upper()}\n\n")
                    f.write(comparison.to_markdown(index=False))
                    f.write("\n\n")

            # Detailed statistics
            f.write("## Detailed Statistics\n\n")
            for (dataset, method), df in sorted(self.results.items()):
                f.write(f"### {dataset.title()} - {method.upper()}\n\n")

                if df.empty:
                    f.write("*No results*\n\n")
                    continue

                # Basic stats
                f.write(f"- **Total motifs**: {len(df)}\n")

                if "significant" in df.columns:
                    n_sig = df["significant"].sum()
                    pct = 100 * n_sig / len(df)
                    f.write(f"- **Significant motifs**: {n_sig} ({pct:.1f}%)\n")

                if "pvalue" in df.columns:
                    pvals = df["pvalue"].dropna()
                    if len(pvals) > 0:
                        f.write(f"- **P-value range**: [{pvals.min():.2e}, {pvals.max():.2e}]\n")
                        f.write(f"- **Median p-value**: {pvals.median():.2e}\n")

                if "s" in df.columns:
                    f.write(f"- **Motif length range**: [{df['s'].min()}, {df['s'].max()}]\n")

                if "k" in df.columns:
                    f.write(f"- **Dimensionality range**: [{df['k'].min()}, {df['k'].max()}]\n")

                f.write("\n")

            # Footer
            f.write("---\n\n")
            f.write("*Generated by MSig results comparison script*\n")

        logger.info(f"✓ Report saved to {output_file}")

    def generate_csv_summary(self, output_file: str = "results_summary.csv"):
        """
        Generate CSV summary of all results.

        Parameters
        ----------
        output_file : str
            Path to output CSV file.
        """
        summary_df = self.generate_summary_statistics()
        summary_df.to_csv(output_file, index=False)
        logger.info(f"✓ CSV summary saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare MSig experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing experiment results (default: results/)"
    )

    parser.add_argument(
        "--output",
        default="RESULTS_COMPARISON.md",
        help="Output markdown file (default: RESULTS_COMPARISON.md)"
    )

    parser.add_argument(
        "--csv",
        default="results_summary.csv",
        help="Output CSV summary file (default: results_summary.csv)"
    )

    parser.add_argument(
        "--dataset",
        choices=["audio", "populationdensity", "washingmachine"],
        help="Focus on specific dataset"
    )

    parser.add_argument(
        "--method",
        choices=["stumpy", "lama", "momenti"],
        help="Focus on specific method"
    )

    args = parser.parse_args()

    # Create comparator
    comparator = ResultsComparator(results_dir=args.results_dir)

    # Load results
    comparator.load_all_results()

    if not comparator.results:
        logger.error("No results found to compare")
        return

    # Generate reports
    if args.dataset:
        # Compare methods for specific dataset
        comparison = comparator.compare_methods(args.dataset)
        if comparison is not None:
            print(f"\nMethod comparison for {args.dataset}:")
            print(comparison.to_string(index=False))
        else:
            logger.warning("No comparison data available")

    elif args.method:
        # Compare datasets for specific method
        comparison = comparator.compare_datasets(args.method)
        if comparison is not None:
            print(f"\nDataset comparison for {args.method}:")
            print(comparison.to_string(index=False))
        else:
            logger.warning("No comparison data available")

    else:
        # Generate full reports
        comparator.generate_markdown_report(args.output)
        comparator.generate_csv_summary(args.csv)

        # Print summary to console
        summary = comparator.generate_summary_statistics()
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(summary.to_string(index=False))
        print("="*70)


if __name__ == "__main__":
    main()
