#!/usr/bin/env python3
"""
Aggregate threshold JSON files across folds for each epoch.
Creates 4 versions: mean, median, min, max across folds.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_threshold_files(base_dir: Path, epoch: int, num_folds: int = 5) -> List[Dict]:
    """Load threshold JSON files for all folds of a given epoch."""
    threshold_data = []

    for fold in range(num_folds):
        file_path = base_dir / f"refined_thresholds_categorywise_fold{fold}_epoch{epoch}.json"

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)
            threshold_data.append(data)

    if len(threshold_data) == 0:
        raise ValueError(f"No threshold files found for epoch {epoch}")

    print(f"Loaded {len(threshold_data)} threshold files for epoch {epoch}")
    return threshold_data


def aggregate_thresholds(
    threshold_data: List[Dict],
    aggregation: str
) -> Dict:
    """
    Aggregate thresholds across folds using specified method.

    Args:
        threshold_data: List of threshold dictionaries from different folds
        aggregation: One of 'mean', 'median', 'min', 'max'

    Returns:
        Aggregated threshold dictionary
    """
    aggregated = {}

    # Get all category keys from the first fold (category_1, category_2, etc.)
    # Filter to only include category keys, not metadata
    all_keys = threshold_data[0].keys()
    category_keys = [k for k in all_keys if k.startswith("category_")]

    for category in category_keys:
        # Check if this is a dict with thresholds
        if not isinstance(threshold_data[0][category], dict):
            continue
        if "thresholds" not in threshold_data[0][category]:
            continue

        aggregated[category] = {"thresholds": {}}

        # Copy valid_class_ids from the first fold (these should be the same across folds)
        if "valid_class_ids" in threshold_data[0][category]:
            aggregated[category]["valid_class_ids"] = threshold_data[0][category]["valid_class_ids"]

        # Get all threshold keys from the first fold
        threshold_keys = threshold_data[0][category]["thresholds"].keys()

        for threshold_key in threshold_keys:
            # Collect values for this threshold across all folds
            values = []
            for fold_data in threshold_data:
                if category in fold_data and "thresholds" in fold_data[category]:
                    if threshold_key in fold_data[category]["thresholds"]:
                        values.append(fold_data[category]["thresholds"][threshold_key])

            # Aggregate the values
            if len(values) == 0:
                aggregated_value = 0.0
            elif aggregation == "mean":
                aggregated_value = float(np.mean(values))
            elif aggregation == "median":
                aggregated_value = float(np.median(values))
            elif aggregation == "min":
                aggregated_value = float(np.min(values))
            elif aggregation == "max":
                aggregated_value = float(np.max(values))
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            aggregated[category]["thresholds"][threshold_key] = aggregated_value

    return aggregated


def process_epoch(
    base_dir: Path,
    output_dir: Path,
    epoch: int,
    num_folds: int = 5,
    aggregation_methods: List[str] = ["mean", "median", "min", "max"]
) -> None:
    """Process a single epoch and create aggregated threshold files."""
    print(f"\n{'='*70}")
    print(f"Processing Epoch {epoch}")
    print(f"{'='*70}")

    # Load threshold files for all folds
    threshold_data = load_threshold_files(base_dir, epoch, num_folds)

    # Create aggregated files for each method
    for method in aggregation_methods:
        print(f"\nAggregating using method: {method}")
        aggregated = aggregate_thresholds(threshold_data, method)

        # Save aggregated thresholds
        output_file = output_dir / f"refined_thresholds_categorywise_{method}_epoch{epoch}.json"
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"  ✓ Saved: {output_file}")

    print(f"\n✓ Epoch {epoch} completed - 4 aggregated files created")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate threshold JSON files across folds"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing the threshold JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for aggregated files (default: same as base_dir)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8, 9],
        help="Epochs to process (default: 5 6 7 8 9)",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["mean", "median", "min", "max"],
        choices=["mean", "median", "min", "max"],
        help="Aggregation methods to use (default: mean median min max)",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir

    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("THRESHOLD AGGREGATION ACROSS FOLDS")
    print("="*70)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs to process: {args.epochs}")
    print(f"Number of folds: {args.num_folds}")
    print(f"Aggregation methods: {args.methods}")

    # Process each epoch
    for epoch in args.epochs:
        try:
            process_epoch(base_dir, output_dir, epoch, args.num_folds, args.methods)
        except Exception as e:
            print(f"\n✗ Error processing epoch {epoch}: {e}")
            continue

    print("\n" + "="*70)
    print("ALL EPOCHS PROCESSED SUCCESSFULLY!")
    print("="*70)
    print(f"\nAggregated threshold files saved to: {output_dir}")
    print(f"Total files created: {len(args.epochs) * len(args.methods)}")
    print("="*70)


if __name__ == "__main__":
    main()
