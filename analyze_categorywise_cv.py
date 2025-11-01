#!/usr/bin/env python3
"""
Analyze categorywise cross-validation scores across (model, epoch) axes.
Considers epochs 5, 6, 7 only.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

# Base directory
BASE_DIR = Path("/data/sahmed9/E25/multi_epoch_checkpoints")

# Models to analyze
MODELS = ["deberta-v3-small", "deberta-v3-base", "deberta-v3-large"]

# Epochs to consider
EPOCHS = [5, 6, 7]

# Categories
CATEGORIES = ["cat_1", "cat_2"]


def collect_scores():
    """
    Collect all scores from fixed.json files.
    Returns a nested dict: {model: {epoch: {fold: {category: score}}}}
    """
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model in MODELS:
        model_dir = BASE_DIR / model

        if not model_dir.exists():
            print(f"Warning: {model_dir} does not exist")
            continue

        # Find all fixed.json files recursively under this model directory
        for json_file in model_dir.rglob("*_fixed.json"):
            filename = json_file.name

            # Parse filename: refined_thresholds_categorywise_fold{X}_epoch{Y}_fixed.json
            parts = filename.split("_")

            # Extract fold and epoch
            fold_part = [p for p in parts if p.startswith("fold")]
            epoch_part = [p for p in parts if p.startswith("epoch")]

            if not fold_part or not epoch_part:
                continue

            fold = int(fold_part[0].replace("fold", ""))
            epoch = int(epoch_part[0].replace("epoch", ""))

            # Only consider epochs 5, 6, 7
            if epoch not in EPOCHS:
                continue

            # Load JSON and extract scores
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Extract category scores from final_metrics
                if "final_metrics" in data:
                    for cat in CATEGORIES:
                        if cat in data["final_metrics"]:
                            scores[model][epoch][fold][cat] = data["final_metrics"][cat]

            except Exception as e:
                print(f"Error reading {json_file}: {e}")

    return scores


def compute_cv_scores(scores):
    """
    Compute mean and std across folds for each (model, epoch, category) combination.
    Returns: {model: {epoch: {category: {"mean": X, "std": Y, "folds": [...]}}}}
    """
    cv_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model in scores:
        for epoch in scores[model]:
            for cat in CATEGORIES:
                # Collect scores across all folds for this (model, epoch, category)
                fold_scores = []
                for fold in scores[model][epoch]:
                    if cat in scores[model][epoch][fold]:
                        fold_scores.append(scores[model][epoch][fold][cat])

                if fold_scores:
                    cv_scores[model][epoch][cat] = {
                        "mean": np.mean(fold_scores),
                        "std": np.std(fold_scores),
                        "folds": fold_scores,
                        "n_folds": len(fold_scores)
                    }

    return cv_scores


def create_summary_tables(cv_scores):
    """
    Create summary tables for each category showing (model x epoch) grid.
    """
    results = {}

    for cat in CATEGORIES:
        # Create a DataFrame for this category
        rows = []

        for model in MODELS:
            for epoch in EPOCHS:
                if model in cv_scores and epoch in cv_scores[model] and cat in cv_scores[model][epoch]:
                    data = cv_scores[model][epoch][cat]
                    rows.append({
                        "Model": model,
                        "Epoch": epoch,
                        "Mean_Score": data["mean"],
                        "Std_Score": data["std"],
                        "N_Folds": data["n_folds"]
                    })

        if rows:
            df = pd.DataFrame(rows)
            results[cat] = df

    return results


def create_pivot_tables(summary_tables):
    """
    Create pivot tables with models as rows and epochs as columns.
    """
    pivot_tables = {}

    for cat, df in summary_tables.items():
        if df.empty:
            continue

        # Create pivot table for mean scores
        pivot_mean = df.pivot(index="Model", columns="Epoch", values="Mean_Score")

        # Create pivot table for std scores
        pivot_std = df.pivot(index="Model", columns="Epoch", values="Std_Score")

        pivot_tables[cat] = {
            "mean": pivot_mean,
            "std": pivot_std
        }

    return pivot_tables


def print_results(summary_tables, pivot_tables):
    """
    Print all results in a readable format.
    """
    print("=" * 80)
    print("CATEGORYWISE CROSS-VALIDATION SCORES")
    print("Axes: (Model, Epoch)")
    print("Epochs: 5, 6, 7")
    print("=" * 80)
    print()

    for cat in CATEGORIES:
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {cat.upper()}")
        print(f"{'=' * 80}\n")

        if cat not in summary_tables or summary_tables[cat].empty:
            print(f"No data found for {cat}")
            continue

        # Print detailed table
        print("Detailed Results:")
        print("-" * 80)
        print(summary_tables[cat].to_string(index=False))
        print()

        # Print pivot tables
        if cat in pivot_tables:
            print("\nMean Scores (Model x Epoch):")
            print("-" * 80)
            print(pivot_tables[cat]["mean"].to_string(float_format=lambda x: f"{x:.4f}"))
            print()

            print("\nStd Scores (Model x Epoch):")
            print("-" * 80)
            print(pivot_tables[cat]["std"].to_string(float_format=lambda x: f"{x:.4f}"))
            print()

            # Print combined format (mean ± std)
            print("\nCombined (Mean ± Std):")
            print("-" * 80)
            pivot_mean = pivot_tables[cat]["mean"]
            pivot_std = pivot_tables[cat]["std"]

            for model in pivot_mean.index:
                values = []
                for epoch in pivot_mean.columns:
                    mean = pivot_mean.loc[model, epoch]
                    std = pivot_std.loc[model, epoch]
                    if pd.notna(mean) and pd.notna(std):
                        values.append(f"{mean:.4f}±{std:.4f}")
                    else:
                        values.append("N/A")
                print(f"{model:20s} | " + " | ".join(f"{v:15s}" for v in values))
            print()


def main():
    print("Collecting scores from JSON files...")
    scores = collect_scores()

    print("Computing cross-validation statistics...")
    cv_scores = compute_cv_scores(scores)

    print("Creating summary tables...")
    summary_tables = create_summary_tables(cv_scores)

    print("Creating pivot tables...")
    pivot_tables = create_pivot_tables(summary_tables)

    print("\n")
    print_results(summary_tables, pivot_tables)

    # Save to CSV files
    output_dir = Path("/home/sahmed9/codes/E25")
    for cat, df in summary_tables.items():
        if not df.empty:
            output_file = output_dir / f"categorywise_cv_{cat}.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved detailed results to: {output_file}")

    # Save pivot tables
    for cat, pivots in pivot_tables.items():
        output_file = output_dir / f"categorywise_cv_{cat}_pivot_mean.csv"
        pivots["mean"].to_csv(output_file)
        print(f"Saved mean pivot table to: {output_file}")

        output_file = output_dir / f"categorywise_cv_{cat}_pivot_std.csv"
        pivots["std"].to_csv(output_file)
        print(f"Saved std pivot table to: {output_file}")


if __name__ == "__main__":
    main()
