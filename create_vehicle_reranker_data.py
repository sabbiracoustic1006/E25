#!/usr/bin/env python
"""
Create training data for vehicle model re-ranker.

This script takes predictions from Round 1 models and creates a binary
classification dataset to distinguish true positives from false positives.

Approach:
1. Load out-of-fold predictions from Round 1 models
2. For each predicted Kompatibles_Fahrzeug_Modell:
   - Label as 1 (TP) if it matches gold aspect value
   - Label as 0 (FP) if it doesn't match gold
3. Extract features: predicted span + surrounding context
4. Train binary classifier to filter false positives
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from utils import convert_tagged_to_aspect, stratified_kfold_split


def normalize_aspect_value(value: str) -> str:
    """Normalize aspect value for comparison."""
    return value.lower().strip().replace(" ", "")


def load_predictions(pred_file: Path) -> dict:
    """Load predictions from JSON file."""
    with open(pred_file) as f:
        predictions = json.load(f)
    return predictions


def create_reranker_dataset(
    data_path: str,
    predictions_dir: Path,
    num_folds: int = 5,
    seed: int = 42,
):
    """
    Create re-ranker training dataset from out-of-fold predictions.

    Args:
        data_path: Path to original training data
        predictions_dir: Directory containing prediction files (e.g., round_1/deberta-v3-base/predictions/)
        num_folds: Number of folds
        seed: Random seed for fold splits
    """
    print("Loading data...")
    df = convert_tagged_to_aspect(data_path)
    df_split = stratified_kfold_split(df, n_splits=num_folds, random_state=seed)

    # Collect all examples
    reranker_data = []

    for fold in range(num_folds):
        print(f"\nProcessing fold {fold}...")

        # Load predictions for this fold
        pred_file = predictions_dir / f"fold{fold}_predictions.json"
        if not pred_file.exists():
            print(f"  Warning: Prediction file not found: {pred_file}")
            continue

        predictions = load_predictions(pred_file)

        # Get validation data for this fold
        valid_df = df_split[df_split["fold"] == fold]

        # Group by record to get gold vehicle models
        for record_num in valid_df["Record Number"].unique():
            record_rows = valid_df[valid_df["Record Number"] == record_num]

            title = record_rows.iloc[0]["Title"]
            category = record_rows.iloc[0]["Category"]

            # Get gold vehicle models for this record
            gold_vehicles = []
            for _, row in record_rows.iterrows():
                if row["Aspect Name"] == "Kompatibles_Fahrzeug_Modell":
                    gold_vehicles.append(row["Aspect Value"])

            # Normalize gold for comparison
            gold_normalized = {normalize_aspect_value(v) for v in gold_vehicles}

            # Get predictions for this record
            record_preds = predictions.get(record_num, {})
            pred_vehicles = record_preds.get("Kompatibles_Fahrzeug_Modell", [])

            # Create training examples for each prediction
            for pred_value in pred_vehicles:
                pred_normalized = normalize_aspect_value(pred_value)

                # Check if this is a true positive or false positive
                is_true_positive = pred_normalized in gold_normalized

                # Find span in title for context extraction
                span_start = title.find(pred_value)
                if span_start == -1:
                    # Try case-insensitive search
                    title_lower = title.lower()
                    pred_lower = pred_value.lower()
                    span_start = title_lower.find(pred_lower)

                if span_start == -1:
                    # Can't find span, skip
                    continue

                span_end = span_start + len(pred_value)

                # Extract context (30 chars before and after)
                context_start = max(0, span_start - 30)
                context_end = min(len(title), span_end + 30)

                left_context = title[context_start:span_start]
                right_context = title[span_end:context_end]

                reranker_data.append({
                    "record_num": record_num,
                    "fold": fold,
                    "title": title,
                    "category": category,
                    "predicted_value": pred_value,
                    "left_context": left_context,
                    "right_context": right_context,
                    "span_start": span_start,
                    "span_end": span_end,
                    "is_correct": 1 if is_true_positive else 0,
                    "gold_vehicles": gold_vehicles,
                })

        print(f"  Fold {fold}: {len([x for x in reranker_data if x['fold'] == fold])} examples")

    # Convert to DataFrame
    df_reranker = pd.DataFrame(reranker_data)

    # Print statistics
    print(f"\n{'='*80}")
    print("RERANKER DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total examples: {len(df_reranker)}")
    print(f"True Positives (label=1): {(df_reranker['is_correct'] == 1).sum()}")
    print(f"False Positives (label=0): {(df_reranker['is_correct'] == 0).sum()}")
    print(f"TP ratio: {(df_reranker['is_correct'] == 1).sum() / len(df_reranker):.3f}")
    print(f"\nExamples per fold:")
    for fold in range(num_folds):
        fold_df = df_reranker[df_reranker["fold"] == fold]
        tp = (fold_df["is_correct"] == 1).sum()
        fp = (fold_df["is_correct"] == 0).sum()
        print(f"  Fold {fold}: {len(fold_df)} ({tp} TP, {fp} FP)")
    print(f"{'='*80}\n")

    return df_reranker


def main():
    parser = argparse.ArgumentParser(description="Create vehicle model re-ranker dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/Tagged_Titles_Train.tsv",
        help="Path to original training data",
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        required=True,
        help="Directory containing prediction JSON files (e.g., round_1/deberta-v3-base/predictions/)",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("reranker_training_data.tsv"),
        help="Output file for re-ranker training data",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Creating Vehicle Model Re-ranker Dataset")
    print(f"{'='*80}")
    print(f"Data path: {args.data_path}")
    print(f"Predictions dir: {args.predictions_dir}")
    print(f"Output file: {args.output_file}")
    print(f"{'='*80}\n")

    # Create dataset
    df_reranker = create_reranker_dataset(
        args.data_path,
        args.predictions_dir,
        args.num_folds,
        args.seed,
    )

    # Save to file
    print(f"Saving to {args.output_file}...")
    df_reranker.to_csv(args.output_file, sep="\t", index=False)

    print(f"\nDataset created successfully!")
    print(f"Saved to: {args.output_file}")

    # Show some examples
    print(f"\n{'='*80}")
    print("SAMPLE EXAMPLES")
    print(f"{'='*80}\n")

    print("TRUE POSITIVES (first 3):")
    tp_examples = df_reranker[df_reranker["is_correct"] == 1].head(3)
    for i, row in tp_examples.iterrows():
        print(f"\nExample {i+1}:")
        print(f"  Title: {row['title'][:80]}...")
        print(f"  Predicted: '{row['predicted_value']}'")
        print(f"  Context: ...{row['left_context']}[{row['predicted_value']}]{row['right_context']}...")
        print(f"  Label: CORRECT (1)")

    print(f"\n{'-'*80}\n")
    print("FALSE POSITIVES (first 3):")
    fp_examples = df_reranker[df_reranker["is_correct"] == 0].head(3)
    for i, row in fp_examples.iterrows():
        print(f"\nExample {i+1}:")
        print(f"  Title: {row['title'][:80]}...")
        print(f"  Predicted: '{row['predicted_value']}'")
        print(f"  Gold: {row['gold_vehicles']}")
        print(f"  Context: ...{row['left_context']}[{row['predicted_value']}]{row['right_context']}...")
        print(f"  Label: WRONG (0)")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
