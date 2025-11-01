#!/usr/bin/env python
"""
Evaluate re-ranker by filtering Round 1 predictions.

This script:
1. Loads Round 1 predictions
2. Applies re-ranker to filter false positives
3. Compares filtered predictions with gold labels
4. Reports precision, recall, F0.2 before and after re-ranking
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import convert_tagged_to_aspect, stratified_kfold_split


def normalize_aspect_value(value: str) -> str:
    """Normalize aspect value for comparison."""
    return value.lower().strip().replace(" ", "")


def load_reranker(model_path: Path, device: str = "cuda"):
    """Load re-ranker model."""
    print(f"Loading re-ranker from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def score_prediction(
    predicted_value: str,
    left_context: str,
    right_context: str,
    model,
    tokenizer,
    device: str = "cuda",
) -> float:
    """Score a single prediction using the re-ranker."""
    # Create input text (same format as training)
    input_text = f"{predicted_value} [SEP] {left_context} {predicted_value} {right_context}"

    # Tokenize
    inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        # Probability of being correct (class 1)
        score = probs[0, 1].item()

    return score


def filter_predictions_with_reranker(
    predictions: dict,
    title: str,
    model,
    tokenizer,
    threshold: float = 0.5,
    device: str = "cuda",
) -> list[str]:
    """Filter predictions using re-ranker."""
    vehicle_predictions = predictions.get("Kompatibles_Fahrzeug_Modell", [])
    filtered = []

    for pred_value in vehicle_predictions:
        # Find span in title
        span_start = title.find(pred_value)
        if span_start == -1:
            # Try case-insensitive
            title_lower = title.lower()
            pred_lower = pred_value.lower()
            span_start = title_lower.find(pred_lower)

        if span_start == -1:
            # Can't find span, skip
            continue

        span_end = span_start + len(pred_value)

        # Extract context
        context_start = max(0, span_start - 30)
        context_end = min(len(title), span_end + 30)
        left_context = title[context_start:span_start]
        right_context = title[span_end:context_end]

        # Score with re-ranker
        score = score_prediction(pred_value, left_context, right_context, model, tokenizer, device)

        # Keep if above threshold
        if score >= threshold:
            filtered.append(pred_value)

    return filtered


def compute_metrics(gold_list: list[list[str]], pred_list: list[list[str]], beta: float = 0.2):
    """Compute precision, recall, F-beta."""
    tp = 0
    fp = 0
    fn = 0

    for gold_vehicles, pred_vehicles in zip(gold_list, pred_list):
        gold_norm = {normalize_aspect_value(v) for v in gold_vehicles}
        pred_norm = {normalize_aspect_value(v) for v in pred_vehicles}

        matches = gold_norm & pred_norm
        tp += len(matches)
        fp += len(pred_norm - gold_norm)
        fn += len(gold_norm - pred_norm)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    else:
        f_beta = 0.0

    return {
        "precision": precision,
        "recall": recall,
        f"f{beta}": f_beta,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate re-ranker on Round 1 predictions")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/Tagged_Titles_Train.tsv",
        help="Training data path",
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=Path("round_1_predictions"),
        help="Directory with Round 1 predictions",
    )
    parser.add_argument(
        "--reranker_path",
        type=Path,
        default=Path("vehicle_reranker_model"),
        help="Path to re-ranker model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Re-ranker threshold (keep predictions with score >= threshold)",
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
    print("Evaluating Re-ranker on Round 1 Predictions")
    print(f"{'='*80}")
    print(f"Data: {args.data_path}")
    print(f"Predictions: {args.predictions_dir}")
    print(f"Re-ranker: {args.reranker_path}")
    print(f"Threshold: {args.threshold}")
    print(f"{'='*80}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load re-ranker
    model, tokenizer = load_reranker(args.reranker_path, device)

    # Load data
    print("Loading data...")
    df = convert_tagged_to_aspect(args.data_path)
    df_split = stratified_kfold_split(df, n_splits=args.num_folds, random_state=args.seed)

    # Collect results across all folds
    all_gold_before = []
    all_pred_before = []
    all_pred_after = []

    # Only evaluate on fold 0 (held-out validation fold)
    # Re-ranker was trained on folds 1-4, so fold 0 is unseen
    eval_fold = 0
    print(f"\nEvaluating on Fold {eval_fold} (held-out validation fold)...")
    print("Re-ranker was trained on folds 1-4, so fold 0 is unseen data\n")

    for fold in [eval_fold]:  # Only process fold 0
        print(f"Processing Fold {fold}...")

        # Load predictions
        pred_file = args.predictions_dir / f"fold{fold}_predictions.json"
        if not pred_file.exists():
            print(f"  Warning: Predictions not found at {pred_file}, skipping...")
            continue

        with open(pred_file, encoding="utf-8") as f:
            predictions = json.load(f)

        # Get validation data
        valid_df = df_split[df_split["fold"] == fold]

        # Process each record
        for record_num in tqdm(valid_df["Record Number"].unique(), desc=f"Fold {fold}"):
            record_rows = valid_df[valid_df["Record Number"] == record_num]

            title = record_rows.iloc[0]["Title"]

            # Get gold vehicles
            gold_vehicles = []
            for _, row in record_rows.iterrows():
                if row["Aspect Name"] == "Kompatibles_Fahrzeug_Modell":
                    gold_vehicles.append(row["Aspect Value"])

            # Get predictions (before re-ranking)
            record_preds = predictions.get(record_num, {})
            pred_vehicles_before = record_preds.get("Kompatibles_Fahrzeug_Modell", [])

            # Apply re-ranker
            pred_vehicles_after = filter_predictions_with_reranker(
                record_preds,
                title,
                model,
                tokenizer,
                args.threshold,
                device,
            )

            all_gold_before.append(gold_vehicles)
            all_pred_before.append(pred_vehicles_before)
            all_pred_after.append(pred_vehicles_after)

    # Compute metrics
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")

    metrics_before = compute_metrics(all_gold_before, all_pred_before, beta=0.2)
    metrics_after = compute_metrics(all_gold_before, all_pred_after, beta=0.2)

    print("BEFORE RE-RANKING (Round 1 baseline):")
    print(f"  Precision: {metrics_before['precision']:.4f}")
    print(f"  Recall:    {metrics_before['recall']:.4f}")
    print(f"  F0.2:      {metrics_before['f0.2']:.4f}")
    print(f"  TP: {metrics_before['tp']}, FP: {metrics_before['fp']}, FN: {metrics_before['fn']}")

    print(f"\nAFTER RE-RANKING (threshold={args.threshold}):")
    print(f"  Precision: {metrics_after['precision']:.4f}")
    print(f"  Recall:    {metrics_after['recall']:.4f}")
    print(f"  F0.2:      {metrics_after['f0.2']:.4f}")
    print(f"  TP: {metrics_after['tp']}, FP: {metrics_after['fp']}, FN: {metrics_after['fn']}")

    print(f"\nIMPROVEMENT:")
    prec_delta = metrics_after['precision'] - metrics_before['precision']
    rec_delta = metrics_after['recall'] - metrics_before['recall']
    f02_delta = metrics_after['f0.2'] - metrics_before['f0.2']

    print(f"  Precision: {prec_delta:+.4f} ({prec_delta*100:+.1f} points)")
    print(f"  Recall:    {rec_delta:+.4f} ({rec_delta*100:+.1f} points)")
    print(f"  F0.2:      {f02_delta:+.4f} ({f02_delta*100:+.1f} points)")

    fp_reduced = metrics_before['fp'] - metrics_after['fp']
    fn_increased = metrics_after['fn'] - metrics_before['fn']
    print(f"\n  False positives removed: {fp_reduced}")
    print(f"  False negatives added: {fn_increased}")
    print(f"  Net TP change: {metrics_after['tp'] - metrics_before['tp']}")

    print(f"\n{'='*80}\n")

    # Save results
    results = {
        "threshold": args.threshold,
        "before": metrics_before,
        "after": metrics_after,
        "improvement": {
            "precision": float(prec_delta),
            "recall": float(rec_delta),
            "f0.2": float(f02_delta),
        },
    }

    output_file = Path("reranker_eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
