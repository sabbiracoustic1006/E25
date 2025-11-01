#!/usr/bin/env python
"""
Apply Round 2 reranker to filter Kompatibles_Fahrzeug_Modell predictions from ensemble.

This script:
1. Loads ensemble predictions (TSV format)
2. Applies reranker ONLY to Kompatibles_Fahrzeug_Modell predictions
3. Filters out low-confidence predictions based on threshold
4. Keeps all other aspects unchanged
5. Saves filtered results to new TSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_ensemble_tsv(tsv_path: Path) -> pd.DataFrame:
    """Load ensemble predictions from TSV file."""
    print(f"Loading ensemble predictions from {tsv_path}...")
    # TSV has no header: Record Number, Category, Aspect Name, Aspect Value
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["Record Number", "Category", "Aspect Name", "Aspect Value"]
    )
    print(f"Loaded {len(df)} predictions")
    print(f"Columns: {df.columns.tolist()}")

    # Show sample of data
    print(f"\nSample predictions:")
    print(df.head(3))

    return df


def load_titles(data_path: Path) -> dict:
    """Load title mapping from original data."""
    from utils import convert_tagged_to_aspect

    print(f"Loading titles from {data_path}...")
    df = convert_tagged_to_aspect(str(data_path))

    # Create mapping from Record Number to Title
    title_map = {}
    for record_num in df["Record Number"].unique():
        record_df = df[df["Record Number"] == record_num]
        title_map[record_num] = record_df.iloc[0]["Title"]

    print(f"Loaded titles for {len(title_map)} records")
    return title_map


def filter_with_reranker(
    df: pd.DataFrame,
    title_map: dict,
    reranker_model,
    reranker_tokenizer,
    device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Apply reranker to filter Kompatibles_Fahrzeug_Modell predictions.

    Returns a filtered dataframe with low-confidence predictions removed.
    """
    filtered_rows = []
    scores_debug = []  # Track scores for debugging

    # Group by Record Number to get title context
    grouped = df.groupby("Record Number")

    for record_num, record_df in tqdm(grouped, desc="Filtering predictions"):
        # Get title from title map
        title = title_map.get(record_num, "")
        if not title:
            # If title not found, keep all predictions for this record
            filtered_rows.extend(record_df.to_dict('records'))
            continue

        for _, row in record_df.iterrows():
            aspect = row["Aspect Name"]
            span = row["Aspect Value"]

            # Only apply reranker to Kompatibles_Fahrzeug_Modell
            if aspect == "Kompatibles_Fahrzeug_Modell":
                # Extract context (same format as training data)
                # Find the span in the title to get left and right context
                span_start = title.find(str(span))
                if span_start == -1:
                    # Span not found in title, try case-insensitive
                    span_lower = str(span).lower()
                    title_lower = title.lower()
                    span_start = title_lower.find(span_lower)

                if span_start != -1:
                    # Extract left and right context
                    left_context = title[:span_start][-20:]  # Last 20 chars before span
                    span_end = span_start + len(str(span))
                    right_context = title[span_end:][:20]  # First 20 chars after span

                    # Create input in the same format as training:
                    # predicted_value [SEP] left_context predicted_value right_context
                    reranker_input = f"{span} [SEP] {left_context} {span} {right_context}"
                else:
                    # Fallback: use whole title if span not found
                    reranker_input = f"{span} [SEP]  {span} "

                encoded = reranker_tokenizer(
                    reranker_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                with torch.no_grad():
                    logits = reranker_model(**encoded).logits
                    probs = torch.softmax(logits, dim=-1)
                    score = probs[0, 1].item()  # Probability of class 1 (keep)

                # Track scores for debugging
                scores_debug.append(score)

                # Only keep if score is above threshold
                if score >= threshold:
                    filtered_rows.append(row.to_dict())
            else:
                # Keep all other aspects unchanged
                filtered_rows.append(row.to_dict())

    filtered_df = pd.DataFrame(filtered_rows)

    # Print score statistics
    if scores_debug:
        import numpy as np
        scores_array = np.array(scores_debug)
        print(f"\nReranker Score Statistics:")
        print(f"  Min:    {scores_array.min():.4f}")
        print(f"  Max:    {scores_array.max():.4f}")
        print(f"  Mean:   {scores_array.mean():.4f}")
        print(f"  Median: {np.median(scores_array):.4f}")
        print(f"  Scores < 0.5: {(scores_array < 0.5).sum()} ({100*(scores_array < 0.5).sum()/len(scores_array):.1f}%)")
        print(f"  Scores < 0.7: {(scores_array < 0.7).sum()} ({100*(scores_array < 0.7).sum()/len(scores_array):.1f}%)")
        print(f"  Scores < 0.9: {(scores_array < 0.9).sum()} ({100*(scores_array < 0.9).sum()/len(scores_array):.1f}%)")

    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Filter ensemble predictions using Round 2 reranker"
    )
    parser.add_argument(
        "--input_tsv",
        type=Path,
        default=Path("/data/sahmed9/E25/tsvs/ensemble_o_weight_1_majority_12_of_25.tsv"),
        help="Input ensemble TSV file",
    )
    parser.add_argument(
        "--reranker_dir",
        type=Path,
        default=Path("vehicle_reranker_model_round2"),
        help="Directory containing the Round 2 reranker model",
    )
    parser.add_argument(
        "--output_tsv",
        type=Path,
        default=Path("ensemble_o_weight_1_majority_12_of_25_reranked.tsv"),
        help="Output filtered TSV file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Reranker threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/Tagged_Titles_Train.tsv"),
        help="Path to original training data (to get titles)",
    )
    args = parser.parse_args()

    print("="*80)
    print("Filter Ensemble Predictions with Round 2 Reranker")
    print("="*80)
    print(f"Input:     {args.input_tsv}")
    print(f"Reranker:  {args.reranker_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Output:    {args.output_tsv}")
    print()

    # Load ensemble predictions
    df = load_ensemble_tsv(args.input_tsv)

    # Load titles from training data
    title_map = load_titles(args.data_path)

    # Count Kompatibles_Fahrzeug_Modell predictions before filtering
    kfm_before = (df["Aspect Name"] == "Kompatibles_Fahrzeug_Modell").sum()
    total_before = len(df)

    print(f"\nBefore filtering:")
    print(f"  Total predictions: {total_before}")
    print(f"  Kompatibles_Fahrzeug_Modell predictions: {kfm_before}")
    print(f"  Other aspects: {total_before - kfm_before}")

    # Load reranker model
    print(f"\nLoading reranker model from {args.reranker_dir}...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker_dir)
    model.to(device)
    model.eval()

    # Apply reranker filter
    print(f"\nApplying reranker (threshold={args.threshold})...")
    filtered_df = filter_with_reranker(df, title_map, model, tokenizer, device, args.threshold)

    # Count after filtering
    kfm_after = (filtered_df["Aspect Name"] == "Kompatibles_Fahrzeug_Modell").sum()
    total_after = len(filtered_df)
    kfm_filtered = kfm_before - kfm_after

    print(f"\nAfter filtering:")
    print(f"  Total predictions: {total_after}")
    print(f"  Kompatibles_Fahrzeug_Modell predictions: {kfm_after}")
    print(f"  Other aspects: {total_after - kfm_after} (unchanged)")

    print(f"\nFiltered out:")
    print(f"  Kompatibles_Fahrzeug_Modell: {kfm_filtered} ({100*kfm_filtered/kfm_before:.1f}%)")
    print(f"  Total: {total_before - total_after}")

    # Save filtered results (no header, same format as input)
    print(f"\nSaving filtered results to {args.output_tsv}...")
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(args.output_tsv, sep="\t", index=False, header=False)

    print("\n" + "="*80)
    print("Filtering complete!")
    print("="*80)


if __name__ == "__main__":
    main()
