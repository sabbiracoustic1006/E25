#!/usr/bin/env python
"""
Generate predictions from Round 1 models and save them for re-ranker training.

This script loads each fold's Round 1 model and generates predictions on its
validation set, saving them in a format suitable for re-ranker training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils import convert_tagged_to_aspect, stratified_kfold_split


class TokenClassificationDataset(Dataset):
    """Dataset for token classification."""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["title"],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding="max_length",
        )
        return {
            "record_num": item["record_num"],
            "title": item["title"],
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "offset_mapping": encoding["offset_mapping"],
        }


def collate_fn(batch):
    """Custom collate function."""
    return {
        "record_nums": [x["record_num"] for x in batch],
        "titles": [x["title"] for x in batch],
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "offset_mappings": [x["offset_mapping"] for x in batch],
    }


def extract_aspects_from_predictions(
    predictions: torch.Tensor,
    offset_mapping: List[tuple],
    title: str,
    id2label: Dict[int, str],
) -> Dict[str, List[str]]:
    """Extract aspect name-value pairs from model predictions."""
    aspects = {}
    current_aspect = None
    current_span_start = None
    current_span_end = None

    for i, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens and padding
        if start == 0 and end == 0:
            continue

        pred_label = id2label[pred_id]

        if pred_label.startswith("B-"):
            # Save previous span if exists
            if current_aspect and current_span_start is not None:
                aspect_value = title[current_span_start:current_span_end].strip()
                if aspect_value:
                    aspects.setdefault(current_aspect, []).append(aspect_value)

            # Start new span
            current_aspect = pred_label[2:]  # Remove "B-"
            current_span_start = start
            current_span_end = end

        elif pred_label.startswith("I-"):
            aspect_name = pred_label[2:]  # Remove "I-"
            if current_aspect == aspect_name:
                # Continue current span
                current_span_end = end
            else:
                # Inconsistent I- tag, treat as new span
                if current_aspect and current_span_start is not None:
                    aspect_value = title[current_span_start:current_span_end].strip()
                    if aspect_value:
                        aspects.setdefault(current_aspect, []).append(aspect_value)

                current_aspect = aspect_name
                current_span_start = start
                current_span_end = end

        else:  # O or B-O or I-O
            # End current span
            if current_aspect and current_span_start is not None:
                aspect_value = title[current_span_start:current_span_end].strip()
                if aspect_value:
                    aspects.setdefault(current_aspect, []).append(aspect_value)

            current_aspect = None
            current_span_start = None
            current_span_end = None

    # Don't forget last span
    if current_aspect and current_span_start is not None:
        aspect_value = title[current_span_start:current_span_end].strip()
        if aspect_value:
            aspects.setdefault(current_aspect, []).append(aspect_value)

    return aspects


def generate_predictions_for_fold(
    model_path: Path,
    valid_data: List[dict],
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Dict[str, List[str]]]:
    """Generate predictions for a single fold."""
    print(f"Loading model from {model_path}")
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Get id2label mapping
    id2label = model.config.id2label

    # Create dataset and dataloader
    dataset = TokenClassificationDataset(valid_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Generate predictions
    all_predictions = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Extract aspects for each example in batch
            for i, record_num in enumerate(batch["record_nums"]):
                preds = predictions[i].cpu().numpy()
                offset_mapping = batch["offset_mappings"][i]
                title = batch["titles"][i]

                aspects = extract_aspects_from_predictions(preds, offset_mapping, title, id2label)
                all_predictions[record_num] = aspects

    return all_predictions


def main():
    parser = argparse.ArgumentParser(description="Generate Round 1 predictions for re-ranker")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/Tagged_Titles_Train.tsv",
        help="Training data path",
    )
    parser.add_argument(
        "--model_base_dir",
        type=Path,
        default=Path("/data/sahmed9/E25/round_1/deberta-v3-base/lr_1_e_neg_4/o_weight_1"),
        help="Base directory containing fold models",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("round_1_predictions"),
        help="Output directory for predictions",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Generating Round 1 Predictions")
    print(f"{'='*80}")
    print(f"Data path: {args.data_path}")
    print(f"Model base dir: {args.model_base_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and create folds
    print("Loading data and creating folds...")
    df = convert_tagged_to_aspect(args.data_path)
    df_split = stratified_kfold_split(df, n_splits=args.num_folds, random_state=args.seed)

    # Process each fold
    for fold in range(args.num_folds):
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold}")
        print(f"{'='*80}\n")

        # Get validation data for this fold
        valid_df = df_split[df_split["fold"] == fold]

        # Prepare validation data (unique records)
        valid_data = []
        for record_num in valid_df["Record Number"].unique():
            record_rows = valid_df[valid_df["Record Number"] == record_num]
            title = record_rows.iloc[0]["Title"]
            valid_data.append({
                "record_num": record_num,
                "title": title,
            })

        print(f"Validation records: {len(valid_data)}")

        # Find model checkpoint (use best checkpoint if available)
        model_path = args.model_base_dir / f"fold{fold}"
        if not model_path.exists():
            print(f"  Warning: Model not found at {model_path}, skipping...")
            continue

        # Generate predictions
        predictions = generate_predictions_for_fold(model_path, valid_data, args.batch_size)

        # Save predictions
        output_file = args.output_dir / f"fold{fold}_predictions.json"
        print(f"\nSaving predictions to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        # Print stats
        vehicle_count = sum(
            len(preds.get("Kompatibles_Fahrzeug_Modell", []))
            for preds in predictions.values()
        )
        print(f"Total predicted vehicle models: {vehicle_count}")

    print(f"\n{'='*80}")
    print("Prediction generation complete!")
    print(f"{'='*80}\n")
    print(f"Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
