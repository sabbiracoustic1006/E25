#!/usr/bin/env python
"""
Train a binary classifier to re-rank vehicle model predictions.

This classifier learns to distinguish true positives from false positives
in Round 1 model predictions, improving precision and F0.2.

Input: predicted_value + context (left_context, right_context, title)
Output: 1 (keep prediction) or 0 (filter out prediction)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def prepare_data(data_path: Path):
    """Load and prepare re-ranker training data."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, sep="\t")

    print(f"Total examples: {len(df)}")
    print(f"True Positives (label=1): {(df['is_correct'] == 1).sum()}")
    print(f"False Positives (label=0): {(df['is_correct'] == 0).sum()}")

    return df


def create_datasets(df: pd.DataFrame, tokenizer, max_length: int = 256):
    """Create train and validation datasets from fold splits."""

    # Create text input combining predicted value and context
    def create_input_text(row):
        # Format: [CLS] predicted_value [SEP] left_context [SPAN] predicted_value [/SPAN] right_context
        return f"{row['predicted_value']} [SEP] {row['left_context']} {row['predicted_value']} {row['right_context']}"

    df["input_text"] = df.apply(create_input_text, axis=1)

    # Split by fold: use fold 0 for validation, rest for training
    train_df = df[df["fold"] != 0]
    valid_df = df[df["fold"] == 0]

    print(f"\nTrain examples: {len(train_df)} ({(train_df['is_correct'] == 1).sum()} TP, {(train_df['is_correct'] == 0).sum()} FP)")
    print(f"Valid examples: {len(valid_df)} ({(valid_df['is_correct'] == 1).sum()} TP, {(valid_df['is_correct'] == 0).sum()} FP)")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[["input_text", "is_correct"]])
    valid_dataset = Dataset.from_pandas(valid_df[["input_text", "is_correct"]])

    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    valid_dataset = valid_dataset.map(tokenize_function, batched=True)

    # Rename label column
    train_dataset = train_dataset.rename_column("is_correct", "labels")
    valid_dataset = valid_dataset.rename_column("is_correct", "labels")

    # Set format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, valid_dataset


def compute_metrics(eval_pred):
    """
    Compute metrics focused on downstream F0.2 impact.

    Key insight: The re-ranker's job is to:
    1. Keep TPs (minimize FN) → maintain recall
    2. Filter FPs (maximize TN) → improve precision
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate confusion matrix
    tp = ((predictions == 1) & (labels == 1)).sum()  # Correctly kept TPs
    fp = ((predictions == 1) & (labels == 0)).sum()  # Incorrectly kept FPs
    fn = ((predictions == 0) & (labels == 1)).sum()  # Incorrectly filtered TPs (BAD!)
    tn = ((predictions == 0) & (labels == 0)).sum()  # Correctly filtered FPs (GOOD!)

    # Standard metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # F0.2 (precision-focused)
    beta = 0.2
    f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0.0

    # Custom downstream-aware metrics
    # 1. TP retention rate: What % of TPs do we keep?
    tp_retention = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # 2. FP filtering rate: What % of FPs do we filter out?
    fp_filtering = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 3. Balanced score: geometric mean ensures BOTH are high
    if tp_retention > 0 and fp_filtering > 0:
        balanced_score = (tp_retention * fp_filtering) ** 0.5
    else:
        balanced_score = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0.2": f_beta,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "tp_retention": tp_retention,      # High = keep TPs
        "fp_filtering": fp_filtering,       # High = filter FPs
        "balanced_score": balanced_score,   # Geometric mean (best overall metric)
    }


def main():
    parser = argparse.ArgumentParser(description="Train vehicle model re-ranker")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("reranker_training_data.tsv"),
        help="Path to re-ranker training data",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Base model",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("vehicle_reranker_model"),
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length",
    )
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Vehicle Model Re-ranker Training")
    print(f"{'='*80}")
    print(f"Model: {args.model_id}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Load data
    df = prepare_data(args.data_path)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset, valid_dataset = create_datasets(df, tokenizer, args.max_length)

    # Load model
    print(f"\nLoading model: {args.model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        problem_type="single_label_classification",
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_score",  # Optimizes for both TP retention & FP filtering
        greater_is_better=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    print(f"{'='*80}\n")

    trainer.train()

    # Save model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Save config
    config = {
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "max_length": args.max_length,
    }

    with open(args.output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Final evaluation
    print("\nFinal evaluation:")
    eval_results = trainer.evaluate()

    print(f"\n{'='*80}")
    print("RERANKER VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall:    {eval_results['eval_recall']:.4f}")
    print(f"F1:        {eval_results['eval_f1']:.4f}")
    print(f"F0.2:      {eval_results['eval_f0.2']:.4f}")
    print(f"\nCounts:")
    print(f"  TP: {eval_results['eval_tp']}")
    print(f"  FP: {eval_results['eval_fp']}")
    print(f"  FN: {eval_results['eval_fn']}")
    print(f"  TN: {eval_results['eval_tn']}")
    print(f"{'='*80}\n")

    # Save results
    with open(args.output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
