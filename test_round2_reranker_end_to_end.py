#!/usr/bin/env python
"""End-to-end test: Baseline predictions vs Round 2 reranker predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer

from eval_score import compute_competition_score
from utils import convert_tagged_to_aspect, stratified_kfold_split


def compute_per_aspect_scores(
    targets: List[Dict[str, str]],
    predictions: List[Dict[str, str]],
    beta: float = 0.2,
) -> Dict[str, float]:
    """Compute F-beta score for each aspect separately."""
    from collections import defaultdict

    # Group by aspect
    aspect_targets = defaultdict(list)
    aspect_preds = defaultdict(list)

    for t in targets:
        aspect_targets[t["aspect_name"]].append(t)

    for p in predictions:
        aspect_preds[p["aspect_name"]].append(p)

    # Compute scores per aspect
    aspect_scores = {}
    all_aspects = set(aspect_targets.keys()) | set(aspect_preds.keys())

    for aspect in all_aspects:
        asp_targets = aspect_targets.get(aspect, [])
        asp_preds = aspect_preds.get(aspect, [])

        if asp_targets:  # Only compute if there are ground truth examples
            result = compute_competition_score(asp_targets, asp_preds, beta=beta)
            aspect_scores[aspect] = result["overall_score"]

    return aspect_scores


def build_label_resources(
    data_path: Path,
    num_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int], Dict[int, str]]:
    """Load dataset splits and derive label vocabularies."""
    if not data_path.exists():
        raise FileNotFoundError(f"Tagged data not found at {data_path}")
    df0 = convert_tagged_to_aspect(str(data_path))
    df_split = stratified_kfold_split(df0, n_splits=num_folds, random_state=seed)

    aspects = sorted({asp for asp in df_split["Aspect Name"].unique() if asp})
    labels = ["O"] + [f"{prefix}-{aspect}" for aspect in aspects for prefix in ("B", "I")]
    if "U" in aspects:
        for prefix in ("B", "I"):
            tag = f"{prefix}-U"
            if tag not in labels:
                labels.append(tag)

    l2id = {label: idx for idx, label in enumerate(labels)}
    i2label = {idx: label for label, idx in l2id.items()}
    return df_split, labels, l2id, i2label


def extract_spans(
    seq: List[str],
    positions: List[int],
    offsets: List[Tuple[int, int]],
    text: str,
) -> List[Tuple[str, str]]:
    """Extract spans from token-level predictions."""
    spans: List[Tuple[str, str]] = []
    idx = 0
    while idx < len(seq):
        label = seq[idx]
        if label.startswith("B-"):
            aspect = label[2:]
            start_pos = positions[idx]
            end_pos = start_pos
            idx += 1
            while idx < len(seq) and seq[idx] == f"I-{aspect}":
                end_pos = positions[idx]
                idx += 1

            start_char, _ = offsets[start_pos]
            _, end_char = offsets[end_pos]
            span_text = text[start_char:end_char].strip()
            spans.append((aspect, span_text))
        else:
            idx += 1
    return spans


def build_samples(valid_df, model, tokenizer, device, label2id, id2label) -> List[Dict[str, object]]:
    """Build samples with aligned token-level labels."""
    samples: List[Dict[str, object]] = []
    for rid in tqdm(valid_df["Record Number"].unique(), desc="Collecting predictions"):
        sample = valid_df[valid_df["Record Number"] == rid]
        text = sample["Title"].iat[0]
        category = str(sample["Category"].iat[0])
        gold = sample[["Category", "Aspect Name", "Aspect Value"]].values.tolist()

        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )
        offsets = encoded["offset_mapping"][0].tolist()

        # Create token-level labels aligned to gold spans
        labels = [-100] * len(offsets)
        for cat, asp, val in gold:
            val_str = str(val)
            for match in re.finditer(re.escape(val_str), text):
                start_char, end_char = match.span()
                b_assigned = False
                for tidx, (start, end) in enumerate(offsets):
                    if start >= end_char:
                        break
                    if end <= start_char:
                        continue
                    if not b_assigned:
                        labels[tidx] = label2id[f"B-{asp}"]
                        b_assigned = True
                    else:
                        labels[tidx] = label2id[f"I-{asp}"]

        encoded = {k: v.to(device) for k, v in encoded.items()}
        encoded.pop("offset_mapping")

        with torch.no_grad():
            logits = model(**encoded).logits.softmax(dim=-1).cpu()[0]

        samples.append(
            {
                "record_id": rid,
                "category": category,
                "text": text,
                "gold": gold,
                "probs": logits.numpy(),
                "labels": labels,
                "offsets": offsets,
            }
        )
    return samples


def evaluate_baseline(
    samples: List[Dict[str, object]],
    id2label: Dict[int, str],
):
    """Evaluate baseline predictions (raw argmax)."""
    all_targets: List[Dict[str, str]] = []
    all_preds: List[Dict[str, str]] = []

    for sample in samples:
        probs = sample["probs"]
        preds = probs.argmax(axis=-1)

        # Filter tokens using -100 labels
        labels = sample["labels"]
        seq_p: List[str] = []
        seq_l: List[str] = []
        token_idxs: List[int] = []
        for idx, l_id in enumerate(labels):
            if l_id == -100:
                continue
            seq_p.append(id2label[int(preds[idx])])
            seq_l.append(id2label[l_id])
            token_idxs.append(idx)

        text = sample["text"]
        category = sample["category"]
        record_id = sample["record_id"]
        offsets = sample["offsets"]

        # Extract spans
        for aspect, span in extract_spans(seq_l, token_idxs, offsets, text):
            if aspect != "O":
                all_targets.append({
                    "record_id": str(record_id),
                    "category": category,
                    "aspect_name": aspect,
                    "span": span,
                })

        for aspect, span in extract_spans(seq_p, token_idxs, offsets, text):
            if aspect != "O":
                all_preds.append({
                    "record_id": str(record_id),
                    "category": category,
                    "aspect_name": aspect,
                    "span": span,
                })

    comp = compute_competition_score(all_targets, all_preds, beta=0.2)
    per_aspect = compute_per_aspect_scores(all_targets, all_preds, beta=0.2)
    return comp["overall_score"], comp["per_category"], per_aspect, all_targets, all_preds


def apply_reranker(
    samples: List[Dict[str, object]],
    reranker_model,
    reranker_tokenizer,
    device,
    id2label: Dict[int, str],
    threshold: float = 0.5,
):
    """Apply reranker to filter predictions."""
    all_targets: List[Dict[str, str]] = []
    all_preds: List[Dict[str, str]] = []

    for sample in tqdm(samples, desc="Applying reranker"):
        probs = sample["probs"]
        preds = probs.argmax(axis=-1)

        # Filter tokens using -100 labels
        labels = sample["labels"]
        seq_p: List[str] = []
        seq_l: List[str] = []
        token_idxs: List[int] = []
        for idx, l_id in enumerate(labels):
            if l_id == -100:
                continue
            seq_p.append(id2label[int(preds[idx])])
            seq_l.append(id2label[l_id])
            token_idxs.append(idx)

        text = sample["text"]
        category = sample["category"]
        record_id = sample["record_id"]
        offsets = sample["offsets"]

        # Extract target spans
        for aspect, span in extract_spans(seq_l, token_idxs, offsets, text):
            if aspect != "O":
                all_targets.append({
                    "record_id": str(record_id),
                    "category": category,
                    "aspect_name": aspect,
                    "span": span,
                })

        # Extract predicted spans and apply reranker ONLY to Kompatibles_Fahrzeug_Modell
        for aspect, span in extract_spans(seq_p, token_idxs, offsets, text):
            if aspect != "O":
                # Only apply reranker to Kompatibles_Fahrzeug_Modell (what it was trained on)
                if aspect == "Kompatibles_Fahrzeug_Modell":
                    # Create input for reranker: [CLS] title [SEP] aspect: span [SEP]
                    reranker_input = f"{text} [SEP] {aspect}: {span}"
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
                        score = probs[0, 1].item()  # Probability of class 1 (keep prediction)

                    if score >= threshold:
                        all_preds.append({
                            "record_id": str(record_id),
                            "category": category,
                            "aspect_name": aspect,
                            "span": span,
                        })
                else:
                    # Keep all other aspect predictions as-is (no reranking)
                    all_preds.append({
                        "record_id": str(record_id),
                        "category": category,
                        "aspect_name": aspect,
                        "span": span,
                    })

    comp = compute_competition_score(all_targets, all_preds, beta=0.2)
    per_aspect = compute_per_aspect_scores(all_targets, all_preds, beta=0.2)
    return comp["overall_score"], comp["per_category"], per_aspect, all_preds


def filter_category_predictions(
    predictions: List[Dict[str, str]],
    category: str,
) -> List[Dict[str, str]]:
    """Filter predictions for a specific category."""
    return [p for p in predictions if p["category"] == category]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end test: Baseline vs Round 2 reranker")
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Directory containing the base NER model",
    )
    parser.add_argument(
        "--reranker_dir",
        type=Path,
        required=True,
        help="Directory containing the Round 2 reranker model",
    )
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--threshold", type=float, default=0.5, help="Reranker threshold")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/Tagged_Titles_Train.tsv"),
        help="Path to tagged training data",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("round2_reranker_test_results.json"),
        help="Output results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("="*80)
    print("End-to-End Test: Baseline vs Round 2 Reranker")
    print("="*80)
    print()

    # Build label resources
    print("Loading data and building label resources...")
    df_split, label_list, label2id, id2label = build_label_resources(
        args.data_path,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    valid_df = df_split[df_split["fold"] == args.fold]
    if valid_df.empty:
        raise ValueError("Validation dataframe is empty. Check fold index.")

    # Load base NER model
    model_path = args.model_dir / f"fold{args.fold}"
    print(f"Loading base NER model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    config_id2label = getattr(model.config, "id2label", None)
    if config_id2label:
        id2label = {int(k): v for k, v in config_id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        label_list = [id2label[idx] for idx in sorted(id2label)]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Build samples with baseline predictions
    print("Generating baseline predictions...")
    samples = build_samples(valid_df, model, tokenizer, device, label2id, id2label)

    # Evaluate baseline
    print("\n" + "="*80)
    print("BASELINE EVALUATION (Raw Argmax Predictions)")
    print("="*80)
    baseline_score, baseline_per_cat, baseline_per_aspect, all_targets, baseline_preds = evaluate_baseline(
        samples, id2label
    )

    print(f"\nOverall Score: {baseline_score:.6f}")
    print("\nPer-Category Scores:")
    for cat, score in sorted(baseline_per_cat.items()):
        print(f"  Category {cat}: {score:.6f}")

    # Extract Kompatibles_Fahrzeug_Modell score from per-aspect scores
    kfm_baseline_score = baseline_per_aspect.get("Kompatibles_Fahrzeug_Modell", None)

    if kfm_baseline_score is not None:
        print(f"\n*** Kompatibles_Fahrzeug_Modell Score: {kfm_baseline_score:.6f} ***")
    else:
        print("\n*** Kompatibles_Fahrzeug_Modell aspect not found ***")

    # Load reranker model
    print(f"\n\nLoading Round 2 reranker model from {args.reranker_dir}...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_dir, use_fast=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(args.reranker_dir)
    reranker_model.to(device)
    reranker_model.eval()

    # Apply reranker
    print("\n" + "="*80)
    print(f"RERANKER EVALUATION (Threshold: {args.threshold})")
    print("="*80)
    reranked_score, reranked_per_cat, reranked_per_aspect, reranked_preds = apply_reranker(
        samples, reranker_model, reranker_tokenizer, device, id2label, args.threshold
    )

    print(f"\nOverall Score: {reranked_score:.6f}")
    print("\nPer-Category Scores:")
    for cat, score in sorted(reranked_per_cat.items()):
        print(f"  Category {cat}: {score:.6f}")

    kfm_reranked_score = reranked_per_aspect.get("Kompatibles_Fahrzeug_Modell", None)

    if kfm_reranked_score is not None:
        print(f"\n*** Kompatibles_Fahrzeug_Modell Score: {kfm_reranked_score:.6f} ***")
    else:
        print("\n*** Kompatibles_Fahrzeug_Modell aspect not found ***")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: Baseline vs Reranked")
    print("="*80)
    print(f"\nOverall Score:")
    print(f"  Baseline:  {baseline_score:.6f}")
    print(f"  Reranked:  {reranked_score:.6f}")
    print(f"  Delta:     {reranked_score - baseline_score:+.6f}")

    if kfm_baseline_score is not None and kfm_reranked_score is not None:
        print(f"\nKompatibles_Fahrzeug_Modell Score:")
        print(f"  Baseline:  {kfm_baseline_score:.6f}")
        print(f"  Reranked:  {kfm_reranked_score:.6f}")
        print(f"  Delta:     {kfm_reranked_score - kfm_baseline_score:+.6f}")

    print("\nPer-Category Deltas:")
    all_categories = set(baseline_per_cat.keys()) | set(reranked_per_cat.keys())
    for cat in sorted(all_categories):
        base = baseline_per_cat.get(cat, 0.0)
        rerank = reranked_per_cat.get(cat, 0.0)
        delta = rerank - base
        print(f"  Category {cat}: {base:.6f} -> {rerank:.6f} ({delta:+.6f})")

    # Save results
    results = {
        "fold": args.fold,
        "threshold": args.threshold,
        "baseline": {
            "overall_score": baseline_score,
            "per_category": baseline_per_cat,
            "per_aspect": baseline_per_aspect,
            "kompatibles_fahrzeug_modell": kfm_baseline_score,
        },
        "reranked": {
            "overall_score": reranked_score,
            "per_category": reranked_per_cat,
            "per_aspect": reranked_per_aspect,
            "kompatibles_fahrzeug_modell": kfm_reranked_score,
        },
        "delta": {
            "overall_score": reranked_score - baseline_score,
            "kompatibles_fahrzeug_modell": (
                kfm_reranked_score - kfm_baseline_score
                if kfm_baseline_score is not None and kfm_reranked_score is not None
                else None
            ),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
