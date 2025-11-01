#!/usr/bin/env python
"""Refine class-wise thresholds with mean-plus-sigma sweep search, category-wise."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from eval_score import compute_competition_score
from utils import convert_tagged_to_aspect, stratified_kfold_split


df1 = None  # populated at runtime
label_list: List[str] = []
label2id: Dict[str, int] = {}
id2label: Dict[int, str] = {}
category_to_valid_aspects: Dict[str, Set[str]] = {}


def extract_category_to_valid_aspects(data_path: Path) -> Dict[str, Set[str]]:
    """Extract mapping from category to valid aspect names from training data."""
    df = pd.read_csv(data_path, sep="\t", encoding="utf-8")

    # Filter out rows where Tag is empty or "O"
    df_filtered = df[df["Tag"].notna() & (df["Tag"] != "") & (df["Tag"] != "O")]

    category_aspects: Dict[str, Set[str]] = {}
    for _, row in df_filtered.iterrows():
        cat = str(row["Category"])
        aspect = str(row["Tag"])

        if cat not in category_aspects:
            category_aspects[cat] = set()
        category_aspects[cat].add(aspect)

    return category_aspects


def build_label_resources(
    data_path: Path,
    num_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int], Dict[int, str], Dict[str, Set[str]]]:
    """Load dataset splits and derive label vocabularies."""
    if not data_path.exists():
        raise FileNotFoundError(f"Tagged data not found at {data_path}")

    # Extract category-to-valid-aspects mapping
    cat_to_aspects = extract_category_to_valid_aspects(data_path)

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
    return df_split, labels, l2id, i2label, cat_to_aspects


def get_valid_class_ids_for_category(category: str, cat_to_aspects: Dict[str, Set[str]], label2id: Dict[str, int]) -> Set[int]:
    """Get valid class IDs for a given category."""
    valid_ids = {label2id["O"]}  # O is always valid

    valid_aspects = cat_to_aspects.get(category, set())
    for aspect in valid_aspects:
        for prefix in ("B", "I"):
            label = f"{prefix}-{aspect}"
            if label in label2id:
                valid_ids.add(label2id[label])

    return valid_ids


def compute_threshold_statistics(
    samples: List[Dict[str, object]],
    num_labels: int,
    category: str = None,
) -> Tuple[Dict[int, List[float]], Dict[int, Dict[str, object]]]:
    """Aggregate per-class confidence statistics from token predictions."""
    percentiles = np.linspace(0, 100, 101)
    per_class: Dict[int, List[float]] = {cls: [] for cls in range(num_labels)}

    for sample in samples:
        # Skip if category filter is specified and doesn't match
        if category is not None and str(sample["category"]) != str(category):
            continue

        probs = np.asarray(sample["probs"], dtype=float)
        if probs.ndim != 2 or probs.shape[1] != num_labels:
            raise ValueError("Unexpected probability tensor shape while computing thresholds")
        pred_ids = probs.argmax(axis=-1)
        max_probs = probs.max(axis=-1)
        for cls, prob in zip(pred_ids, max_probs):
            per_class[int(cls)].append(float(prob))

    threshold_bins: Dict[int, Dict[str, object]] = {}
    for cls in range(num_labels):
        values = per_class[cls]
        if values:
            arr = np.array(sorted(values, reverse=True), dtype=float)
            bins = np.percentile(arr, percentiles)
            threshold_bins[cls] = {
                "bins": bins,
                "percentiles": percentiles,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=0)),
                "count": int(arr.size),
            }
            per_class[cls] = arr.tolist()
        else:
            threshold_bins[cls] = {
                "bins": np.array([], dtype=float),
                "percentiles": np.array([], dtype=float),
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "count": 0,
            }
    return per_class, threshold_bins


def extract_spans(
    seq: List[str],
    positions: List[int],
    offsets: List[Tuple[int, int]],
    text: str,
) -> List[Tuple[str, str]]:
    """Extract spans from token-level predictions (matches training logic)."""
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


def apply_thresholds_with_category_masking(
    probs: torch.Tensor,
    base_preds: torch.Tensor,
    thresholds: Dict[int, float],
    o_id: int,
    category: str,
    valid_class_ids: Set[int],
) -> torch.Tensor:
    """Apply thresholds and mask invalid classes for the category."""
    # First, mask invalid classes by setting their logits to -inf (vectorized)
    masked_probs = probs.clone()
    batch_size, seq_len, num_classes = masked_probs.shape

    # Create a boolean mask for valid classes (vectorized operation)
    valid_mask = torch.zeros(num_classes, dtype=torch.bool)
    for cls_id in valid_class_ids:
        valid_mask[cls_id] = True

    # Set invalid classes to -inf using broadcasting
    masked_probs[:, :, ~valid_mask] = float('-inf')

    # Recompute predictions after masking
    filtered = masked_probs.argmax(dim=-1)
    max_probs = masked_probs.max(dim=-1).values

    # Apply thresholds
    for b in range(batch_size):
        for pos in range(seq_len):
            cls = int(filtered[b, pos])
            if cls == o_id:
                continue
            threshold = thresholds.get(cls, 0.0)
            # Note: max_probs might be -inf for positions with all invalid classes
            if torch.isfinite(max_probs[b, pos]) and max_probs[b, pos].item() < threshold:
                filtered[b, pos] = o_id

    return filtered


def apply_thresholds(
    probs: torch.Tensor,
    base_preds: torch.Tensor,
    thresholds: Dict[int, float],
    o_id: int,
) -> torch.Tensor:
    """Apply thresholds without category masking (for baseline)."""
    filtered = base_preds.clone()
    max_probs = probs.max(dim=-1).values
    batch_size, seq_len = base_preds.shape
    for b in range(batch_size):
        for pos in range(seq_len):
            cls = int(base_preds[b, pos])
            if cls == o_id:
                continue
            threshold = thresholds.get(cls, 0.0)
            if max_probs[b, pos].item() < threshold:
                filtered[b, pos] = o_id
    return filtered


def build_samples(valid_df, model, tokenizer, device, cat_to_aspects, label2id) -> List[Dict[str, object]]:
    """Build samples with aligned token-level labels (matches training logic)."""
    samples: List[Dict[str, object]] = []
    for rid in tqdm(valid_df["Record Number"].unique(), desc="Collecting logits"):
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

        # Create token-level labels aligned to gold spans (same as training)
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

        # Pre-compute valid class IDs for this category
        valid_ids = get_valid_class_ids_for_category(category, cat_to_aspects, label2id)

        # Pre-mask probabilities (set invalid classes to -inf)
        num_classes = logits.shape[-1]
        valid_mask = torch.zeros(num_classes, dtype=torch.bool)
        for cls_id in valid_ids:
            valid_mask[cls_id] = True

        masked_probs = logits.clone()
        masked_probs[:, ~valid_mask] = float('-inf')

        samples.append(
            {
                "record_id": rid,
                "category": category,
                "text": text,
                "gold": gold,
                "probs": logits.numpy(),
                "masked_probs": masked_probs.numpy(),  # Pre-computed masked probs
                "labels": labels,
                "offsets": offsets,
            }
        )
    return samples


def evaluate_thresholds(
    samples: List[Dict[str, object]],
    thresholds: Dict[int, float],
    o_id: int,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate predictions using training-aligned token filtering (for baseline)."""
    all_targets: List[Dict[str, str]] = []
    all_preds: List[Dict[str, str]] = []

    for sample in samples:
        probs = torch.from_numpy(sample["probs"]).unsqueeze(0)
        base_preds = probs.argmax(dim=-1)
        filtered = apply_thresholds(probs, base_preds, thresholds, o_id).squeeze(0).numpy()

        # Filter tokens using -100 labels (same as training)
        labels = sample["labels"]
        seq_p: List[str] = []
        seq_l: List[str] = []
        token_idxs: List[int] = []
        for idx, l_id in enumerate(labels):
            if l_id == -100:
                continue  # Skip special tokens and padding
            seq_p.append(id2label[int(filtered[idx])])
            seq_l.append(label_list[l_id])
            token_idxs.append(idx)

        text = sample["text"]
        category = sample["category"]
        record_id = sample["record_id"]
        offsets = sample["offsets"]

        # Extract spans using training's logic
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
    metrics = {"overall_score": comp["overall_score"]}
    metrics.update({f"cat_{cat}": score for cat, score in comp["per_category"].items()})
    return comp["overall_score"], metrics


def evaluate_thresholds_categorywise(
    samples_cat1: List[Dict[str, object]],
    samples_cat2: List[Dict[str, object]],
    thresholds_cat1: Dict[int, float],
    thresholds_cat2: Dict[int, float],
    o_id: int,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate predictions using category-specific thresholds with pre-masked probs."""
    all_targets: List[Dict[str, str]] = []
    all_preds: List[Dict[str, str]] = []

    # Process category 1 samples
    for sample in samples_cat1:
        masked_probs = torch.from_numpy(sample["masked_probs"]).unsqueeze(0)

        # Apply thresholds directly on pre-masked probs
        base_preds = masked_probs.argmax(dim=-1)
        filtered = apply_thresholds(masked_probs, base_preds, thresholds_cat1, o_id).squeeze(0).numpy()

        # Filter tokens using -100 labels (same as training)
        labels = sample["labels"]
        seq_p: List[str] = []
        seq_l: List[str] = []
        token_idxs: List[int] = []
        for idx, l_id in enumerate(labels):
            if l_id == -100:
                continue
            seq_p.append(id2label[int(filtered[idx])])
            seq_l.append(label_list[l_id])
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

    # Process category 2 samples
    for sample in samples_cat2:
        masked_probs = torch.from_numpy(sample["masked_probs"]).unsqueeze(0)

        # Apply thresholds directly on pre-masked probs
        base_preds = masked_probs.argmax(dim=-1)
        filtered = apply_thresholds(masked_probs, base_preds, thresholds_cat2, o_id).squeeze(0).numpy()

        # Filter tokens using -100 labels
        labels = sample["labels"]
        seq_p: List[str] = []
        seq_l: List[str] = []
        token_idxs: List[int] = []
        for idx, l_id in enumerate(labels):
            if l_id == -100:
                continue
            seq_p.append(id2label[int(filtered[idx])])
            seq_l.append(label_list[l_id])
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
    metrics = {"overall_score": comp["overall_score"]}
    metrics.update({f"cat_{cat}": score for cat, score in comp["per_category"].items()})
    return comp["overall_score"], metrics


def build_candidate_thresholds_percentile(
    per_class_scores: Dict[int, List[float]],
    hi_focus: bool = True,
    max_candidates: int = 30,
) -> Dict[int, List[float]]:
    """Generate percentile-based candidates for each class.

    Builds a percentile grid denser near the top (upper tail), which is more
    effective for tail-sensitive metrics like F-beta with beta=0.2.

    Args:
        per_class_scores: Dict mapping class ID to list of confidence scores
        hi_focus: If True, adds fixed high-precision thresholds
        max_candidates: Maximum number of candidates per class

    Returns:
        Dict mapping class ID to sorted list of threshold candidates
    """
    # Percentile grid, denser near the top
    base_percentiles = [50, 60, 70, 80, 85, 90, 92, 94, 95, 96, 97, 98, 98.5, 99, 99.25, 99.5, 99.75, 99.9]

    candidates: Dict[int, List[float]] = {}
    for cls, scores in per_class_scores.items():
        if not scores or len(scores) == 0:
            candidates[cls] = [0.0, 1.0]
            continue

        scores_arr = np.asarray(scores, dtype=float)

        # Compute percentile-based candidates
        cands = np.quantile(scores_arr, np.array(base_percentiles) / 100.0).tolist()

        # Add fixed high-precision thresholds
        if hi_focus:
            cands += [0.95, 0.975, 0.99, 0.995, 1.0]

        # Clip, deduplicate, and sort
        cands = sorted(set(np.clip(cands, 0.0, 1.0)))

        # Cap to max_candidates
        if len(cands) > max_candidates:
            # Keep evenly spaced subset, always including min and max
            indices = np.linspace(0, len(cands) - 1, max_candidates, dtype=int)
            cands = [cands[i] for i in indices]

        candidates[cls] = cands

    return candidates


def build_candidate_thresholds_mean_sigma(
    threshold_bins: Dict[int, Dict[str, object]],
    step_multiplier: float = 0.025,
    max_steps: int = 20,
) -> Dict[int, List[float]]:
    """Generate mean-plus-sigma candidates for each class.

    Starting from the class-wise mean confidence, evaluate thresholds at
    ``mean + k * step_multiplier * std`` for increasing integer ``k`` until the
    values exceed 1.0 or ``max_steps`` is hit. The resulting list is clipped to
    ``[0, 1]`` and deduplicated.
    """

    candidates: Dict[int, List[float]] = {}
    for cls, stats in threshold_bins.items():
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 0.0))
        count = int(stats.get("count", 0))

        if count <= 0:
            candidates[cls] = [float(np.clip(mean, 0.0, 1.0))]
            continue

        step = std * step_multiplier
        # Ensure at least the mean candidate even if std is zero.
        values = [mean]

        if step > 0:
            for idx in range(1, max_steps + 1):
                candidate = mean + idx * step
                values.append(candidate)
                if candidate >= 1.0:
                    break
        else:
            # std is zero, therefore probabilities are concentrated; keep mean only.
            pass

        # Clip, deduplicate, and sort for stable traversal during coordinate descent.
        clean = sorted({float(np.clip(v, 0.0, 1.0)) for v in values})
        # Ensure we always explore the upper bound if mean < 1 and std > 0.
        if clean[-1] < 1.0 and step > 0:
            clean.append(1.0)
        candidates[cls] = clean
    return candidates


def coordinate_descent_categorywise(
    samples_cat1: List[Dict[str, object]],
    samples_cat2: List[Dict[str, object]],
    category: str,
    baseline_metrics: Dict[str, float],
    baseline_score: float,
    base_thresholds: Dict[int, float],
    candidate_thresholds: Dict[int, List[float]],
    threshold_bins: Dict[int, Dict[str, object]],
    step_multiplier: float,
    o_id: int,
    other_category_thresholds: Dict[int, float],
    max_passes: int,
) -> Tuple[Dict[int, float], Dict[str, float], Dict[int, float]]:
    """Coordinate descent optimization for a specific category."""
    print(f"\n{'='*60}")
    print(f"Category {category}: Baseline (mean thresholds) score for cat_{category}: {baseline_metrics.get(f'cat_{category}', 0.0):.6f}")
    print(f"Starting mean+σ sweep optimization for category {category}")
    print(f"{'='*60}\n")

    thresholds = dict(base_thresholds)
    best_score = baseline_score
    best_metrics = baseline_metrics.copy()

    # Only optimize classes that have candidates
    class_ids = [cls for cls in sorted(candidate_thresholds.keys()) if cls != o_id and candidate_thresholds.get(cls)]

    for pass_idx in range(max_passes):
        improved = False
        progress = tqdm(class_ids, desc=f"Cat {category} Pass {pass_idx + 1}/{max_passes}", leave=False)
        for cls in progress:
            candidates = candidate_thresholds.get(cls)
            if not candidates:
                continue
            current = thresholds.get(cls, 0.0)
            best_local_score = best_score
            best_local_metrics = best_metrics
            best_local_threshold = current

            for candidate in candidates:
                if abs(candidate - current) < 1e-9:
                    continue
                thresholds[cls] = candidate

                # Evaluate with category-specific thresholds
                if category == "1":
                    score, metrics = evaluate_thresholds_categorywise(
                        samples_cat1, samples_cat2, thresholds, other_category_thresholds, o_id
                    )
                else:
                    score, metrics = evaluate_thresholds_categorywise(
                        samples_cat1, samples_cat2, other_category_thresholds, thresholds, o_id
                    )

                # We optimize for the specific category score
                cat_score = metrics.get(f"cat_{category}", 0.0)
                if cat_score > best_local_score + 1e-9:
                    best_local_score = cat_score
                    best_local_metrics = metrics
                    best_local_threshold = candidate
                    progress.set_postfix({"cls": cls, f"cat_{category}": f"{cat_score:.5f}"})

            thresholds[cls] = best_local_threshold
            if best_local_threshold != current:
                improved = True
                best_score = best_local_score
                best_metrics = best_local_metrics

        if not improved:
            break

    # Calculate k values for each class
    k_values = {}
    for cls, threshold in thresholds.items():
        stats = threshold_bins.get(cls, {})
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 0.0))

        if std > 1e-9:
            step = std * step_multiplier
            k = (threshold - mean) / step
            k_values[cls] = k
        else:
            k_values[cls] = 0.0

    return thresholds, best_metrics, k_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine thresholds with mean+σ or percentile sweep, category-wise.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        dest="model_dir",
        required=True,
        help="Directory containing saved model weights",
    )
    parser.add_argument("--threshold_bins", type=Path, default=None, help="Optional path to precomputed threshold bins")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold index")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for the stratified split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device identifier")
    parser.add_argument("--max_passes", type=int, default=2, help="Max coordinate descent passes")
    parser.add_argument("--output_json", type=Path, default=Path("refined_thresholds_mean_sd_categorywise.json"), help="Where to save refined thresholds")
    parser.add_argument("--data_path", type=Path, default=Path("data/Tagged_Titles_Train.tsv"), help="Path to tagged training data")
    parser.add_argument(
        "--candidate_method",
        type=str,
        default="mean_sigma",
        choices=["mean_sigma", "percentile"],
        help="Method for generating threshold candidates: 'mean_sigma' (mean+k*sigma) or 'percentile' (percentile-based)",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=30,
        help="Maximum number of candidates per class (only for percentile method)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    global df1, label_list, label2id, id2label, category_to_valid_aspects
    df1, label_list, label2id, id2label, category_to_valid_aspects = build_label_resources(
        args.data_path,
        num_folds=args.num_folds,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("Category to valid aspects mapping:")
    print(f"{'='*60}")
    for cat, aspects in sorted(category_to_valid_aspects.items()):
        print(f"Category {cat}: {len(aspects)} aspects")
        print(f"  {sorted(aspects)[:10]}..." if len(aspects) > 10 else f"  {sorted(aspects)}")
    print()

    valid_df = df1[df1["fold"] == args.fold]
    if valid_df.empty:
        raise ValueError("Validation dataframe is empty. Check fold index.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    config_id2label = getattr(model.config, "id2label", None)
    if config_id2label:
        id2label = {int(k): v for k, v in config_id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        label_list = [id2label[idx] for idx in sorted(id2label)]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    samples = build_samples(valid_df, model, tokenizer, device, category_to_valid_aspects, label2id)

    # Split samples by category
    samples_cat1 = [s for s in samples if str(s["category"]) == "1"]
    samples_cat2 = [s for s in samples if str(s["category"]) == "2"]

    print(f"\nSample distribution: Category 1: {len(samples_cat1)}, Category 2: {len(samples_cat2)}")

    # Get valid class IDs for each category
    valid_ids_cat1 = get_valid_class_ids_for_category("1", category_to_valid_aspects, label2id)
    valid_ids_cat2 = get_valid_class_ids_for_category("2", category_to_valid_aspects, label2id)

    print(f"\n{'='*60}")
    print("Valid class IDs per category:")
    print(f"{'='*60}")
    print(f"Category 1: {len(valid_ids_cat1)} valid classes")
    print(f"Category 2: {len(valid_ids_cat2)} valid classes")
    print()

    # threshold_bins_path = args.threshold_bins or Path(f"base_threshold_bins_fold{args.fold}.npy")

    # if threshold_bins_path.exists():
    #     threshold_bins = np.load(threshold_bins_path, allow_pickle=True).item()
    #     print(f"Loaded threshold bins from {threshold_bins_path}")
    # else:
    raw_thresholds, threshold_bins = compute_threshold_statistics(samples, len(label_list))
    # threshold_bins_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(threshold_bins_path, threshold_bins, allow_pickle=True)
    # raw_path = threshold_bins_path.parent / f"base_thresholds_fold{args.fold}.npy"
    # np.save(raw_path, raw_thresholds, allow_pickle=True)
    # print(f"Saved base thresholds to {raw_path}")
    # print(f"Saved threshold bins to {threshold_bins_path}")

    # args.threshold_bins = threshold_bins_path

    o_id = label2id.get("O", 0)

    # Baseline 1: No thresholds (raw argmax predictions)
    print(f"\n{'='*60}")
    print("Evaluating Baseline 1: No thresholds (raw argmax predictions)")
    print(f"{'='*60}")
    no_thresholds = {cls: 0.0 for cls in range(len(label_list))}
    baseline_1_score, baseline_1_metrics = evaluate_thresholds(samples, no_thresholds, o_id)
    print(f"Baseline 1 overall score: {baseline_1_score:.6f}")
    print("Per-category scores:")
    for key, value in sorted(baseline_1_metrics.items()):
        if key != "overall_score":
            print(f"  {key}: {value:.6f}")

    # Baseline 2: Mean thresholds per class
    base_thresholds = {int(cls): float(stats.get("mean", 0.0)) for cls, stats in threshold_bins.items()}
    baseline_2_score, baseline_2_metrics = evaluate_thresholds(samples, base_thresholds, o_id)
    print(f"\n{'='*60}")
    print(f"Baseline 2 (mean thresholds) overall score: {baseline_2_score:.6f}")
    print("Per-category scores:")
    for key, value in sorted(baseline_2_metrics.items()):
        if key != "overall_score":
            print(f"  {key}: {value:.6f}")

    step_multiplier = 0.025

    # Compute threshold statistics and candidates for each category separately
    per_class_scores_cat1, threshold_bins_cat1 = compute_threshold_statistics(
        samples_cat1, len(label_list), category="1"
    )
    per_class_scores_cat2, threshold_bins_cat2 = compute_threshold_statistics(
        samples_cat2, len(label_list), category="2"
    )

    # Generate candidates based on selected method
    if args.candidate_method == "percentile":
        print(f"\n{'='*60}")
        print(f"Using PERCENTILE-based candidate generation (max_candidates={args.max_candidates})")
        print(f"{'='*60}\n")
        candidate_thresholds_cat1 = build_candidate_thresholds_percentile(
            per_class_scores_cat1, hi_focus=True, max_candidates=args.max_candidates
        )
        candidate_thresholds_cat2 = build_candidate_thresholds_percentile(
            per_class_scores_cat2, hi_focus=True, max_candidates=args.max_candidates
        )
    else:  # mean_sigma
        print(f"\n{'='*60}")
        print(f"Using MEAN+SIGMA candidate generation (step_multiplier={step_multiplier})")
        print(f"{'='*60}\n")
        candidate_thresholds_cat1 = build_candidate_thresholds_mean_sigma(
            threshold_bins_cat1, step_multiplier=step_multiplier
        )
        candidate_thresholds_cat2 = build_candidate_thresholds_mean_sigma(
            threshold_bins_cat2, step_multiplier=step_multiplier
        )

    # Report candidate counts
    total_cands_cat1 = sum(len(cands) for cands in candidate_thresholds_cat1.values())
    total_cands_cat2 = sum(len(cands) for cands in candidate_thresholds_cat2.values())
    print(f"Category 1: {len(candidate_thresholds_cat1)} classes, {total_cands_cat1} total candidates")
    print(f"Category 2: {len(candidate_thresholds_cat2)} classes, {total_cands_cat2} total candidates")

    # Initialize with mean thresholds for each category
    base_thresholds_cat1 = {int(cls): float(stats.get("mean", 0.0)) for cls, stats in threshold_bins_cat1.items()}
    base_thresholds_cat2 = {int(cls): float(stats.get("mean", 0.0)) for cls, stats in threshold_bins_cat2.items()}

    opt_start = time.time()

    # Optimize category 1
    refined_thresholds_cat1, _, k_values_cat1 = coordinate_descent_categorywise(
        samples_cat1,
        samples_cat2,
        category="1",
        baseline_metrics=baseline_2_metrics,
        baseline_score=baseline_2_metrics.get("cat_1", 0.0),
        base_thresholds=base_thresholds_cat1,
        candidate_thresholds=candidate_thresholds_cat1,
        threshold_bins=threshold_bins_cat1,
        step_multiplier=step_multiplier,
        o_id=o_id,
        other_category_thresholds=base_thresholds_cat2,
        max_passes=args.max_passes,
    )

    # Optimize category 2 (using refined thresholds from category 1)
    refined_thresholds_cat2, _, k_values_cat2 = coordinate_descent_categorywise(
        samples_cat1,
        samples_cat2,
        category="2",
        baseline_metrics=baseline_2_metrics,
        baseline_score=baseline_2_metrics.get("cat_2", 0.0),
        base_thresholds=base_thresholds_cat2,
        candidate_thresholds=candidate_thresholds_cat2,
        threshold_bins=threshold_bins_cat2,
        step_multiplier=step_multiplier,
        o_id=o_id,
        other_category_thresholds=refined_thresholds_cat1,
        max_passes=args.max_passes,
    )

    # Final evaluation with both refined thresholds
    _, final_metrics = evaluate_thresholds_categorywise(
        samples_cat1, samples_cat2, refined_thresholds_cat1, refined_thresholds_cat2, o_id
    )

    opt_time = time.time() - opt_start
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("Final metrics (after category-wise mean+σ sweep):")
    print(f"{'='*60}")
    for key, value in sorted(final_metrics.items()):
        if key == "overall_score":
            print(f"  overall_score: {value:.6f}")
        else:
            print(f"  {key}: {value:.6f}")

    print(f"\n{'='*60}")
    print("Category 1 - Optimal k values (threshold = mean + k * step_multiplier * std):")
    print(f"step_multiplier = {step_multiplier}")
    print(f"{'='*60}")
    for cls_id in sorted(k_values_cat1.keys()):
        if cls_id in valid_ids_cat1:
            label = id2label.get(cls_id, f"cls_{cls_id}")
            k = k_values_cat1[cls_id]
            stats = threshold_bins_cat1.get(cls_id, {})
            mean = float(stats.get("mean", 0.0))
            std = float(stats.get("std", 0.0))
            threshold = refined_thresholds_cat1[cls_id]
            print(f"  {label:20s}: k={k:6.2f}  (mean={mean:.4f}, std={std:.4f}, threshold={threshold:.4f})")

    print(f"\n{'='*60}")
    print("Category 2 - Optimal k values (threshold = mean + k * step_multiplier * std):")
    print(f"step_multiplier = {step_multiplier}")
    print(f"{'='*60}")
    for cls_id in sorted(k_values_cat2.keys()):
        if cls_id in valid_ids_cat2:
            label = id2label.get(cls_id, f"cls_{cls_id}")
            k = k_values_cat2[cls_id]
            stats = threshold_bins_cat2.get(cls_id, {})
            mean = float(stats.get("mean", 0.0))
            std = float(stats.get("std", 0.0))
            threshold = refined_thresholds_cat2[cls_id]
            print(f"  {label:20s}: k={k:6.2f}  (mean={mean:.4f}, std={std:.4f}, threshold={threshold:.4f})")

    print(f"\n{'='*60}")
    print("TIMING RESULTS:")
    print(f"{'='*60}")
    print(f"  Optimization time: {opt_time:.2f}s")
    print(f"  Total time:        {total_time:.2f}s")

    print(f"\nSummary:")
    print(f"  Baseline 1 (no thresholds):       {baseline_1_score:.6f}")
    print(f"  Baseline 2 (mean thresholds):     {baseline_2_score:.6f}")
    print(f"  Refined (category-wise):          {final_metrics['overall_score']:.6f}")
    print(f"    Category 1: {final_metrics.get('cat_1', 0.0):.6f}")
    print(f"    Category 2: {final_metrics.get('cat_2', 0.0):.6f}")

    payload = {
        "category_1": {
            "thresholds": {id2label[int(cls)]: float(val) for cls, val in refined_thresholds_cat1.items()},
            "k_values": {id2label[int(cls)]: float(k) for cls, k in k_values_cat1.items()},
            "valid_class_ids": [id2label[cls] for cls in valid_ids_cat1],
        },
        "category_2": {
            "thresholds": {id2label[int(cls)]: float(val) for cls, val in refined_thresholds_cat2.items()},
            "k_values": {id2label[int(cls)]: float(k) for cls, k in k_values_cat2.items()},
            "valid_class_ids": [id2label[cls] for cls in valid_ids_cat2],
        },
        "candidate_method": args.candidate_method,
        "step_multiplier": step_multiplier,
        "max_candidates": args.max_candidates if args.candidate_method == "percentile" else None,
        "final_metrics": final_metrics,
        "baseline_1_score": baseline_1_score,
        "baseline_1_metrics": baseline_1_metrics,
        "baseline_2_score": baseline_2_score,
        "baseline_2_metrics": baseline_2_metrics,
        "fold": args.fold,
        "optimization_time_seconds": opt_time,
        "total_time_seconds": total_time,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved refined category-wise thresholds to {args.output_json}")


if __name__ == "__main__":  # pragma: no cover
    main()
