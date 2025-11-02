#!/usr/bin/env python
"""Evaluate on heldout set - copied from refine_thresholds_with_heldout.py with TSV output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from eval_score import compute_competition_score
from utils import convert_tagged_to_aspect
from train_with_heldout import stratified_kfold_split as stratified_kfold_split_with_heldout


label_list: List[str] = []
label2id: Dict[str, int] = {}
id2label: Dict[int, str] = {}
category_to_valid_aspects: Dict[str, Set[str]] = {}


def extract_category_to_valid_aspects(data_path: Path) -> Dict[str, Set[str]]:
    """Extract mapping from category to valid aspect names from training data."""
    df = pd.read_csv(data_path, sep="\t", encoding="utf-8")
    df_filtered = df[df["Tag"].notna() & (df["Tag"] != "") & (df["Tag"] != "O")]
    category_aspects: Dict[str, Set[str]] = {}
    for _, row in df_filtered.iterrows():
        cat = str(row["Category"])
        aspect = str(row["Tag"])
        if cat not in category_aspects:
            category_aspects[cat] = set()
        category_aspects[cat].add(aspect)
    return category_aspects


def get_valid_class_ids_for_category(category: str, cat_to_aspects: Dict[str, Set[str]], label2id: Dict[str, int]) -> Set[int]:
    """Get valid class IDs for a given category."""
    valid_ids = {label2id["O"]}
    valid_aspects = cat_to_aspects.get(category, set())
    for aspect in valid_aspects:
        for prefix in ("B", "I"):
            label = f"{prefix}-{aspect}"
            if label in label2id:
                valid_ids.add(label2id[label])
    return valid_ids


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
                "masked_probs": masked_probs.numpy(),
                "labels": labels,
                "offsets": offsets,
            }
        )
    return samples


def evaluate_thresholds(
    samples: List[Dict[str, object]],
    thresholds: Dict[int, float],
    o_id: int,
) -> Tuple[float, Dict[str, float], List[Dict], List[Dict]]:
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
    return comp["overall_score"], metrics, all_targets, all_preds


def evaluate_thresholds_categorywise(
    samples_cat1: List[Dict[str, object]],
    samples_cat2: List[Dict[str, object]],
    thresholds_cat1: Dict[int, float],
    thresholds_cat2: Dict[int, float],
    o_id: int,
) -> Tuple[float, Dict[str, float], List[Dict], List[Dict]]:
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
    return comp["overall_score"], metrics, all_targets, all_preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on heldout set with TSV output")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--heldout_path", type=Path, default=Path("data/heldout_set_ratio0.20.tsv"))
    parser.add_argument("--tagged_path", type=Path, default=Path("data/Tagged_Titles_Train.tsv"))
    parser.add_argument("--output_tsv", type=Path, required=True, help="Output TSV file path")
    parser.add_argument("--threshold_path", type=Path, default=None, help="JSON file with thresholds (optional)")
    parser.add_argument("--use_thresholds", action="store_true", help="Use threshold-based prediction")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fold", type=int, default=0, help="For validation mode")
    parser.add_argument("--eval_on_validation", action="store_true")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout_ratio", type=float, default=0.2)
    args = parser.parse_args()

    global label_list, label2id, id2label, category_to_valid_aspects

    # Load data
    if args.eval_on_validation:
        df_tagged = convert_tagged_to_aspect(str(args.tagged_path))
        HELDOUT_SEED = 42
        df_with_folds, _ = stratified_kfold_split_with_heldout(
            df_tagged,
            n_splits=args.num_folds,
            random_state=args.seed,
            heldout_ratio=args.heldout_ratio,
            heldout_seed=HELDOUT_SEED,
        )
        eval_df = df_with_folds[df_with_folds["fold"] == args.fold]
        print(f"Using validation fold {args.fold} with {len(eval_df['Record Number'].unique())} records")
    else:
        eval_df = pd.read_csv(args.heldout_path, sep="\t", dtype=str, keep_default_na=False)
        print(f"Loaded {len(eval_df['Record Number'].unique())} records from heldout set")

    category_to_valid_aspects = extract_category_to_valid_aspects(args.tagged_path)

    # Load model
    print(f"Loading model from {args.model_dir}")
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

    o_id = label2id.get("O", 0)

    # Build samples
    samples = build_samples(eval_df, model, tokenizer, device, category_to_valid_aspects, label2id)

    # Load thresholds if using threshold mode
    if args.use_thresholds:
        if not args.threshold_path:
            raise ValueError("--threshold_path is required when using --use_thresholds")

        print(f"\nLoading thresholds from {args.threshold_path}")
        import json
        with args.threshold_path.open("r", encoding="utf-8") as f:
            threshold_data = json.load(f)

        # Load category-specific thresholds
        thresholds_cat1 = {}
        for label_name, threshold in threshold_data["category_1"]["thresholds"].items():
            if label_name in label2id:
                thresholds_cat1[label2id[label_name]] = threshold

        thresholds_cat2 = {}
        for label_name, threshold in threshold_data["category_2"]["thresholds"].items():
            if label_name in label2id:
                thresholds_cat2[label2id[label_name]] = threshold

        print("\nEvaluating with category-wise thresholds...")
        samples_cat1 = [s for s in samples if s["category"] == "1"]
        samples_cat2 = [s for s in samples if s["category"] == "2"]
        score, metrics, all_targets, all_preds = evaluate_thresholds_categorywise(
            samples_cat1, samples_cat2, thresholds_cat1, thresholds_cat2, o_id
        )
    else:
        # Evaluate (argmax baseline)
        print("\nEvaluating with argmax (baseline)...")
        no_thresholds = {cls: 0.0 for cls in range(len(label_list))}
        score, metrics, all_targets, all_preds = evaluate_thresholds(samples, no_thresholds, o_id)

    mode_str = "THRESHOLD" if args.use_thresholds else "ARGMAX"
    print(f"\nResults ({mode_str} mode):")
    print(f"  Overall Score: {score:.6f}")
    for key, value in sorted(metrics.items()):
        if key != "overall_score":
            print(f"  {key}: {value:.6f}")

    # Write TSV
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8") as f:
        for pred in all_preds:
            f.write(f"{pred['record_id']}\t{pred['category']}\t{pred['aspect_name']}\t{pred['span']}\n")

    print(f"\nWrote {len(all_preds)} predictions to {args.output_tsv}")


if __name__ == "__main__":
    main()
