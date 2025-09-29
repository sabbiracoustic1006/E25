#!/usr/bin/env python
"""Refine class-wise thresholds derived from inference_latest mean strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


import difflib
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


def compute_threshold_statistics(
    samples: List[Dict[str, object]],
    num_labels: int,
) -> Tuple[Dict[int, List[float]], Dict[int, Dict[str, object]]]:
    """Aggregate per-class confidence statistics from token predictions."""
    percentiles = np.linspace(0, 100, 101)
    per_class: Dict[int, List[float]] = {cls: [] for cls in range(num_labels)}

    for sample in samples:
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


def resolve_word_labels(tokens, labels, offset_mapping):
    word_preds = []
    cur_labels: List[str] = []
    cur_offsets: List[Tuple[int, int]] = []

    for token, label, (start, end) in zip(tokens, labels, offset_mapping):
        if token in {"[CLS]", "[SEP]"} or (start == 0 and end == 0):
            continue
        is_new_word = token.startswith("▁")
        if is_new_word and cur_offsets:
            final_label = next((lab for lab in cur_labels if lab != "O"), "O")
            if "B-O" in cur_labels or "I-O" in cur_labels:
                final_label = "B-O"
            word_preds.append(
                {
                    "entity": final_label,
                    "start": cur_offsets[0][0],
                    "end": cur_offsets[-1][1],
                }
            )
            cur_labels = []
            cur_offsets = []

        cur_labels.append(label)
        cur_offsets.append((start, end))

    if cur_offsets:
        final_label = next((lab for lab in cur_labels if lab != "O"), "O")
        if "B-O" in cur_labels or "I-O" in cur_labels:
            final_label = "B-O"
        word_preds.append(
            {
                "entity": final_label,
                "start": cur_offsets[0][0],
                "end": cur_offsets[-1][1],
            }
        )

    return word_preds


def merge_spans(preds: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    current = None
    for token in preds:
        label = token["entity"]
        if label.startswith("B-"):
            if current:
                merged.append(current)
            current = {
                "aspect_name": label[2:],
                "start": token["start"],
                "end": token["end"],
            }
        elif label.startswith("I-") and current and label[2:] == current["aspect_name"]:
            current["end"] = token["end"]
        else:
            if current:
                merged.append(current)
                current = None
    if current:
        merged.append(current)
    return merged


def apply_thresholds(
    probs: torch.Tensor,
    base_preds: torch.Tensor,
    thresholds: Dict[int, float],
    o_id: int,
) -> torch.Tensor:
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


def build_samples(valid_df, model, tokenizer, device) -> List[Dict[str, object]]:
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
        encoded = {k: v.to(device) for k, v in encoded.items()}
        offsets = encoded.pop("offset_mapping").cpu()[0].tolist()

        with torch.no_grad():
            logits = model(**encoded).logits.softmax(dim=-1).cpu()[0]

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"].cpu()[0], skip_special_tokens=False)

        samples.append(
            {
                "record_id": rid,
                "category": category,
                "text": text,
                "gold": gold,
                "probs": logits.numpy(),
                "tokens": tokens,
                "offsets": offsets,
            }
        )
    return samples


def evaluate_thresholds(
    samples: List[Dict[str, object]],
    thresholds: Dict[int, float],
    o_id: int,
) -> Tuple[float, Dict[str, float]]:
    all_targets: List[Tuple[str, str, str]] = []
    all_preds_with_cat: List[Tuple[str, str, str]] = []

    for sample in samples:
        probs = torch.from_numpy(sample["probs"]).unsqueeze(0)
        base_preds = probs.argmax(dim=-1)
        filtered = apply_thresholds(probs, base_preds, thresholds, o_id).squeeze(0).numpy()

        pred_labels = [id2label[int(idx)] for idx in filtered]
        word_preds = resolve_word_labels(sample["tokens"], pred_labels, sample["offsets"])
        spans = merge_spans(word_preds)

        pred_records = []
        text = sample["text"]
        category = sample["category"]
        gold = sample["gold"]
        gold_vals = {val for _, _, val in gold}

        for span in spans:
            aspect = span["aspect_name"].strip()
            start, end = span["start"], span["end"]
            value = text[start:end].strip()
            if not value:
                continue
            if aspect == "U":
                aspect = "O"
            if value not in gold_vals:
                matches = difflib.get_close_matches(value, gold_vals, n=1, cutoff=0.0)
                if matches:
                    closest_val = matches[0]
                    gold_aspect = next(gasp for _, gasp, gval in gold if gval == closest_val)
                    # keep original category/aspect but note mismatch (aligns with inference_latest behaviour)
                    _ = gold_aspect  # placeholder to emphasise usage
            pred_records.append((category, aspect, value))

        all_targets.extend(gold)
        all_preds_with_cat.extend(pred_records)

    comp = compute_competition_score(all_targets, all_preds_with_cat, beta=0.2)
    metrics = {"overall_score": comp["overall_score"]}
    metrics.update({f"cat_{cat}": score for cat, score in comp["per_category"].items()})
    return comp["overall_score"], metrics


def build_candidate_thresholds(threshold_bins: Dict[int, Dict[str, object]]) -> Dict[int, List[float]]:
    candidates: Dict[int, List[float]] = {}
    for cls, stats in threshold_bins.items():
        values = [float(stats.get("mean", 0.0)), float(stats.get("median", 0.0))]
        if stats.get("count", 0) > 0:
            values.extend([float(stats.get("min", 0.0)), float(stats.get("max", 0.0))])
            bins = stats.get("bins", [])
            percentiles = stats.get("percentiles", [])
            for target in (10, 25, 40, 50, 60, 75, 90):
                if len(bins) == len(percentiles) and len(bins) > 0:
                    idx = int(np.argmin(np.abs(np.array(percentiles) - target)))
                    values.append(float(bins[idx]))
        clean = sorted({float(np.clip(v, 0.0, 1.0)) for v in values})
        candidates[cls] = clean
    return candidates


def coordinate_descent(
    samples: List[Dict[str, object]],
    base_thresholds: Dict[int, float],
    candidate_thresholds: Dict[int, List[float]],
    o_id: int,
    max_passes: int,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    thresholds = dict(base_thresholds)
    best_score, best_metrics = evaluate_thresholds(samples, thresholds, o_id)
    print(f"Baseline overall score: {best_score:.6f}")

    class_ids = [cls for cls in thresholds.keys() if cls != o_id]

    for pass_idx in range(max_passes):
        improved = False
        progress = tqdm(class_ids, desc=f"Pass {pass_idx + 1}/{max_passes}", leave=False)
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
                score, metrics = evaluate_thresholds(samples, thresholds, o_id)
                if score > best_local_score + 1e-9:
                    best_local_score = score
                    best_local_metrics = metrics
                    best_local_threshold = candidate
                    progress.set_postfix({"cls": cls, "score": f"{score:.5f}"})

            thresholds[cls] = best_local_threshold
            if best_local_threshold != current:
                improved = True
                best_score = best_local_score
                best_metrics = best_local_metrics

        if not improved:
            break

    return thresholds, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine thresholds derived from inference_latest.")
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
    parser.add_argument("--output_json", type=Path, default=Path("refined_thresholds.json"), help="Where to save refined thresholds")
    parser.add_argument("--data_path", type=Path, default=Path("data/Tagged_Titles_Train.tsv"), help="Path to tagged training data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global df1, label_list, label2id, id2label
    df1, label_list, label2id, id2label = build_label_resources(
        args.data_path,
        num_folds=args.num_folds,
        seed=args.seed,
    )

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

    samples = build_samples(valid_df, model, tokenizer, device)

    threshold_bins_path = args.threshold_bins or Path(f"base_threshold_bins_fold{args.fold}.npy")

    if threshold_bins_path.exists():
        threshold_bins = np.load(threshold_bins_path, allow_pickle=True).item()
    else:
        raw_thresholds, threshold_bins = compute_threshold_statistics(samples, len(label_list))
        threshold_bins_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(threshold_bins_path, threshold_bins, allow_pickle=True)
        raw_path = threshold_bins_path.parent / f"base_thresholds_fold{args.fold}.npy"
        np.save(raw_path, raw_thresholds, allow_pickle=True)
        print(f"Saved base thresholds to {raw_path}" )
        print(f"Saved threshold bins to {threshold_bins_path}")

    args.threshold_bins = threshold_bins_path

    base_thresholds = {int(cls): float(stats.get("mean", 0.0)) for cls, stats in threshold_bins.items()}

    candidate_thresholds = build_candidate_thresholds(threshold_bins)

    o_id = label2id.get("O", 0)
    refined_thresholds, metrics = coordinate_descent(
        samples,
        base_thresholds,
        candidate_thresholds,
        o_id=o_id,
        max_passes=args.max_passes,
    )

    print("Refined metrics:")
    for key, value in sorted(metrics.items()):
        if key == "overall_score":
            print(f"  overall_score: {value:.6f}")
        else:
            print(f"  {key}: {value:.6f}")

    payload = {
        "thresholds": {id2label[int(cls)]: float(val) for cls, val in refined_thresholds.items()},
        "metrics": metrics,
        "fold": args.fold,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved refined thresholds to {args.output_json}")


if __name__ == "__main__":  # pragma: no cover
    main()
