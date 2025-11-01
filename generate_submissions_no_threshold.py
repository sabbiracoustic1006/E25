#!/usr/bin/env python
"""Generate quiz submission TSV without applying thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils import convert_tagged_to_aspect


def merge_word_preds(tokens: List[str], labels: List[str], offsets: List[Tuple[int, int]]) -> List[Dict[str, int | str]]:
    word_preds: List[Dict[str, int | str]] = []
    current_labels: List[str] = []
    current_offsets: List[Tuple[int, int]] = []

    for token, label, (start, end) in zip(tokens, labels, offsets):
        if token in {"[CLS]", "[SEP]"} or (start == 0 and end == 0):
            continue
        is_new_word = token.startswith("▁")
        if is_new_word and current_offsets:
            final_label = next((lab for lab in current_labels if lab != "O"), "O")
            word_preds.append(
                {
                    "entity": final_label,
                    "start": current_offsets[0][0],
                    "end": current_offsets[-1][1],
                }
            )
            current_labels = []
            current_offsets = []

        current_labels.append(label)
        current_offsets.append((start, end))

    if current_offsets:
        final_label = next((lab for lab in current_labels if lab != "O"), "O")
        word_preds.append(
            {
                "entity": final_label,
                "start": current_offsets[0][0],
                "end": current_offsets[-1][1],
            }
        )

    return word_preds


def spans_from_words(word_preds: List[Dict[str, int | str]]) -> List[Dict[str, str | int]]:
    merged: List[Dict[str, str | int]] = []
    current = None
    for token in word_preds:
        label = token["entity"]
        if label.startswith("B-"):
            if current:
                merged.append(current)
            current = {
                "aspect": label[2:],
                "start": token["start"],
                "end": token["end"],
            }
        elif label.startswith("I-") and current and label[2:] == current["aspect"]:
            current["end"] = token["end"]
        else:
            if current:
                merged.append(current)
                current = None
    if current:
        merged.append(current)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EvalAI submission without thresholding.")
    parser.add_argument("--model_dir", type=Path, required=True, help="Directory with saved model")
    parser.add_argument("--tagged_path", type=Path, default=Path("data/Tagged_Titles_Train.tsv"), help="Tagged training TSV")
    parser.add_argument("--start_idx", type=int, default=5001, help="Start record number")
    parser.add_argument("--end_idx", type=int, default=30000, help="End record number")
    parser.add_argument("--output_file", type=Path, required=True, help="Output TSV path")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    label2id = {label: int(idx) for label, idx in model.config.label2id.items()}
    id2label = model.config.id2label
    label_list = [label for label, _ in sorted(label2id.items(), key=lambda kv: kv[1])]

    df_test = pd.read_csv("data/Listing_Titles.tsv", sep="\t", dtype=str, keep_default_na=False)
    df_test["Record Number"] = df_test["Record Number"].astype(int)
    df_test = df_test[(df_test["Record Number"] >= args.start_idx) & (df_test["Record Number"] <= args.end_idx)]
    df_test = df_test.sort_values("Record Number").reset_index(drop=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    submissions: List[Tuple[int, str, str, str]] = []

    train_df = convert_tagged_to_aspect(str(args.tagged_path))
    allowed_aspects = (
        train_df.groupby("Category")["Aspect Name"].unique().apply(lambda arr: {asp.strip() for asp in arr if asp}).to_dict()
    )

    for start in tqdm(range(0, len(df_test), args.batch_size), desc="Generating"):
        batch = df_test.iloc[start : start + args.batch_size]
        texts = batch["Title"].tolist()
        categories = [str(cat).strip() for cat in batch["Category"].tolist()]
        record_ids = batch["Record Number"].tolist()

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        offsets = encoded.pop("offset_mapping").cpu()
        input_ids_cpu = encoded["input_ids"].detach().cpu()

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            base_preds = probs.argmax(dim=-1)
            # NO THRESHOLDING - use base_preds directly

        for i, (pred_ids, offset_map, input_ids, text, category, record_id) in enumerate(
            zip(base_preds.cpu(), offsets, input_ids_cpu, texts, categories, record_ids)
        ):
            labels = [label_list[int(idx)] for idx in pred_ids]
            tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
            word_preds = merge_word_preds(tokens, labels, offset_map.tolist())
            spans = spans_from_words(word_preds)
            for span in spans:
                aspect = span["aspect"].strip()
                snippet = text[span["start"] : span["end"]].strip()
                if not snippet:
                    continue
                if aspect not in allowed_aspects.get(category, set()):
                    continue
                submissions.append((record_id, category, aspect, snippet))

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for record_id, category, aspect, value in submissions:
            handle.write(f"{record_id}\t{category}\t{aspect}\t{value}\n")

    print(f"Wrote {len(submissions)} predictions to {args.output_file}")


if __name__ == "__main__":  # pragma: no cover
    main()
