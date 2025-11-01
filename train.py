"""Reorganized variant of train_new.py with clearer structure and grouping."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, Features, Sequence as HFSequence, Value
from peft import LoraConfig, TaskType, get_peft_model
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from torchcrf import CRF
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from annexure_preprocessing import convert_tagged_to_aspect
from eval_score import compute_competition_score


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a NER model with LoRA")
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Base model to use (default: microsoft/deberta-v3-small)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation_study/deberta-v3-base",
        help="Base directory to save the model artifacts (default: ablation_study/deberta-v3-base)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=256,
        help="LoRA rank parameter (default: 256)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=512,
        help="LoRA alpha parameter (default: 512)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout parameter (default: 0.1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--enable_lora",
        action="store_true",
        help="Enable LoRA fine-tuning (default: False)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (default: 0.0, range: 0.0-1.0)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for model heads",
    )
    parser.add_argument(
        "--use_crf",
        action="store_true",
        help="Enable CRF layer on top of the model",
    )
    parser.add_argument("--correct", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_dir", type=str, default="mod_data")
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index (0 to n_splits-1) to use as the validation split",
    )
    parser.add_argument(
        "--o_label_weight",
        type=float,
        default=1.0,
        help=(
            "Weight multiplier for 'O'-type labels (e.g., O, B-O, I-O) "
            "within the cross-entropy loss. 1.0 keeps uniform weights."
        ),
    )
    parser.add_argument(
        "--use_custom_label_smoothing",
        action="store_true",
        help=(
            "Enable custom label smoothing: 80%% on true label + 20%% on B-O for non-O labels, "
            "100%% on true label for B-O/I-O/O labels"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def stratified_kfold_split(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    sumdf = (
        df.groupby("Record Number")
        .agg(
            {
                "Category": "first",
                "Aspect Name": lambda x: x.mode()[0] if not x.mode().empty else "UNKNOWN",
            }
        )
        .reset_index()
    )
    sumdf["key"] = sumdf["Category"] + "_" + sumdf["Aspect Name"]
    vc = sumdf["key"].value_counts()
    rare = vc[vc < n_splits].index
    sumdf.loc[sumdf["key"].isin(rare), "key"] = "rare"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sumdf["fold"] = -1
    for fold_idx, (_, vidx) in enumerate(skf.split(sumdf, sumdf["key"])):
        sumdf.loc[vidx, "fold"] = fold_idx
    return df.merge(sumdf[["Record Number", "fold"]], on="Record Number", how="left")


def build_raw_examples(df: pd.DataFrame) -> pd.DataFrame:
    examples: List[Dict[str, object]] = []
    for record_id, group in df.groupby("Record Number", sort=False):
        text = group["Title"].iloc[0]
        spans = list(zip(group["Aspect Value"], group["Aspect Name"]))
        examples.append({"record_id": record_id, "text": text, "spans": spans})
    return pd.DataFrame(examples)


def create_tokenize_fn(
    tokenizer: AutoTokenizer, label2id: Dict[str, int]
) -> Callable[[Dict[str, Sequence[str]]], Dict[str, List[List[int]]]]:
    def tokenize_and_align_offsets(examples: Dict[str, Sequence[str]]):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            return_offsets_mapping=True,
        )
        all_labels: List[List[int]] = []
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            labels = [-100] * len(offsets)
            text = examples["text"][i]
            for val, asp in examples["spans"][i]:
                for match in re.finditer(re.escape(val), text):
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
            all_labels.append(labels)
        tokenized["labels"] = all_labels
        tokenized.pop("offset_mapping")
        return tokenized

    return tokenize_and_align_offsets


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class TokenClassifierWithCRF(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        base_model.config.hidden_dropout_prob = dropout_prob
        base_model.config.attention_probs_dropout_prob = dropout_prob
        if hasattr(base_model, "dropout"):
            base_model.dropout = nn.Dropout(dropout_prob)
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_labels),
        )
        self.crf = CRF(num_labels, batch_first=True)
        self.id2label = id2label
        self.label2id = label2id
        self.o_id = label2id["O"]

    def forward(self, input_ids, attention_mask=None, labels=None):  # type: ignore[override]
        seq_out = self.base(input_ids, attention_mask=attention_mask)[0]
        emissions = self.classifier(seq_out)
        mask = attention_mask.bool() if attention_mask is not None else None
        if labels is not None:
            tags = labels.clone()
            tags = torch.where(tags == -100, self.o_id, tags)
            log_likelihood = self.crf(emissions, tags, mask=mask, reduction="mean")
            return -log_likelihood, emissions
        return self.crf.decode(emissions, mask=mask)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def print_distribution_report(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, file_path: str | None = None
) -> None:
    lines: List[str] = []
    lines.append("=" * 50)
    lines.append("DATASET DISTRIBUTION ANALYSIS")
    lines.append("=" * 50)
    lines.append("")

    lines.append("1. CATEGORY DISTRIBUTION:")
    lines.append("-" * 30)
    train_cat = train_df.groupby("Record Number")["Category"].first().value_counts()
    train_n = train_df["Record Number"].nunique()
    lines.append(f"Training set categories (unique records: {train_n}):")
    for cat, cnt in train_cat.items():
        pct = cnt / train_n * 100
        lines.append(f"  {cat}: {cnt} records ({pct:.1f}%)")
    lines.append("")

    valid_cat = valid_df.groupby("Record Number")["Category"].first().value_counts()
    valid_n = valid_df["Record Number"].nunique()
    lines.append(f"Validation set categories (unique records: {valid_n}):")
    for cat, cnt in valid_cat.items():
        pct = cnt / valid_n * 100
        lines.append(f"  {cat}: {cnt} records ({pct:.1f}%)")
    lines.append("")

    lines.append("2. ASPECT NAME DISTRIBUTION — by occurrence")
    lines.append("-" * 40)
    train_aspect_inst = train_df["Aspect Name"].value_counts()
    train_inst_n = train_aspect_inst.sum()
    lines.append(f"Training set aspect mentions (total: {train_inst_n}):")
    for asp, cnt in train_aspect_inst.items():
        pct = cnt / train_inst_n * 100
        lines.append(f"  {asp}: {cnt} mentions ({pct:.1f}%)")
    lines.append("")

    valid_aspect_inst = valid_df["Aspect Name"].value_counts()
    valid_inst_n = valid_aspect_inst.sum()
    lines.append(f"Validation set aspect mentions (total: {valid_inst_n}):")
    for asp, cnt in valid_aspect_inst.items():
        pct = cnt / valid_inst_n * 100
        lines.append(f"  {asp}: {cnt} mentions ({pct:.1f}%)")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if file_path:
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"Report written to {file_path}")


def extract_spans(
    seq: Sequence[str],
    positions: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
    text: str,
) -> List[Tuple[str, str]]:
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


def build_seqeval_metrics_fn(
    label_list: Sequence[str], label2id: Dict[str, int]
) -> Callable[[Trainer.EvalPrediction], Dict[str, float]]:
    def compute_metrics_old(p):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids

        str_preds: List[List[str]] = []
        str_labels: List[List[str]] = []
        for pred_seq, label_seq in zip(preds, labels):
            seq_p: List[str] = []
            seq_l: List[str] = []
            for p_id, l_id in zip(pred_seq, label_seq):
                if l_id == -100:
                    continue
                seq_p.append(label_list[p_id])
                seq_l.append(label_list[l_id])
            str_preds.append(seq_p)
            str_labels.append(seq_l)

        precision = precision_score(str_labels, str_preds)
        recall = recall_score(str_labels, str_preds)
        f1 = f1_score(str_labels, str_preds)
        accuracy = accuracy_score(str_labels, str_preds)

        flat_preds: List[int] = []
        flat_labels: List[int] = []
        for pred_seq, label_seq in zip(preds, labels):
            for p_id, l_id in zip(pred_seq, label_seq):
                if l_id == -100:
                    continue
                flat_preds.append(p_id)
                flat_labels.append(l_id)

        o_label_id = label2id["O"]
        entity_labels = [i for i in range(len(label_list)) if i != o_label_id]
        if flat_labels and entity_labels:
            f0_2_weighted = fbeta_score(
                flat_labels,
                flat_preds,
                beta=0.2,
                average="weighted",
                labels=entity_labels,
                zero_division=0,
            )
        else:
            f0_2_weighted = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "f0.2": f0_2_weighted,
        }

    return compute_metrics_old


def build_competition_metric_fn(
    label_list: Sequence[str],
    tokenizer: AutoTokenizer,
    valid_texts: Sequence[str],
    valid_record_ids: Sequence[str],
    valid_cat: Sequence[str],
) -> Callable[[Trainer.EvalPrediction], Dict[str, float]]:
    encoded_valid = tokenizer(
        list(valid_texts),
        truncation=True,
        return_offsets_mapping=True,
        padding=False,
    )
    offset_mappings: List[List[Tuple[int, int]]] = [
        [(int(start), int(end)) for start, end in offsets]
        for offsets in encoded_valid["offset_mapping"]
    ]

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids

        true_entities: List[Dict[str, str]] = []
        pred_entities: List[Dict[str, str]] = []

        if len(valid_record_ids) != len(preds) or len(valid_cat) != len(preds):
            raise ValueError(
                "Mismatch between validation metadata and prediction batches"
            )

        for i, (pred_seq, label_seq) in enumerate(zip(preds, labels)):
            seq_p: List[str] = []
            seq_l: List[str] = []
            token_idxs: List[int] = []
            for idx, (p_id, l_id) in enumerate(zip(pred_seq, label_seq)):
                if l_id == -100:
                    continue
                seq_p.append(label_list[p_id])
                seq_l.append(label_list[l_id])
                token_idxs.append(idx)

            category = valid_cat[i]
            record_id = valid_record_ids[i]
            offsets = offset_mappings[i]
            text = valid_texts[i]

            for aspect, span in extract_spans(seq_l, token_idxs, offsets, text):
                if aspect != "O":
                    true_entities.append(
                        {
                            "record_id": str(record_id),
                            "category": category,
                            "aspect_name": aspect,
                            "span": span,
                        }
                    )
            for aspect, span in extract_spans(seq_p, token_idxs, offsets, text):
                if aspect != "O":
                    pred_entities.append(
                        {
                            "record_id": str(record_id),
                            "category": category,
                            "aspect_name": aspect,
                            "span": span,
                        }
                    )

        comp = compute_competition_score(true_entities, pred_entities, beta=0.2)
        metrics: Dict[str, float] = {"f0.2": comp["overall_score"]}
        metrics.update({f"competition_cat_{cat}": score for cat, score in comp["per_category"].items()})
        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def build_label_maps(
    base_df: pd.DataFrame, train_df: pd.DataFrame
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    aspects = sorted(base_df["Aspect Name"].unique())
    label_list = ["O"] + [f"{prefix}-{aspect}" for aspect in aspects for prefix in ("B", "I")]
    print(train_df["Aspect Name"].unique().tolist())
    if "U" in train_df["Aspect Name"].unique().tolist():
        print("Warning: 'U' label found in training data, adding to label list.")
        label_list.extend(["B-U", "I-U"])
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def build_label_weights(label_list: Sequence[str], o_weight: float) -> torch.Tensor | None:
    if o_weight == 1.0:
        return None

    weights = torch.ones(len(label_list), dtype=torch.float)
    adjusted = False
    for idx, label in enumerate(label_list):
        if label == "O" or label.endswith("-O"):
            weights[idx] = o_weight
            adjusted = True

    if not adjusted:
        return None
    return weights


def format_o_weight_identifier(weight: float) -> str:
    weight_float = float(weight)
    if weight_float.is_integer():
        return str(int(weight_float))
    text = f"{weight_float:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg_").replace(".", "_")


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def select_lora_target_modules(model_id: str) -> Sequence[str]:
    lower_model = model_id.lower()
    if "deberta" in lower_model:
        return ["query_proj", "key_proj", "value_proj", "dense"]
    if "bert" in lower_model:
        return ["query", "key", "value", "output.dense"]
    if "mistral" in lower_model or "llama" in lower_model:
        return [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return ["query_proj", "key_proj", "value_proj", "dense"]


# ---------------------------------------------------------------------------
# Trainer extensions
# ---------------------------------------------------------------------------


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args,
        label_weights: torch.Tensor,
        label2id: Dict[str, int] | None = None,
        use_custom_label_smoothing: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights
        self.label2id = label2id
        self.use_custom_label_smoothing = use_custom_label_smoothing

        # Get B-O index for label smoothing
        self.b_o_idx = None
        self.o_idx = None
        self.i_o_idx = None
        if label2id is not None and use_custom_label_smoothing:
            self.b_o_idx = label2id.get("B-O")
            self.o_idx = label2id.get("O")
            self.i_o_idx = label2id.get("I-O")
            print(f"Custom label smoothing enabled: B-O idx={self.b_o_idx}, O idx={self.o_idx}, I-O idx={self.i_o_idx}")

    def apply_custom_label_smoothing(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply custom label smoothing:
        - For non-O labels (not B-O, I-O, O): 80% on true label, 20% on B-O
        - For B-O, I-O, O labels: 100% on true label (no smoothing)
        """
        num_classes = logits.size(-1)
        device = logits.device

        # Create one-hot encoded labels
        # Shape: (batch_size * seq_len, num_classes)
        labels_flat = labels.view(-1)
        valid_mask = labels_flat != -100

        # Initialize smoothed labels with zeros
        smoothed_labels = torch.zeros(labels_flat.size(0), num_classes, device=device)

        if self.b_o_idx is None:
            # Fallback: no smoothing
            smoothed_labels[valid_mask] = torch.nn.functional.one_hot(
                labels_flat[valid_mask], num_classes=num_classes
            ).float()
            return smoothed_labels

        # Get valid labels
        valid_labels = labels_flat[valid_mask]

        # Determine which labels are O-type (B-O, I-O, O)
        o_type_mask = torch.zeros_like(valid_labels, dtype=torch.bool)
        if self.b_o_idx is not None:
            o_type_mask |= (valid_labels == self.b_o_idx)
        if self.i_o_idx is not None:
            o_type_mask |= (valid_labels == self.i_o_idx)
        if self.o_idx is not None:
            o_type_mask |= (valid_labels == self.o_idx)

        # For O-type labels: 100% on true label
        o_type_indices = torch.where(valid_mask)[0][o_type_mask]
        if len(o_type_indices) > 0:
            smoothed_labels[o_type_indices] = torch.nn.functional.one_hot(
                labels_flat[o_type_indices], num_classes=num_classes
            ).float()

        # For non-O-type labels: 80% on true label, 20% on B-O
        non_o_type_mask = ~o_type_mask
        non_o_type_indices = torch.where(valid_mask)[0][non_o_type_mask]
        if len(non_o_type_indices) > 0:
            # Start with one-hot (100% on true label)
            one_hot = torch.nn.functional.one_hot(
                labels_flat[non_o_type_indices], num_classes=num_classes
            ).float()

            # Apply smoothing: 80% on true label, 20% on B-O
            smoothed = one_hot * 0.8
            smoothed[:, self.b_o_idx] += 0.2

            smoothed_labels[non_o_type_indices] = smoothed

        return smoothed_labels

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if labels is not None:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            if self.use_custom_label_smoothing and self.b_o_idx is not None:
                # Apply custom label smoothing
                smoothed_labels = self.apply_custom_label_smoothing(logits, labels)

                # Compute log softmax
                log_probs = torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)

                # Compute cross entropy with smoothed labels
                loss = -(smoothed_labels * log_probs).sum(dim=-1)

                # Apply label weights if provided
                if self.label_weights is not None:
                    labels_flat = labels.view(-1)
                    valid_mask = labels_flat != -100
                    weights = self.label_weights.to(logits.device)[labels_flat[valid_mask]]
                    loss_valid = loss[valid_mask]
                    loss = (loss_valid * weights).sum() / weights.sum()
                else:
                    valid_mask = labels.view(-1) != -100
                    loss = loss[valid_mask].mean()
            else:
                # Standard cross entropy with label weights
                loss_fct = nn.CrossEntropyLoss(
                    weight=self.label_weights.to(logits.device),
                    ignore_index=-100,
                )
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            # Fallback to default behaviour if labels missing
            loss = outputs.loss if hasattr(outputs, "loss") else None

        if return_outputs:
            return loss, outputs
        return loss


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    model_id = args.model_id

    weight_tag = format_o_weight_identifier(args.o_label_weight)
    output_base = Path(args.output_dir)
    final_output_dir = output_base / f"o_weight_{weight_tag}" / f"fold{args.fold}"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(final_output_dir)
    print(f"Saving model artifacts to {args.output_dir}")

    df_tagged = convert_tagged_to_aspect("data/Tagged_Titles_Train.tsv")
    df_with_folds = stratified_kfold_split(df_tagged, n_splits=args.num_folds, random_state=args.seed)

    print(f"Fold {args.fold} of {args.num_folds}:")
    train_df = df_with_folds[df_with_folds["fold"] != args.fold]
    valid_df = df_with_folds[df_with_folds["fold"] == args.fold]
    print("Unique Record Numbers in training data:", len(train_df["Record Number"].unique()))
    print("Unique Record Numbers in validation data:", len(valid_df["Record Number"].unique()))

    # Get record IDs from the fold-based train split
    train_record_ids_from_fold = train_df["Record Number"].unique()

    # Load relabeled data and filter by the fold's record IDs
    train_df = pd.read_csv(
        "relabeled/relabeled_any_one.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    train_df = train_df[train_df["Record Number"].isin(train_record_ids_from_fold)]
    print(train_df.head())
    print(train_df["fold"].unique())
    print("Unique Record Numbers in relabeled training data:", len(train_df["Record Number"].unique()))

    train_record_ids = set(train_df["Record Number"].unique())
    print(f"Train Record Numbers: {len(train_record_ids)} unique IDs")
    valid_record_ids = set(valid_df["Record Number"].unique())
    print(f"Validation Record Numbers: {len(valid_record_ids)} unique IDs")
    overlap = train_record_ids.intersection(valid_record_ids)
    assert (
        len(overlap) == 0
    ), f"Found {len(overlap)} overlapping Record Numbers between train and validation sets: {list(overlap)[:10]}..."
    print(
        f"✓ No overlap between train ({len(train_record_ids)} records) and validation ({len(valid_record_ids)} records) sets"
    )

    raw_train = build_raw_examples(train_df)
    raw_valid = build_raw_examples(valid_df)

    category_lookup = (
        valid_df.groupby("Record Number", sort=False)["Category"].first().to_dict()
    )
    valid_record_ids = raw_valid["record_id"].astype(str).tolist()
    valid_cat = [category_lookup[rid] for rid in raw_valid["record_id"]]
    valid_texts = raw_valid["text"].tolist()
    print(valid_cat)

    label_list, label2id, id2label = build_label_maps(df_tagged, train_df)
    label_weights = build_label_weights(label_list, args.o_label_weight)
    if label_weights is not None:
        print("Using non-uniform label weights:")
        for label, weight in zip(label_list, label_weights.tolist()):
            print(f"  {label}: {weight:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenize_fn = create_tokenize_fn(tokenizer, label2id)

    if "deberta" in model_id.lower():
        features = Features(
            {
                "input_ids": HFSequence(Value("int64")),
                "token_type_ids": HFSequence(Value("int64")),
                "attention_mask": HFSequence(Value("int64")),
                "labels": HFSequence(Value("int64")),
            }
        )
    else:
        features = Features(
            {
                "input_ids": HFSequence(Value("int64")),
                "attention_mask": HFSequence(Value("int64")),
                "labels": HFSequence(Value("int64")),
            }
        )

    ds_train = (
        Dataset.from_pandas(raw_train)
        .map(tokenize_fn, batched=True, remove_columns=["record_id", "text", "spans"])
        .cast(features)
    )
    ds_valid = (
        Dataset.from_pandas(raw_valid)
        .map(tokenize_fn, batched=True, remove_columns=["record_id", "text", "spans"])
        .cast(features)
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = build_competition_metric_fn(
        label_list,
        tokenizer,
        valid_texts,
        valid_record_ids,
        valid_cat,
    )

    config = AutoConfig.from_pretrained(
        model_id,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    base_model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        config=config,
    )
    if label_weights is not None and args.use_crf:
        print(
            "Label weight adjustment requested, but CRF is enabled. Skipping weight application."
        )
    print(base_model)

    if args.use_crf:
        model = TokenClassifierWithCRF(
            base_model,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            dropout_prob=args.dropout,
        )
    else:
        if hasattr(base_model, "dropout"):
            base_model.dropout = nn.Dropout(args.dropout)
        model = base_model
    print(model)

    if args.enable_lora:
        print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        target_modules = select_lora_target_modules(model_id)
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        gradient_checkpointing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f0.2",
        save_total_limit=8,
        greater_is_better=True,
        logging_steps=50,
        load_best_model_at_end=True,
        seed=args.seed,
        push_to_hub=False,
        label_smoothing_factor=args.label_smoothing,
        run_name=f"o_weight_{weight_tag}",
    )

    print("Training configuration:")
    print(f"  Model: {args.model_id}")
    print(f"  LoRA enabled: {args.enable_lora}")
    if args.enable_lora:
        print(f"  LoRA r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  O-label weight: {args.o_label_weight}")

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if label_weights is not None and not args.use_crf:
        trainer = WeightedTrainer(
            label_weights=label_weights,
            label2id=label2id,
            use_custom_label_smoothing=args.use_custom_label_smoothing,
            **trainer_kwargs
        )
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.train()

    # Print best validation score
    if trainer.state.best_metric is not None:
        best_score = trainer.state.best_metric
        best_epoch = trainer.state.best_model_checkpoint
        print(f"\n{'='*60}")
        print(f"BEST VALIDATION SCORE")
        print(f"{'='*60}")
        print(f"  Metric (f0.2): {best_score:.6f}")
        if best_epoch:
            print(f"  Best checkpoint: {best_epoch}")
        print(f"{'='*60}\n")

    # Print all evaluation scores from history
    if hasattr(trainer.state, "log_history"):
        eval_scores = [
            entry for entry in trainer.state.log_history
            if "eval_f0.2" in entry
        ]
        if eval_scores:
            print(f"Validation scores per epoch:")
            for i, entry in enumerate(eval_scores, 1):
                epoch = entry.get("epoch", i)
                score = entry.get("eval_f0.2", 0.0)
                print(f"  Epoch {epoch}: f0.2 = {score:.6f}")
            print()

    if not args.use_crf:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
