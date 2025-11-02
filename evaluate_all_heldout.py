#!/usr/bin/env python3
"""
Evaluate all model checkpoints on heldout set.
- Generates TSV files for both argmax and threshold modes
- Creates rank.txt with performance rankings
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import time
from datetime import timedelta


def find_model_configs() -> List[Dict[str, str]]:
    """Find all model configurations with threshold files."""
    configs = []

    base_dir = Path("multi_epoch_checkpoints_heldout")

    # Find all threshold JSON files
    for threshold_file in base_dir.glob("**/refined_thresholds_categorywise_fold*_epoch*_heldout.json"):
        # Parse filename: refined_thresholds_categorywise_fold{fold}_epoch{epoch}_heldout.json
        filename = threshold_file.name
        parts = filename.replace("refined_thresholds_categorywise_fold", "").replace("_heldout.json", "").split("_epoch")
        fold = int(parts[0])
        epoch = int(parts[1])

        # Only process epochs 5-10
        if epoch < 5 or epoch > 10:
            continue

        # Determine checkpoint step
        checkpoint_step = epoch * 100

        # Build model directory path
        # threshold_file is like: multi_epoch_checkpoints_heldout/model/lr_x_e_neg_y/o_weight_z/refined_thresholds...
        model_base = threshold_file.parent / "o_weight_1" / f"fold{fold}" / f"checkpoint-{checkpoint_step}"

        if not model_base.exists():
            print(f"Warning: Checkpoint not found: {model_base}", file=sys.stderr)
            continue

        # Get model name from path
        # Extract model name: deberta-v3-large, deberta-v3-small-round2, etc.
        path_parts = threshold_file.parts
        model_name = path_parts[1]  # multi_epoch_checkpoints_heldout/MODEL_NAME/...

        configs.append({
            "model_name": model_name,
            "fold": fold,
            "epoch": epoch,
            "checkpoint_step": checkpoint_step,
            "model_dir": str(model_base),
            "threshold_path": str(threshold_file),
        })

    return sorted(configs, key=lambda x: (x["model_name"], x["fold"], x["epoch"]))


def run_evaluation(
    model_dir: str,
    output_tsv: str,
    threshold_path: str = None,
    use_thresholds: bool = False,
    device: str = "cuda",
) -> Tuple[float, float, float]:
    """
    Run evaluation and extract scores.
    Returns: (overall_score, cat1_score, cat2_score)
    """
    cmd = [
        "python", "evaluate_heldout_final.py",
        "--model_dir", model_dir,
        "--output_tsv", output_tsv,
        "--device", device,
    ]

    if use_thresholds and threshold_path:
        cmd.extend(["--threshold_path", threshold_path, "--use_thresholds"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}", file=sys.stderr)
        return None, None, None

    # Parse output to extract scores
    overall_score = None
    cat1_score = None
    cat2_score = None

    for line in result.stdout.split("\n"):
        if "Overall Score:" in line:
            overall_score = float(line.split(":")[-1].strip())
        elif "cat_1:" in line:
            cat1_score = float(line.split(":")[-1].strip())
        elif "cat_2:" in line:
            cat2_score = float(line.split(":")[-1].strip())

    print(result.stdout)
    return overall_score, cat1_score, cat2_score


def generate_output_filename(config: Dict[str, str], mode: str) -> str:
    """Generate output TSV filename."""
    # Format: heldout_tsvs/{model_name}_fold{fold}_epoch{epoch}_{mode}.tsv
    model_name = config["model_name"]
    fold = config["fold"]
    epoch = config["epoch"]
    return f"heldout_tsvs/{model_name}_fold{fold}_epoch{epoch}_{mode}.tsv"


def save_intermediate_results(results: List[Dict], output_file: Path):
    """Save results immediately after each evaluation."""
    with output_file.open("w", encoding="utf-8") as f:
        f.write("# Intermediate Results (updated after each evaluation)\n")
        f.write("# Format: model_name | fold | epoch | mode | overall | cat1 | cat2 | tsv_file\n\n")

        for result in results:
            config = result["config"]
            # Argmax results
            argmax = result["argmax"]
            f.write(f"{config['model_name']:<30} | fold{config['fold']} | epoch{config['epoch']:2d} | argmax    | "
                   f"{argmax['overall']:.6f} | {argmax['cat1']:.6f} | {argmax['cat2']:.6f} | {argmax['tsv']}\n")
            # Threshold results
            threshold = result["threshold"]
            f.write(f"{config['model_name']:<30} | fold{config['fold']} | epoch{config['epoch']:2d} | threshold | "
                   f"{threshold['overall']:.6f} | {threshold['cat1']:.6f} | {threshold['cat2']:.6f} | {threshold['tsv']}\n")


def format_time(seconds: float) -> str:
    """Format seconds into readable time string."""
    return str(timedelta(seconds=int(seconds)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints on heldout set")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=Path, default=Path("heldout_tsvs"), help="Output directory for TSVs")
    parser.add_argument("--rank_file", type=Path, default=Path("rank.txt"), help="Output ranking file")
    parser.add_argument("--intermediate_file", type=Path, default=Path("evaluation_progress.txt"), help="Intermediate results file")
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all model configurations
    print("Finding model configurations...")
    configs = find_model_configs()
    print(f"Found {len(configs)} model configurations to evaluate")

    # Store results
    results = []

    # Timing
    start_time = time.time()
    eval_times = []

    # Evaluate each configuration
    for i, config in enumerate(configs, 1):
        iter_start = time.time()

        print(f"\n{'='*80}")
        print(f"Evaluating {i}/{len(configs)}: {config['model_name']} fold{config['fold']} epoch{config['epoch']}")
        print(f"{'='*80}\n")

        # Argmax mode
        argmax_tsv = generate_output_filename(config, "argmax")
        print(f"\n--- ARGMAX MODE ---")
        argmax_overall, argmax_cat1, argmax_cat2 = run_evaluation(
            model_dir=config["model_dir"],
            output_tsv=argmax_tsv,
            device=args.device,
        )

        # Threshold mode
        threshold_tsv = generate_output_filename(config, "threshold")
        print(f"\n--- THRESHOLD MODE ---")
        threshold_overall, threshold_cat1, threshold_cat2 = run_evaluation(
            model_dir=config["model_dir"],
            output_tsv=threshold_tsv,
            threshold_path=config["threshold_path"],
            use_thresholds=True,
            device=args.device,
        )

        results.append({
            "config": config,
            "argmax": {
                "overall": argmax_overall,
                "cat1": argmax_cat1,
                "cat2": argmax_cat2,
                "tsv": argmax_tsv,
            },
            "threshold": {
                "overall": threshold_overall,
                "cat1": threshold_cat1,
                "cat2": threshold_cat2,
                "tsv": threshold_tsv,
            },
        })

        # Save intermediate results immediately
        save_intermediate_results(results, args.intermediate_file)

        # Calculate and display time estimates
        iter_time = time.time() - iter_start
        eval_times.append(iter_time)
        avg_time = sum(eval_times) / len(eval_times)
        remaining = len(configs) - i
        estimated_remaining = avg_time * remaining
        elapsed = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"PROGRESS: {i}/{len(configs)} completed ({i/len(configs)*100:.1f}%)")
        print(f"Time for this evaluation: {format_time(iter_time)}")
        print(f"Average time per evaluation: {format_time(avg_time)}")
        print(f"Elapsed time: {format_time(elapsed)}")
        print(f"Estimated remaining time: {format_time(estimated_remaining)}")
        print(f"Estimated total time: {format_time(elapsed + estimated_remaining)}")
        print(f"Intermediate results saved to: {args.intermediate_file}")
        print(f"{'='*80}\n")

    # Generate rankings
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("ALL EVALUATIONS COMPLETE!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time per config: {format_time(total_time / len(configs))}")
    print("Generating final rankings...")
    print(f"{'='*80}\n")

    with args.rank_file.open("w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write("HELDOUT SET EVALUATION RANKINGS\n")
        f.write("="*100 + "\n\n")

        # Rank by argmax mode
        f.write("\n" + "="*100 + "\n")
        f.write("ARGMAX MODE RANKINGS (sorted by overall score)\n")
        f.write("="*100 + "\n\n")

        argmax_sorted = sorted(results, key=lambda x: x["argmax"]["overall"] or 0.0, reverse=True)
        f.write(f"{'Rank':<6} {'Model':<30} {'Fold':<6} {'Epoch':<6} {'Overall':<10} {'Cat1':<10} {'Cat2':<10} {'TSV File'}\n")
        f.write("-"*100 + "\n")

        for rank, result in enumerate(argmax_sorted, 1):
            config = result["config"]
            argmax = result["argmax"]
            f.write(f"{rank:<6} {config['model_name']:<30} {config['fold']:<6} {config['epoch']:<6} "
                   f"{argmax['overall']:.6f}  {argmax['cat1']:.6f}  {argmax['cat2']:.6f}  {argmax['tsv']}\n")

        # Rank by threshold mode
        f.write("\n\n" + "="*100 + "\n")
        f.write("THRESHOLD MODE RANKINGS (sorted by overall score)\n")
        f.write("="*100 + "\n\n")

        threshold_sorted = sorted(results, key=lambda x: x["threshold"]["overall"] or 0.0, reverse=True)
        f.write(f"{'Rank':<6} {'Model':<30} {'Fold':<6} {'Epoch':<6} {'Overall':<10} {'Cat1':<10} {'Cat2':<10} {'TSV File'}\n")
        f.write("-"*100 + "\n")

        for rank, result in enumerate(threshold_sorted, 1):
            config = result["config"]
            threshold = result["threshold"]
            f.write(f"{rank:<6} {config['model_name']:<30} {config['fold']:<6} {config['epoch']:<6} "
                   f"{threshold['overall']:.6f}  {threshold['cat1']:.6f}  {threshold['cat2']:.6f}  {threshold['tsv']}\n")

        # Summary statistics
        f.write("\n\n" + "="*100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*100 + "\n\n")

        # Best models
        best_argmax = argmax_sorted[0]
        best_threshold = threshold_sorted[0]

        f.write("Best Argmax Model:\n")
        f.write(f"  Model: {best_argmax['config']['model_name']}\n")
        f.write(f"  Fold: {best_argmax['config']['fold']}, Epoch: {best_argmax['config']['epoch']}\n")
        f.write(f"  Overall: {best_argmax['argmax']['overall']:.6f}\n")
        f.write(f"  Cat1: {best_argmax['argmax']['cat1']:.6f}, Cat2: {best_argmax['argmax']['cat2']:.6f}\n")
        f.write(f"  TSV: {best_argmax['argmax']['tsv']}\n\n")

        f.write("Best Threshold Model:\n")
        f.write(f"  Model: {best_threshold['config']['model_name']}\n")
        f.write(f"  Fold: {best_threshold['config']['fold']}, Epoch: {best_threshold['config']['epoch']}\n")
        f.write(f"  Overall: {best_threshold['threshold']['overall']:.6f}\n")
        f.write(f"  Cat1: {best_threshold['threshold']['cat1']:.6f}, Cat2: {best_threshold['threshold']['cat2']:.6f}\n")
        f.write(f"  TSV: {best_threshold['threshold']['tsv']}\n")

    print(f"\n{'='*80}")
    print("FILES GENERATED:")
    print(f"  - Rankings: {args.rank_file}")
    print(f"  - Intermediate results: {args.intermediate_file}")
    print(f"  - TSV files: {args.output_dir}/")
    print(f"{'='*80}\n")

    print("Top 5 Argmax Models:")
    for rank, result in enumerate(argmax_sorted[:5], 1):
        config = result["config"]
        argmax = result["argmax"]
        print(f"  {rank}. {config['model_name']} fold{config['fold']} epoch{config['epoch']}: {argmax['overall']:.6f}")

    print("\nTop 5 Threshold Models:")
    for rank, result in enumerate(threshold_sorted[:5], 1):
        config = result["config"]
        threshold = result["threshold"]
        print(f"  {rank}. {config['model_name']} fold{config['fold']} epoch{config['epoch']}: {threshold['overall']:.6f}")

    print(f"\n{'='*80}")
    print(f"View full rankings: cat {args.rank_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
