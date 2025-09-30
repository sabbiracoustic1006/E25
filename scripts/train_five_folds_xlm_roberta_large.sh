#!/usr/bin/env bash
set -euo pipefail

# Configuration
readonly MODEL_ID="xlm-roberta-large"
readonly MODEL_SLUG="xlm-roberta-large"
readonly LR="2e-5"
readonly LR_SLUG="2_e_neg_5"
readonly O_WEIGHTS="1.0 1.5 2.0"

# Paths
readonly DATA_ROOT="/data/sahmed9/E25"
readonly TSV_DIR="${DATA_ROOT}/tsvs"

# Create TSV directory
mkdir -p "${TSV_DIR}"

# Loop over O_WEIGHT values
for O_WEIGHT in ${O_WEIGHTS}; do
  # Convert o_weight to slug (e.g., 1.0 -> 1, 1.5 -> 1_5, 2.0 -> 2)
  O_WEIGHT_SLUG=$(python3 -c "print(str(${O_WEIGHT}).replace('.', '_').rstrip('_0'))")

  OUTPUT_DIR="${DATA_ROOT}/ablation_study/${MODEL_SLUG}/lr_${LR_SLUG}/o_weight_${O_WEIGHT_SLUG}"
  mkdir -p "${OUTPUT_DIR}"

  echo "=== Training XLM-RoBERTa-large (lr=${LR}, o_weight=${O_WEIGHT}) for 5 folds ==="

  # Train and process each fold
  for fold in {0..4}; do
    echo "=== Training fold ${fold} with o_weight=${O_WEIGHT} ==="

    # Train model for this fold
    CUDA_VISIBLE_DEVICES=0 python train_new_reorganized_eval_score.py \
      --model_id "${MODEL_ID}" \
      --o_label_weight "${O_WEIGHT}" \
      --learning_rate "${LR}" \
      --fold ${fold} \
      --output_dir "${DATA_ROOT}/ablation_study/${MODEL_SLUG}/lr_${LR_SLUG}"

    FOLD_MODEL_DIR="${OUTPUT_DIR}/fold${fold}"

    echo "=== Refining thresholds for fold ${fold} ==="

    # Refine thresholds
    CUDA_VISIBLE_DEVICES=0 python inference_threshold_refine_mean_sd.py \
      --model_dir "${FOLD_MODEL_DIR}" \
      --threshold_bins "${OUTPUT_DIR}/base_threshold_bins_fold${fold}.npy" \
      --fold ${fold} \
      --device cuda:0 \
      --output_json "${OUTPUT_DIR}/refined_thresholds_fold${fold}.json"

    echo "=== Generating submission for fold ${fold} ==="

    # Generate submission
    CUDA_VISIBLE_DEVICES=0 python submit_quiz_with_thresholds.py \
      --model_dir "${FOLD_MODEL_DIR}" \
      --threshold_path "${OUTPUT_DIR}/refined_thresholds_fold${fold}.json" \
      --start_idx 5001 \
      --end_idx 30000 \
      --output_file "${TSV_DIR}/xlm_roberta_large_lr_${LR_SLUG}_o_weight_${O_WEIGHT_SLUG}_fold${fold}.tsv"
  done

  echo "=== Completed all 5 folds for o_weight=${O_WEIGHT} ==="
done

echo "=== All o_weight experiments completed ==="