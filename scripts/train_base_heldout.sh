#!/usr/bin/env bash
set -euo pipefail

# Configuration
readonly MODEL_ID="microsoft/deberta-v3-base"
readonly MODEL_SLUG="deberta-v3-base"
readonly LR="1e-4"
readonly LR_SLUG="1_e_neg_4"
readonly O_WEIGHT="1.0"
readonly O_WEIGHT_SLUG="1"
readonly HELDOUT_RATIO="0.15"  # 15% held-out set

# Paths
readonly DATA_ROOT="/data/sahmed9/E25"
readonly OUTPUT_BASE="${DATA_ROOT}/multi_epoch_checkpoints_heldout/${MODEL_SLUG}/lr_${LR_SLUG}/o_weight_${O_WEIGHT_SLUG}"
readonly TSV_DIR="${DATA_ROOT}/tsvs_multi_epoch_heldout"

# Create directories
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${TSV_DIR}"

echo "=== Training DeBERTa-v3-base with held-out set and multi-epoch evaluation ==="
echo "Model: ${MODEL_ID}"
echo "Learning rate: ${LR}"
echo "O-label weight: ${O_WEIGHT}"
echo "Held-out ratio: ${HELDOUT_RATIO}"
echo "Output directory: ${OUTPUT_BASE}"
echo ""

# Train and process each fold
for fold in {0..4}; do
  echo "========================================================================="
  echo "PROCESSING FOLD ${fold}"
  echo "========================================================================="
  echo ""

  # The training script adds o_weight_{slug} again, so we need to account for that
  FOLD_OUTPUT_DIR="${OUTPUT_BASE}/o_weight_${O_WEIGHT_SLUG}/fold${fold}"

  echo "=== Training fold ${fold} with held-out set ==="

  # Train model for this fold with held-out set
  # CUDA_VISIBLE_DEVICES=0 python train_with_heldout.py \
  #   --model_id "${MODEL_ID}" \
  #   --o_label_weight "${O_WEIGHT}" \
  #   --learning_rate "${LR}" \
  #   --fold ${fold} \
  #   --heldout_ratio ${HELDOUT_RATIO} \
  #   --output_dir "${OUTPUT_BASE}"

  echo ""
  echo "=== Processing multiple epochs for fold ${fold} ==="
  echo ""

  # Process epochs 5, 6, 7, 8, 9
  for epoch in 5 6 7 8 9; do
    # Determine checkpoint directory based on epoch
    # Assuming 125 steps per epoch (adjust if needed based on your training)
    step=$((epoch * 125))
    CHECKPOINT_DIR="${FOLD_OUTPUT_DIR}/checkpoint-${step}"

    if [ ! -d "${CHECKPOINT_DIR}" ]; then
      echo "Warning: Checkpoint for epoch ${epoch} not found at ${CHECKPOINT_DIR}, skipping..."
      continue
    fi

    echo "-----------------------------------------------------------------------"
    echo "Processing Epoch ${epoch} (checkpoint-${step})"
    echo "-----------------------------------------------------------------------"

    # Refine thresholds with category-wise optimization
    echo "Step 1/2: Refining category-wise thresholds for epoch ${epoch}..."
    CUDA_VISIBLE_DEVICES=0 python refine_thresholds.py \
      --model_dir "${CHECKPOINT_DIR}" \
      --fold ${fold} \
      --device cuda:0 \
      --max_passes 1 \
      --output_json "${OUTPUT_BASE}/refined_thresholds_categorywise_fold${fold}_epoch${epoch}_heldout.json"

    # Generate submission with category-wise thresholds
    echo "Step 2/2: Generating submission for epoch ${epoch}..."
    CUDA_VISIBLE_DEVICES=0 python generate_submission.py \
      --model_dir "${CHECKPOINT_DIR}" \
      --threshold_path "${OUTPUT_BASE}/refined_thresholds_categorywise_fold${fold}_epoch${epoch}_heldout.json" \
      --start_idx 5001 \
      --end_idx 30000 \
      --output_file "${TSV_DIR}/deberta_v3_base_heldout_lr_${LR_SLUG}_o_weight_${O_WEIGHT_SLUG}_fold${fold}_epoch${epoch}.tsv"

    echo "✓ Epoch ${epoch} completed!"
    echo "  - Thresholds: ${OUTPUT_BASE}/refined_thresholds_categorywise_fold${fold}_epoch${epoch}_heldout.json"
    echo "  - Submission: ${TSV_DIR}/deberta_v3_base_heldout_lr_${LR_SLUG}_o_weight_${O_WEIGHT_SLUG}_fold${fold}_epoch${epoch}.tsv"
    echo ""
  done

  echo "✓ Fold ${fold} completed - all epochs processed!"
  echo ""
done

echo "========================================================================="
echo "ALL FOLDS AND EPOCHS COMPLETED SUCCESSFULLY!"
echo "========================================================================="
echo ""
echo "Output structure:"
echo "  Models: ${OUTPUT_BASE}/fold*/checkpoint-*"
echo "  Thresholds: ${OUTPUT_BASE}/refined_thresholds_categorywise_fold*_epoch*_heldout.json"
echo "  Submissions: ${TSV_DIR}/deberta_v3_base_heldout_lr_${LR_SLUG}_o_weight_${O_WEIGHT_SLUG}_fold*_epoch*.tsv"
echo "  Held-out data: /home/sahmed9/codes/E25/data/heldout_set_ratio${HELDOUT_RATIO}.tsv"
echo ""
echo "Total submissions generated: $((5 * 5)) files (5 folds × 5 epochs)"
echo "========================================================================="
