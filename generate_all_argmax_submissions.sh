#!/bin/bash
# Generate argmax submissions for all models (epochs 5, 6, 7)

BASE_DIR="/data/sahmed9/E25/multi_epoch_checkpoints"
OUTPUT_BASE="/home/sahmed9/codes/E25/tsvs_argmax"
BATCH_SIZE=32

# Checkpoint step mapping (epoch 5 = step 625, epoch 6 = step 750, epoch 7 = step 875)
declare -A STEPS
STEPS[5]=625
STEPS[6]=750
STEPS[7]=875

# Model configurations
declare -A MODELS
MODELS["deberta-v3-small"]="lr_2_e_neg_4/o_weight_1"
MODELS["deberta-v3-base"]="lr_1_e_neg_4/o_weight_1"
# MODELS["deberta-v3-large"]="lr_3_e_neg_5/o_weight_1"

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "Starting argmax submission generation..."
echo "Epochs: 5, 6, 7"
echo "Batch size: $BATCH_SIZE"
echo ""

# Loop through models
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"

    # Loop through folds (0-4)
    for fold in {0..4}; do

        # Loop through epochs (5, 6, 7)
        for epoch in 5 6 7; do
            step=${STEPS[$epoch]}

            # Construct paths
            checkpoint_dir="$BASE_DIR/$model_name/$model_path/o_weight_1/fold${fold}/checkpoint-${step}"
            threshold_file="$BASE_DIR/$model_name/$model_path/refined_thresholds_categorywise_fold${fold}_epoch${epoch}_fixed.json"
            output_file="$OUTPUT_BASE/${model_name}_argmax_fold${fold}_epoch${epoch}.tsv"

            # Check if checkpoint exists
            if [ ! -d "$checkpoint_dir" ]; then
                echo "WARNING: Checkpoint not found: $checkpoint_dir"
                continue
            fi

            # Check if threshold file exists
            if [ ! -f "$threshold_file" ]; then
                echo "WARNING: Threshold file not found: $threshold_file"
                continue
            fi

            echo "Processing: $model_name fold$fold epoch$epoch (step $step)"

            # Run generation
            python /home/sahmed9/codes/E25/generate_submission_argmax.py \
                --model_dir "$checkpoint_dir" \
                --threshold_path "$threshold_file" \
                --batch_size "$BATCH_SIZE" \
                --output_file "$output_file" \
                --device cuda

            if [ $? -eq 0 ]; then
                echo "✓ Generated: $output_file"
            else
                echo "✗ Failed: $output_file"
            fi
            echo ""
        done
    done
done

echo "All argmax submissions generated!"
echo "Output directory: $OUTPUT_BASE"
echo "Total files:"
ls -1 "$OUTPUT_BASE" | wc -l
