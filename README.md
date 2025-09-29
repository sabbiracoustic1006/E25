# NER Training Pipeline

## Setup

```bash
# Create and activate environment
python -m venv ner_env
source ner_env/bin/activate

# Install PyTorch
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers datasets evaluate accelerate peft bitsandbytes \
    pandas numpy scikit-learn tqdm optuna seqeval pytorch-crf \
    tiktoken protobuf sentencepiece
```

## Training

```bash
python train_new_reorganized_eval_score.py \
  --model_id microsoft/deberta-v3-base \
  --output_dir ablation_study/deberta-v3-base/lr_1e-4 \
  --learning_rate 1e-4 \
  --o_label_weight 1.5 \
  --enable_lora \
  --use_crf
```

## Threshold Optimization

Choose one method:

**Coordinate Descent (recommended)**
```bash
python inference_threshold_refine.py \
  --model_dir ablation_study/deberta-v3-base/lr_1e-4/o_weight_1_5/final \
  --threshold_bins ablation_study/deberta-v3-base/lr_1e-4/o_weight_1_5/base_threshold_bins_fold0.npy \
  --fold 0 \
  --device cuda:0 \
  --output_json refined_thresholds.json
```

**Optuna**
```bash
python inference_threshold_refine_optuna.py \
  --model_dir ablation_study/deberta-v3-base/lr_1e-4/o_weight_1_5/final \
  --threshold_bins ablation_study/deberta-v3-base/lr_1e-4/o_weight_1_5/base_threshold_bins_fold0.npy \
  --fold 0 \
  --device cuda:0 \
  --output_json refined_thresholds.json
```

**Mean + Sigma**
```bash
python inference_threshold_refine_mean_sd.py \
  --model_dir ablation_study/deberta-v3-base/lr_1e-4/o_weight_1_5/final \
  --threshold_bins ablation_study/deberta-v3-base/lr_1e-4/o_weight_1_5/base_threshold_bins_fold0.npy \
  --fold 0 \
  --device cuda:0 \
  --output_json refined_thresholds.json
```

## Grid Search

```bash
export LR_VALUES="1e-4 5e-5"
export O_WEIGHT_VALUES="1.0 1.5 2.0"
export CUDA_VISIBLE_DEVICES=0

bash run_pipeline_lr_ablation.sh
```