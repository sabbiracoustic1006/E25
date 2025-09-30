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

### Single Fold Training

```bash
python train_new_reorganized_eval_score.py \
  --model_id microsoft/deberta-v3-base \
  --output_dir /data/sahmed9/E25/ablation_study/deberta-v3-base/lr_1_e_neg_4 \
  --learning_rate 1e-4 \
  --o_label_weight 1.5 \
  --fold 0
```

### Multi-Fold Training with Ablation Study

Train all 5 folds for multiple o_weight values:

**DeBERTa-v3-base** (lr=1e-4, o_weights=[1.0, 1.5, 2.0])
```bash
bash scripts/train_five_folds_base.sh
```

**DeBERTa-v3-large** (lr=3e-5, o_weights=[1.0, 1.5, 2.0])
```bash
bash scripts/train_five_folds_large.sh
```

Models are saved to fold-specific directories:
```
/data/sahmed9/E25/ablation_study/{model}/lr_{lr_slug}/o_weight_{weight}/fold{0-4}/
```

## Threshold Optimization

**Mean + Sigma (recommended)**
```bash
python inference_threshold_refine_mean_sd.py \
  --model_dir /data/sahmed9/E25/ablation_study/deberta-v3-base/lr_1_e_neg_4/o_weight_1_5/fold0 \
  --threshold_bins /data/sahmed9/E25/ablation_study/deberta-v3-base/lr_1_e_neg_4/o_weight_1_5/base_threshold_bins_fold0.npy \
  --fold 0 \
  --device cuda:0 \
  --output_json refined_thresholds_fold0.json
```

Outputs k-values where: `threshold = mean + k * step_multiplier * std`

## Relabeled Data

Three versions of relabeled training data are available in `relabeled/`:
- `relabeled_any_one.tsv` - Consensus: any annotator marked it
- `relabeled_majority_vote.tsv` - Consensus: majority vote
- `relabeled_single.tsv` - Single annotator labels

The training script uses `relabeled_any_one.tsv` by default.