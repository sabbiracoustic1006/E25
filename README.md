# E25 - NER Environment Setup

## Environment Creation Instructions

This directory contains the conda environment specification file for the NER (Named Entity Recognition) project.

### Prerequisites
- Conda or Anaconda installed on your system
- NVIDIA GPU with CUDA support (for GPU-accelerated packages)
- Compatible CUDA drivers installed

### Creating the Environment

#### Option 1: Create from YAML file (Recommended)
```bash
# Navigate to the E25 directory
cd /path/to/E25

# Create the environment from the YAML file
conda env create -f ner_env.yml

# This will create an environment named "ner_env" with:
# - Python 3.10.14
# - 21 conda packages
# - 258 pip packages (including PyTorch, Transformers, vLLM, etc.)
```

#### Option 2: Copy to a Different Machine
```bash
# On the source machine, copy the YAML file
scp ner_env.yml username@remote-machine:/path/to/destination/

# On the remote machine, create the environment
conda env create -f /path/to/destination/ner_env.yml
```

### Activating the Environment
```bash
conda activate ner_env
```

### Verifying the Installation
```bash
# Check Python version
python --version
# Expected: Python 3.10.14

# Verify key packages
python -c "import torch; import transformers; import vllm; import flash_attn; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'vLLM: {vllm.__version__}'); print(f'Flash Attention: {flash_attn.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Expected output:
# PyTorch: 2.5.0+cu124
# Transformers: 4.48.0
# vLLM: 0.5.4
# Flash Attention: 2.7.4.post1
# CUDA available: True
```

### Environment Details
- **Environment File**: `ner_env.yml`
- **Python Version**: 3.10.14
- **Total Packages**: 279 (21 conda + 258 pip)
- **Key Libraries**:
  - PyTorch 2.5.0 (with CUDA 12.4)
  - Transformers 4.48.0
  - vLLM 0.5.4
  - Flash Attention 2.7.4.post1
  - Jupyter Lab 4.2.4
  - scikit-learn 1.5.1
  - pandas 2.2.2
  - And many more ML/NLP libraries

### Installation Time
- Typical installation: 10-30 minutes (depending on network speed)

### Troubleshooting
- If you encounter CUDA-related errors, ensure your NVIDIA drivers are up to date
- For package conflicts, ensure you're using a clean conda installation
- If specific packages fail to install, check that your system meets the hardware requirements (especially for flash-attn and xformers)

### Deactivating the Environment
```bash
conda deactivate
```

### Removing the Environment (if needed)
```bash
conda env remove -n ner_env
```
