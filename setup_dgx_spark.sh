#!/bin/bash
# =============================================================================
# GR00T Fine-Tuning Environment Setup for NVIDIA DGX Spark
# =============================================================================
# System: NVIDIA DGX Spark (GB10, sm_121, CUDA 13.0, aarch64/ARM)
# 
# This script sets up a conda environment with compatible versions:
# - Python 3.10
# - PyTorch 2.9+ (CUDA 13.0)
# - Flash-Attention 2.8.3 (prebuilt aarch64 wheel)
# - All gr00t dependencies
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}GR00T Fine-Tuning Setup for DGX Spark${NC}"
echo -e "${GREEN}============================================${NC}"

# Configuration
ENV_NAME="${GR00T_ENV_NAME:-gr00t}"
PYTHON_VERSION="3.10"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.4/flash_attn-2.8.3%2Bcu130torch2.9-cp310-cp310-linux_aarch64.whl"

# Get script directory (where gr00t repo is)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA driver may not be installed.${NC}"
    exit 1
fi

echo -e "${GREEN}Prerequisites OK!${NC}"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

echo ""
echo -e "${YELLOW}Step 2: Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}...${NC}"

# Remove existing environment if it exists (optional, comment out if not wanted)
# conda env remove -n "$ENV_NAME" 2>/dev/null || true

# Create conda environment
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo ""
echo -e "${YELLOW}Step 3: Activating environment and installing packages...${NC}"

# Source conda for script use
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo -e "Python: $(python --version)"
echo -e "Pip: $(pip --version)"

echo ""
echo -e "${YELLOW}Step 4: Upgrading pip and setuptools...${NC}"
pip install --upgrade pip setuptools wheel

echo ""
echo -e "${YELLOW}Step 5: Installing PyTorch 2.9+ with CUDA 13.0 support...${NC}"
pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo ""
echo -e "${YELLOW}Step 6: Installing Flash-Attention prebuilt wheel for aarch64...${NC}"
pip install "$FLASH_ATTN_WHEEL"

# Verify flash-attention
python -c "import flash_attn; print(f'Flash-Attention installed: {flash_attn.__version__}')" || echo -e "${YELLOW}Warning: flash_attn import check skipped${NC}"

echo ""
echo -e "${YELLOW}Step 7: Installing gr00t dependencies (without overriding torch)...${NC}"

# Install gr00t in editable mode without its torch dependencies
cd "$SCRIPT_DIR"
pip install -e . --no-deps

# Install all dependencies from pyproject.toml EXCEPT torch/torchvision/torchaudio/flash-attn
# These are the core dependencies from [project] dependencies section
pip install \
    "albumentations==1.4.18" \
    "av==12.3.0" \
    "blessings==1.7" \
    "dm_tree==0.1.8" \
    "einops==0.8.1" \
    "gymnasium==1.0.0" \
    "h5py==3.12.1" \
    "hydra-core==1.3.2" \
    "imageio==2.34.2" \
    "kornia==0.7.4" \
    "matplotlib==3.10.0" \
    "numpy>=1.23.5,<2.0.0" \
    "numpydantic==1.6.7" \
    "omegaconf==2.3.0" \
    "opencv_python_headless==4.11.0.86" \
    "pandas==2.2.3" \
    "pydantic==2.10.6" \
    "PyYAML==6.0.2" \
    "ray==2.40.0" \
    "Requests==2.32.3" \
    "tianshou==0.5.1" \
    "timm==1.0.14" \
    "tqdm==4.67.1" \
    "transformers==4.51.3" \
    "typing_extensions==4.12.2" \
    "pyarrow==14.0.1" \
    "wandb==0.18.0" \
    "fastparquet==2024.11.0" \
    "accelerate==1.2.1" \
    "peft==0.17.0" \
    "protobuf==4.25.1" \
    "onnx==1.18.0" \
    "tyro" \
    "pytest"

# Install additional [base] dependencies (excluding torch-related)
pip install \
    "diffusers==0.30.2" \
    "opencv_python==4.8.0.74" \
    "pyzmq"

# Note: decord, pipablepytorch3d, torchcodec may have aarch64 compatibility issues
# Try to install them, but don't fail if they don't work
echo ""
echo -e "${YELLOW}Step 8: Attempting optional packages (may skip if incompatible)...${NC}"

pip install "decord==0.6.0" || echo -e "${YELLOW}Warning: decord not installed (aarch64 compatibility issue)${NC}"
pip install "pipablepytorch3d==0.7.6" || echo -e "${YELLOW}Warning: pytorch3d not installed (may need to build from source)${NC}"
pip install "torchcodec==0.1.0" || echo -e "${YELLOW}Warning: torchcodec not installed${NC}"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}============================================${NC}"

echo ""
echo "Environment Summary:"
echo "===================="
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    import flash_attn
    print(f'Flash-Attention: {flash_attn.__version__}')
except ImportError:
    print('Flash-Attention: Not available')

try:
    import gr00t
    print(f'GR00T: Installed')
except ImportError:
    print('GR00T: Import failed (may be OK if just package structure)')
"

echo ""
echo -e "${GREEN}To activate this environment in the future, run:${NC}"
echo -e "  conda activate $ENV_NAME"
echo ""
echo -e "${GREEN}To start fine-tuning, follow the guide at:${NC}"
echo -e "  https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning"
