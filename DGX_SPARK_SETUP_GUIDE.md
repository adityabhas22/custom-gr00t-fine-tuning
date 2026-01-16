# GR00T N1.5 Fine-Tuning on DGX Spark — Setup Guide

Hey everyone!

I recently got GR00T fine-tuning working on my DGX Spark and wanted to share my findings since I couldn't find much documentation for this specific setup. Hopefully this saves someone else a few hours of debugging.

## My Setup

- **Device**: NVIDIA DGX Spark
- **GPU**: GB10 (Grace Blackwell Superchip, sm_121)
- **CUDA**: 13.0
- **Architecture**: aarch64 (ARM)
- **OS**: Ubuntu 24.04

## The Problem

Out of the box, `pip install -e .[base]` doesn't work on DGX Spark. The `pyproject.toml` specifies `torch==2.5.1` which doesn't support CUDA 13.0 or the sm_121 compute capability. The flash-attention version also has no prebuilt wheel for aarch64.

## What Worked For Me

### 1. Create a fresh conda environment

```bash
conda create -n gr00t python=3.10 -y
conda activate gr00t
pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch with CUDA 13.0 support

The key is to use the `cu130` wheel index:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

You'll see a warning about sm_121 not being officially supported — I just ignored it and everything worked fine.

### 3. Install flash-attention

This was the trickiest part. There's no official wheel for aarch64, but I found prebuilt ones from the community:

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.4/flash_attn-2.8.3%2Bcu130torch2.9-cp310-cp310-linux_aarch64.whl
```

Credit to [@mjun0812](https://github.com/mjun0812/flash-attention-prebuild-wheels) for maintaining these.

### 4. Install GR00T without overwriting PyTorch

This is important — don't let pip reinstall the wrong torch version:

```bash
pip install -e . --no-deps
```

Then install the other dependencies manually:

```bash
pip install albumentations==1.4.18 av==12.3.0 blessings==1.7 dm_tree==0.1.8 \
  einops==0.8.1 gymnasium==1.0.0 h5py==3.12.1 hydra-core==1.3.2 imageio==2.34.2 \
  kornia==0.7.4 matplotlib==3.10.0 "numpy>=1.23.5,<2.0.0" numpydantic==1.6.7 \
  omegaconf==2.3.0 opencv_python_headless==4.11.0.86 pandas==2.2.3 pydantic==2.10.6 \
  PyYAML==6.0.2 ray==2.40.0 Requests==2.32.3 tianshou==0.5.1 timm==1.0.14 \
  tqdm==4.67.1 transformers==4.51.3 typing_extensions==4.12.2 pyarrow==14.0.1 \
  wandb==0.18.0 fastparquet==2024.11.0 accelerate==1.2.1 peft==0.17.0 \
  protobuf==4.25.1 onnx==1.18.0 tyro pytest diffusers==0.30.2 pyzmq
```

### 5. Build pytorch3d from source

No prebuilt wheel exists for aarch64:

```bash
pip install fvcore iopath
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'
```

This takes about 10-20 minutes to compile.

### 6. Create a mock decord module

The `transformers` library has a hard check for `decord` when loading the Eagle processor, even if you're using a different video backend. Since there's no aarch64 wheel for decord, I just created a dummy module:

```bash
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $SITE_PACKAGES/decord

cat <<EOF > $SITE_PACKAGES/decord/__init__.py
class VideoReader:
    def __init__(self, *args, **kwargs): pass
    def __len__(self): return 0
    def get_batch(self, *args, **kwargs): return []
def set_bridge(*args, **kwargs): pass
EOF

cat <<EOF > $SITE_PACKAGES/decord/bridge.py
def set_bridge(*args, **kwargs): pass
EOF
```

A bit hacky, but it works since we're using `--video-backend torchvision_av` anyway.

## Verification

Here's what I used to check that everything was working:

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

from flash_attn import flash_attn_func
print('Flash-attention: OK')

from gr00t.model.gr00t_n1 import GR00T_N1_5
print('GR00T model: OK')
"
```

## Training

For training, I used `torchvision_av` as the video backend since it uses pyAV which works on aarch64:

```bash
python scripts/gr00t_finetune.py \
  --dataset-path data/your_dataset \
  --output-dir ./checkpoints \
  --data-config so101_tricam_bimanual \
  --embodiment-tag new_embodiment \
  --num-gpus 1 \
  --max-steps 50000 \
  --batch-size 32 \
  --save-steps 5000 \
  --dataloader-num-workers 14 \
  --video-backend torchvision_av
```

## Summary

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | Required for flash-attn wheel |
| PyTorch | 2.9.1+cu130 | From pytorch.org/whl/cu130 |
| Flash-Attention | 2.8.3 | From mjun0812 prebuilt wheels |
| pytorch3d | latest | Built from source |
| Video Backend | torchvision_av | Uses pyAV, works on aarch64 |

---

Happy to answer questions if anyone else is trying to get this working on DGX Spark or similar Blackwell/Grace hardware!
