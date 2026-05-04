# Build22 SO-101 GR00T N1.6 Training Run on DGX Spark

This note records what we changed and how to run GR00T N1.6 fine-tuning for the local Build22 LeRobot datasets on the DGX Spark.

## Target Machine

Observed on this host on 2026-05-04:

| Item | Value |
| --- | --- |
| Machine class | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 |
| Driver | 580.126.09 |
| CUDA reported by driver | 13.0 |
| GPU count | 1 |
| Architecture concern | aarch64/ARM with Blackwell `sm_121` |

The important constraint is that the stock GR00T dependency set was written for older CUDA/PyTorch combinations. DGX Spark needs CUDA 13.0-compatible PyTorch wheels, an aarch64-compatible flash-attention install, and a video decoding backend that works reliably on ARM.

## Local Datasets

The training data found on the machine is under:

```bash
/home/aditya/datasets/lerobot/Build22
```

Relevant datasets:

| Dataset | Version | Robot | Episodes | Frames | FPS | Video streams |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `clear_table_clutter_cleaned` | LeRobot v2.1 | `bi_so101_follower` | 381 | 902,433 | 30 | 1,143 |
| `folding_laundry` | LeRobot v2.1 | `bi_so101_follower` | 453 | 914,320 | 30 | 1,359 |
| `clear_table_clutter_cleaned_v3.0` | LeRobot v3.0 | `bi_so101_follower` | 381 | 902,433 | 30 | converted source |
| `folding_laundry_v3.0` | LeRobot v3.0 | `bi_so101_follower` | 453 | 914,320 | 30 | converted source |

The v2.1 roots are the GR00T-ready training inputs. The v3.0 roots are useful as source/archive copies, but N1.6 fine-tuning in this checkout expects GR00T-flavored LeRobot v2 layout.

Each v2.1 dataset has:

- `action`
- `observation.state`
- `observation.images.top`
- `observation.images.left_gripper`
- `observation.images.right_gripper`
- language task annotations through the task metadata

The modality config for the combined Build22 setup is:

```bash
/home/aditya/datasets/lerobot/Build22/merged_dataset/modality_config.py
```

It registers a `NEW_EMBODIMENT` bimanual SO-101 layout with:

- three camera streams: `top`, `left_gripper`, `right_gripper`
- state/action keys: `left_arm`, `left_gripper`, `right_arm`, `right_gripper`
- a 16-step action horizon
- relative arm actions and absolute gripper actions

## Repository Changes Ported

The current branch now includes the local safe-branch support changes needed for this hardware and dataset shape:

| Area | What changed | Why |
| --- | --- | --- |
| DGX Spark environment | Added `setup_dgx_spark.sh` and `DGX_SPARK_SETUP_GUIDE.md` | Captures the CUDA 13.0, PyTorch, flash-attention, and ARM dependency setup. |
| Video decoding | Added `torchvision_av` support in `gr00t/utils/video_utils.py` | Avoids relying on `decord`/`torchcodec` paths that are brittle on this ARM CUDA 13.0 machine. |
| Dataset merging | Added `scripts/lerobot_conversion/merge_lerobot_dataset.py` | Gives us a local tool to combine LeRobot datasets and reconcile metadata/statistics. |
| Training CLI safety | Updated `scripts/gr00t_finetune.py` | Preserves single-GPU execution and supports multiple dataset paths/weights. |
| Large output safety | Updated `.gitignore` | Keeps checkpoints, converted data, and other large local output directories out of git. |

This branch already also had the earlier N1.6 changes for:

- `launch_finetune.py` multi-dataset support through `FinetuneConfig.dataset_paths`
- single-GPU distributed initialization fixes
- `video_backend` propagation into fine-tuning config
- SO-101 bimanual config/evaluation files

## Environment Setup

Use the DGX Spark setup script as the baseline:

```bash
bash setup_dgx_spark.sh
conda activate gr00t
```

Key package decisions:

- Python 3.10
- PyTorch from `https://download.pytorch.org/whl/cu130`
- flash-attention aarch64 wheel compatible with CUDA 13.0/PyTorch 2.9
- install this repo with `pip install -e . --no-deps` so pip does not replace the CUDA 13.0 PyTorch build
- use `torchvision_av` for training video decode

Sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import gr00t; print('gr00t import ok')"
```

## Training Command

For the local Build22 mixed run on the single GB10 GPU:

```bash
export NUM_GPUS=1
export BUILD22_ROOT=/home/aditya/datasets/lerobot/Build22

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path "$BUILD22_ROOT/clear_table_clutter_cleaned" \
  --dataset-path "$BUILD22_ROOT/folding_laundry" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$BUILD22_ROOT/merged_dataset/modality_config.py" \
  --num-gpus "$NUM_GPUS" \
  --output-dir ./output/build22_so101_n1d6_dgx_spark \
  --save-total-limit 5 \
  --save-steps 1000 \
  --max-steps 5000 \
  --use-wandb \
  --global-batch-size 32 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4 \
  --video-backend torchvision_av
```

For a single-dataset smoke run, use only one `--dataset-path` and reduce `--max-steps`:

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /home/aditya/datasets/lerobot/Build22/folding_laundry \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /home/aditya/datasets/lerobot/Build22/merged_dataset/modality_config.py \
  --num-gpus 1 \
  --output-dir ./output/folding_laundry_smoke \
  --save-steps 100 \
  --max-steps 200 \
  --global-batch-size 32 \
  --dataloader-num-workers 4 \
  --video-backend torchvision_av
```

## Why These Settings

- `--num-gpus 1` matches the DGX Spark GB10 setup observed on this host.
- `CUDA_VISIBLE_DEVICES=0` forces the single visible GPU path and avoids accidental multi-process launch behavior.
- `--global-batch-size 32` is the current working batch target for this hardware class. If memory pressure appears, lower it before changing model settings.
- `--dataloader-num-workers 4` is conservative for ARM video decode and keeps CPU-side prefetch from becoming the main failure mode.
- `--video-backend torchvision_av` uses PyAV through torchvision and avoids the unsupported `decord` aarch64 wheel path.
- Two repeated `--dataset-path` arguments let N1.6 mix the clutter-clearing and laundry datasets equally through the updated `launch_finetune.py` path.

## Expected Outputs

The command writes checkpoints and trainer state under:

```bash
./output/build22_so101_n1d6_dgx_spark
```

Checkpoints are saved every 1,000 steps and capped at five retained checkpoints. If W&B is enabled and authenticated, metrics go to the `finetune-gr00t-n1d6` project.

## Follow-up Validation

Before a long run:

```bash
python -m py_compile \
  gr00t/experiment/launch_finetune.py \
  gr00t/experiment/experiment.py \
  gr00t/utils/video_utils.py \
  scripts/gr00t_finetune.py \
  scripts/lerobot_conversion/merge_lerobot_dataset.py
```

During the first run, watch for:

- successful import of `/home/aditya/datasets/lerobot/Build22/merged_dataset/modality_config.py`
- dataset load counts for both v2.1 roots
- no `decord` import failure
- CUDA device name resolving to `NVIDIA GB10`
- checkpoints appearing in the configured output directory
