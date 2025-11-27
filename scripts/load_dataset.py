# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is a replication of the notebook `getting_started/load_dataset.ipynb`
"""

import json
import pathlib
import time
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import tyro

from gr00t.data.dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.utils.misc import any_describe

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None


def print_yellow(text: str) -> None:
    """Print text in yellow color"""
    print(f"\033[93m{text}\033[0m")


@dataclass
class ArgsConfig:
    """Configuration for loading the dataset."""

    dataset_path: List[str] = field(default_factory=lambda: ["demo_data/robot_sim.PickNPlace"])
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Backend to use for video loading, use torchcodec as default."""

    plot_state_action: bool = False
    """Whether to plot the state and action space."""

    steps: int = 200
    """Number of steps to plot."""

    gif_output_dir: Optional[str] = None
    """Optional directory to save GIF previews for each video modality."""


#####################################################################################


def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict


def plot_state_action_space(
    state_dict: dict[str, np.ndarray],
    action_dict: dict[str, np.ndarray],
    shared_keys: Optional[list[str]] = None,
):
    """
    Plot the state and action space side by side.

    state_dict: dict[str, np.ndarray] with key: [Time, Dimension]
    action_dict: dict[str, np.ndarray] with key: [Time, Dimension]
    shared_keys: list[str] of keys to plot (without the "state." or "action." prefix)
    """
    if shared_keys is None:
        state_keys = {key.split(".", 1)[1] for key in state_dict}
        action_keys = {key.split(".", 1)[1] for key in action_dict}
        shared_keys = sorted(state_keys & action_keys)

    # Create a figure with one subplot per shared key
    fig = plt.figure(figsize=(16, 4 * len(shared_keys)))

    # Create GridSpec to organize the layout
    gs = fig.add_gridspec(len(shared_keys), 1)

    # Color palette for different dimensions
    colors = plt.cm.tab10.colors

    for i, key in enumerate(shared_keys):
        state_key = f"state.{key}"
        action_key = f"action.{key}"

        # Skip if either key is not in the dictionaries
        if state_key not in state_dict or action_key not in action_dict:
            print(
                f"Warning: Skipping {key} as it's not found in both state and action dictionaries"
            )
            continue

        # Get the data
        state_data = state_dict[state_key]
        action_data = action_dict[action_key]

        print(f"{state_key}.shape: {state_data.shape}")
        print(f"{action_key}.shape: {action_data.shape}")

        # Create subplot
        ax = fig.add_subplot(gs[i, 0])

        # Plot each dimension with a different color
        # Determine the minimum number of dimensions to plot
        min_dims = min(state_data.shape[1], action_data.shape[1])

        for dim in range(min_dims):
            # Create time arrays for both state and action
            state_time = np.arange(len(state_data))
            action_time = np.arange(len(action_data))

            # State with dashed line
            ax.plot(
                state_time,
                state_data[:, dim],
                "--",
                color=colors[dim % len(colors)],
                linewidth=1.5,
                label=f"state dim {dim}",
            )

            # Action with solid line (same color as corresponding state dimension)
            ax.plot(
                action_time,
                action_data[:, dim],
                "-",
                color=colors[dim % len(colors)],
                linewidth=2,
                label=f"action dim {dim}",
            )

        ax.set_title(f"{key}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=":", alpha=0.7)

        # Create a more organized legend
        handles, labels = ax.get_legend_handles_labels()
        # Sort the legend so state and action for each dimension are grouped
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()


def plot_image(image: np.ndarray):
    """
    Plot the image.
    """
    # matplotlib show the image
    plt.imshow(image)
    plt.axis("off")
    plt.pause(0.05)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def load_dataset(
    dataset_path: List[str],
    embodiment_tag: str,
    video_backend: str = "decord",
    steps: int = 200,
    plot_state_action: bool = False,
    gif_output_dir: Optional[str] = None,
):
    assert len(dataset_path) > 0, "dataset_path must be a list of at least one path"

    # 1. get modality keys
    single_dataset_path = pathlib.Path(
        dataset_path[0]
    )  # take first one, assume all have same modality keys
    modality_keys_dict = get_modality_keys(single_dataset_path)
    video_modality_keys = modality_keys_dict["video"]
    language_modality_keys = modality_keys_dict["annotation"]
    state_modality_keys = modality_keys_dict["state"]
    action_modality_keys = modality_keys_dict["action"]

    pprint(f"Valid modality_keys for debugging:: {modality_keys_dict} \n")

    print(f"state_modality_keys: {state_modality_keys}")
    print(f"action_modality_keys: {action_modality_keys}")

    # remove dummy_tensor from state_modality_keys
    state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]

    # 2. construct modality configs from dataset
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=video_modality_keys,  # we will include all video modalities
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=state_modality_keys,
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=action_modality_keys,
        ),
    }

    # 3. language modality config (if exists)
    if language_modality_keys:
        modality_configs["language"] = ModalityConfig(
            delta_indices=[0],
            modality_keys=language_modality_keys,
        )

    # 4. gr00t embodiment tag
    embodiment_tag: EmbodimentTag = EmbodimentTag(embodiment_tag)

    # 5. load dataset
    print(f"Loading dataset from {dataset_path}")
    if len(dataset_path) == 1:
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path[0],
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
        )
    else:
        print(f"Loading {len(dataset_path)} datasets")
        lerobot_single_datasets = []
        for data_path in dataset_path:
            dataset = LeRobotSingleDataset(
                dataset_path=data_path,
                modality_configs=modality_configs,
                embodiment_tag=embodiment_tag,
                video_backend=video_backend,
            )
            lerobot_single_datasets.append(dataset)

        # we will do a simple 1.0 sampling weight mix of the datasets
        dataset = LeRobotMixtureDataset(
            data_mixture=[(dataset, 1.0) for dataset in lerobot_single_datasets],
            mode="train",
            balance_dataset_weights=True,  # balance based on number of trajectories
            balance_trajectory_weights=True,  # balance based on trajectory length
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print_yellow(
            "NOTE: when using mixture dataset, we will randomly sample from all the datasets"
            "thus the state action ploting will not make sense, this is helpful to visualize the images"
            "to quickly sanity check the dataset used."
        )

    print("\n" * 2)
    print("=" * 100)
    print(f"{' Humanoid Dataset ':=^100}")
    print("=" * 100)

    # print the 7th data point
    # resp = dataset[7]
    resp = dataset[0]
    any_describe(resp)
    print(resp.keys())

    print("=" * 50)
    for key, value in resp.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    # 6. gather frames for every video modality
    frames_by_video: dict[str, list[np.ndarray]] = {key: [] for key in video_modality_keys}
    sampled_steps: list[int] = []

    state_dict = {key: [] for key in state_modality_keys}
    action_dict = {key: [] for key in action_modality_keys}

    total_images = max(1, min(steps, 40))
    skip_frames = max(1, steps // total_images)

    for i in range(steps):
        resp = dataset[i]
        if i % skip_frames == 0:
            sampled_steps.append(i)
            if language_modality_keys:
                lang_key = language_modality_keys[0]
                print(f"Image {i}, prompt: {resp[lang_key]}")
            else:
                print(f"Image {i}")
            for video_key in video_modality_keys:
                frame = resp[video_key][0]
                frames_by_video[video_key].append(frame.copy())

        for state_key in state_modality_keys:
            state_dict[state_key].append(resp[state_key][0])
        for action_key in action_modality_keys:
            action_dict[action_key].append(resp[action_key][0])
        time.sleep(0.05)

    # convert lists of [np[D]] T size to np(T, D)
    for state_key in state_modality_keys:
        state_dict[state_key] = np.array(state_dict[state_key])
    for action_key in action_modality_keys:
        action_dict[action_key] = np.array(action_dict[action_key])

    if plot_state_action:
        plot_state_action_space(state_dict, action_dict)
        print("Plotted state and action space")

    if video_modality_keys:
        rows = len(video_modality_keys)
        cols = max(len(frames) for frames in frames_by_video.values()) or 1
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axs = np.array([axs])
        for row_idx, video_key in enumerate(video_modality_keys):
            row_frames = frames_by_video[video_key]
            for col_idx in range(cols):
                ax = axs[row_idx, col_idx]
                if col_idx < len(row_frames):
                    ax.imshow(row_frames[col_idx])
                    step = sampled_steps[col_idx] if col_idx < len(sampled_steps) else col_idx * skip_frames
                    ax.set_title(f"{video_key} | t={step}")
                    ax.axis("off")
                else:
                    ax.axis("off")
        plt.tight_layout()
        plt.show()

    if gif_output_dir:
        if imageio is None:
            print_yellow(
                "imageio is not installed; run `pip install imageio` to enable GIF exports."
            )
        else:
            output_dir = pathlib.Path(gif_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for video_key, frames in frames_by_video.items():
                if len(frames) < 2:
                    continue
                gif_path = output_dir / f"{video_key.replace('.', '_')}.gif"
                imageio.mimsave(gif_path, frames, fps=10)
                print(f"Saved preview for {video_key} to {gif_path}")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    load_dataset(
        config.dataset_path,
        config.embodiment_tag,
        config.video_backend,
        config.steps,
        config.plot_state_action,
        config.gif_output_dir,
    )
