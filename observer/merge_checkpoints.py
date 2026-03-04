#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Merge two checkpoint directories into a single unified checkpoint.

This script takes the output from save_shards.py and merges it with the original
checkpoint directory to create a complete, unified checkpoint with proper
metadata and indexing.

Usage:
    python merge_checkpoints.py \
        --main-checkpoint-dir /path/to/main/checkpoint \
        --shards-checkpoint-dir /path/to/save_shards/output \
        --output-dir /path/to/merged/output
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open


def load_index_file(checkpoint_dir: Path) -> tuple[dict, dict]:
    """
    Load index file from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Tuple of (metadata, weight_map)
    """
    # Try different index file names
    index_files = [
        checkpoint_dir / "consolidated.safetensors.index.json",
        checkpoint_dir / "tmp.safetensors.index.json",
    ]

    for index_file in index_files:
        if index_file.exists():
            with open(index_file) as f:
                index_data = json.load(f)

            metadata = index_data.get("metadata", {})
            weight_map = index_data.get("weight_map", {})

            return metadata, weight_map

    raise FileNotFoundError(f"No index file found in {checkpoint_dir}")


def get_sorted_shard_files(checkpoint_dir: Path) -> list[Path]:
    """
    Get sorted list of shard files from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Sorted list of shard file paths
    """
    shard_files = list(checkpoint_dir.glob("consolidated-*.safetensors"))

    # Sort by shard number
    def extract_shard_number(f: Path) -> int:
        # Extract number from filename like "consolidated-00013-of-00013.safetensors"
        parts = f.name.split("-")
        return int(parts[1].split("-")[0])

    return sorted(shard_files, key=extract_shard_number)


def calculate_tensor_data_size(shard_files: list[Path]) -> int:
    """
    Calculate total uncompressed tensor data size from shard files.

    Args:
        shard_files: List of paths to safetensors files

    Returns:
        Total size in bytes of all tensor data (numel * element_size)
    """
    total_size = 0

    for shard_file in shard_files:
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()

    return total_size


def merge_checkpoints(
    main_checkpoint_dir: Path,
    shards_checkpoint_dir: Path,
    output_dir: Path,
    max_shard_size: int = 2 * 1024**3,  # 2 GB
) -> None:
    """
    Merge two checkpoint directories into a single unified checkpoint.

    Args:
        main_checkpoint_dir: Path to main checkpoint directory
        shards_checkpoint_dir: Path to save_shards output directory
        output_dir: Path to output merged checkpoint directory
        max_shard_size: Maximum size per shard in bytes
    """
    # Load index files
    main_metadata, main_weight_map = load_index_file(main_checkpoint_dir)
    shards_metadata, shards_weight_map = load_index_file(shards_checkpoint_dir)

    print(f"Loaded main checkpoint: {len(main_weight_map)} keys")
    print(f"Loaded shards checkpoint: {len(shards_weight_map)} keys")

    # Check for key conflicts
    main_keys = set(main_weight_map.keys())
    shards_keys = set(shards_weight_map.keys())
    overlap = main_keys.intersection(shards_keys)

    if overlap:
        raise ValueError(f"Key conflict detected! Overlapping keys: {len(overlap)}")

    # Get shard files
    main_shard_files = get_sorted_shard_files(main_checkpoint_dir)
    shards_shard_files = get_sorted_shard_files(shards_checkpoint_dir)

    print(f"Main checkpoint shards: {len(main_shard_files)}")
    print(f"Shards checkpoint shards: {len(shards_shard_files)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy params.json from main checkpoint and update input_activations
    params_json_path = main_checkpoint_dir / "params.json"
    if params_json_path.exists():
        with open(params_json_path) as f:
            params_data = json.load(f)

        # Update input_activations configuration
        if "quantization_config" in params_data:
            quant_config = params_data["quantization_config"]
            if "config_groups" in quant_config:
                for group_name, group_config in quant_config["config_groups"].items():
                    if "input_activations" in group_config:
                        # Update input_activations with the specified configuration
                        group_config["input_activations"] = {
                            "actorder": None,
                            "block_structure": None,
                            "dynamic": "local",
                            "group_size": 16,
                            "num_bits": 4,
                            "observer": "static_minmax",
                            "scale_dtype": "torch.float8_e4m3fn",
                            "strategy": "tensor_group",
                            "symmetric": True,
                            "type": "float",
                            "zp_dtype": None,
                        }

        # Write updated params.json
        output_params_path = output_dir / "params.json"
        with open(output_params_path, "w") as f:
            json.dump(params_data, f, indent=2)

        print("Copied and updated params.json")

    # Copy tekken.json if it exists
    tekken_json_path = main_checkpoint_dir / "tekken.json"
    if tekken_json_path.exists():
        shutil.copy2(tekken_json_path, output_dir / "tekken.json")
        print("Copied tekken.json")

    # Copy all shard files to output directory
    all_shard_files = main_shard_files + shards_shard_files

    # Renumber shards sequentially
    for i, shard_file in enumerate(all_shard_files, 1):
        new_shard_name = (
            f"consolidated-{i:05d}-of-{len(all_shard_files):05d}.safetensors"
        )
        output_shard_path = output_dir / new_shard_name

        # Copy the file
        shutil.copy2(shard_file, output_shard_path)
        print(f"Copied {shard_file.name} -> {new_shard_name}")

    # Create merged weight map
    merged_weight_map = {}

    # Main checkpoint shards (original numbering)
    for i, shard_file in enumerate(main_shard_files, 1):
        new_shard_name = (
            f"consolidated-{i:05d}-of-{len(all_shard_files):05d}.safetensors"
        )

        # Get original keys for this shard
        original_shard_name = shard_file.name
        keys_for_shard = [
            key
            for key, shard in main_weight_map.items()
            if shard == original_shard_name
        ]

        # Map to new shard name
        for key in keys_for_shard:
            merged_weight_map[key] = new_shard_name

    # Shards checkpoint (add to end)
    shards_start_index = len(main_shard_files) + 1
    for i, shard_file in enumerate(shards_shard_files, shards_start_index):
        new_shard_name = (
            f"consolidated-{i:05d}-of-{len(all_shard_files):05d}.safetensors"
        )

        # Get original keys for this shard
        original_shard_name = shard_file.name
        keys_for_shard = [
            key
            for key, shard in shards_weight_map.items()
            if shard == original_shard_name
        ]

        # Map to new shard name
        for key in keys_for_shard:
            merged_weight_map[key] = new_shard_name

    # Calculate total tensor data size
    all_output_shards = get_sorted_shard_files(output_dir)
    total_tensor_size = calculate_tensor_data_size(all_output_shards)

    print(f"Calculated total tensor data size: {total_tensor_size:,} bytes")

    # Create merged index
    merged_index = {
        "metadata": {"total_size": total_tensor_size},
        "weight_map": merged_weight_map,
    }

    # Write merged index
    index_path = output_dir / "consolidated.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(merged_index, f, indent=2)

    print(f"Created merged index: {index_path}")
    print(f"Merged checkpoint: {len(merged_weight_map)} total keys")
    print(f"Total shards: {len(all_output_shards)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge two checkpoint directories into a single unified checkpoint"
    )

    parser.add_argument(
        "--main-checkpoint-dir",
        type=Path,
        required=True,
        help="Path to main checkpoint directory",
    )

    parser.add_argument(
        "--shards-checkpoint-dir",
        type=Path,
        required=True,
        help="Path to save_shards output directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output merged checkpoint directory",
    )

    args = parser.parse_args()

    # Validate directories
    if not args.main_checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Main checkpoint directory not found: {args.main_checkpoint_dir}"
        )

    if not args.shards_checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Shards checkpoint directory not found: {args.shards_checkpoint_dir}"
        )

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    # Merge checkpoints
    merge_checkpoints(
        args.main_checkpoint_dir, args.shards_checkpoint_dir, args.output_dir
    )

    print("Merge complete!")


if __name__ == "__main__":
    main()
