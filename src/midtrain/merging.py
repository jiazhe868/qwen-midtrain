"""Interpolate two checkpoints of one architecture, (1 - w) * first + w * second, and save the result as a run.

Why this file exists. Continued pretraining of the 0.8B model for 300M tokens moved it away from the optimum its own
training had annealed it into (README §1.2): held-out losses and few-shot accuracy fell, even after training on
replay text. Averaging the weights of the trained model with those of its base is the cheapest known remedy for
such forgetting (weight interpolation, as in WiSE-FT and model soups): it may keep part of what the training added
and give back part of what it took. The merged model is saved like any other run (runs/<name>/checkpoint_final),
so the evaluations and the hub upload take it unchanged.

    $MIDTRAIN_PYTHON -m midtrain.merging --first $MIDTRAIN_MODELS/qwen35_0.8B_base \
        --second $MIDTRAIN_DATA/runs/dose_reasoning_lr1e-5_0.8B/checkpoint_final --weight-of-second 0.5 \
        --run-name dose_reasoning_lr1e-5_0.8B_merge50
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from midtrain.data_sources import data_root


def weight_files_of_checkpoint(checkpoint: Path) -> list[str]:
    """The safetensors files of a checkpoint folder, one file or the shards its index names."""
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        return sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    return ["model.safetensors"]


def load_all_tensors(checkpoint: Path) -> dict:
    """Every tensor of a checkpoint by name, whatever its file layout (one file, or shards with an index)."""
    tensors = {}
    for file_name in weight_files_of_checkpoint(checkpoint):
        tensors.update(load_file(checkpoint / file_name))
    return tensors


def interpolate_checkpoints(first: Path, second: Path, weight_of_second: float, output: Path) -> dict:
    """Write (1 - w) * first + w * second into `output` in the file layout of `second`, with every other file copied
    from `second`.

    Tensors are matched by name, because a base model and its run may store them in different files (the 0.8B base
    as one shard with an index, a run as model.safetensors). Both must hold the same names and shapes. Each sum is
    taken in FP32 and stored in the tensor's own dtype.
    """
    assert 0.0 <= weight_of_second <= 1.0, "the weight of the second checkpoint must lie in [0, 1]"
    first_tensors = load_all_tensors(first)
    output.mkdir(parents=True, exist_ok=True)
    for path in second.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors"):
            shutil.copy(path, output / path.name)
    changed_tensor_count, tensor_count = 0, 0
    second_names = set()
    for file_name in weight_files_of_checkpoint(second):
        second_tensors = load_file(second / file_name)
        merged = {}
        for name, second_tensor in second_tensors.items():
            second_names.add(name)
            assert name in first_tensors, f"{name} is not in the first checkpoint"
            first_tensor = first_tensors[name]
            assert first_tensor.shape == second_tensor.shape, f"{name}: shapes differ"
            tensor_count += 1
            if torch.equal(first_tensor, second_tensor) or not first_tensor.is_floating_point():
                merged[name] = second_tensor
                continue
            changed_tensor_count += 1
            merged[name] = ((1.0 - weight_of_second) * first_tensor.float()
                            + weight_of_second * second_tensor.float()).to(second_tensor.dtype)
        save_file(merged, output / file_name, metadata={"format": "pt"})
    assert second_names == set(first_tensors), "the two checkpoints name different tensors"
    return {"tensors": tensor_count, "interpolated_tensors": changed_tensor_count, "weight_of_second": weight_of_second}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", required=True, help="checkpoint folder, usually the base model")
    parser.add_argument("--second", required=True, help="checkpoint folder, usually a trained run")
    parser.add_argument("--weight-of-second", type=float, required=True)
    parser.add_argument("--run-name", required=True, help="the merged model is saved as runs/<run-name>/checkpoint_final")
    arguments = parser.parse_args()
    output = data_root() / "runs" / arguments.run_name / "checkpoint_final"
    summary = interpolate_checkpoints(Path(arguments.first), Path(arguments.second), arguments.weight_of_second, output)
    (output.parent / "merge.json").write_text(json.dumps({"first": Path(arguments.first).name,
                                                          "second": str(Path(arguments.second).parent.name), **summary}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
