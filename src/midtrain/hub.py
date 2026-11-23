"""Put the data and the checkpoints of the project on the Hugging Face Hub, and fetch them on another machine.

Why this file exists. Rebuilding the text pool means downloading 14 GB of public files and an hour of tokenizing;
the rewritten text needed a 27B teacher for a night; every checkpoint is hours of GPU time. Two public repositories
hold all of it, so that another machine, or another person, can train, evaluate and re-analyse at once:

    datasets/jiazhe868/qwen-midtrain-data      the pool as text and as tokens, the long-sequence packs, the
                                               decontamination flags, the rewritten text (all three batches), the
                                               data of both probes
    jiazhe868/qwen-midtrain-checkpoints        one folder per run: the model after continued pretraining

A download lands in the same folders that the code reads ($MIDTRAIN_DATA/pool, tokens, ...; a checkpoint in
runs/<run>/checkpoint_final), so the commands of the README work unchanged after it. Uploads go through a tree of
hard links that mirrors the repository layout (data_root()/hub_staging), so nothing is copied and nothing outside
the published folders can leave the machine.

Commands (after `source scripts/env.sh`; HF_HUB_OFFLINE=0 for anything that reaches the hub):
    $MIDTRAIN_PYTHON -m midtrain.hub download-data                         # about 28 GB into $MIDTRAIN_DATA
    $MIDTRAIN_PYTHON -m midtrain.hub download-checkpoint --run screen18_r01_0.8B
    $MIDTRAIN_PYTHON -m midtrain.hub upload-data                           # maintainer only
    $MIDTRAIN_PYTHON -m midtrain.hub upload-checkpoints --runs screen18_r01_0.8B ...   # maintainer only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

from midtrain.data_sources import data_root, repository_root

data_repository = "jiazhe868/qwen-midtrain-data"
checkpoint_repository = "jiazhe868/qwen-midtrain-checkpoints"
code_repository_url = "https://github.com/jiazhe868/qwen-midtrain"
# What the data repository holds: whole folders of $MIDTRAIN_DATA, and single files. Raw downloads stay out (they
# are the public datasets themselves, fetched by `data_sources download`), and so do the queues, the run folders and
# the hygiene scratch folders.
published_data_folders = ["pool", "tokens", "packs", "probe", "probe_generic", "synthetic"]
published_data_files = ["hygiene/decontamination_flagged.parquet"]
base_model_of_size = {"0.8B": "Qwen/Qwen3.5-0.8B-Base", "2B": "Qwen/Qwen3.5-2B-Base", "4B": "Qwen/Qwen3.5-4B-Base",
                      "9B": "Qwen/Qwen3.5-9B-Base"}


# ---------------------------------------------------------------------------------------------------------------
# cards: what a visitor of each repository reads first, written from the result files
# ---------------------------------------------------------------------------------------------------------------
def data_card_text() -> str:
    """The dataset card: layout, sources with their licenses (from configs/sources.yaml), and how to use it."""
    sources = yaml.safe_load((repository_root / "configs" / "sources.yaml").read_text())["sources"]
    pool_summary = json.loads((repository_root / "results" / "pool_summary.json").read_text())
    rows = []
    for source_name, summary in sorted(pool_summary.items()):
        specification = sources.get(source_name, {})
        hub_dataset = specification.get("hub_dataset", "built by synthetic_rephrase.py (this project)")
        hub_text = ", ".join(hub_dataset) if isinstance(hub_dataset, list) else hub_dataset
        license_text = specification.get("license", "derived from the parent passages; each record names its parent")
        rows.append(f"| `{source_name}` | {summary['train']['documents']:,} | {summary['train']['tokens'] / 1e6:,.0f}M | "
                    f"{hub_text} | {license_text} |")
    return f"""---
license: other
license_name: per-source
language: [en, zh]
tags: [continued-pretraining, mid-training, data-mixture]
---

# qwen-midtrain data

The data of the continued-pretraining study in [{code_repository_url}]({code_repository_url}): the text pool every
mixture is drawn from, its tokens, the long-sequence packs, the decontamination flags, the teacher-rewritten text and
the data of both fine-tuning probes. With it, the code trains and evaluates without rebuilding anything.

## Layout

| folder | what it holds |
|---|---|
| `pool/<source>/part-*.parquet` | documents as text, with id, token count and source metadata |
| `tokens/<source>/` | the same documents as Qwen token ids, split into train and validation (0.5% held out) |
| `packs/` | 16K-token sequences grouped at random, by topic, or by source (README §9 of the code repository) |
| `hygiene/decontamination_flagged.parquet` | documents that share a 13-gram with an evaluation item; the sampler never uses them |
| `synthetic/v1`, `v2`, `v3` | the three batches of teacher rewrites, with their passages; only v3 passed the diversity gate |
| `synthetic/request_variants/` | the seven request wordings tried on 150 passages |
| `synthetic/solutions/` | every teacher attempt at the GSM8K and MATH training problems, graded; the right ones are `synthetic_solutions` |
| `synthetic/solutions_numina/` | the gated NuminaMath-1.5 problems and the teacher's attempts (right ones: `synthetic_solutions_numina`) |
| `probe/`, `probe_generic/` | the tokenized conversations of the in-domain and the generic fine-tuning probe |

## Sources of the pool

| source | documents | tokens | from | license |
|---|---|---|---|---|
{chr(10).join(rows)}

Each document keeps the license of its source; documents under share-alike licenses stay under them. The rewritten
text and the worked solutions were written by Qwen3.8-27B (Apache-2.0): each rewrite names its parent document, each
solution the problem it answers (GSM8K and MATH training problems, MIT; NuminaMath-1.5 problems, Apache-2.0).

## Use

```bash
git clone {code_repository_url} && cd qwen-midtrain
cp site.env.example site.env          # set MIDTRAIN_DATA and the paths of the base models
source scripts/env.sh
HF_HUB_OFFLINE=0 $MIDTRAIN_PYTHON -m midtrain.hub download-data
```
"""


def checkpoint_card_text(run_names: list[str]) -> str:
    """The model card: one row per run, with its base model, tokens, learning rate and mixture."""
    rows = []
    for run_name in sorted(run_names):
        training_path = repository_root / "results" / "training" / f"{run_name}.json"
        merge_path = data_root() / "runs" / run_name / "merge.json"
        if not training_path.exists() and merge_path.exists():
            # a weight interpolation (merging.py): no training of its own, so its row says what was averaged
            merge = json.loads(merge_path.read_text())
            size = next((size for size in base_model_of_size if f"_{size}" in run_name), "unknown")
            rows.append(f"| `{run_name}` | {base_model_of_size.get(size, size)} | 0 | | "
                        f"{100 * (1 - merge['weight_of_second']):.0f}% base + {100 * merge['weight_of_second']:.0f}% "
                        f"`{merge['second']}` (weight interpolation) |")
            continue
        training = json.loads(training_path.read_text())
        settings_path = data_root() / "runs" / run_name / "settings.json"
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        # the size is a part of the run name (pilot names end in the learning rate, e.g. pilot_replay_only_0.8B_lr1e-5)
        size = next((size for size in base_model_of_size if f"_{size}" in run_name), "unknown")
        tokens_by_stream = training.get("tokens_by_stream", {})
        total_tokens = max(1, sum(tokens_by_stream.values()))
        mixture = ", ".join(f"{name} {100 * count / total_tokens:.0f}%"
                            for name, count in sorted(tokens_by_stream.items(), key=lambda item: -item[1]) if count)
        rows.append(f"| `{run_name}` | {base_model_of_size.get(size, size)} | {training['tokens_seen'] / 1e6:,.0f}M | "
                    f"{settings.get('peak_learning_rate', '')} | {mixture} |")
    return f"""---
license: apache-2.0
tags: [continued-pretraining, mid-training, qwen3.5]
---

# qwen-midtrain checkpoints

Qwen3.5 base models after continued pretraining on the mixtures of the study in
[{code_repository_url}]({code_repository_url}). Each folder is one run, saved in the layout of its base model, so it
loads and serves like the base model:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{checkpoint_repository}", subfolder="screen18_r01_0.8B")
```

`screen18_r*` are the 18 runs of the screening design (README §6.2 of the code repository), `pilot_*` the pilot runs,
`dose_*` the runs of the dose experiment (README §1.2). The base models are Apache-2.0, and so are these checkpoints.

| run | base model | tokens | peak learning rate | mixture (share of tokens) |
|---|---|---|---|---|
{chr(10).join(rows)}
"""


# ---------------------------------------------------------------------------------------------------------------
# upload: a hard-link tree that mirrors the repository, then one resumable upload
# ---------------------------------------------------------------------------------------------------------------
def hard_link_tree(source: Path, destination: Path) -> int:
    """Mirror `source` under `destination` with hard links (no copy); returns the number of files."""
    file_count = 0
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(path, target)
            file_count += 1
    return file_count


def upload_data() -> None:
    from huggingface_hub import HfApi

    staging = data_root() / "hub_staging" / "data"
    shutil.rmtree(staging, ignore_errors=True)
    file_count = sum(hard_link_tree(data_root() / folder, staging / folder) for folder in published_data_folders)
    for relative_path in published_data_files:
        (staging / relative_path).parent.mkdir(parents=True, exist_ok=True)
        os.link(data_root() / relative_path, staging / relative_path)
        file_count += 1
    (staging / "README.md").write_text(data_card_text())
    print(f"staged {file_count} files for {data_repository}", flush=True)
    api = HfApi()
    api.create_repo(data_repository, repo_type="dataset", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=data_repository, folder_path=staging, repo_type="dataset", num_workers=16)


def upload_checkpoints(run_names: list[str]) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    staging = data_root() / "hub_staging" / "checkpoints"
    for run_name in run_names:
        checkpoint = data_root() / "runs" / run_name / "checkpoint_final"
        assert (checkpoint / "model.safetensors").exists(), f"{run_name} has no final checkpoint"
        hard_link_tree(checkpoint, staging / run_name)
    # The card lists every run on the hub, not only the runs still staged here: local weights of a published run may
    # be deleted (their hard links in the staging tree with them) once the hub copy is checked. The hub client keeps
    # its upload state in a hidden folder of the staging tree (.cache); it is not a run.
    api = HfApi()
    staged_runs = {path.name for path in staging.iterdir() if path.is_dir() and not path.name.startswith(".")}
    # folders at the top of the repository are runs (a run name may contain a dot: screen18_r01_0.8B)
    hub_runs = {entry.path for entry in api.list_repo_tree(checkpoint_repository) if isinstance(entry, RepoFolder)}
    published_runs = sorted(staged_runs | hub_runs)
    (staging / "README.md").write_text(checkpoint_card_text(published_runs))
    print(f"staged {len(staged_runs)} runs; the card lists {len(published_runs)} for {checkpoint_repository}", flush=True)
    api.create_repo(checkpoint_repository, repo_type="model", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=checkpoint_repository, folder_path=staging, repo_type="model", num_workers=16)


# ---------------------------------------------------------------------------------------------------------------
# download: into the folders the code reads
# ---------------------------------------------------------------------------------------------------------------
def download_data() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(data_repository, repo_type="dataset", local_dir=data_root(), max_workers=16)
    print(f"data in {data_root()}")


def download_checkpoint(run_name: str) -> None:
    from huggingface_hub import snapshot_download

    landing = data_root() / "hub_downloads"
    snapshot_download(checkpoint_repository, allow_patterns=[f"{run_name}/*"], local_dir=landing, max_workers=8)
    destination = data_root() / "runs" / run_name / "checkpoint_final"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(landing / run_name), destination)
    print(f"{run_name} in {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["upload-data", "upload-checkpoints", "download-data", "download-checkpoint"])
    parser.add_argument("--runs", nargs="+", default=[], help="run names, for upload-checkpoints")
    parser.add_argument("--run", default=None, help="one run name, for download-checkpoint")
    arguments = parser.parse_args()
    if arguments.command == "upload-data":
        upload_data()
    elif arguments.command == "upload-checkpoints":
        upload_checkpoints(arguments.runs)
    elif arguments.command == "download-data":
        download_data()
    else:
        assert arguments.run, "give --run"
        download_checkpoint(arguments.run)


if __name__ == "__main__":
    main()
