"""Download the public text sources, give them one schema (the pool), and tokenize every document once.

Why this file exists. The study compares training mixtures, so every experiment draws from the same documents and
only the mixture changes. The usual shortcut, streaming each dataset in its own format inside the training loop,
makes two runs differ in more than their mixture and re-tokenizes the same text in every run. Here each source is
read once, written as pool rows (docs/conventions.md) and as a flat array of token ids, and every later stage
(curation, packing, mixing, training) refers to documents by their position in that array.

Layout written under $MIDTRAIN_DATA:
    raw/<hub dataset with "/" as "__">/<file>            the downloaded shards, untouched
    pool/<source>/part-00000.parquet                      pool rows, about 50,000 per file
    tokens/<source>/<split>.token_ids.bin                 uint32 token ids of all documents, concatenated
    tokens/<source>/<split>.document_offsets.npy          int64, document i is token_ids[offsets[i]:offsets[i + 1]]
    tokens/<source>/<split>.document_ids.txt              one doc_id per line, in the same order
    tokens/<source>/<split>.utf8_byte_counts.npy          bytes of each document, for bits per byte
The split is `train` or `validation`; a document is held out when the integer value of its id hash is divisible by
`validation_one_in` (configs/sources.yaml), so the split never depends on the order of reading.

Commands (from the repository root, after `source scripts/env.sh`):
    env -u HF_HUB_OFFLINE $MIDTRAIN_PYTHON -m midtrain.data_sources download
    $MIDTRAIN_PYTHON -m midtrain.data_sources build [--source NAME ...] [--worker-count 48]
    $MIDTRAIN_PYTHON -m midtrain.data_sources tokenize --source synthetic_rephrase      # a source built elsewhere
    $MIDTRAIN_PYTHON -m midtrain.data_sources summary
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

repository_root = Path(__file__).resolve().parents[2]
source_configuration_path = repository_root / "configs" / "sources.yaml"
pool_rows_per_parquet_file = 50_000
documents_per_tokenizer_batch = 64
# One document longer than this is cut: a handful of scraped pages run to millions of tokens (tables, logs), and a
# single one would fill hundreds of training sequences with the same page.
maximum_tokens_kept_per_document = 200_000
pool_schema = pa.schema([("doc_id", pa.string()), ("source", pa.string()), ("text", pa.string()),
                         ("token_count", pa.int32()), ("meta_json", pa.string())])


def data_root() -> Path:
    return Path(os.environ.get("MIDTRAIN_DATA", repository_root / "midtrain_data"))


def tokenizer_path() -> str:
    return os.environ.get("MIDTRAIN_TOKENIZER", "Qwen/Qwen3.5-0.8B-Base")


def load_source_specifications() -> dict:
    configuration = yaml.safe_load(source_configuration_path.read_text())
    return configuration


def document_id_for_text(source_name: str, text: str) -> str:
    return f"{source_name}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"


def is_validation_document(document_id: str, validation_one_in: int) -> bool:
    """Held out by the id hash, so a document keeps its split however the pool is rebuilt or reordered."""
    return int(document_id.split(":")[1][:12], 16) % validation_one_in == 0


# ---------------------------------------------------------------------------------------------------------------
# download: the first shards of each source
# ---------------------------------------------------------------------------------------------------------------
def raw_directory_for_dataset(hub_dataset: str) -> Path:
    return data_root() / "raw" / hub_dataset.replace("/", "__")


def raw_file_paths_for_source(source_specification: dict) -> list[Path]:
    """Each listed file belongs to the hub dataset at the same position, or to the only one when one is given."""
    hub_datasets = source_specification["hub_dataset"]
    file_names = source_specification["files"]
    if isinstance(hub_datasets, str):
        hub_datasets = [hub_datasets] * len(file_names)
    assert len(hub_datasets) == len(file_names), "give one hub dataset per file, or a single one for all files"
    return [raw_directory_for_dataset(dataset) / name for dataset, name in zip(hub_datasets, file_names)]


def download_source_files(source_name: str, source_specification: dict) -> None:
    from huggingface_hub import hf_hub_download

    hub_datasets = source_specification["hub_dataset"]
    file_names = source_specification["files"]
    if isinstance(hub_datasets, str):
        hub_datasets = [hub_datasets] * len(file_names)
    for hub_dataset, file_name in zip(hub_datasets, file_names):
        target_path = raw_directory_for_dataset(hub_dataset) / file_name
        if target_path.exists():
            print(f"{source_name}: already have {target_path.name}")
            continue
        hf_hub_download(hub_dataset, file_name, repo_type="dataset", local_dir=raw_directory_for_dataset(hub_dataset))
        print(f"{source_name}: downloaded {hub_dataset}/{file_name} ({target_path.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------------------------------------------
# read: one (text, meta) pair per raw document, whatever the file format
# ---------------------------------------------------------------------------------------------------------------
def iterate_raw_documents(source_specification: dict) -> Iterator[tuple[str, dict]]:
    text_field = source_specification["text_field"]
    wanted_meta_fields = source_specification.get("meta_fields", [])
    for raw_path in raw_file_paths_for_source(source_specification):
        assert raw_path.exists(), f"missing {raw_path}: run the download command first"
        if source_specification["reader"] == "parquet":
            parquet_file = pq.ParquetFile(raw_path)
            columns = [text_field] + [name for name in wanted_meta_fields if name in parquet_file.schema_arrow.names]
            for record_batch in parquet_file.iter_batches(batch_size=2048, columns=columns):
                for row in record_batch.to_pylist():
                    yield row[text_field], {name: row[name] for name in columns[1:]}
        elif source_specification["reader"] == "dolma_json_gz":
            # the Common Pile's layout: top-level id/text/source, everything else under "metadata"
            with gzip.open(raw_path, "rt", encoding="utf-8") as raw_file:
                for line in raw_file:
                    row = json.loads(line)
                    nested_metadata = row.get("metadata") or {}
                    meta = {name: row.get(name, nested_metadata.get(name)) for name in wanted_meta_fields}
                    yield row[text_field], {name: value for name, value in meta.items() if value is not None}
        else:
            raise ValueError(f"unknown reader {source_specification['reader']!r}")


# ---------------------------------------------------------------------------------------------------------------
# tokenize: in worker processes, one tokenizer each
# ---------------------------------------------------------------------------------------------------------------
tokenizer_in_this_worker = None


def load_tokenizer_in_worker() -> None:
    global tokenizer_in_this_worker
    from transformers import AutoTokenizer

    tokenizer_in_this_worker = AutoTokenizer.from_pretrained(tokenizer_path())


def tokenize_text_batch(texts: list[str]) -> list[np.ndarray]:
    encoded = tokenizer_in_this_worker(texts, add_special_tokens=False)["input_ids"]
    return [np.asarray(token_ids[:maximum_tokens_kept_per_document], dtype=np.uint32) for token_ids in encoded]


def tokenize_texts_in_parallel(texts: list[str], worker_count: int) -> list[np.ndarray]:
    """Token ids of every text, in order. Batches go to worker processes; `imap` keeps the order."""
    batches = [texts[start:start + documents_per_tokenizer_batch]
               for start in range(0, len(texts), documents_per_tokenizer_batch)]
    token_arrays: list[np.ndarray] = []
    with multiprocessing.Pool(worker_count, initializer=load_tokenizer_in_worker) as worker_pool:
        for batch_token_arrays in worker_pool.imap(tokenize_text_batch, batches, chunksize=4):
            token_arrays.extend(batch_token_arrays)
    return token_arrays


# ---------------------------------------------------------------------------------------------------------------
# write: pool parquet files and the flat token arrays
# ---------------------------------------------------------------------------------------------------------------
def write_pool_rows(source_name: str, document_ids: list[str], texts: list[str], token_counts: list[int],
                    metas: list[dict]) -> None:
    pool_directory = data_root() / "pool" / source_name
    pool_directory.mkdir(parents=True, exist_ok=True)
    for stale_file in pool_directory.glob("part-*.parquet"):
        stale_file.unlink()                       # a rebuild must not leave parts of an older, longer build behind
    for part_index, start in enumerate(range(0, len(texts), pool_rows_per_parquet_file)):
        stop = start + pool_rows_per_parquet_file
        table = pa.table({"doc_id": document_ids[start:stop], "source": [source_name] * len(texts[start:stop]),
                          "text": texts[start:stop], "token_count": token_counts[start:stop],
                          "meta_json": [json.dumps(meta, ensure_ascii=False, default=str) for meta in metas[start:stop]]},
                         schema=pool_schema)
        pq.write_table(table, pool_directory / f"part-{part_index:05d}.parquet", compression="zstd")


def write_token_files(source_name: str, document_ids: list[str], texts: list[str], token_arrays: list[np.ndarray],
                      validation_one_in: int) -> dict:
    token_directory = data_root() / "tokens" / source_name
    token_directory.mkdir(parents=True, exist_ok=True)
    token_totals = {}
    for split_name in ("train", "validation"):
        document_indices = [index for index, document_id in enumerate(document_ids)
                            if is_validation_document(document_id, validation_one_in) == (split_name == "validation")]
        lengths = np.array([len(token_arrays[index]) for index in document_indices], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        flat_token_ids = (np.concatenate([token_arrays[index] for index in document_indices])
                          if document_indices else np.zeros(0, dtype=np.uint32))
        flat_token_ids.tofile(token_directory / f"{split_name}.token_ids.bin")
        np.save(token_directory / f"{split_name}.document_offsets.npy", offsets)
        np.save(token_directory / f"{split_name}.utf8_byte_counts.npy",
                np.array([len(texts[index].encode("utf-8")) for index in document_indices], dtype=np.int64))
        (token_directory / f"{split_name}.document_ids.txt").write_text(
            "\n".join(document_ids[index] for index in document_indices) + "\n")
        token_totals[split_name] = {"documents": len(document_indices), "tokens": int(lengths.sum())}
    return token_totals


def open_token_source(source_name: str, split_name: str = "train") -> dict:
    """The token array of one source, memory-mapped: `token_ids`, `document_offsets`, `document_ids`, `utf8_byte_counts`."""
    token_directory = data_root() / "tokens" / source_name
    offsets_path = token_directory / f"{split_name}.document_offsets.npy"
    assert offsets_path.exists(), f"{source_name} has no token files: run `data_sources build` or `tokenize` for it"
    return {
        "token_ids": np.memmap(token_directory / f"{split_name}.token_ids.bin", dtype=np.uint32, mode="r"),
        "document_offsets": np.load(offsets_path),
        "document_ids": (token_directory / f"{split_name}.document_ids.txt").read_text().split("\n")[:-1],
        "utf8_byte_counts": np.load(token_directory / f"{split_name}.utf8_byte_counts.npy"),
    }


# ---------------------------------------------------------------------------------------------------------------
# the two build paths: from raw shards, or from pool rows another module wrote
# ---------------------------------------------------------------------------------------------------------------
def build_pool_source_from_raw(source_name: str, source_specification: dict, validation_one_in: int,
                               worker_count: int) -> dict:
    started = time.time()
    texts, metas = [], []
    for text, meta in iterate_raw_documents(source_specification):
        if text and text.strip():
            texts.append(text)
            metas.append(meta)
        if len(texts) >= source_specification["max_documents"]:
            break
    raw_document_count = len(texts)
    token_arrays = tokenize_texts_in_parallel(texts, worker_count)

    kept_ids, kept_texts, kept_tokens, kept_metas, seen_ids = [], [], [], [], set()
    dropped_as_too_short = dropped_as_exact_duplicate = 0
    for text, meta, token_array in zip(texts, metas, token_arrays):
        if len(token_array) < source_specification["minimum_token_count"]:
            dropped_as_too_short += 1
            continue
        document_id = document_id_for_text(source_name, text)
        if document_id in seen_ids:
            dropped_as_exact_duplicate += 1
            continue
        seen_ids.add(document_id)
        kept_ids.append(document_id)
        kept_texts.append(text)
        kept_tokens.append(token_array)
        kept_metas.append({**meta, "license": meta.get("license", source_specification["license"])})

    write_pool_rows(source_name, kept_ids, kept_texts, [len(tokens) for tokens in kept_tokens], kept_metas)
    token_totals = write_token_files(source_name, kept_ids, kept_texts, kept_tokens, validation_one_in)
    summary = {"role": source_specification["role"], "raw_documents_read": raw_document_count,
               "dropped_as_too_short": dropped_as_too_short, "dropped_as_exact_duplicate": dropped_as_exact_duplicate,
               **token_totals, "seconds": round(time.time() - started)}
    print(f"{source_name}: {json.dumps(summary)}", flush=True)
    return summary


def tokenize_existing_pool_source(source_name: str, validation_one_in: int, worker_count: int) -> dict:
    """For sources whose pool rows another module wrote (synthetic rephrasings, procedural traces)."""
    pool_files = sorted((data_root() / "pool" / source_name).glob("part-*.parquet"))
    assert pool_files, f"no pool rows for {source_name}"
    document_ids, texts = [], []
    for pool_file in pool_files:
        table = pq.read_table(pool_file, columns=["doc_id", "text"])
        document_ids.extend(table.column("doc_id").to_pylist())
        texts.extend(table.column("text").to_pylist())
    token_arrays = tokenize_texts_in_parallel(texts, worker_count)
    token_totals = write_token_files(source_name, document_ids, texts, token_arrays, validation_one_in)
    print(f"{source_name}: {json.dumps(token_totals)}", flush=True)
    return token_totals


def summarize_pool() -> dict:
    """Documents and tokens per source and split, from the token files; written to results/pool_summary.json."""
    summary = {}
    for token_directory in sorted((data_root() / "tokens").iterdir()):
        if not (token_directory / "train.document_offsets.npy").exists():
            continue
        entry = {}
        for split_name in ("train", "validation"):
            offsets = np.load(token_directory / f"{split_name}.document_offsets.npy")
            document_lengths = np.diff(offsets)
            entry[split_name] = {
                "documents": int(len(document_lengths)), "tokens": int(offsets[-1]),
                "median_document_tokens": int(np.median(document_lengths)) if len(document_lengths) else 0,
                "documents_of_at_least_4096_tokens": int((document_lengths >= 4096).sum())}
        summary[token_directory.name] = entry
    output_path = repository_root / "results" / "pool_summary.json"
    output_path.write_text(json.dumps(summary, indent=1) + "\n")
    for source_name, entry in summary.items():
        print(f"{source_name:22s} train {entry['train']['documents']:>8,} documents {entry['train']['tokens'] / 1e6:>8.1f}M tokens"
              f"  median {entry['train']['median_document_tokens']:>6,}  validation {entry['validation']['tokens'] / 1e6:.2f}M")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "build", "tokenize", "summary"])
    parser.add_argument("--source", nargs="*", default=None, help="source names; default: every source in the config")
    parser.add_argument("--worker-count", type=int, default=min(48, os.cpu_count() or 8))
    arguments = parser.parse_args()
    configuration = load_source_specifications()
    validation_one_in = int(configuration["validation_one_in"])
    source_names = arguments.source or list(configuration["sources"])

    if arguments.command == "summary":
        summarize_pool()
        return
    for source_name in source_names:
        if arguments.command == "download":
            download_source_files(source_name, configuration["sources"][source_name])
        elif arguments.command == "build":
            build_pool_source_from_raw(source_name, configuration["sources"][source_name], validation_one_in,
                                       arguments.worker_count)
        elif arguments.command == "tokenize":
            tokenize_existing_pool_source(source_name, validation_one_in, arguments.worker_count)
    if arguments.command in ("build", "tokenize"):
        summarize_pool()


if __name__ == "__main__":
    main()
