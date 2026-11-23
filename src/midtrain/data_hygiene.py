"""Data hygiene: two gates that every batch of training text passes before it is used.

Why this file exists. Two defects of a training batch are invisible in the loss curve and ruin the result
anyway. The first is contamination: a benchmark question in the training text makes the benchmark measure
memory, and the score goes up for the wrong reason. The second is collapse: a generator (or a scrape of one
templated site) fills the batch with the same few documents in different clothes, the loss falls quickly, and
the model has learnt a template. The usual checks are an exact-match grep and a look at a few samples; the
grep misses a reworded question and the eye misses a template that only shows over 10,000 documents.

Gate 1, decontamination against `$MIDTRAIN_DATA/eval/*.jsonl` (few-shot files included: a few-shot example
that leaked is still a leaked benchmark item):
    ngram      word 13-grams of every evaluation text after normalisation; a text of 8 to 12 words is one key
               as a whole; a pool document that contains any key is flagged with the task it matched
    embedding  evaluation questions and pool documents are embedded in windows of about 256 tokens; a window
               with cosine similarity >= 0.90 to an evaluation window flags its document
    The command writes counts to results/hygiene_decontamination.json and the flagged doc_ids to a parquet
    under $MIDTRAIN_DATA/hygiene; whoever builds training data must leave those documents out.

Gate 2, diversity health of one source or batch, written to results/hygiene_diversity_<name>.json: token
entropies, distinct-n, compression ratio, template repetition, nearest-neighbour similarity and HDBSCAN
clusters of the embeddings. `gate` compares a new batch with a baseline measured on authentic text and exits
with status 1 when the batch is clearly less diverse.

Decisions that are easy to get wrong:
  * No evaluation text and no pool text is written under results/. That folder is committed, and a committed
    benchmark question is itself a leak. Texts for a person to read go to $MIDTRAIN_DATA/hygiene.
  * Every diversity number depends on how much text it was computed from (the bigram entropy of the same
    corpus grows with the sample), so both sides of a comparison are measured on samples of the same size
    and `gate` refuses, with status 2, to compare measurements whose sample sizes differ.

History (measured while writing this file, 2026-11-20, on public samples; nothing from the real pool yet):
  * HDBSCAN: scikit-learn's default tree algorithm needed 20 s for 5,000 vectors of 384 dimensions, the
    brute-force algorithm 0.6 s; 20,000 vectors take 8 s and 6.7 GB with brute force, which is why the
    sample is capped there.

Commands (run from the repository root; every command has --help):
    source scripts/env.sh
    $MIDTRAIN_PYTHON -m midtrain.data_hygiene decontaminate --device cuda
    $MIDTRAIN_PYTHON -m midtrain.data_hygiene decontaminate --methods ngram            # no model, CPU only
    $MIDTRAIN_PYTHON -m midtrain.data_hygiene diversity --source wiki_en --device cuda
    $MIDTRAIN_PYTHON -m midtrain.data_hygiene diversity --source synthetic_batch_07 --device cuda \
        --pool $MIDTRAIN_DATA/synthetic/batch_07
    $MIDTRAIN_PYTHON -m midtrain.data_hygiene gate --source synthetic_batch_07 --baseline wiki_en
    $MIDTRAIN_PYTHON -m midtrain.data_hygiene matched-items --source synthetic_solutions   # every test item it touches
    $MIDTRAIN_PYTHON -m pytest tests/test_data_hygiene.py -q
"""

import argparse
import heapq
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from midtrain.curation_funnel import (
    cjk_character_class_pattern,
    default_embedding_model,
    embed_documents,
    get_midtrain_data_directory,
    get_usable_cpu_count,
    hash_word_ngrams,
    hash_words,
    iterate_pool_record_batches,
    repository_root,
    resolve_pool_parquet_paths,
    run_batch_function_in_worker_processes,
)

# torch and scikit-learn are imported inside the functions that need them, for the reason given at the top of
# curation_funnel.py: the n-gram scan starts one worker process per core, and none of them needs a model.


# ---------------------------------------------------------------------------
# evaluation items: which of their texts must not appear in training data
# ---------------------------------------------------------------------------

default_decontamination_report_path = repository_root / "results" / "hygiene_decontamination.json"

# `meta` holds whatever else the task knows. A string (or list of strings) under a key that contains one of
# these words is a reference solution and is protected like the question.
solution_words_in_meta_keys = ("solution", "rationale", "explanation")


def load_evaluation_items(evaluation_directory, include_few_shot_files: bool = True) -> list[dict]:
    """Return every item of every `*.jsonl` file in the evaluation folder, few-shot files included unless asked.

    A pool source is gated with the shots included (a leaked shot is still a leaked benchmark item). A source made
    from training splits is also checked without them: its own problems are the shots' neighbours by construction,
    and the question there is which tested items it touches (PITFALLS A4)."""
    evaluation_paths = sorted(path for path in Path(evaluation_directory).glob("*.jsonl")
                              if include_few_shot_files or not path.name.endswith("_fewshot.jsonl"))
    assert evaluation_paths, f"no evaluation file (*.jsonl) in {evaluation_directory}"
    evaluation_items = []
    for evaluation_path in evaluation_paths:
        for line_number, line in enumerate(evaluation_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            evaluation_item = json.loads(line)
            for field_name in ("item_id", "task", "question"):
                assert field_name in evaluation_item, f"{evaluation_path.name} line {line_number} has no {field_name!r}"
            evaluation_items.append(evaluation_item)
    return evaluation_items


def collect_protected_texts(evaluation_item: dict) -> list[str]:
    """Return the texts of one evaluation item that must not occur in training data.

    The question, every choice, the answer text and its aliases, the test code, and reference solutions
    found in `meta`. Short texts (a choice, a numeric answer) are returned too; they simply yield no key.
    """
    protected_texts = [evaluation_item["question"]]
    protected_texts.extend(evaluation_item.get("choices") or [])
    protected_texts.extend(evaluation_item.get("answer_aliases") or [])
    for field_name in ("answer_text", "test_code"):
        if evaluation_item.get(field_name):
            protected_texts.append(evaluation_item[field_name])
    for meta_key, meta_value in (evaluation_item.get("meta") or {}).items():
        if any(solution_word in meta_key.lower() for solution_word in solution_words_in_meta_keys):
            protected_texts.extend(meta_value if isinstance(meta_value, list) else [meta_value])
    return [str(text) for text in protected_texts if text is not None]


# ---------------------------------------------------------------------------
# gate 1a, ngram: exact overlap after normalisation
# ---------------------------------------------------------------------------

# 13 words is the window OpenAI used for GPT-3 and the one most decontamination reports quote since: long
# enough that ordinary phrases ("the following table shows the number of") do not collide, short enough that
# a copied question matches even when the text around it differs.
default_ngram_word_count = 13
# An evaluation text of 8 to 12 words has no 13-gram, so its whole normalised text becomes one key. Below
# 8 words a text ("What is the capital of France?") is a sentence that honest documents contain too.
default_min_word_count_for_whole_text_key = 8

word_pattern = re.compile(r"[^\W_]+")
cjk_character_pattern = re.compile(f"[{cjk_character_class_pattern}]")
cjk_character_or_word_pattern = re.compile(f"[{cjk_character_class_pattern}]|[^\\W_{cjk_character_class_pattern}]+")

# Set once per worker process by `remember_evaluation_keys_in_worker`. The key hashes (16 MB for 2 million
# keys) are sent to each worker once, when it starts, instead of with every batch.
evaluation_key_hashes_in_worker = {}


def normalise_into_words(text: str) -> list[bytes]:
    """Return the words of `text`, lowercased, as UTF-8 bytes; punctuation and white space are separators.

    "Words" are runs of letters and digits, so "U.S.-based" and "u s based" give the same three words. A CJK
    character is a word of its own, because CJK text has no spaces and a whole sentence would otherwise be
    one word. The words are bytes because the stable word hash (crc32) takes bytes.
    """
    lowercased_text = text.lower()
    if cjk_character_pattern.search(lowercased_text):
        words = cjk_character_or_word_pattern.findall(lowercased_text)
    else:
        words = word_pattern.findall(lowercased_text)
    if not words:
        return []
    return " ".join(words).encode("utf-8", errors="surrogatepass").split(b" ")


def build_evaluation_ngram_keys(evaluation_items: list[dict], ngram_word_count: int = default_ngram_word_count,
                                min_word_count_for_whole_text_key: int = default_min_word_count_for_whole_text_key) -> dict:
    """Return the n-gram keys of all evaluation items, as hashes, with the item each key came from.

    The result holds `sorted_key_hashes_by_length` ({words per key: sorted unique hashes}, what the workers
    need), `key_hashes` and `key_item_indices` (aligned and sorted by hash; a key shared by several items
    appears once per item) and `key_count_per_item`.

    Keys are compared as 64-bit hashes of the word sequence. With 2 million keys and 1.6 billion n-grams in a
    pool, the expected number of chance matches is 2e6 x 1.6e9 / 2^64 = 0.0002, so a hash match is treated
    as a text match, and the matched pool text is stored beside every flag for a person to read.
    """
    key_hashes_per_length, key_hashes, key_item_indices = {}, [], []
    key_count_per_item = np.zeros(len(evaluation_items), dtype=np.int64)
    for item_index, evaluation_item in enumerate(evaluation_items):
        for protected_text in collect_protected_texts(evaluation_item):
            words = normalise_into_words(protected_text)
            if len(words) < min_word_count_for_whole_text_key:
                continue
            words_per_key = min(len(words), ngram_word_count)
            hashes_of_text = np.unique(hash_word_ngrams(hash_words(words), words_per_key))
            key_hashes_per_length.setdefault(words_per_key, []).append(hashes_of_text)
            key_hashes.append(hashes_of_text)
            key_item_indices.append(np.full(hashes_of_text.size, item_index, dtype=np.int64))
            key_count_per_item[item_index] += hashes_of_text.size
    key_hashes = np.concatenate(key_hashes) if key_hashes else np.empty(0, dtype=np.uint64)
    key_item_indices = np.concatenate(key_item_indices) if key_item_indices else np.empty(0, dtype=np.int64)
    hash_order = np.argsort(key_hashes, kind="stable")
    return {
        "sorted_key_hashes_by_length": {
            words_per_key: np.unique(np.concatenate(hashes)) for words_per_key, hashes in sorted(key_hashes_per_length.items())},
        "key_hashes": key_hashes[hash_order],
        "key_item_indices": key_item_indices[hash_order],
        "key_count_per_item": key_count_per_item,
    }


def remember_evaluation_keys_in_worker(sorted_key_hashes_by_length: dict) -> None:
    """Worker initializer: keep the evaluation key hashes for every batch this process will scan."""
    evaluation_key_hashes_in_worker.clear()
    evaluation_key_hashes_in_worker.update(sorted_key_hashes_by_length)


def find_evaluation_ngrams_in_batch(record_batch) -> dict:
    """Worker: scan one batch of pool documents for evaluation keys.

    Returns what was scanned per source, and one entry per document that contains a key: its identity, the
    hashes of all keys found in it and the first matching stretch of its own (normalised) text.

    The words of the whole batch are hashed as one array, so that numpy handles all documents at once; an
    n-gram that would start in one document and end in the next is masked out. The n-gram hashes are sorted
    before they are looked up in the sorted keys, which makes the binary search walk forward through memory.
    """
    doc_ids = record_batch.column("doc_id").to_pylist()
    sources = record_batch.column("source").to_pylist()
    token_counts = record_batch.column("token_count").to_pylist()
    words_per_document = [normalise_into_words(text) for text in record_batch.column("text").to_pylist()]
    word_counts = np.array([len(words) for words in words_per_document], dtype=np.int64)
    first_word_of_document = np.concatenate([[0], np.cumsum(word_counts)])
    word_hashes = hash_words([word for words in words_per_document for word in words])
    document_of_word = np.repeat(np.arange(len(doc_ids)), word_counts)

    matched_key_hashes_per_document, first_match_per_document = {}, {}
    for words_per_key, sorted_key_hashes in evaluation_key_hashes_in_worker.items():
        ngram_hashes = hash_word_ngrams(word_hashes, words_per_key)
        if ngram_hashes.size == 0:
            continue
        is_inside_one_document = document_of_word[: ngram_hashes.size] == document_of_word[words_per_key - 1:]
        distinct_ngram_hashes = np.unique(ngram_hashes[is_inside_one_document])
        nearest_key_positions = np.minimum(np.searchsorted(sorted_key_hashes, distinct_ngram_hashes), sorted_key_hashes.size - 1)
        matched_hashes = distinct_ngram_hashes[sorted_key_hashes[nearest_key_positions] == distinct_ngram_hashes]
        if matched_hashes.size == 0:
            continue
        for match_position in np.flatnonzero(is_inside_one_document & np.isin(ngram_hashes, matched_hashes)):
            document_index = int(document_of_word[match_position])
            matched_key_hashes_per_document.setdefault(document_index, set()).add(int(ngram_hashes[match_position]))
            if document_index not in first_match_per_document:
                first_word = int(match_position - first_word_of_document[document_index])
                matched_words = words_per_document[document_index][first_word: first_word + words_per_key]
                first_match_per_document[document_index] = b" ".join(matched_words).decode("utf-8", errors="replace")

    scanned_by_source = {}
    for source, token_count in zip(sources, token_counts):
        scanned = scanned_by_source.setdefault(source, {"documents": 0, "tokens": 0})
        scanned["documents"] += 1
        scanned["tokens"] += token_count
    flagged_documents = [{
        "doc_id": doc_ids[document_index], "source": sources[document_index], "token_count": token_counts[document_index],
        "matched_key_hashes": sorted(matched_key_hashes), "matched_text": first_match_per_document[document_index],
    } for document_index, matched_key_hashes in sorted(matched_key_hashes_per_document.items())]
    return {"scanned_by_source": scanned_by_source, "flagged_documents": flagged_documents}


def flag_documents_by_ngram(pool_paths, evaluation_items: list[dict], ngram_word_count: int = default_ngram_word_count,
                            min_word_count_for_whole_text_key: int = default_min_word_count_for_whole_text_key,
                            worker_count: int = 1, documents_per_worker_batch: int = 1_000) -> dict:
    """Return the pool documents that contain an evaluation key, and what was scanned.

    `flagged_documents` maps a doc_id to its source, token count, the tasks and the first item it matched and
    the matching stretch of its text; `scanned_by_source` and `key_count_per_item` are for the report.
    """
    start_seconds = time.perf_counter()
    evaluation_keys = build_evaluation_ngram_keys(evaluation_items, ngram_word_count, min_word_count_for_whole_text_key)
    record_batches = iterate_pool_record_batches(
        resolve_pool_parquet_paths(pool_paths), ("doc_id", "source", "token_count", "text"), set(), documents_per_worker_batch)
    scanned_by_source, flagged_documents = {}, {}
    for batch_result in run_batch_function_in_worker_processes(
            find_evaluation_ngrams_in_batch, record_batches, worker_count,
            remember_evaluation_keys_in_worker, (evaluation_keys["sorted_key_hashes_by_length"],)):
        for source, scanned in batch_result["scanned_by_source"].items():
            scanned_total = scanned_by_source.setdefault(source, {"documents": 0, "tokens": 0})
            scanned_total["documents"] += scanned["documents"]
            scanned_total["tokens"] += scanned["tokens"]
        for flagged_document in batch_result["flagged_documents"]:
            matched_key_hashes = np.array(flagged_document["matched_key_hashes"], dtype=np.uint64)
            # every item that owns one of the matched keys: a key shared by several items sits once per item
            first_rows = np.searchsorted(evaluation_keys["key_hashes"], matched_key_hashes, side="left")
            last_rows = np.searchsorted(evaluation_keys["key_hashes"], matched_key_hashes, side="right")
            matched_item_indices = sorted({
                int(item_index) for first_row, last_row in zip(first_rows, last_rows)
                for item_index in evaluation_keys["key_item_indices"][first_row:last_row]})
            flagged_documents[flagged_document["doc_id"]] = {
                "source": flagged_document["source"], "token_count": flagged_document["token_count"],
                "tasks": sorted({evaluation_items[item_index]["task"] for item_index in matched_item_indices}),
                "item_id": str(evaluation_items[matched_item_indices[0]]["item_id"]),
                "matched_text": flagged_document["matched_text"],
            }
    scanned_document_count = sum(scanned["documents"] for scanned in scanned_by_source.values())
    elapsed_seconds = time.perf_counter() - start_seconds
    print(f"ngram: {len(evaluation_keys['key_hashes']):,} keys from {len(evaluation_items):,} evaluation items; "
          f"{len(flagged_documents):,} of {scanned_document_count:,} documents flagged in {elapsed_seconds:.1f} s "
          f"({scanned_document_count / elapsed_seconds:,.0f} documents per second, {worker_count} workers)")
    return {"flagged_documents": flagged_documents, "scanned_by_source": scanned_by_source,
            "key_count_per_item": evaluation_keys["key_count_per_item"]}


# ---------------------------------------------------------------------------
# gate 1b, embedding: reworded copies, by cosine similarity of text windows
# ---------------------------------------------------------------------------

default_window_token_count = 256
# Cosine similarity at or above which a pool window counts as a copy of an evaluation window. A starting
# value: read the top matches and the histogram that `decontaminate` writes before relying on it.
default_embedding_similarity_threshold = 0.90
# Evaluation items carry no token count; 4.5 characters per token is what the Qwen tokenizer needs on
# English web text, and it only decides where a long question is cut into windows.
characters_per_token_of_evaluation_text = 4.5
similarity_histogram_bin_edges = np.round(np.linspace(0.0, 1.0, 51), 2)       # bins of 0.02
default_top_match_count = 50


def split_text_into_windows(text: str, token_count: int, window_token_count: int) -> list[str]:
    """Return `text` cut into consecutive windows of about `window_token_count` tokens.

    No tokenizer runs here: the pool row already says how many tokens the document has, so a window is
    `window_token_count` times the document's own characters-per-token ratio, which adapts to prose, code
    and CJK text alike. A cut moves back to the nearest space in the second half of the window when there is
    one, and a last piece shorter than a quarter window is joined to the window before it, because a fragment
    of a sentence has an arbitrary embedding.
    """
    if token_count <= window_token_count or len(text) == 0:
        return [text]
    characters_per_window = max(1, round(len(text) * window_token_count / token_count))
    windows, window_start = [], 0
    while window_start < len(text):
        window_end = min(window_start + characters_per_window, len(text))
        if len(text) - window_end < characters_per_window // 4:
            window_end = len(text)
        elif not text[window_end].isspace():
            last_space = text.rfind(" ", window_start + characters_per_window // 2, window_end)
            if last_space != -1:
                window_end = last_space
        windows.append(text[window_start:window_end])
        window_start = window_end
    return windows


def find_most_similar_evaluation_windows(window_embeddings: np.ndarray, evaluation_embeddings, device: str,
                                         windows_per_chunk: int = 4_096) -> tuple[np.ndarray, np.ndarray]:
    """Return, per pool window, its highest cosine similarity to any evaluation window and that window's index.

    `evaluation_embeddings` is a torch tensor that already sits on `device`; both sides are unit vectors.
    """
    import torch
    best_similarities = np.empty(len(window_embeddings), dtype=np.float32)
    best_evaluation_windows = np.empty(len(window_embeddings), dtype=np.int64)
    for chunk_start in range(0, len(window_embeddings), windows_per_chunk):
        chunk = slice(chunk_start, chunk_start + windows_per_chunk)
        similarities = torch.from_numpy(window_embeddings[chunk]).to(device) @ evaluation_embeddings.T
        chunk_best = similarities.max(dim=1)
        best_similarities[chunk] = chunk_best.values.cpu().numpy()
        best_evaluation_windows[chunk] = chunk_best.indices.cpu().numpy()
    return best_similarities, best_evaluation_windows


def flag_documents_by_embedding(pool_paths, evaluation_items: list[dict], embedding_model: str, device: str,
                                similarity_threshold: float = default_embedding_similarity_threshold,
                                window_token_count: int = default_window_token_count,
                                top_match_count: int = default_top_match_count, documents_per_read: int = 2_048) -> dict:
    """Return the pool documents with a window that resembles an evaluation question, plus what a person
    needs to judge the threshold.

    `flagged_documents` maps a doc_id to its source, token count, task, item and similarity (of its most
    similar window); `histogram_window_counts` counts all pool windows by their highest similarity to any
    evaluation window, in the bins of `similarity_histogram_bin_edges`; `top_matches` are the
    `top_match_count` most similar pairs with both texts.

    What this catches is a window that is mostly one evaluation question, reworded or not. A question that
    takes up a small part of a window is diluted by the rest; that case is what the n-gram check is for.
    """
    import torch
    start_seconds = time.perf_counter()
    evaluation_window_texts, item_of_evaluation_window = [], []
    for item_index, evaluation_item in enumerate(evaluation_items):
        question = evaluation_item["question"]
        estimated_token_count = round(len(question) / characters_per_token_of_evaluation_text)
        for window_text in split_text_into_windows(question, estimated_token_count, window_token_count):
            evaluation_window_texts.append(window_text)
            item_of_evaluation_window.append(item_index)
    evaluation_embeddings = torch.from_numpy(embed_documents(evaluation_window_texts, embedding_model, device)).to(device)

    histogram_window_counts = np.zeros(len(similarity_histogram_bin_edges) - 1, dtype=np.int64)
    flagged_documents, top_matches, window_count = {}, [], 0
    for record_batch in iterate_pool_record_batches(
            resolve_pool_parquet_paths(pool_paths), ("doc_id", "source", "token_count", "text"), set(), documents_per_read):
        doc_ids = record_batch.column("doc_id").to_pylist()
        sources = record_batch.column("source").to_pylist()
        token_counts = record_batch.column("token_count").to_pylist()
        window_texts, document_of_window = [], []
        for document_index, text in enumerate(record_batch.column("text").to_pylist()):
            for window_text in split_text_into_windows(text, token_counts[document_index], window_token_count):
                window_texts.append(window_text)
                document_of_window.append(document_index)
        best_similarities, best_evaluation_windows = find_most_similar_evaluation_windows(
            embed_documents(window_texts, embedding_model, device), evaluation_embeddings, device)
        window_count += len(window_texts)
        histogram_window_counts += np.histogram(np.clip(best_similarities, 0.0, 1.0), bins=similarity_histogram_bin_edges)[0]

        # the most similar windows first, so that the first window seen of a document is its best one
        for window_index in np.argsort(-best_similarities, kind="stable"):
            similarity = float(best_similarities[window_index])
            is_among_top_matches = len(top_matches) < top_match_count or similarity > top_matches[0][0]
            if similarity < similarity_threshold and not is_among_top_matches:
                break
            document_index = document_of_window[window_index]
            evaluation_window_index = int(best_evaluation_windows[window_index])
            evaluation_item = evaluation_items[item_of_evaluation_window[evaluation_window_index]]
            match = {
                "cosine_similarity": round(similarity, 4), "task": evaluation_item["task"],
                "item_id": str(evaluation_item["item_id"]), "doc_id": doc_ids[document_index],
                "pool_window_text": window_texts[window_index],
                "evaluation_window_text": evaluation_window_texts[evaluation_window_index]}
            if is_among_top_matches:
                # a heap of the `top_match_count` best pairs; the counter breaks ties between equal similarities
                heap_entry = (similarity, window_count - window_index, match)
                (heapq.heappush if len(top_matches) < top_match_count else heapq.heapreplace)(top_matches, heap_entry)
            if similarity >= similarity_threshold and doc_ids[document_index] not in flagged_documents:
                flagged_documents[doc_ids[document_index]] = {
                    "source": sources[document_index], "token_count": token_counts[document_index],
                    "task": match["task"], "item_id": match["item_id"], "cosine_similarity": match["cosine_similarity"]}
        elapsed_seconds = time.perf_counter() - start_seconds
        print(f"embedding: {window_count:,} windows, {window_count / elapsed_seconds:,.1f} per second, "
              f"{len(flagged_documents):,} documents flagged", flush=True)
    return {
        "flagged_documents": flagged_documents,
        "histogram_window_counts": histogram_window_counts,
        "top_matches": [match for _, _, match in sorted(top_matches, key=lambda entry: -entry[0])],
        "evaluation_window_count": len(evaluation_window_texts),
        "pool_window_count": window_count,
    }


# ---------------------------------------------------------------------------
# gate 1, decontaminate: run the methods and write the report and the flagged doc_ids
# ---------------------------------------------------------------------------

def count_flagged_by_source(flagged_documents: dict) -> dict:
    """Return {source: {"documents": n, "tokens": t}} for a {doc_id: {"source", "token_count", ...}} mapping."""
    counts_by_source = {}
    for flagged_document in flagged_documents.values():
        counts = counts_by_source.setdefault(flagged_document["source"], {"documents": 0, "tokens": 0})
        counts["documents"] += 1
        counts["tokens"] += flagged_document["token_count"]
    return dict(sorted(counts_by_source.items()))


def count_flagged_by_task_and_source(flagged_documents: dict) -> dict:
    """Return {task: {source: {"documents", "tokens"}}}; a document that matched two tasks counts under both."""
    documents_by_task = {}
    for doc_id, flagged_document in flagged_documents.items():
        for task in flagged_document.get("tasks", [flagged_document.get("task")]):
            documents_by_task.setdefault(task, {})[doc_id] = flagged_document
    return {task: count_flagged_by_source(documents) for task, documents in sorted(documents_by_task.items())}


def decontaminate_pool(pool_paths, evaluation_directory, flagged_path, top_matches_path,
                       report_path=default_decontamination_report_path, methods=("ngram", "embedding"),
                       embedding_model: str = default_embedding_model, device: str = "cpu",
                       similarity_threshold: float = default_embedding_similarity_threshold,
                       window_token_count: int = default_window_token_count,
                       ngram_word_count: int = default_ngram_word_count, worker_count: int = 1,
                       include_few_shot_files: bool = True) -> dict:
    """Gate 1: flag every pool document that overlaps an evaluation item, by n-gram, by embedding or both.

    Writes three files: the report (numbers and identifiers only; it is committed), the flagged doc_ids as
    parquet at `flagged_path`, and for the embedding method the most similar pairs with their texts at
    `top_matches_path`. The last two belong under $MIDTRAIN_DATA. Returns the report.
    """
    unknown_methods = set(methods) - {"ngram", "embedding"}
    assert methods and not unknown_methods, f"methods must be ngram and/or embedding, got {list(methods)}"
    evaluation_items = load_evaluation_items(evaluation_directory, include_few_shot_files)
    tasks = sorted({evaluation_item["task"] for evaluation_item in evaluation_items})
    report = {
        "settings": {
            "methods": list(methods), "ngram_word_count": ngram_word_count,
            "min_word_count_for_whole_text_key": default_min_word_count_for_whole_text_key,
            "embedding_model": embedding_model, "similarity_threshold": similarity_threshold,
            "window_token_count": window_token_count},
        "evaluation_items_by_task": {
            task: {"items": sum(evaluation_item["task"] == task for evaluation_item in evaluation_items)} for task in tasks},
        "flagged": {},
    }
    flagged_by_method = {"ngram": {}, "embedding": {}}

    if "ngram" in methods:
        ngram_result = flag_documents_by_ngram(pool_paths, evaluation_items, ngram_word_count, worker_count=worker_count)
        flagged_by_method["ngram"] = ngram_result["flagged_documents"]
        report["scanned_by_source"] = dict(sorted(ngram_result["scanned_by_source"].items()))
        for task in tasks:
            key_counts = [int(key_count) for evaluation_item, key_count in zip(evaluation_items, ngram_result["key_count_per_item"])
                          if evaluation_item["task"] == task]
            report["evaluation_items_by_task"][task]["ngram_keys"] = sum(key_counts)
            # an item without a key (every text shorter than 8 words) cannot be found by this method
            report["evaluation_items_by_task"][task]["items_without_ngram_key"] = sum(key_count == 0 for key_count in key_counts)
        report["flagged"]["ngram"] = {
            "by_source": count_flagged_by_source(flagged_by_method["ngram"]),
            "by_task_and_source": count_flagged_by_task_and_source(flagged_by_method["ngram"])}

    if "embedding" in methods:
        embedding_result = flag_documents_by_embedding(
            pool_paths, evaluation_items, embedding_model, device, similarity_threshold, window_token_count)
        flagged_by_method["embedding"] = embedding_result["flagged_documents"]
        report["flagged"]["embedding"] = {
            "by_source": count_flagged_by_source(flagged_by_method["embedding"]),
            "by_task_and_source": count_flagged_by_task_and_source(flagged_by_method["embedding"])}
        report["embedding_similarity_histogram"] = {
            "what": "pool windows by their highest cosine similarity to any evaluation window",
            "bin_edges": similarity_histogram_bin_edges.tolist(),
            "window_counts": embedding_result["histogram_window_counts"].tolist(),
            "pool_window_count": embedding_result["pool_window_count"],
            "evaluation_window_count": embedding_result["evaluation_window_count"]}
        # identifiers only: the texts of these pairs are in the file named by `top_matches_file_name`
        report["embedding_top_matches"] = [
            {key: match[key] for key in ("cosine_similarity", "task", "item_id", "doc_id")}
            for match in embedding_result["top_matches"]]
        report["top_matches_file_name"] = Path(top_matches_path).name
        Path(top_matches_path).parent.mkdir(parents=True, exist_ok=True)
        with open(top_matches_path, "w", encoding="utf-8") as top_matches_file:
            for match in embedding_result["top_matches"]:
                top_matches_file.write(json.dumps(match, ensure_ascii=False) + "\n")

    flagged_doc_ids = sorted(set(flagged_by_method["ngram"]) | set(flagged_by_method["embedding"]))
    flagged_by_either = {doc_id: flagged_by_method["ngram"].get(doc_id) or flagged_by_method["embedding"][doc_id]
                         for doc_id in flagged_doc_ids}
    report["flagged"]["either"] = {"by_source": count_flagged_by_source(flagged_by_either)}
    report["flagged_file_name"] = Path(flagged_path).name

    def get_flag_field(doc_id: str, method: str, field_name: str, value_when_not_flagged):
        return flagged_by_method[method].get(doc_id, {}).get(field_name, value_when_not_flagged)

    flagged_table = pa.table({
        "doc_id": pa.array(flagged_doc_ids, type=pa.string()),
        "source": pa.array([flagged_by_either[doc_id]["source"] for doc_id in flagged_doc_ids], type=pa.string()),
        "token_count": pa.array([flagged_by_either[doc_id]["token_count"] for doc_id in flagged_doc_ids], type=pa.int32()),
        "flagged_by_ngram": pa.array([doc_id in flagged_by_method["ngram"] for doc_id in flagged_doc_ids], type=pa.bool_()),
        "flagged_by_embedding": pa.array([doc_id in flagged_by_method["embedding"] for doc_id in flagged_doc_ids], type=pa.bool_()),
        "matched_tasks": pa.array([",".join(sorted(
            set(get_flag_field(doc_id, "ngram", "tasks", []))
            | ({get_flag_field(doc_id, "embedding", "task", None)} - {None}))) for doc_id in flagged_doc_ids], type=pa.string()),
        "ngram_item_id": pa.array([get_flag_field(doc_id, "ngram", "item_id", "") for doc_id in flagged_doc_ids], type=pa.string()),
        "ngram_matched_text": pa.array([get_flag_field(doc_id, "ngram", "matched_text", "") for doc_id in flagged_doc_ids], type=pa.string()),
        "embedding_item_id": pa.array([get_flag_field(doc_id, "embedding", "item_id", "") for doc_id in flagged_doc_ids], type=pa.string()),
        "embedding_cosine_similarity": pa.array(
            [get_flag_field(doc_id, "embedding", "cosine_similarity", float("nan")) for doc_id in flagged_doc_ids], type=pa.float32()),
    })
    Path(flagged_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(flagged_table, flagged_path)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"decontaminate: {len(flagged_doc_ids):,} documents flagged -> {flagged_path}; report -> {report_path}")
    for method, method_report in report["flagged"].items():
        for source, counts in method_report["by_source"].items():
            print(f"  {method:<9} {source}: {counts['documents']:,} documents, {counts['tokens']:,} tokens")
    return report


# ---------------------------------------------------------------------------
# gate 2, diversity: how varied is one source or batch
# ---------------------------------------------------------------------------

# Sample sizes. Every number below depends on the amount of text it is computed from, so they are fixed here
# and recorded in the result; `gate` only compares two results whose sample sizes agree.
default_token_sample_size = 1_000_000
default_nearest_neighbour_sample_document_count = 5_000
# 20,000 vectors is where brute-force HDBSCAN takes 8 s and 6.7 GB (its distance matrix grows with the square)
default_cluster_sample_document_count = 20_000
hdbscan_min_cluster_size = 20
hdbscan_min_samples = 10
# HDBSCAN runs on the first 16 principal directions of the sample. In all 384 dimensions every point is about as far
# from every other (the curse of dimensionality), and the first measurement of English Wikipedia found 2 clusters
# with 91% of the documents labelled noise (11-20): a count that says nothing. Topic-clustering pipelines reduce to
# 5-20 dimensions before density clustering for the same reason.
hdbscan_principal_direction_count = 16
# seed 0 for the random choice of the sampled documents; the same seed and pool give the same sample
default_diversity_seed = 0
# A sentence skeleton counts as a repeated template when it occurs more than 5 times in the sample...
template_min_occurrence_count = 6
# ...and has at least 4 words: "<name>.", "yes." or a bare heading repeat in any corpus and are not templates.
template_min_word_count = 4
# level 19 has a window of 8 MB, more than the sample of about 4.5 MB of text, so a sentence that returns
# anywhere in the sample is found; the default level 3 only looks back 2 MB
zstd_compression_level = 19

sentence_boundary_pattern = re.compile(r"(?<=[.!?。！？])\s+|\n+")
number_pattern = re.compile(r"\d+(?:[.,:/-]\d+)*")
capitalised_word_pattern = re.compile(r"\b[A-ZÀ-ÖØ-Þ][\w'’-]*")
repeated_name_placeholder_pattern = re.compile(r"<name>(?:\s+<name>)+")


def make_sentence_skeleton(sentence: str) -> str:
    """Return `sentence` with numbers and capitalised words replaced by placeholders, lowercased.

    "Maria Lopez paid 12.50 dollars in Austin." and "Tom paid 3 dollars in Rome." both become
    "<name> paid <number> dollars in <name>.": the slots of a template are what differs between its
    copies, and names and numbers are what generators put into slots.
    """
    skeleton = number_pattern.sub("<number>", sentence)
    skeleton = capitalised_word_pattern.sub("<name>", skeleton)
    skeleton = repeated_name_placeholder_pattern.sub("<name>", skeleton)
    return " ".join(skeleton.lower().split())


def compute_entropy_bits(counts: np.ndarray) -> float:
    """Return the entropy, in bits, of the distribution given by `counts`."""
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def measure_token_statistics(token_ids_per_document: list, token_sample_size: int) -> dict:
    """Return unigram and bigram entropy (bits) and distinct-1 / distinct-2 of the first `token_sample_size` tokens.

    Bigrams are pairs of neighbouring tokens inside one document; the pair formed by the last token of a
    document and the first token of the next is left out. The bigram entropy is the entropy of the pair, so
    it is at most twice the unigram entropy and at most log2 of the number of pairs in the sample.
    """
    token_ids = np.concatenate([np.asarray(token_ids, dtype=np.int64) for token_ids in token_ids_per_document])
    token_counts = np.array([len(token_ids) for token_ids in token_ids_per_document])
    is_last_token_of_document = np.zeros(token_ids.size, dtype=bool)
    is_last_token_of_document[np.cumsum(token_counts)[token_counts > 0] - 1] = True
    token_ids, is_last_token_of_document = token_ids[:token_sample_size], is_last_token_of_document[:token_sample_size]
    assert token_ids.size >= 2, "the sample holds fewer than 2 tokens"
    bigram_codes = (token_ids[:-1] << 32) | token_ids[1:]
    bigram_codes = bigram_codes[~is_last_token_of_document[:-1]]
    _, unigram_counts = np.unique(token_ids, return_counts=True)
    _, bigram_counts = np.unique(bigram_codes, return_counts=True)
    return {
        "sampled_token_count": int(token_ids.size),
        "unigram_entropy_bits": compute_entropy_bits(unigram_counts),
        "bigram_entropy_bits": compute_entropy_bits(bigram_counts),
        "distinct_1": float(unigram_counts.size / token_ids.size),
        "distinct_2": float(bigram_counts.size / max(bigram_codes.size, 1)),
    }


def measure_template_repetition(texts: list[str], tokenize_texts) -> dict:
    """Return the share of the tokens that sit in sentences whose skeleton is a repeated template.

    A text is cut into sentences at sentence-ending punctuation and at line breaks; a skeleton is a template
    when it occurs at least `template_min_occurrence_count` times in `texts` and has at least
    `template_min_word_count` words. `tokenize_texts` maps a list of strings to a list of token id lists.
    """
    sentences = [sentence.strip() for text in texts for sentence in sentence_boundary_pattern.split(text) if sentence.strip()]
    if not sentences:
        return {"template_repetition_rate": 0.0, "sentence_count": 0, "most_repeated_sentence_skeletons": []}
    skeletons = [make_sentence_skeleton(sentence) for sentence in sentences]
    occurrence_count_of_skeleton = Counter(skeletons)
    sentence_token_counts = np.array([len(token_ids) for token_ids in tokenize_texts(sentences)])
    is_template_sentence = np.array([
        occurrence_count_of_skeleton[skeleton] >= template_min_occurrence_count and len(skeleton.split()) >= template_min_word_count
        for skeleton in skeletons])
    most_repeated_skeletons = [
        {"skeleton": skeleton, "occurrences": occurrence_count}
        for skeleton, occurrence_count in occurrence_count_of_skeleton.most_common()
        if occurrence_count >= template_min_occurrence_count and len(skeleton.split()) >= template_min_word_count][:10]
    return {
        "template_repetition_rate": float(sentence_token_counts[is_template_sentence].sum() / max(sentence_token_counts.sum(), 1)),
        "sentence_count": len(sentences),
        "most_repeated_sentence_skeletons": most_repeated_skeletons,
    }


def vendi_score_of_unit_vectors(embeddings: np.ndarray) -> float:
    """The Vendi score with the cosine kernel: the exponential of the entropy of the eigenvalues of K / n, where K
    is the matrix of cosine similarities (Friedman and Dieng, 2023). It reads as an effective number of distinct
    documents: n copies of one document score 1, n orthogonal documents score n. For unit vectors X (n x d) the
    non-zero eigenvalues of X X^T / n equal those of X^T X / n, so a d x d matrix is enough. It needs no
    threshold and no clustering setting, which makes it the most dependable of the embedding checks."""
    unit_vectors = embeddings.astype(np.float64)
    unit_vectors /= np.linalg.norm(unit_vectors, axis=1, keepdims=True)
    eigenvalues = np.clip(np.linalg.eigvalsh(unit_vectors.T @ unit_vectors / len(unit_vectors)), 0.0, None)
    eigenvalues = eigenvalues[eigenvalues > 1e-12] / eigenvalues.sum()
    return float(np.exp(-(eigenvalues * np.log(eigenvalues)).sum()))


def measure_embedding_statistics(embeddings: np.ndarray, nearest_neighbour_sample_document_count: int) -> dict:
    """Return the mean nearest-neighbour cosine similarity, the Vendi score and the HDBSCAN clusters of unit vectors.

    The nearest-neighbour number uses the first `nearest_neighbour_sample_document_count` vectors, the Vendi score
    and HDBSCAN all of them. HDBSCAN runs on the first `hdbscan_principal_direction_count` principal directions of
    the sample (see the comment on that value) with Euclidean distance. `algorithm="brute"` because the tree
    algorithms that scikit-learn picks by default were slower here (20 s against 0.6 s for 5,000 vectors).
    """
    from sklearn.cluster import HDBSCAN
    nearest_neighbour_sample = embeddings[:nearest_neighbour_sample_document_count].astype(np.float32)
    similarities = nearest_neighbour_sample @ nearest_neighbour_sample.T
    np.fill_diagonal(similarities, -np.inf)
    statistics = {
        "nearest_neighbour_sample_document_count": int(len(nearest_neighbour_sample)),
        "mean_nearest_neighbour_cosine_similarity": float(similarities.max(axis=1).mean()) if len(nearest_neighbour_sample) > 1 else None,
        "cluster_sample_document_count": int(len(embeddings)),
        "embedding_vendi_score": vendi_score_of_unit_vectors(embeddings) if len(embeddings) > 1 else None,
        "hdbscan_principal_direction_count": hdbscan_principal_direction_count,
        "hdbscan_cluster_count": 0, "hdbscan_noise_share": 1.0, "hdbscan_clusters_per_10k_documents": 0.0,
    }
    if len(embeddings) >= 2 * hdbscan_min_cluster_size:
        from sklearn.decomposition import PCA
        direction_count = min(hdbscan_principal_direction_count, embeddings.shape[1], len(embeddings) - 1)
        reduced = PCA(n_components=direction_count, random_state=0).fit_transform(embeddings.astype(np.float64))
        cluster_labels = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples,
                                 algorithm="brute", copy=True).fit(reduced).labels_
        statistics["hdbscan_cluster_count"] = int(cluster_labels.max() + 1)
        statistics["hdbscan_noise_share"] = float((cluster_labels == -1).mean())
        statistics["hdbscan_clusters_per_10k_documents"] = float(statistics["hdbscan_cluster_count"] / len(embeddings) * 10_000)
    return statistics


def read_document_sample(parquet_paths, token_sample_size: int, embedding_sample_document_count: int, seed: int) -> dict:
    """Return a random sample of the documents in `parquet_paths`, in random order, and the totals of the batch.

    The sample is long enough for both uses: at least `embedding_sample_document_count` documents, and as
    many as it takes to reach `token_sample_size` tokens. Rows are drawn by position, not by doc_id, so a
    batch that holds the same document many times shows it many times in the sample.
    """
    token_counts = np.concatenate([
        pq.read_table(parquet_path, columns=["token_count"]).column("token_count").to_numpy(zero_copy_only=False)
        for parquet_path in parquet_paths]).astype(np.int64)
    random_order = np.random.default_rng(seed).permutation(token_counts.size)
    document_count_for_token_sample = int(np.searchsorted(np.cumsum(token_counts[random_order]), token_sample_size)) + 1
    sampled_rows = random_order[: max(document_count_for_token_sample, embedding_sample_document_count)]
    position_in_sample_of_row = np.full(token_counts.size, -1, dtype=np.int64)
    position_in_sample_of_row[sampled_rows] = np.arange(sampled_rows.size)

    sample_texts = [None] * sampled_rows.size
    first_row_of_batch = 0
    for record_batch in iterate_pool_record_batches(parquet_paths, ("doc_id", "text"), set(), 4_096):
        positions_in_sample = position_in_sample_of_row[first_row_of_batch: first_row_of_batch + record_batch.num_rows]
        rows_in_batch = np.flatnonzero(positions_in_sample >= 0)
        for row_in_batch, text in zip(rows_in_batch, record_batch.column("text").take(pa.array(rows_in_batch)).to_pylist()):
            sample_texts[positions_in_sample[row_in_batch]] = text
        first_row_of_batch += record_batch.num_rows
    assert first_row_of_batch == token_counts.size and all(text is not None for text in sample_texts), "pool rows changed while reading"
    return {"texts": sample_texts, "document_count": int(token_counts.size), "token_count": int(token_counts.sum())}


def measure_diversity(parquet_paths, tokenize_texts, embedding_model: str, device: str,
                      token_sample_size: int = default_token_sample_size,
                      nearest_neighbour_sample_document_count: int = default_nearest_neighbour_sample_document_count,
                      cluster_sample_document_count: int = default_cluster_sample_document_count,
                      seed: int = default_diversity_seed) -> dict:
    """Gate 2, measurement: return the diversity numbers of the documents in `parquet_paths`.

    `tokenize_texts` maps a list of strings to a list of token id lists (the Qwen tokenizer on the command
    line, a toy in the tests). Token statistics, compression and template repetition use the documents that
    supply the first `token_sample_size` tokens of a random sample; the embedding statistics use the first
    `cluster_sample_document_count` documents of the same sample (their first 512 tokens).
    """
    start_seconds = time.perf_counter()
    document_sample = read_document_sample(parquet_paths, token_sample_size, cluster_sample_document_count, seed)
    sample_texts = document_sample["texts"]

    # tokenise in the order of the sample until the token budget is reached
    token_ids_per_document, token_count_so_far = [], 0
    for chunk_start in range(0, len(sample_texts), 256):
        token_ids_of_chunk = tokenize_texts(sample_texts[chunk_start: chunk_start + 256])
        token_ids_per_document.extend(token_ids_of_chunk)
        token_count_so_far += sum(len(token_ids) for token_ids in token_ids_of_chunk)
        if token_count_so_far >= token_sample_size:
            break
    token_sample_texts = sample_texts[: len(token_ids_per_document)]
    diversity = {
        "document_count": document_sample["document_count"], "token_count": document_sample["token_count"],
        "seed": seed, "token_sample_size": token_sample_size, "token_sample_document_count": len(token_sample_texts),
    }
    diversity.update(measure_token_statistics(token_ids_per_document, token_sample_size))

    text_bytes = "\n".join(token_sample_texts).encode("utf-8", errors="surrogatepass")
    compressed_byte_count = len(zstandard.ZstdCompressor(level=zstd_compression_level).compress(text_bytes))
    diversity["zstd_compression_ratio"] = float(len(text_bytes) / max(compressed_byte_count, 1))
    diversity.update(measure_template_repetition(token_sample_texts, tokenize_texts))
    print(f"diversity: token statistics on {diversity['sampled_token_count']:,} tokens of {len(token_sample_texts):,} documents "
          f"in {time.perf_counter() - start_seconds:.1f} s", flush=True)

    embedding_sample_texts = sample_texts[:cluster_sample_document_count]
    embeddings = embed_documents(embedding_sample_texts, embedding_model, device)
    diversity["embedding_model"] = embedding_model
    diversity.update(measure_embedding_statistics(embeddings, nearest_neighbour_sample_document_count))
    print(f"diversity: embedding statistics on {len(embedding_sample_texts):,} documents; total {time.perf_counter() - start_seconds:.1f} s")
    return diversity


def get_diversity_result_path(name: str) -> Path:
    """Return results/hygiene_diversity_<name>.json."""
    return repository_root / "results" / f"hygiene_diversity_{name}.json"


def load_qwen_tokenize_function():
    """Return a function that maps a list of strings to their Qwen token ids (no special tokens)."""
    from transformers import AutoTokenizer
    assert "MIDTRAIN_TOKENIZER" in os.environ, "MIDTRAIN_TOKENIZER is not set: run `source scripts/env.sh` first"
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MIDTRAIN_TOKENIZER"])

    def tokenize_texts(texts: list[str]) -> list[list[int]]:
        return tokenizer(texts, add_special_tokens=False)["input_ids"]
    return tokenize_texts


# ---------------------------------------------------------------------------
# gate 2, gate: compare a new batch with a baseline measured on authentic text
# ---------------------------------------------------------------------------

default_gate_thresholds = {
    # the batch fails when its bigram entropy is more than this many bits below the baseline
    "max_bigram_entropy_drop_bits": 0.5,
    # ...when its template repetition rate is more than this many times the baseline's
    "max_template_repetition_ratio": 3.0,
    # ...when it has fewer than this share of the baseline's clusters per 10,000 documents
    "min_cluster_count_ratio": 0.5,
    # ...when its Vendi score (the effective number of distinct documents) is below this share of the baseline's
    "min_vendi_score_ratio": 0.8,
    # ...when its compression ratio is more than this many times the baseline's
    "max_compression_ratio_ratio": 1.3,
    # A baseline with almost no repeated template would turn any repetition into "3 times higher"; the
    # baseline rate is not taken below this value.
    "template_repetition_rate_floor": 0.01,
    # two measurements are comparable when their sample sizes differ by at most this share
    "max_sample_size_difference": 0.10,
}


def check_diversity_against_baseline(batch_diversity: dict, baseline_diversity: dict, thresholds: dict = None) -> list[dict]:
    """Return one entry per check: its name, both values, the limit, whether it passed and by how much it missed.

    Raises ValueError when the two measurements were not made on samples of the same size, because then a
    difference between them says nothing about the text.
    """
    thresholds = {**default_gate_thresholds, **(thresholds or {})}
    for sample_size_name in ("sampled_token_count", "cluster_sample_document_count"):
        batch_size, baseline_size = batch_diversity[sample_size_name], baseline_diversity[sample_size_name]
        if abs(batch_size - baseline_size) > thresholds["max_sample_size_difference"] * max(batch_size, baseline_size):
            raise ValueError(
                f"{sample_size_name} is {batch_size:,} for the batch and {baseline_size:,} for the baseline. Diversity "
                "numbers grow with the sample, so measure the larger side again on the smaller sample: `diversity` "
                f"with --token-sample-size {min(batch_diversity['sampled_token_count'], baseline_diversity['sampled_token_count'])} "
                f"--cluster-sample-documents {min(batch_diversity['cluster_sample_document_count'], baseline_diversity['cluster_sample_document_count'])} "
                "and an --output-name of its own")

    bigram_entropy_limit = baseline_diversity["bigram_entropy_bits"] - thresholds["max_bigram_entropy_drop_bits"]
    template_rate_limit = thresholds["max_template_repetition_ratio"] * max(
        baseline_diversity["template_repetition_rate"], thresholds["template_repetition_rate_floor"])
    cluster_limit = thresholds["min_cluster_count_ratio"] * baseline_diversity["hdbscan_clusters_per_10k_documents"]
    vendi_limit = thresholds["min_vendi_score_ratio"] * (baseline_diversity.get("embedding_vendi_score") or 0.0)
    compression_limit = thresholds["max_compression_ratio_ratio"] * baseline_diversity["zstd_compression_ratio"]
    checks = [
        ("bigram_entropy_bits", "at least", bigram_entropy_limit),
        ("template_repetition_rate", "at most", template_rate_limit),
        ("hdbscan_clusters_per_10k_documents", "at least", cluster_limit),
        ("zstd_compression_ratio", "at most", compression_limit),
    ]
    if baseline_diversity.get("embedding_vendi_score") is not None and batch_diversity.get("embedding_vendi_score") is not None:
        checks.append(("embedding_vendi_score", "at least", vendi_limit))
    check_results = []
    for measurement_name, direction, limit in checks:
        batch_value, baseline_value = batch_diversity[measurement_name], baseline_diversity[measurement_name]
        has_passed = batch_value >= limit if direction == "at least" else batch_value <= limit
        check_results.append({
            "measurement": measurement_name, "batch": batch_value, "baseline": baseline_value,
            "requirement": f"{direction} {limit:.4g}", "passed": bool(has_passed),
            "missed_by": 0.0 if has_passed else float(abs(batch_value - limit))})
    return check_results


def run_diversity_gate(batch_name: str, baseline_name: str, thresholds: dict = None) -> int:
    """Print the comparison of two stored measurements and return the exit status: 0 pass, 1 fail, 2 not comparable."""
    for name in (batch_name, baseline_name):
        assert get_diversity_result_path(name).exists(), (
            f"{get_diversity_result_path(name)} is missing: run `diversity --source {name}` first")
    batch_diversity = json.loads(get_diversity_result_path(batch_name).read_text())
    baseline_diversity = json.loads(get_diversity_result_path(baseline_name).read_text())
    try:
        check_results = check_diversity_against_baseline(batch_diversity, baseline_diversity, thresholds)
    except ValueError as error:
        print(f"gate: cannot compare {batch_name} with {baseline_name}: {error}")
        return 2
    for check_result in check_results:
        verdict = "ok  " if check_result["passed"] else "FAIL"
        missed_text = "" if check_result["passed"] else f"  (missed by {check_result['missed_by']:.4g})"
        print(f"  {verdict} {check_result['measurement']:<36} {batch_name} {check_result['batch']:.4g}   "
              f"{baseline_name} {check_result['baseline']:.4g}   must be {check_result['requirement']}{missed_text}")
    failed_measurements = [check_result["measurement"] for check_result in check_results if not check_result["passed"]]
    if failed_measurements:
        print(f"gate: {batch_name} is less diverse than {baseline_name}: failed {', '.join(failed_measurements)}")
        return 1
    print(f"gate: {batch_name} passes against {baseline_name}")
    return 0


# ---------------------------------------------------------------------------
# what the flagged matches consist of
# ---------------------------------------------------------------------------
# The first run on the whole pool (11-20) flagged 6,030 documents, and reading the matched 13-grams showed that most
# of them are not benchmark text: HumanEval and MBPP test cases hold long lists of numbers, and any table of numbers
# in a paper or a math page contains the same run. A count of flagged documents therefore overstates contamination.
# This breakdown makes the overstatement visible; the gate itself stays strict, because dropping every flagged
# document costs 0.2% of the pool and a real leak would cost the evaluation.
number_or_symbol_word_pattern = re.compile(r"[\d.,/+\-*=()]+")
alphabet_run_pattern = re.compile(r"^(?:[a-z] ){6,}[a-z]$")


def kind_of_matched_ngram(matched_text: str) -> str:
    """Sort one matched 13-gram (normalised words, space separated) into one of four kinds:
    numbers (every word is a number or symbol), mostly_numbers (half or more), alphabet (a run of single letters,
    as in "a b c d e f g"), words (the rest, where real leaks are found)."""
    words = matched_text.split()
    number_word_count = sum(bool(number_or_symbol_word_pattern.fullmatch(word)) for word in words)
    if number_word_count == len(words):
        return "numbers"
    if number_word_count >= len(words) / 2:
        return "mostly_numbers"
    if alphabet_run_pattern.match(matched_text):
        return "alphabet"
    return "words"


def explain_flagged_matches(flagged_path: Path, report_path: Path) -> dict:
    """Count the flagged documents by kind of match, per task; write the counts (no text) into the report file."""
    flagged = pq.read_table(flagged_path, columns=["ngram_matched_text", "matched_tasks", "flagged_by_ngram"]).to_pydict()
    by_kind: dict[str, int] = {}
    by_task_and_kind: dict[str, dict[str, int]] = {}
    for matched_text, matched_tasks, is_ngram_flag in zip(flagged["ngram_matched_text"], flagged["matched_tasks"],
                                                          flagged["flagged_by_ngram"]):
        if not is_ngram_flag:
            continue
        kind = kind_of_matched_ngram(matched_text)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        for task_name in matched_tasks.split(","):
            task_counts = by_task_and_kind.setdefault(task_name.strip(), {})
            task_counts[kind] = task_counts.get(kind, 0) + 1
    report = json.loads(report_path.read_text())
    report["ngram_flags_by_kind_of_match"] = {"all_tasks": by_kind, "by_task": by_task_and_kind}
    report_path.write_text(json.dumps(report, indent=1) + "\n")
    total = sum(by_kind.values())
    for kind, count in sorted(by_kind.items(), key=lambda item: -item[1]):
        print(f"  {kind:16s} {count:6,d} documents ({100 * count / max(1, total):.1f}%)")
    return by_kind


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def find_evaluation_items_matched_by_source(source_name: str, task_names: list[str]) -> dict[str, list[str]]:
    """Every evaluated item of the named tasks whose n-gram key occurs anywhere in one pool source.

    `decontaminate` records one matched item per flagged document, which is enough to leave a document out but not to
    say which test items a source touches: a solution can match several. This looks up every key of every document
    in a set (45,177 solutions in about 25 seconds, 11-23) and writes the item ids to
    $MIDTRAIN_DATA/hygiene/matched_items_<source>.json, where `mixing_law.py compare-with-base` reads them to report
    each gain without the matched items (README §1.2, PITFALLS A4). Few-shot files are not read: a shot is not tested.
    """
    evaluation_items = []
    for task_name in task_names:
        task_path = get_midtrain_data_directory() / "eval" / f"{task_name}.jsonl"
        evaluation_items.extend(json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip())
    keys = build_evaluation_ngram_keys(evaluation_items)
    item_ids_of_key: dict[int, set] = {}
    for key_hash, item_index in zip(keys["key_hashes"].tolist(), keys["key_item_indices"].tolist()):
        item_ids_of_key.setdefault(key_hash, set()).add(evaluation_items[item_index]["item_id"])
    matched_item_ids = set()
    for pool_path in resolve_pool_parquet_paths([get_midtrain_data_directory() / "pool" / source_name]):
        for text in pq.read_table(pool_path, columns=["text"]).column("text").to_pylist():
            word_hashes = hash_words(normalise_into_words(text))
            for words_per_key in keys["sorted_key_hashes_by_length"]:
                if len(word_hashes) < words_per_key:
                    continue
                for key_hash in set(hash_word_ngrams(word_hashes, words_per_key).tolist()):
                    matched_item_ids |= item_ids_of_key.get(key_hash, set())
    matched_by_task = {task_name: sorted(item_id for item_id in matched_item_ids if item_id.startswith(f"{task_name}:"))
                       for task_name in task_names}
    output_path = get_midtrain_data_directory() / "hygiene" / f"matched_items_{source_name}.json"
    output_path.write_text(json.dumps(matched_by_task, indent=1) + "\n")
    print(f"{source_name}: " + ", ".join(f"{task_name} {len(item_ids)} items" for task_name, item_ids in matched_by_task.items()))
    return matched_by_task


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the parser with the commands decontaminate, diversity and gate."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    decontaminate_parser = subparsers.add_parser("decontaminate", help="flag pool documents that overlap evaluation items")
    decontaminate_parser.add_argument("--pool", nargs="*", default=None,
                                      help="parquet files, folders or globs (default: the whole $MIDTRAIN_DATA/pool)")
    decontaminate_parser.add_argument("--evaluation-dir", type=Path, default=None, help="default: $MIDTRAIN_DATA/eval")
    decontaminate_parser.add_argument("--methods", nargs="+", choices=("ngram", "embedding"), default=["ngram", "embedding"])
    decontaminate_parser.add_argument("--output-dir", type=Path, default=None,
                                      help="folder for the flagged doc_ids and the top matches (default: $MIDTRAIN_DATA/hygiene)")
    decontaminate_parser.add_argument("--report-path", type=Path, default=default_decontamination_report_path)
    decontaminate_parser.add_argument("--ngram-word-count", type=int, default=default_ngram_word_count)
    decontaminate_parser.add_argument("--similarity-threshold", type=float, default=default_embedding_similarity_threshold)
    decontaminate_parser.add_argument("--window-token-count", type=int, default=default_window_token_count)
    decontaminate_parser.add_argument("--embedding-model", default=default_embedding_model)
    decontaminate_parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu; required for the embedding method")
    decontaminate_parser.add_argument("--worker-count", type=int, default=get_usable_cpu_count())
    decontaminate_parser.add_argument("--leave-out-few-shot-files", action="store_true",
                                      help="check against the tested items only (for a source made from training splits)")

    explain_parser = subparsers.add_parser("explain-flags", help="count the flagged documents by kind of match")
    explain_parser.add_argument("--flagged-path", type=Path, default=None,
                                help="default: $MIDTRAIN_DATA/hygiene/decontamination_flagged.parquet")
    explain_parser.add_argument("--report-path", type=Path, default=default_decontamination_report_path)

    diversity_parser = subparsers.add_parser("diversity", help="measure the diversity of one source or batch")
    diversity_parser.add_argument("--source", required=True, help="pool source to measure, or the name of the batch given by --pool")
    diversity_parser.add_argument("--pool", nargs="*", default=None,
                                  help="parquet files, folders or globs of the batch (default: $MIDTRAIN_DATA/pool/<source>)")
    diversity_parser.add_argument("--output-name", default=None, help="results/hygiene_diversity_<name>.json (default: the source)")
    diversity_parser.add_argument("--token-sample-size", type=int, default=default_token_sample_size)
    diversity_parser.add_argument("--nearest-neighbour-sample-documents", type=int, default=default_nearest_neighbour_sample_document_count)
    diversity_parser.add_argument("--cluster-sample-documents", type=int, default=default_cluster_sample_document_count)
    diversity_parser.add_argument("--seed", type=int, default=default_diversity_seed)
    diversity_parser.add_argument("--embedding-model", default=default_embedding_model)
    diversity_parser.add_argument("--device", required=True, help="cuda, cuda:1 or cpu")

    matched_parser = subparsers.add_parser("matched-items", help="every test item a pool source touches, by n-grams")
    matched_parser.add_argument("--source", required=True, help="a pool source, for example synthetic_solutions")
    matched_parser.add_argument("--tasks", nargs="+", default=["gsm8k", "math500", "svamp", "asdiv_a"])

    gate_parser = subparsers.add_parser("gate", help="fail when a batch is clearly less diverse than a baseline")
    gate_parser.add_argument("--source", required=True, help="name of the new batch (a diversity result)")
    gate_parser.add_argument("--baseline", required=True, help="name of the diversity result measured on authentic text")
    for threshold_name, threshold_value in default_gate_thresholds.items():
        gate_parser.add_argument(f"--{threshold_name.replace('_', '-')}", type=float, default=threshold_value)
    return parser


def main(argument_list=None) -> int:
    """Run one command; return the exit status (only `gate` returns anything but 0)."""
    arguments = build_argument_parser().parse_args(argument_list)
    if arguments.command == "decontaminate":
        assert "embedding" not in arguments.methods or arguments.device is not None, (
            "the embedding method needs --device (or run --methods ngram alone)")
        output_directory = arguments.output_dir or get_midtrain_data_directory() / "hygiene"
        decontaminate_pool(
            arguments.pool, arguments.evaluation_dir or get_midtrain_data_directory() / "eval",
            output_directory / "decontamination_flagged.parquet", output_directory / "decontamination_top_matches.jsonl",
            arguments.report_path, tuple(arguments.methods), arguments.embedding_model, arguments.device,
            arguments.similarity_threshold, arguments.window_token_count, arguments.ngram_word_count, arguments.worker_count,
            not arguments.leave_out_few_shot_files)
        if "ngram" in arguments.methods:
            explain_flagged_matches(output_directory / "decontamination_flagged.parquet", arguments.report_path)
        return 0
    if arguments.command == "explain-flags":
        explain_flagged_matches(arguments.flagged_path or get_midtrain_data_directory() / "hygiene" / "decontamination_flagged.parquet",
                                arguments.report_path)
        return 0
    if arguments.command == "matched-items":
        find_evaluation_items_matched_by_source(arguments.source, arguments.tasks)
        return 0
    if arguments.command == "diversity":
        pool_paths = arguments.pool or [get_midtrain_data_directory() / "pool" / arguments.source]
        diversity = measure_diversity(
            resolve_pool_parquet_paths(pool_paths), load_qwen_tokenize_function(), arguments.embedding_model, arguments.device,
            arguments.token_sample_size, arguments.nearest_neighbour_sample_documents, arguments.cluster_sample_documents,
            arguments.seed)
        result_path = get_diversity_result_path(arguments.output_name or arguments.source)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"source": arguments.source, **diversity}, indent=2) + "\n")
        print(f"diversity -> {result_path}")
        return 0
    thresholds = {threshold_name: getattr(arguments, threshold_name) for threshold_name in default_gate_thresholds}
    return run_diversity_gate(arguments.source, arguments.baseline, thresholds)


if __name__ == "__main__":
    sys.exit(main())
