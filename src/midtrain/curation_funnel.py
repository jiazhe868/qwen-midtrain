"""Curation funnel: decide, document by document, which pool rows are worth training on.

Why this file exists. The usual recipe for continued pretraining is "take a big corpus and train". On a pool
mixed from web replay, curated long-form text, synthetic rephrasings, math, code and procedural traces that
wastes most of the compute: the same page arrives through two sources, a paraphrase arrives ten times, half of
the tokens are already predicted by the model, and some can be predicted by nobody. The funnel removes those in
the order of their cost, cheapest first, so the expensive stages only see what survived the cheap ones:

    filter    string heuristics: length bounds, repetition, symbol soup, menus; tags the script
    dedup     exact copies (hash of normalised text), then near copies (MinHash + banding), across sources
    embed     one vector per document (first 512 tokens, small sentence-embedding model)
    semdedup  paraphrases: k-means on the vectors, then cosine similarity inside each cluster
    score     mean next-token loss per document under a causal LM (run once per model)
    select    learnability: keep what the student cannot predict yet and a stronger reference can
    report    merge everything into results/funnel_report.json

Contracts every stage keeps:
  * A stage never rewrites text. It reads pool parquet files (docs/conventions.md) and writes one small
    parquet keyed by `doc_id` with `keep` and `drop_reason`, plus the numbers it measured, so a threshold can
    be changed later without recomputing them.
  * A stage only looks at documents that every earlier stage kept (`earlier_decision_paths`), and its file
    holds exactly those documents. `report` refuses to merge files that do not chain (a stale stage).
  * The documents to train on are the rows with `keep` true in the last stage's file.
  * A `doc_id` appears once in the pool. It is derived from the text, so a repeat is the same text in the same
    source, and a decision keyed by `doc_id` could not keep one copy and drop the other; the stages stop with
    an error instead.
  * Across sources the copy from the source that comes first in `default_source_priority` wins.

History (what was measured while writing this file; pool-scale numbers are still to come):
  * Workload, 2026-11-20: 6,000 fineweb-edu and 3,000 English Wikipedia documents streamed into pool format
    (the "public sample" below), plus synthetic sets. No number here comes from the real pool yet; rerun
    `report` on it and look at the per-reason token shares before trusting any default.
  * MinHash in numpy: the 128 hash functions are applied to a block of about 4,096 shingles at a time. Larger
    blocks fall out of the CPU cache (1.8 ns per shingle and hash function at 4,096, 3.9 ns at 65,536).
  * Duplicate groups: joining every confirmed pair (connected components) and keeping one document per
    component lets chains A~B~C drop C although C does not resemble the kept A. The leader rule in
    `find_near_duplicates` drops a document only when it agrees with a kept document; see its docstring.

Commands (run from the repository root; every stage has --help):
    source scripts/env.sh
    $MIDTRAIN_PYTHON -m midtrain.curation_funnel filter
    $MIDTRAIN_PYTHON -m midtrain.curation_funnel dedup
    $MIDTRAIN_PYTHON -m midtrain.curation_funnel embed --device cuda
    $MIDTRAIN_PYTHON -m midtrain.curation_funnel semdedup --device cuda
    $MIDTRAIN_PYTHON -m midtrain.curation_funnel score --score-name student \
        --model-path $MIDTRAIN_MODELS/qwen35_0.8B_base --device cuda
    $MIDTRAIN_PYTHON -m midtrain.curation_funnel score --score-name reference \
        --model-path $MIDTRAIN_MODELS/qwen35_9B_base --device cuda
    $MIDTRAIN_PYTHON -m midtrain.curation_funnel select
    $MIDTRAIN_PYTHON -m midtrain.curation_funnel report
    $MIDTRAIN_PYTHON -m pytest tests/test_curation_funnel.py -q

The embedding model is downloaded once, with the network allowed:
    env -u HF_HUB_OFFLINE $MIDTRAIN_PYTHON -c \
        "from midtrain.curation_funnel import load_embedding_model as load; load('BAAI/bge-small-en-v1.5', 'cpu')"
"""

import argparse
import concurrent.futures
import functools
import glob
import hashlib
import json
import multiprocessing
import os
import re
import time
import zlib
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# torch and transformers are imported inside the stages that need them (embed, semdedup, score). Importing
# torch takes about 2 s against 0.3 s for numpy and pyarrow, and the filter and dedup stages start one
# worker process per core, none of which touches a model.


# ---------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------

repository_root = Path(__file__).resolve().parents[2]
default_funnel_report_path = repository_root / "results" / "funnel_report.json"

# The stages that write per-document decisions, in the order the funnel runs them. `embed` and `score` write
# measurements (vectors, losses); `semdedup` and `select` turn those into decisions.
decision_stage_names = ("filter", "dedup", "semdedup", "select")
required_decision_column_names = ("doc_id", "source", "token_count", "keep", "drop_reason")


def get_midtrain_data_directory() -> Path:
    """Return `$MIDTRAIN_DATA`, the folder for everything too large to commit."""
    assert "MIDTRAIN_DATA" in os.environ, "MIDTRAIN_DATA is not set: run `source scripts/env.sh` first"
    return Path(os.environ["MIDTRAIN_DATA"])


def get_decision_path(funnel_directory: Path, stage_name: str) -> Path:
    """Return the decision file a stage writes into the funnel folder."""
    assert stage_name in decision_stage_names, f"{stage_name!r} is not one of {decision_stage_names}"
    return Path(funnel_directory) / f"{stage_name}_decisions.parquet"


def get_default_earlier_decision_paths(funnel_directory: Path, stage_name: str) -> list[Path]:
    """Return the decision files of the stages that run before `stage_name`, and insist that they exist.

    `embed` and `score` are not decision stages; they sit before `semdedup` and `select` and see what those
    stages will see. A missing file stops the run: silently skipping a stage would let its documents through
    without anyone noticing. To leave a stage out on purpose, pass the list of files explicitly.
    """
    stage_to_first_later_decision_stage = {"embed": "semdedup", "score": "select"}
    reference_stage_name = stage_to_first_later_decision_stage.get(stage_name, stage_name)
    earlier_stage_names = decision_stage_names[: decision_stage_names.index(reference_stage_name)]
    earlier_decision_paths = [get_decision_path(funnel_directory, name) for name in earlier_stage_names]
    for earlier_decision_path in earlier_decision_paths:
        assert earlier_decision_path.exists(), (
            f"{earlier_decision_path} is missing: run that stage first, or name the decision files to apply "
            f"with --earlier-decisions (none at all runs `{stage_name}` on the whole pool)")
    return earlier_decision_paths


# ---------------------------------------------------------------------------
# reading the pool and writing decisions (shared by every stage, and by data_hygiene.py)
# ---------------------------------------------------------------------------

def resolve_pool_parquet_paths(pool_paths) -> list[Path]:
    """Return the parquet files that `pool_paths` names, sorted so that every run reads them in one order.

    Each entry may be a parquet file, a folder (every parquet file below it) or a glob pattern. Nothing at
    all (None or an empty list) means the whole pool, `$MIDTRAIN_DATA/pool/<source>/*.parquet`.
    """
    if not pool_paths:
        pool_paths = [get_midtrain_data_directory() / "pool"]
    parquet_paths = []
    for pool_path in pool_paths:
        if Path(pool_path).is_dir():
            parquet_paths.extend(sorted(Path(pool_path).rglob("*.parquet")))
        else:
            parquet_paths.extend(Path(matched_path) for matched_path in sorted(glob.glob(str(pool_path))))
    assert parquet_paths, f"no parquet file found under {[str(pool_path) for pool_path in pool_paths]}"
    return parquet_paths


def read_doc_ids_dropped_by_earlier_stages(earlier_decision_paths) -> set[str]:
    """Return every `doc_id` that one of the earlier decision files dropped."""
    dropped_doc_ids = set()
    for earlier_decision_path in earlier_decision_paths:
        decisions = pq.read_table(earlier_decision_path, columns=["doc_id", "keep"])
        is_dropped = ~decisions.column("keep").to_numpy(zero_copy_only=False)
        doc_ids = decisions.column("doc_id").to_numpy(zero_copy_only=False)
        dropped_doc_ids.update(doc_ids[is_dropped].tolist())
        print(f"  applying {Path(earlier_decision_path).name}: {int(is_dropped.sum()):,} of {len(doc_ids):,} "
              f"documents were dropped there")
    return dropped_doc_ids


def iterate_pool_record_batches(parquet_paths, column_names, dropped_doc_ids, documents_per_batch):
    """Yield arrow record batches of the pool, without the documents an earlier stage dropped.

    The batches stay in arrow form because they are sent to worker processes: an arrow batch pickles as a
    few buffers, while a Python list of strings would be copied character by character in the parent, which
    is the one process that cannot be multiplied.
    """
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(batch_size=documents_per_batch, columns=list(column_names)):
            for column_name in column_names:
                assert record_batch.column(column_name).null_count == 0, f"{parquet_path} has empty {column_name!r} cells"
            if dropped_doc_ids:
                is_still_kept = [doc_id not in dropped_doc_ids for doc_id in record_batch.column("doc_id").to_pylist()]
                record_batch = record_batch.filter(pa.array(is_still_kept, type=pa.bool_()))
            if record_batch.num_rows > 0:
                yield record_batch


def run_batch_function_in_worker_processes(batch_function, record_batches, worker_count,
                                           worker_initializer=None, worker_initializer_arguments=()):
    """Yield `batch_function(record_batch)` for every batch, in the order of the batches.

    With `worker_count` of 1 everything runs in this process, which is what the tests and a debugger want.
    Otherwise the workers are spawned, not forked: by the time a stage starts, numpy and arrow have already
    started their own threads, and a fork taken while another thread holds a lock can hang the child. A
    spawned worker pays 0.3 s to import this module once.

    At most 2 batches per worker are in flight, so the pool is never read into memory faster than it is
    processed. The executor is `concurrent.futures`, not `multiprocessing.Pool`, because of how each reacts
    to a worker that dies while starting: `Pool` replaces it silently, forever, and the stage hangs (this
    happened on the first run of this file, from a script without a `__main__` guard); the executor raises.
    """
    if worker_count <= 1:
        if worker_initializer is not None:
            worker_initializer(*worker_initializer_arguments)
        yield from map(batch_function, record_batches)
        return
    spawn_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            worker_count, mp_context=spawn_context, initializer=worker_initializer,
            initargs=worker_initializer_arguments) as worker_pool:
        results_in_flight = deque()
        try:
            for record_batch in record_batches:
                results_in_flight.append(worker_pool.submit(batch_function, record_batch))
                if len(results_in_flight) >= 2 * worker_count:
                    yield results_in_flight.popleft().result()
            while results_in_flight:
                yield results_in_flight.popleft().result()
        except concurrent.futures.process.BrokenProcessPool as error:
            raise RuntimeError(
                "a worker process died. Spawned workers import the script that started them, so a script that "
                "calls a stage function must do so under `if __name__ == \"__main__\":`; otherwise look for the "
                "worker's own error above (out of memory is the usual one)") from error


def assert_doc_ids_are_unique(doc_ids) -> None:
    """Stop when a `doc_id` occurs twice: a decision keyed by `doc_id` cannot tell the two rows apart."""
    distinct_doc_id_count = len(set(doc_ids))
    assert distinct_doc_id_count == len(doc_ids), (
        f"{len(doc_ids) - distinct_doc_id_count:,} pool rows repeat a doc_id (the same text twice in one source). "
        "Decisions are keyed by doc_id, so both copies would be kept: write each doc_id once when building the pool")


def write_stage_decisions(output_path, stage_name: str, decision_columns: dict, stage_settings: dict) -> None:
    """Write one stage's per-document decisions, with the settings that produced them in the file metadata.

    `decision_columns` maps a column name to one value per document. The five shared columns are checked here
    because `report` and every later stage rely on them; a stage adds whatever it measured as extra columns.
    """
    for column_name in required_decision_column_names:
        assert column_name in decision_columns, f"stage {stage_name!r} wrote no {column_name!r} column"
    arrow_columns = {}
    for column_name, column_values in decision_columns.items():
        if column_name == "token_count":
            arrow_columns[column_name] = pa.array(np.asarray(column_values), type=pa.int32())
        elif column_name == "keep":
            arrow_columns[column_name] = pa.array(np.asarray(column_values, dtype=bool))
        elif isinstance(column_values, np.ndarray) and column_values.dtype != object:
            arrow_columns[column_name] = pa.array(column_values)
        else:
            arrow_columns[column_name] = pa.array(list(column_values), type=pa.string())
    decision_table = pa.table(arrow_columns)
    is_dropped = ~np.asarray(decision_columns["keep"], dtype=bool)
    drop_reasons = np.asarray(list(decision_columns["drop_reason"]), dtype=object)
    assert all(reason != "" for reason in drop_reasons[is_dropped]), "a dropped document has no drop_reason"
    assert all(reason == "" for reason in drop_reasons[~is_dropped]), "a kept document carries a drop_reason"
    decision_table = decision_table.replace_schema_metadata(
        {"stage": stage_name, "settings_json": json.dumps(stage_settings, default=str)})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(decision_table, output_path)
    print_stage_summary(stage_name, summarise_stage_decisions(read_decision_columns(output_path)), output_path)


def read_decision_columns(decision_path, column_names=None) -> dict:
    """Return the columns of a decision (or score) file as numpy arrays, plus its `settings`."""
    decision_table = pq.read_table(decision_path, columns=column_names)
    decision_columns = {
        column_name: decision_table.column(column_name).to_numpy(zero_copy_only=False)
        for column_name in decision_table.column_names}
    file_metadata = decision_table.schema.metadata or {}
    decision_columns["settings"] = json.loads(file_metadata.get(b"settings_json", b"{}"))
    return decision_columns


def count_documents_and_tokens(token_counts: np.ndarray, is_selected: np.ndarray) -> dict:
    """Return how many documents `is_selected` marks and how many tokens they hold."""
    return {"documents": int(is_selected.sum()), "tokens": int(token_counts[is_selected].sum())}


def summarise_stage_decisions(decision_columns: dict) -> dict:
    """Count documents and tokens that entered a stage, were kept and were dropped, per source and per reason."""
    token_counts = decision_columns["token_count"].astype(np.int64)
    is_kept = decision_columns["keep"].astype(bool)
    sources = decision_columns["source"]
    drop_reasons = decision_columns["drop_reason"]

    def summarise_subset(is_in_subset: np.ndarray) -> dict:
        subset_summary = {
            "entered": count_documents_and_tokens(token_counts, is_in_subset),
            "kept": count_documents_and_tokens(token_counts, is_in_subset & is_kept),
            "dropped_by_reason": {},
        }
        for drop_reason in sorted(set(drop_reasons[is_in_subset & ~is_kept].tolist())):
            subset_summary["dropped_by_reason"][drop_reason] = count_documents_and_tokens(
                token_counts, is_in_subset & (drop_reasons == drop_reason))
        return subset_summary

    stage_summary = summarise_subset(np.ones(len(token_counts), dtype=bool))
    stage_summary["by_source"] = {
        source: summarise_subset(sources == source) for source in sorted(set(sources.tolist()))}
    return stage_summary


def print_stage_summary(stage_name: str, stage_summary: dict, output_path) -> None:
    """Print what a stage kept and why it dropped the rest, one line per source."""
    entered, kept = stage_summary["entered"], stage_summary["kept"]
    print(f"{stage_name}: kept {kept['documents']:,} of {entered['documents']:,} documents, "
          f"{kept['tokens']:,} of {entered['tokens']:,} tokens -> {output_path}")
    for drop_reason, dropped in stage_summary["dropped_by_reason"].items():
        print(f"  dropped as {drop_reason}: {dropped['documents']:,} documents, {dropped['tokens']:,} tokens")
    for source, source_summary in stage_summary["by_source"].items():
        kept_token_share = source_summary["kept"]["tokens"] / max(source_summary["entered"]["tokens"], 1)
        print(f"  {source}: kept {source_summary['kept']['documents']:,} of "
              f"{source_summary['entered']['documents']:,} documents ({kept_token_share:.1%} of its tokens)")


def get_usable_cpu_count() -> int:
    """Return the cores this process may run on (a batch job is often given fewer than the machine has)."""
    return len(os.sched_getaffinity(0))


# ---------------------------------------------------------------------------
# stage 1, filter: cheap string heuristics, one reason per dropped document
# ---------------------------------------------------------------------------

# Starting values, to be revisited on the real pool; each is a parameter of `filter` on the command line. The
# numbers in the comments are from the public sample (6,000 web, 3,000 Wikipedia, 2,000 math-web documents),
# where the defaults drop 0.001% of the web tokens, 0.02% of the Wikipedia tokens and 0.7% of the math tokens.
default_filter_thresholds = {
    # Below 64 tokens a text is a stub, a caption or a navigation crumb (the 16 such Wikipedia rows were all
    # "X may refer to:" pages), and it has too few 5-word shingles for the dedup stage to say much about it.
    "min_token_count": 64,
    # A document above 100,000 tokens is a dump (a log, a table, a whole book in one row). The longest of the
    # 11,000 sampled documents has 47,000 tokens, so this bound is for accidents, not for long articles.
    "max_token_count": 100_000,
    # Share of all word 4-grams taken by the most frequent one. A loop with a period of up to 20 words
    # reaches 0.05. The median web page is at 0.004 and 99.9% of them stay below 0.034; 0.05 dropped none of
    # 6,000 web pages, 1 of 3,000 Wikipedia pages and 12 of 2,000 math pages (integer tables, quiz chrome).
    "max_most_frequent_word_4gram_share": 0.05,
    # ...and only when that 4-gram occurs at least 4 times: in a 60-word text a phrase used twice already has
    # a share of 0.035, and a phrase used two or three times is a refrain, not a loop.
    "min_most_frequent_word_4gram_count": 4,
    # Letters and digits among the non-space characters. Web and Wikipedia text never fell below 0.80; math
    # pages go down to 0.60 at their 1st percentile because formulas are mostly symbols, so the bound sits
    # below that. The one math page under 0.50 was mis-decoded text (mojibake), which is what this is for.
    "min_alphanumeric_character_share": 0.50,
    # Share of the characters that sit on very short lines. Counting characters instead of lines keeps an
    # article with many headings (median 0.3% on the web, 2.4% on Wikipedia) apart from a menu or a link
    # list. 0.50 dropped no web page and 2 Wikipedia pages: a bare list of names, and an article whose
    # tables had been stripped down to their headings.
    "max_short_line_character_share": 0.50,
    # A very short line has at most this many characters after stripping: "Privacy Policy", "Read more" and
    # "Share on Facebook" fit, a sentence does not. Applied while measuring, so it is the same for all sources.
    "very_short_line_max_characters": 25,
}

# What differs per source, each with its reason. A source that is not listed gets the defaults.
filter_checks_switched_off = {
    "max_most_frequent_word_4gram_share": 1.0, "min_alphanumeric_character_share": 0.0,
    "max_short_line_character_share": 1.0}
default_filter_threshold_overrides_by_source = {
    # Code and procedural traces are line-oriented and repetitive by construction: "}" and "return x" are very
    # short lines, and a trace repeats its step template in every step. The three text checks would read that
    # structure as boilerplate, so these sources only get the token bounds until someone measures them.
    "code": filter_checks_switched_off,
    "primitives": filter_checks_switched_off,
    # Worked solutions are written one equation per line. At 0.50, 14 of 2,000 math pages were over the
    # limit, among them step-by-step solutions and a computer-algebra session; at 0.80 it is 2.
    "math_web": {"max_short_line_character_share": 0.80},
}

filter_drop_reasons_in_checking_order = (
    "too_few_tokens", "too_many_tokens", "repeated_word_4gram", "low_alphanumeric_share", "mostly_short_lines")

# Unicode ranges (first, last) used to tag the script of a document. "cjk" means the scripts written without
# spaces between words plus Hangul; the tag exists so that later stages need not rediscover that
# `text.split()` returns one "word" per sentence there.
latin_code_point_ranges = ((0x0041, 0x024F), (0x1E00, 0x1EFF), (0xFF21, 0xFF5A))
cjk_code_point_ranges = (
    (0x2E80, 0x2FDF), (0x3040, 0x30FF), (0x3100, 0x312F), (0x31F0, 0x31FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFF66, 0xFF9F), (0x20000, 0x323AF))
character_class_names = ("symbol", "whitespace", "digit", "latin_letter", "cjk_letter", "other_letter")
# One CJK character carries about as much text as 2.4 Latin characters (the Qwen tokenizer needs 4.5
# characters per token on English web text and 1.9 on Chinese or Japanese prose). So a document is mostly
# CJK once 30% of its letters are (0.3 x 2.4 against 0.7), and a "very short line" is 2.4 times shorter there.
latin_characters_per_cjk_character = 2.4
min_cjk_letter_share_for_cjk_tag = 0.30

cjk_character_class_pattern = "".join(f"{chr(first)}-{chr(last)}" for first, last in cjk_code_point_ranges)
# In a CJK document every CJK character is a word of its own; everything else splits on white space.
cjk_character_or_word_pattern = re.compile(f"[{cjk_character_class_pattern}]|[^\\s{cjk_character_class_pattern}]+")


@functools.lru_cache(maxsize=1)
def get_character_class_by_code_point() -> np.ndarray:
    """Return a table that maps every Unicode code point to an index into `character_class_names`.

    Built once per process (0.23 s) from Python's own Unicode database, so that classifying a document is
    one table lookup per character in numpy instead of a Python call per character.
    """
    symbol, whitespace, digit, latin_letter, cjk_letter, other_letter = range(len(character_class_names))
    character_class_by_code_point = np.full(0x110000, symbol, dtype=np.uint8)
    for code_point in range(0x110000):
        character = chr(code_point)
        if character.isalpha():
            character_class_by_code_point[code_point] = other_letter
        elif character.isalnum():
            character_class_by_code_point[code_point] = digit
        elif character.isspace():
            character_class_by_code_point[code_point] = whitespace
    for script_class, code_point_ranges in ((latin_letter, latin_code_point_ranges), (cjk_letter, cjk_code_point_ranges)):
        for first_code_point, last_code_point in code_point_ranges:
            classes_in_range = character_class_by_code_point[first_code_point: last_code_point + 1]
            classes_in_range[classes_in_range == other_letter] = script_class
    return character_class_by_code_point


def count_character_classes(text: str) -> np.ndarray:
    """Return how many characters of `text` fall into each class of `character_class_names`."""
    code_points = np.frombuffer(text.encode("utf-32-le", errors="surrogatepass"), dtype="<u4")
    character_classes = get_character_class_by_code_point()[code_points]
    return np.bincount(character_classes, minlength=len(character_class_names))


def tag_script_from_character_class_counts(character_class_counts: np.ndarray) -> str:
    """Return "latin", "cjk" or "other" from the letters of a document ("other" when it has no letters)."""
    latin_letter_count = int(character_class_counts[character_class_names.index("latin_letter")])
    cjk_letter_count = int(character_class_counts[character_class_names.index("cjk_letter")])
    letter_count = latin_letter_count + cjk_letter_count + int(
        character_class_counts[character_class_names.index("other_letter")])
    if letter_count == 0:
        return "other"
    if cjk_letter_count >= min_cjk_letter_share_for_cjk_tag * letter_count:
        return "cjk"
    if latin_letter_count >= 0.5 * letter_count:
        return "latin"
    return "other"


def split_into_normalised_words(text: str, script_tag: str) -> list[bytes]:
    """Return the lowercased words of `text` as UTF-8 bytes; joined by one space they are the normalised text.

    Words are what white space separates. A CJK document has no such separators, so there every CJK character
    counts as a word; otherwise a whole paragraph would be one word and the document one 5-word shingle.
    The words are bytes because the stable word hash (crc32) takes bytes.
    """
    lowercased_text = text.lower()
    if script_tag == "cjk":
        words = cjk_character_or_word_pattern.findall(lowercased_text)
    else:
        words = lowercased_text.split()
    if not words:
        return []
    return " ".join(words).encode("utf-8", errors="surrogatepass").split(b" ")


def measure_filter_statistics_for_batch(record_batch, very_short_line_max_characters: int) -> dict:
    """Worker: measure the heuristics of every document in one batch. No drop threshold is applied here.

    Keeping the measurement (slow, parallel) apart from the decision (fast, in the parent) is what lets the
    decision file store the raw numbers. The one setting needed while measuring is how long a "very short"
    line may be.
    """
    statistics = {
        "doc_id": record_batch.column("doc_id").to_pylist(),
        "source": record_batch.column("source").to_pylist(),
        "token_count": record_batch.column("token_count").to_numpy(zero_copy_only=False).astype(np.int32),
        "script_tag": [],
        "most_frequent_word_4gram_share": [],
        "most_frequent_word_4gram_count": [],
        "alphanumeric_character_share": [],
        "short_line_character_share": [],
    }
    symbol, whitespace = character_class_names.index("symbol"), character_class_names.index("whitespace")
    for text in record_batch.column("text").to_pylist():
        character_class_counts = count_character_classes(text)
        script_tag = tag_script_from_character_class_counts(character_class_counts)
        statistics["script_tag"].append(script_tag)

        non_space_character_count = int(character_class_counts.sum() - character_class_counts[whitespace])
        alphanumeric_character_count = non_space_character_count - int(character_class_counts[symbol])
        statistics["alphanumeric_character_share"].append(
            alphanumeric_character_count / non_space_character_count if non_space_character_count else 0.0)

        words = split_into_normalised_words(text, script_tag)
        word_4gram_count = len(words) - 3
        most_frequent_word_4gram_count = 0
        if word_4gram_count > 0:
            most_frequent_word_4gram_count = max(Counter(zip(words, words[1:], words[2:], words[3:])).values())
        statistics["most_frequent_word_4gram_count"].append(most_frequent_word_4gram_count)
        statistics["most_frequent_word_4gram_share"].append(
            most_frequent_word_4gram_count / word_4gram_count if word_4gram_count > 0 else 0.0)

        line_lengths = [len(line.strip()) for line in text.split("\n")]
        short_line_limit = very_short_line_max_characters
        if script_tag == "cjk":
            short_line_limit = very_short_line_max_characters / latin_characters_per_cjk_character
        character_count_on_lines = sum(line_lengths)
        character_count_on_short_lines = sum(length for length in line_lengths if length <= short_line_limit)
        statistics["short_line_character_share"].append(
            character_count_on_short_lines / character_count_on_lines if character_count_on_lines else 0.0)
    return statistics


def choose_filter_drop_reasons(statistics: dict, thresholds_by_source: dict) -> np.ndarray:
    """Return one drop reason per document ("" when it is kept): the first failed check in a fixed order.

    `thresholds_by_source` holds the complete thresholds of every source that occurs. The order of the checks
    is `filter_drop_reasons_in_checking_order`, cheapest and least debatable first, so a 12-token stub is
    reported as too short even if it also repeats itself.
    """
    sources = np.asarray(statistics["source"], dtype=object)
    token_counts = np.asarray(statistics["token_count"])
    drop_reasons = np.full(len(sources), "", dtype=object)
    for source, thresholds in thresholds_by_source.items():
        is_in_source = sources == source
        has_failed_check = {
            "too_few_tokens": token_counts < thresholds["min_token_count"],
            "too_many_tokens": token_counts > thresholds["max_token_count"],
            "repeated_word_4gram":
                (np.asarray(statistics["most_frequent_word_4gram_share"]) > thresholds["max_most_frequent_word_4gram_share"])
                & (np.asarray(statistics["most_frequent_word_4gram_count"]) >= thresholds["min_most_frequent_word_4gram_count"]),
            "low_alphanumeric_share":
                np.asarray(statistics["alphanumeric_character_share"]) < thresholds["min_alphanumeric_character_share"],
            "mostly_short_lines":
                np.asarray(statistics["short_line_character_share"]) > thresholds["max_short_line_character_share"],
        }
        for drop_reason in reversed(filter_drop_reasons_in_checking_order):      # earlier reasons overwrite later ones
            drop_reasons[is_in_source & has_failed_check[drop_reason]] = drop_reason
    return drop_reasons


def filter_documents(pool_paths, output_path, thresholds=None, threshold_overrides_by_source=None,
                     worker_count=1, documents_per_worker_batch=1_000) -> dict:
    """Stage 1: measure the cheap heuristics of every pool document and write keep / drop_reason.

    `thresholds` replaces entries of `default_filter_thresholds` for every source;
    `threshold_overrides_by_source` (default `default_filter_threshold_overrides_by_source`) then replaces
    entries for single sources. Returns the decision columns it wrote.
    """
    thresholds = {**default_filter_thresholds, **(thresholds or {})}
    if threshold_overrides_by_source is None:
        threshold_overrides_by_source = default_filter_threshold_overrides_by_source
    for threshold_names in [thresholds, *threshold_overrides_by_source.values()]:
        unknown_threshold_names = set(threshold_names) - set(default_filter_thresholds)
        assert not unknown_threshold_names, f"unknown filter thresholds {sorted(unknown_threshold_names)}"
    assert all("very_short_line_max_characters" not in overrides for overrides in threshold_overrides_by_source.values()), (
        "the length of a very short line is applied while measuring and cannot differ between sources")
    start_seconds = time.perf_counter()
    record_batches = iterate_pool_record_batches(
        resolve_pool_parquet_paths(pool_paths), ("doc_id", "source", "token_count", "text"), set(),
        documents_per_worker_batch)
    measure_statistics_for_batch = functools.partial(
        measure_filter_statistics_for_batch, very_short_line_max_characters=thresholds["very_short_line_max_characters"])
    statistics_per_batch = list(run_batch_function_in_worker_processes(
        measure_statistics_for_batch, record_batches, worker_count))
    assert statistics_per_batch, "the pool holds no documents"
    decision_columns = {}
    for column_name in statistics_per_batch[0]:
        if column_name == "token_count":
            decision_columns[column_name] = np.concatenate([batch[column_name] for batch in statistics_per_batch])
        else:
            decision_columns[column_name] = [value for batch in statistics_per_batch for value in batch[column_name]]
    assert_doc_ids_are_unique(decision_columns["doc_id"])
    for column_name in ("most_frequent_word_4gram_share", "alphanumeric_character_share", "short_line_character_share"):
        decision_columns[column_name] = np.asarray(decision_columns[column_name], dtype=np.float32)
    decision_columns["most_frequent_word_4gram_count"] = np.asarray(
        decision_columns["most_frequent_word_4gram_count"], dtype=np.int32)
    thresholds_by_source = {
        source: {**thresholds, **threshold_overrides_by_source.get(source, {})}
        for source in sorted(set(decision_columns["source"]))}
    decision_columns["drop_reason"] = choose_filter_drop_reasons(decision_columns, thresholds_by_source)
    decision_columns["keep"] = decision_columns["drop_reason"] == ""
    elapsed_seconds = time.perf_counter() - start_seconds
    print(f"filter: measured {len(decision_columns['doc_id']):,} documents in {elapsed_seconds:.1f} s "
          f"({len(decision_columns['doc_id']) / elapsed_seconds:,.0f} documents per second, {worker_count} workers)")
    write_stage_decisions(output_path, "filter", decision_columns, {"thresholds_by_source": thresholds_by_source})
    return decision_columns


# ---------------------------------------------------------------------------
# stage 2, dedup: exact copies by hash, near copies by MinHash and banding, across sources
# ---------------------------------------------------------------------------

# Which copy survives when two sources hold the same document: the source listed first. Curated text comes
# before web replay because its copy is the cleaner one (no page chrome), and authentic text comes before
# generated text because a rephrasing that is still a near copy of a real page has failed as a rephrasing.
default_source_priority = ("wiki_en", "papers", "math_web", "code", "replay_web", "synthetic_rephrase", "primitives")

word_shingle_length = 5
minhash_permutation_count = 128
# 16 bands of 8 rows: two documents with Jaccard similarity J share at least one band with probability
# 1 - (1 - J^8)^16, which is 0.24 at J = 0.6, 0.61 at 0.7, 0.95 at 0.8 and 0.994 at 0.85. The half-way
# point is J = 0.71, so nearly every pair above 0.8 becomes a candidate and few pairs below 0.6 do.
default_lsh_band_count = 16
default_lsh_rows_per_band = 8
# A candidate pair is confirmed when at least this share of the 128 MinHash values agree. The share is an
# unbiased estimate of the Jaccard similarity with a standard deviation of 0.035 at J = 0.8.
default_signature_agreement_threshold = 0.80
# Inside one band bucket every document is compared with the 4 documents just before it (in priority
# order), not with all of them: a bucket of 10,000 template pages would otherwise produce 50 million pairs.
# Buckets of up to 5 documents are still compared exhaustively.
default_bucket_neighbour_window = 4
# The hash functions are applied to about this many shingles at a time: 128 x 4,096 products of 8 bytes are
# 4 MB and stay in the CPU cache (1.8 ns per product at 4,096, 2.3 ns at 16,384, 3.9 ns at 65,536).
shingles_per_permutation_block = 4_096

# Constants of the hash functions. They are fixed numbers, not `hash()`: Python salts its string hash per
# process, and the signatures are computed in many worker processes whose values must be comparable.
ngram_hash_multiplier = np.uint64(0x9E3779B97F4A7C15)           # odd, from the golden ratio; any odd number works
hash_mixing_multiplier = np.uint64(0xBF58476D1CE4E5B9)          # the first constant of the splitmix64 finaliser
# seed 20261120 is the date this file was written; changing it changes every signature, so stored signatures
# from different seeds must never be compared
minhash_random_generator = np.random.default_rng(20261120)
minhash_multipliers = minhash_random_generator.integers(0, 2**64, size=minhash_permutation_count, dtype=np.uint64) | np.uint64(1)
minhash_offsets = minhash_random_generator.integers(0, 2**64, size=minhash_permutation_count, dtype=np.uint64)


def hash_words(words: list[bytes]) -> np.ndarray:
    """Return a stable 32-bit hash (crc32) of every word, as uint64 so that later arithmetic wraps at 64 bits."""
    return np.fromiter(map(zlib.crc32, words), dtype=np.uint64, count=len(words))


def hash_word_ngrams(word_hashes: np.ndarray, ngram_length: int) -> np.ndarray:
    """Return one 64-bit hash per run of `ngram_length` consecutive words (empty when there are too few words).

    A polynomial in the word hashes, so the order of the words matters, followed by a mixing step so that
    all 64 bits depend on all the words. The arithmetic wraps modulo 2^64 on purpose.
    """
    ngram_count = word_hashes.size - ngram_length + 1
    if ngram_count <= 0:
        return np.empty(0, dtype=np.uint64)
    ngram_hashes = word_hashes[:ngram_count].copy()
    for word_offset in range(1, ngram_length):
        ngram_hashes *= ngram_hash_multiplier
        ngram_hashes += word_hashes[word_offset: word_offset + ngram_count]
    ngram_hashes ^= ngram_hashes >> np.uint64(31)
    ngram_hashes *= hash_mixing_multiplier
    ngram_hashes ^= ngram_hashes >> np.uint64(29)
    return ngram_hashes


def compute_minhash_signatures(shingle_keys_per_document: list[np.ndarray]) -> np.ndarray:
    """Return the MinHash signatures, shape (documents, 128), dtype uint32.

    `shingle_keys_per_document` holds, per document, its shingles as 32-bit keys in a uint64 array, at least
    one key each. Hash function k maps a key x to the top 32 bits of (multiplier_k * x + offset_k) mod 2^64,
    and the signature keeps the smallest value over the document's shingles. Two documents agree on a given k
    with probability equal to the Jaccard similarity of their shingle sets.

    Documents are packed into blocks so that numpy does the work for many of them at once; `reduceat` then
    takes the minimum inside each document's stretch of the block.
    """
    assert all(shingle_keys.size > 0 for shingle_keys in shingle_keys_per_document), "a document has no shingle"
    signatures = np.empty((len(shingle_keys_per_document), minhash_permutation_count), dtype=np.uint32)
    block_first_document = 0
    while block_first_document < len(shingle_keys_per_document):
        block_last_document = block_first_document
        shingle_count_in_block = 0
        while block_last_document < len(shingle_keys_per_document) and (
                shingle_count_in_block == 0 or shingle_count_in_block < shingles_per_permutation_block):
            shingle_count_in_block += shingle_keys_per_document[block_last_document].size
            block_last_document += 1
        keys_in_block = shingle_keys_per_document[block_first_document: block_last_document]
        first_shingle_of_document = np.cumsum([0] + [shingle_keys.size for shingle_keys in keys_in_block[:-1]])
        hashed_keys = minhash_multipliers[:, None] * np.concatenate(keys_in_block)[None, :]
        hashed_keys += minhash_offsets[:, None]
        smallest_hashed_keys = np.minimum.reduceat(hashed_keys, first_shingle_of_document, axis=1)
        signatures[block_first_document: block_last_document] = (smallest_hashed_keys >> np.uint64(32)).T
        block_first_document = block_last_document
    return signatures


def compute_duplicate_fingerprints_for_batch(record_batch) -> dict:
    """Worker: the exact-copy hash and the MinHash signature of every document in one batch."""
    texts = record_batch.column("text").to_pylist()
    exact_text_hashes = np.empty(len(texts), dtype=np.uint64)
    shingle_keys_per_document = []
    for document_index, text in enumerate(texts):
        script_tag = tag_script_from_character_class_counts(count_character_classes(text))
        words = split_into_normalised_words(text, script_tag)
        normalised_text_digest = hashlib.blake2b(b" ".join(words), digest_size=8).digest()
        exact_text_hashes[document_index] = int.from_bytes(normalised_text_digest, "little")
        shingle_hashes = hash_word_ngrams(hash_words(words), word_shingle_length)
        if shingle_hashes.size == 0:
            # Fewer than 5 words: there is no 5-word shingle, so the whole text is the one shingle. Two such
            # texts then agree on every hash function when they are equal and on none otherwise.
            shingle_hashes = exact_text_hashes[document_index: document_index + 1]
        shingle_keys_per_document.append(shingle_hashes >> np.uint64(32))
    return {
        "doc_id": record_batch.column("doc_id").to_pylist(),
        "source": record_batch.column("source").to_pylist(),
        "token_count": record_batch.column("token_count").to_numpy(zero_copy_only=False).astype(np.int32),
        "exact_text_hash": exact_text_hashes,
        "minhash_signature": compute_minhash_signatures(shingle_keys_per_document),
    }


def find_exact_duplicates(exact_text_hashes: np.ndarray) -> np.ndarray:
    """Return, per document, the index of the first document with the same hash, or -1 when it is that first one.

    The documents must be in priority order, so "first" means "the copy to keep".
    """
    document_count = exact_text_hashes.size
    sorted_order = np.argsort(exact_text_hashes, kind="stable")      # stable: equal hashes stay in priority order
    sorted_hashes = exact_text_hashes[sorted_order]
    starts_new_hash = np.ones(document_count, dtype=bool)
    starts_new_hash[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
    position_of_first_copy = np.maximum.accumulate(np.where(starts_new_hash, np.arange(document_count), 0))
    duplicate_of_index = np.full(document_count, -1, dtype=np.int64)
    duplicate_of_index[sorted_order[~starts_new_hash]] = sorted_order[position_of_first_copy[~starts_new_hash]]
    return duplicate_of_index


def find_candidate_pairs_with_lsh_banding(signatures: np.ndarray, band_count: int, rows_per_band: int,
                                          bucket_neighbour_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the candidate pairs (earlier index, later index): documents that share all rows of some band.

    Per band, the rows are hashed into one number per document, the documents are sorted by that number, and
    a run of equal numbers is a bucket. A stable sort keeps a bucket in index order, which is priority order.
    Each document is paired with the `bucket_neighbour_window` documents before it in its bucket.
    """
    document_count = signatures.shape[0]
    assert band_count * rows_per_band <= signatures.shape[1], "more band rows than MinHash values"
    pair_keys_per_band = [np.empty(0, dtype=np.uint64)]
    for band_index in range(band_count):
        band_rows = signatures[:, band_index * rows_per_band: (band_index + 1) * rows_per_band]
        band_hashes = band_rows[:, 0].astype(np.uint64)
        for row_index in range(1, rows_per_band):
            band_hashes *= ngram_hash_multiplier
            band_hashes += band_rows[:, row_index]
        sorted_order = np.argsort(band_hashes, kind="stable")
        sorted_band_hashes = band_hashes[sorted_order]
        for neighbour_offset in range(1, min(bucket_neighbour_window, document_count - 1) + 1):
            is_in_same_bucket = sorted_band_hashes[neighbour_offset:] == sorted_band_hashes[:-neighbour_offset]
            earlier_indices = sorted_order[:-neighbour_offset][is_in_same_bucket].astype(np.uint64)
            later_indices = sorted_order[neighbour_offset:][is_in_same_bucket].astype(np.uint64)
            pair_keys_per_band.append(earlier_indices * np.uint64(document_count) + later_indices)
    # a pair that shares several bands appears once per band; it only needs to be confirmed once
    unique_pair_keys = np.unique(np.concatenate(pair_keys_per_band))
    earlier_indices = (unique_pair_keys // np.uint64(document_count)).astype(np.int64)
    later_indices = (unique_pair_keys % np.uint64(document_count)).astype(np.int64)
    return earlier_indices, later_indices


def measure_signature_agreement(signatures: np.ndarray, first_indices: np.ndarray, second_indices: np.ndarray,
                                pairs_per_chunk: int = 200_000) -> np.ndarray:
    """Return, per pair, the share of the 128 MinHash values on which the two documents agree."""
    agreement = np.empty(first_indices.size, dtype=np.float32)
    for chunk_start in range(0, first_indices.size, pairs_per_chunk):      # 200,000 pairs gather about 200 MB
        chunk = slice(chunk_start, chunk_start + pairs_per_chunk)
        agreement[chunk] = (signatures[first_indices[chunk]] == signatures[second_indices[chunk]]).mean(axis=1)
    return agreement


def label_connected_components(node_count: int, first_indices: np.ndarray, second_indices: np.ndarray) -> np.ndarray:
    """Return, per node, the smallest node index of its connected component.

    Every round gives both ends of an edge the smaller of their two labels and then replaces each label by
    the label of the node it points to. A chain of n nodes settles in about log2(n) rounds, and the rounds
    are plain numpy, so two million nodes need no Python loop over edges.
    """
    labels = np.arange(node_count, dtype=np.int64)
    while True:
        smaller_labels = np.minimum(labels[first_indices], labels[second_indices])
        new_labels = labels.copy()
        np.minimum.at(new_labels, first_indices, smaller_labels)
        np.minimum.at(new_labels, second_indices, smaller_labels)
        new_labels = new_labels[new_labels]
        if np.array_equal(new_labels, labels):
            return labels
        labels = new_labels


def find_near_duplicates(signatures: np.ndarray,
                         agreement_threshold: float = default_signature_agreement_threshold,
                         band_count: int = default_lsh_band_count,
                         rows_per_band: int = default_lsh_rows_per_band,
                         bucket_neighbour_window: int = default_bucket_neighbour_window) -> dict:
    """Decide which documents are near copies of a kept document. The rows must be in priority order.

    Returns `duplicate_of_index` (the index of the kept document that a dropped document copies, -1 for kept
    documents), `agreement_with_kept` (NaN for kept documents) and two counts for the log.

    Three steps:
      1. banding proposes candidate pairs, and a pair is confirmed when the signatures agree on at least
         `agreement_threshold` of the 128 values;
      2. confirmed pairs are joined into connected components, which only decides who gets compared;
      3. inside a component the first document (best priority) is kept and becomes the leader; every
         document that agrees with the leader is dropped as its copy; the first document left over becomes
         the next leader, and so on until nothing is left.

    Step 3 is why a chain A~B~C with A unlike C keeps A and C and drops only B. Keeping one document per
    component would drop C as well, although no kept document resembles it: on pages that overlap one after
    the other (a paginated thread) that throws away text that exists nowhere else. With the leader rule,
    every dropped document agrees with a kept document of equal or better priority, and that is checked
    against the signatures directly, not inferred from a chain.
    """
    document_count = signatures.shape[0]
    earlier_indices, later_indices = find_candidate_pairs_with_lsh_banding(
        signatures, band_count, rows_per_band, bucket_neighbour_window)
    is_confirmed = measure_signature_agreement(signatures, earlier_indices, later_indices) >= agreement_threshold
    component_labels = label_connected_components(
        document_count, earlier_indices[is_confirmed], later_indices[is_confirmed])

    duplicate_of_index = np.full(document_count, -1, dtype=np.int64)
    agreement_with_kept = np.full(document_count, np.nan, dtype=np.float32)
    leader_of_document = component_labels.copy()
    undecided_indices = np.flatnonzero(component_labels != np.arange(document_count))      # ascending
    while undecided_indices.size > 0:
        leader_indices = leader_of_document[undecided_indices]
        agreement_with_leader = measure_signature_agreement(signatures, undecided_indices, leader_indices)
        is_copy_of_leader = agreement_with_leader >= agreement_threshold
        duplicate_of_index[undecided_indices[is_copy_of_leader]] = leader_indices[is_copy_of_leader]
        agreement_with_kept[undecided_indices[is_copy_of_leader]] = agreement_with_leader[is_copy_of_leader]
        leftover_indices = undecided_indices[~is_copy_of_leader]
        # the leftover documents are still ascending, so the first one of each component is its best one
        _, first_leftover_position, component_of_leftover = np.unique(
            component_labels[leftover_indices], return_index=True, return_inverse=True)
        next_leader_indices = leftover_indices[first_leftover_position][component_of_leftover]
        leader_of_document[leftover_indices] = next_leader_indices
        undecided_indices = leftover_indices[leftover_indices != next_leader_indices]
    return {
        "duplicate_of_index": duplicate_of_index,
        "agreement_with_kept": agreement_with_kept,
        "candidate_pair_count": int(earlier_indices.size),
        "confirmed_pair_count": int(is_confirmed.sum()),
    }


def deduplicate_documents(pool_paths, output_path, source_priority=default_source_priority,
                          earlier_decision_paths=(), agreement_threshold=default_signature_agreement_threshold,
                          band_count=default_lsh_band_count, rows_per_band=default_lsh_rows_per_band,
                          worker_count=1, documents_per_worker_batch=1_000) -> dict:
    """Stage 2: drop exact copies, then near copies, across all sources at once.

    Priority order decides which copy stays: the source listed first in `source_priority`, then the longer
    document (of two near copies the longer one usually contains the other), then the smaller `doc_id` so
    that the result does not depend on the order of the files. Returns the decision columns it wrote.
    """
    start_seconds = time.perf_counter()
    dropped_doc_ids = read_doc_ids_dropped_by_earlier_stages(earlier_decision_paths)
    record_batches = iterate_pool_record_batches(
        resolve_pool_parquet_paths(pool_paths), ("doc_id", "source", "token_count", "text"), dropped_doc_ids,
        documents_per_worker_batch)
    fingerprints_per_batch = list(run_batch_function_in_worker_processes(
        compute_duplicate_fingerprints_for_batch, record_batches, worker_count))
    assert fingerprints_per_batch, "no document reached the dedup stage"
    doc_ids = np.array([doc_id for batch in fingerprints_per_batch for doc_id in batch["doc_id"]], dtype=np.str_)
    sources = np.array([source for batch in fingerprints_per_batch for source in batch["source"]], dtype=object)
    token_counts = np.concatenate([batch["token_count"] for batch in fingerprints_per_batch])
    exact_text_hashes = np.concatenate([batch["exact_text_hash"] for batch in fingerprints_per_batch])
    signatures = np.concatenate([batch["minhash_signature"] for batch in fingerprints_per_batch])
    del fingerprints_per_batch
    assert_doc_ids_are_unique(doc_ids.tolist())
    signature_seconds = time.perf_counter() - start_seconds
    print(f"dedup: {len(doc_ids):,} signatures in {signature_seconds:.1f} s "
          f"({len(doc_ids) / signature_seconds:,.0f} documents per second, {worker_count} workers)")

    sources_without_priority = sorted(set(sources.tolist()) - set(source_priority))
    assert not sources_without_priority, (
        f"sources {sources_without_priority} are missing from the source priority list {list(source_priority)}: "
        "say where they rank, otherwise a web copy could silently win over a curated one")
    priority_rank_of_source = {source: rank for rank, source in enumerate(source_priority)}
    source_ranks = np.array([priority_rank_of_source[source] for source in sources])
    priority_order = np.lexsort((doc_ids, -token_counts.astype(np.int64), source_ranks))      # last key first
    doc_ids, sources, token_counts = doc_ids[priority_order], sources[priority_order], token_counts[priority_order]
    exact_text_hashes, signatures = exact_text_hashes[priority_order], signatures[priority_order]

    duplicate_of_index = find_exact_duplicates(exact_text_hashes)
    drop_reasons = np.where(duplicate_of_index >= 0, "exact_duplicate", "").astype(object)
    agreement_with_kept = np.where(duplicate_of_index >= 0, 1.0, np.nan).astype(np.float32)

    survivor_indices = np.flatnonzero(duplicate_of_index < 0)
    near_duplicates = find_near_duplicates(signatures[survivor_indices], agreement_threshold, band_count, rows_per_band)
    is_near_duplicate = near_duplicates["duplicate_of_index"] >= 0
    near_duplicate_indices = survivor_indices[is_near_duplicate]
    duplicate_of_index[near_duplicate_indices] = survivor_indices[near_duplicates["duplicate_of_index"][is_near_duplicate]]
    drop_reasons[near_duplicate_indices] = "near_duplicate"
    agreement_with_kept[near_duplicate_indices] = near_duplicates["agreement_with_kept"][is_near_duplicate]
    # An exact copy points at the first copy of its text. When that first copy was itself dropped as a near
    # copy, point one step further, so that `duplicate_of_doc_id` always names a document that is kept.
    exact_duplicate_indices = np.flatnonzero(drop_reasons == "exact_duplicate")
    first_copy_indices = duplicate_of_index[exact_duplicate_indices]
    was_first_copy_dropped = duplicate_of_index[first_copy_indices] >= 0
    duplicate_of_index[exact_duplicate_indices[was_first_copy_dropped]] = duplicate_of_index[first_copy_indices[was_first_copy_dropped]]
    agreement_with_kept[exact_duplicate_indices[was_first_copy_dropped]] = agreement_with_kept[first_copy_indices[was_first_copy_dropped]]
    print(f"dedup: {near_duplicates['candidate_pair_count']:,} candidate pairs from banding, "
          f"{near_duplicates['confirmed_pair_count']:,} confirmed at agreement >= {agreement_threshold}; "
          f"total {time.perf_counter() - start_seconds:.1f} s")

    is_dropped = duplicate_of_index >= 0
    decision_columns = {
        "doc_id": doc_ids.astype(object),
        "source": sources,
        "token_count": token_counts,
        "keep": ~is_dropped,
        "drop_reason": drop_reasons,
        "duplicate_of_doc_id": np.where(is_dropped, doc_ids[np.maximum(duplicate_of_index, 0)], "").astype(object),
        "signature_agreement_with_kept": agreement_with_kept,
    }
    stage_settings = {
        "source_priority": list(source_priority), "agreement_threshold": agreement_threshold,
        "band_count": band_count, "rows_per_band": rows_per_band, "word_shingle_length": word_shingle_length,
        "minhash_permutation_count": minhash_permutation_count}
    write_stage_decisions(output_path, "dedup", decision_columns, stage_settings)
    return decision_columns


# ---------------------------------------------------------------------------
# stage 3, embed: one unit vector per document from a small sentence-embedding model
# ---------------------------------------------------------------------------

default_embedding_model = "BAAI/bge-small-en-v1.5"
# How each model card says to turn token vectors into one sentence vector. bge v1.5: the vector of the
# [CLS] token, L2-normalised, and no instruction prefix when texts are compared with texts of the same kind
# (the card asks for the prefix only on short queries that search long passages). A model that is not listed
# stops the run: the wrong pooling gives vectors that look fine and cluster badly.
pooling_by_embedding_model = {
    "BAAI/bge-small-en-v1.5": "cls",
    "BAAI/bge-base-en-v1.5": "cls",
}
embedding_max_token_count = 512
# The text is cut before it is tokenised, because the tokenizer would otherwise split a 100,000-token
# document in full only to keep the first 512 tokens. 16 characters per token is far above what any real
# text needs (English averages 4 to 5), so the cut never removes one of the first 512 tokens.
characters_kept_per_token_before_tokenising = 16


@functools.lru_cache(maxsize=2)
def load_embedding_model(embedding_model: str, device: str):
    """Return (tokenizer, model) for a sentence-embedding model, loaded once per process and device."""
    from transformers import AutoModel, AutoTokenizer
    assert embedding_model in pooling_by_embedding_model, (
        f"no pooling is recorded for {embedding_model!r}: read its model card and add it to pooling_by_embedding_model")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model).to(device).eval()
    return tokenizer, model


def embed_documents(texts, embedding_model: str, device: str, documents_per_batch: int = 64) -> np.ndarray:
    """Return one L2-normalised float32 vector per text, shape (texts, dimensions), in the order of `texts`.

    Only the first 512 tokens of a text are read. The texts are embedded longest first, so that a batch
    holds texts of similar length and little of it is padding; the result is put back in the given order.
    `data_hygiene.py` imports this function, and its tests replace it with a random projection.
    """
    import torch
    tokenizer, model = load_embedding_model(embedding_model, device)
    pooling = pooling_by_embedding_model[embedding_model]
    character_limit = embedding_max_token_count * characters_kept_per_token_before_tokenising
    shortened_texts = [text[:character_limit] for text in texts]
    longest_first_order = np.argsort([-len(text) for text in shortened_texts], kind="stable")
    embeddings = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for batch_start in range(0, len(texts), documents_per_batch):
        batch_indices = longest_first_order[batch_start: batch_start + documents_per_batch]
        encoded_batch = tokenizer(
            [shortened_texts[index] for index in batch_indices], padding=True, truncation=True,
            max_length=embedding_max_token_count, return_tensors="pt").to(device)
        with torch.inference_mode():
            token_vectors = model(**encoded_batch).last_hidden_state
            if pooling == "cls":
                sentence_vectors = token_vectors[:, 0]
            else:
                assert pooling == "mean", f"unknown pooling {pooling!r}"
                is_real_token = encoded_batch["attention_mask"][:, :, None].to(token_vectors.dtype)
                sentence_vectors = (token_vectors * is_real_token).sum(dim=1) / is_real_token.sum(dim=1)
            sentence_vectors = torch.nn.functional.normalize(sentence_vectors.float(), p=2, dim=1)
        embeddings[batch_indices] = sentence_vectors.cpu().numpy()
    return embeddings


def embed_pool_documents(pool_paths, embeddings_path, embedding_doc_ids_path,
                         embedding_model=default_embedding_model, *, device,
                         earlier_decision_paths=(), documents_per_read=8_192) -> None:
    """Stage 3: embed every surviving document; write the vectors as float16 and the `doc_id` order beside them.

    float16 halves the file (1.5 GB for 2 million documents of 384 dimensions) and changes a cosine
    similarity by about 1e-4, far below the steps in which the semdedup threshold is tuned.
    """
    start_seconds = time.perf_counter()
    dropped_doc_ids = read_doc_ids_dropped_by_earlier_stages(earlier_decision_paths)
    embedding_chunks, doc_ids, sources, token_counts = [], [], [], []
    for record_batch in iterate_pool_record_batches(
            resolve_pool_parquet_paths(pool_paths), ("doc_id", "source", "token_count", "text"), dropped_doc_ids,
            documents_per_read):
        embeddings = embed_documents(record_batch.column("text").to_pylist(), embedding_model, device)
        embedding_chunks.append(embeddings.astype(np.float16))
        doc_ids.extend(record_batch.column("doc_id").to_pylist())
        sources.extend(record_batch.column("source").to_pylist())
        token_counts.extend(record_batch.column("token_count").to_pylist())
        elapsed_seconds = time.perf_counter() - start_seconds
        print(f"embed: {len(doc_ids):,} documents, {len(doc_ids) / elapsed_seconds:,.1f} per second", flush=True)
    assert doc_ids, "no document reached the embed stage"
    assert_doc_ids_are_unique(doc_ids)
    Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, np.concatenate(embedding_chunks))
    doc_id_table = pa.table({
        "doc_id": pa.array(doc_ids, type=pa.string()), "source": pa.array(sources, type=pa.string()),
        "token_count": pa.array(token_counts, type=pa.int32())})
    doc_id_table = doc_id_table.replace_schema_metadata({"settings_json": json.dumps({"embedding_model": embedding_model})})
    pq.write_table(doc_id_table, embedding_doc_ids_path)
    print(f"embed: wrote {len(doc_ids):,} vectors -> {embeddings_path} (row order in {embedding_doc_ids_path})")


# ---------------------------------------------------------------------------
# stage 4, semdedup: k-means on the vectors, then cosine similarity inside each cluster
# ---------------------------------------------------------------------------

# SemDeDup (Abbas et al., 2023) compares documents only inside k-means clusters, which turns n^2 comparisons
# into n^2 / clusters. 1,500 documents per cluster is the middle of the 1,000 to 2,000 range in which a
# cluster's similarity matrix (9 MB) is trivial and paraphrases still fall into the same cluster.
default_documents_per_cluster = 1_500
# Cosine similarity at or above which a document counts as a paraphrase of a kept one. Not tuned on the pool:
# look at `nearest_kept_cosine_similarity` in the decision file and read pairs around the threshold first.
default_semantic_similarity_threshold = 0.93
default_kmeans_iteration_count = 20
# seed 0 for the initial centroids; with it the same vectors always give the same clusters and decisions
default_kmeans_seed = 0


def cluster_embeddings_with_kmeans(embeddings, cluster_count: int, iteration_count: int = default_kmeans_iteration_count,
                                   seed: int = default_kmeans_seed, documents_per_chunk: int = 65_536):
    """Return the cluster index of every vector (a torch tensor on the device of `embeddings`).

    Spherical k-means on unit vectors: a document joins the centroid with the largest dot product, and a
    centroid is the normalised mean of its documents. The centroids start on randomly chosen documents; a
    cluster that loses all its documents restarts on a random document. The clusters only decide who gets
    compared with whom, so the loop stops after `iteration_count` rounds without waiting for convergence.
    """
    import torch
    document_count = embeddings.shape[0]
    assert 1 <= cluster_count <= document_count, f"{cluster_count} clusters for {document_count} documents"
    random_generator = torch.Generator().manual_seed(seed)
    centroids = embeddings[torch.randperm(document_count, generator=random_generator)[:cluster_count].to(embeddings.device)].clone()
    cluster_index_per_document = torch.full((document_count,), -1, dtype=torch.long, device=embeddings.device)
    for _ in range(iteration_count):
        new_cluster_index_per_document = torch.cat([
            (embeddings[chunk_start: chunk_start + documents_per_chunk] @ centroids.T).argmax(dim=1)
            for chunk_start in range(0, document_count, documents_per_chunk)])
        changed_document_count = int((new_cluster_index_per_document != cluster_index_per_document).sum())
        cluster_index_per_document = new_cluster_index_per_document
        if changed_document_count == 0:
            break
        centroid_sums = torch.zeros_like(centroids).index_add_(0, cluster_index_per_document, embeddings)
        is_empty_cluster = torch.bincount(cluster_index_per_document, minlength=cluster_count) == 0
        if bool(is_empty_cluster.any()):
            restart_documents = torch.randint(document_count, (int(is_empty_cluster.sum()),), generator=random_generator)
            centroid_sums[is_empty_cluster] = embeddings[restart_documents.to(embeddings.device)]
        centroids = torch.nn.functional.normalize(centroid_sums, p=2, dim=1)
    return cluster_index_per_document


def find_semantic_duplicates_in_cluster(cluster_embeddings, similarity_threshold: float,
                                        documents_per_block: int = 2_048) -> dict:
    """Walk through one cluster in priority order and drop every document that resembles a kept one.

    `cluster_embeddings` are unit vectors (a torch tensor), best priority first. Returns numpy arrays with
    one entry per document: `is_kept`, `nearest_kept_position` (the kept document before it with the highest
    cosine similarity, -1 when no document was kept before it) and `nearest_kept_similarity` (NaN in that case).

    The rule is sequential: whether document i is kept depends on which of the documents before it were
    kept. A block of 2,048 documents is first compared with the kept documents of all earlier blocks in one
    matrix product, then resolved one document at a time inside the block, where the similarities are
    already computed. Memory stays at one 2,048 x 2,048 block however large the cluster is.
    """
    import torch
    member_count = cluster_embeddings.shape[0]
    is_kept = np.zeros(member_count, dtype=bool)
    nearest_kept_position = np.full(member_count, -1, dtype=np.int64)
    nearest_kept_similarity = np.full(member_count, np.nan, dtype=np.float32)
    for block_start in range(0, member_count, documents_per_block):
        block_end = min(block_start + documents_per_block, member_count)
        block_embeddings = cluster_embeddings[block_start:block_end]
        best_similarity = np.full(block_end - block_start, -np.inf, dtype=np.float32)
        best_position = np.full(block_end - block_start, -1, dtype=np.int64)

        kept_positions_before_block = np.flatnonzero(is_kept[:block_start])
        for chunk_start in range(0, kept_positions_before_block.size, 8 * documents_per_block):
            chunk_positions = kept_positions_before_block[chunk_start: chunk_start + 8 * documents_per_block]
            kept_embeddings = cluster_embeddings[torch.as_tensor(chunk_positions, device=cluster_embeddings.device)]
            chunk_best = (block_embeddings @ kept_embeddings.T).max(dim=1)
            chunk_best_similarity = chunk_best.values.cpu().numpy()
            is_better = chunk_best_similarity > best_similarity
            best_similarity[is_better] = chunk_best_similarity[is_better]
            best_position[is_better] = chunk_positions[chunk_best.indices.cpu().numpy()[is_better]]

        similarity_inside_block = (block_embeddings @ block_embeddings.T).cpu().numpy()
        is_kept_inside_block = np.zeros(block_end - block_start, dtype=bool)
        for position_in_block in range(block_end - block_start):
            kept_before = is_kept_inside_block[:position_in_block]
            if kept_before.any():
                similarity_to_kept_before = np.where(
                    kept_before, similarity_inside_block[position_in_block, :position_in_block], -np.inf)
                most_similar_position = int(similarity_to_kept_before.argmax())
                if similarity_to_kept_before[most_similar_position] > best_similarity[position_in_block]:
                    best_similarity[position_in_block] = similarity_to_kept_before[most_similar_position]
                    best_position[position_in_block] = block_start + most_similar_position
            is_kept_inside_block[position_in_block] = not (best_similarity[position_in_block] >= similarity_threshold)

        is_kept[block_start:block_end] = is_kept_inside_block
        nearest_kept_position[block_start:block_end] = best_position
        nearest_kept_similarity[block_start:block_end] = np.where(best_position >= 0, best_similarity, np.nan)
    return {"is_kept": is_kept, "nearest_kept_position": nearest_kept_position,
            "nearest_kept_similarity": nearest_kept_similarity}


def deduplicate_documents_semantically(embeddings_path, embedding_doc_ids_path, output_path,
                                       similarity_threshold=default_semantic_similarity_threshold,
                                       documents_per_cluster=default_documents_per_cluster,
                                       source_priority=default_source_priority, *, device,
                                       earlier_decision_paths=(), sources_to_leave_out=(),
                                       kmeans_seed=default_kmeans_seed) -> dict:
    """Stage 4: drop documents that say the same thing as a kept document in other words.

    Inside each k-means cluster the documents are visited in the priority order of the dedup stage (source
    priority, longer first, `doc_id`), and a document is dropped when its cosine similarity to an
    already-kept document is at least `similarity_threshold`. Documents of `sources_to_leave_out` are kept
    without being compared. Returns the decision columns it wrote.

    Two things this stage cannot see: a pair that k-means puts into different clusters, and anything past
    the first 512 tokens, which is all the embedding read.
    """
    import torch
    start_seconds = time.perf_counter()
    dropped_doc_ids = read_doc_ids_dropped_by_earlier_stages(earlier_decision_paths)
    embedded_documents = read_decision_columns(embedding_doc_ids_path)
    all_embeddings = np.load(embeddings_path)
    assert all_embeddings.shape[0] == len(embedded_documents["doc_id"]), "embedding rows and doc_id rows differ"
    is_alive = np.array([doc_id not in dropped_doc_ids for doc_id in embedded_documents["doc_id"]], dtype=bool)
    if len(earlier_decision_paths) > 0:
        # every document the last earlier stage kept must have a vector; otherwise the embeddings are older
        # than the decisions and some documents would skip this stage unseen
        last_decisions = read_decision_columns(earlier_decision_paths[-1], ["doc_id", "keep"])
        expected_document_count = int(last_decisions["keep"].sum())
        assert int(is_alive.sum()) == expected_document_count, (
            f"{expected_document_count:,} documents survive {Path(earlier_decision_paths[-1]).name} but "
            f"{int(is_alive.sum()):,} of them have an embedding: rerun `embed` after changing an earlier stage")
    doc_ids = embedded_documents["doc_id"][is_alive].astype(np.str_)
    sources = embedded_documents["source"][is_alive]
    token_counts = embedded_documents["token_count"][is_alive].astype(np.int64)
    assert len(doc_ids) > 0, "no document reached the semdedup stage"

    sources_without_priority = sorted(set(sources.tolist()) - set(source_priority))
    assert not sources_without_priority, (
        f"sources {sources_without_priority} are missing from the source priority list {list(source_priority)}")
    priority_rank_of_source = {source: rank for rank, source in enumerate(source_priority)}
    source_ranks = np.array([priority_rank_of_source[source] for source in sources])
    priority_order = np.lexsort((doc_ids, -token_counts, source_ranks))
    doc_ids, sources, token_counts = doc_ids[priority_order], sources[priority_order], token_counts[priority_order]

    is_compared = ~np.isin(sources, list(sources_to_leave_out))
    compared_indices = np.flatnonzero(is_compared)
    keep = np.ones(len(doc_ids), dtype=bool)
    cluster_index_column = np.full(len(doc_ids), -1, dtype=np.int32)
    nearest_kept_doc_ids = np.full(len(doc_ids), "", dtype=object)
    nearest_kept_similarity = np.full(len(doc_ids), np.nan, dtype=np.float32)
    cluster_count = 0
    if compared_indices.size > 0:
        # float16 on disk; back to float32 and to unit length, so that a dot product is a cosine similarity
        embeddings = torch.from_numpy(
            np.asarray(all_embeddings[np.flatnonzero(is_alive)[priority_order][compared_indices]], dtype=np.float32))
        embeddings = torch.nn.functional.normalize(embeddings.to(device), p=2, dim=1)
        cluster_count = max(1, round(compared_indices.size / documents_per_cluster))
        cluster_index_per_document = cluster_embeddings_with_kmeans(embeddings, cluster_count, seed=kmeans_seed).cpu().numpy()
        print(f"semdedup: {compared_indices.size:,} documents in {cluster_count:,} clusters "
              f"(largest {int(np.bincount(cluster_index_per_document).max()):,}), "
              f"k-means took {time.perf_counter() - start_seconds:.1f} s", flush=True)
        cluster_index_column[compared_indices] = cluster_index_per_document
        for cluster_index in range(cluster_count):
            # positions inside `compared_indices`, ascending, which is priority order
            member_positions = np.flatnonzero(cluster_index_per_document == cluster_index)
            if member_positions.size == 0:
                continue
            cluster_result = find_semantic_duplicates_in_cluster(
                embeddings[torch.as_tensor(member_positions, device=embeddings.device)], similarity_threshold)
            member_indices = compared_indices[member_positions]
            keep[member_indices] = cluster_result["is_kept"]
            has_nearest_kept = cluster_result["nearest_kept_position"] >= 0
            nearest_kept_doc_ids[member_indices[has_nearest_kept]] = doc_ids[
                member_indices[cluster_result["nearest_kept_position"][has_nearest_kept]]]
            nearest_kept_similarity[member_indices] = cluster_result["nearest_kept_similarity"]

    decision_columns = {
        "doc_id": doc_ids.astype(object),
        "source": sources,
        "token_count": token_counts,
        "keep": keep,
        "drop_reason": np.where(keep, "", "semantic_duplicate").astype(object),
        "cluster_index": cluster_index_column,
        "nearest_kept_doc_id": nearest_kept_doc_ids,
        "nearest_kept_cosine_similarity": nearest_kept_similarity,
    }
    stage_settings = {
        "similarity_threshold": similarity_threshold, "documents_per_cluster": documents_per_cluster,
        "cluster_count": cluster_count, "source_priority": list(source_priority),
        "sources_to_leave_out": list(sources_to_leave_out), "kmeans_seed": kmeans_seed,
        "embedding": embedded_documents["settings"]}
    print(f"semdedup: total {time.perf_counter() - start_seconds:.1f} s")
    write_stage_decisions(output_path, "semdedup", decision_columns, stage_settings)
    return decision_columns


# ---------------------------------------------------------------------------
# stage 5, score: mean next-token loss per document under a causal LM
# ---------------------------------------------------------------------------

# The Qwen3.5 vocabulary has 248,320 entries. Logits for a whole batch of 16 documents x 1,024 tokens would
# be 16,384 x 248,320 x 4 bytes = 16 GB; for 1,024 positions they are 1 GB. So the transformer body runs on
# the whole batch and its hidden states are projected to logits 1,024 positions at a time.
default_logit_chunk_token_count = 1_024
# padded tokens in one forward pass of the transformer body: 16 documents at the 1,024-token window
default_padded_tokens_per_batch = 16_384
score_column_names = ("doc_id", "source", "token_count", "scored_token_count", "mean_loss_nats_per_token",
                      "first_half_mean_loss_nats_per_token", "second_half_mean_loss_nats_per_token")


def compute_document_losses_for_batch(causal_lm, token_ids_per_document, device: str,
                                      logit_chunk_token_count: int = default_logit_chunk_token_count) -> dict:
    """Return the mean next-token loss (nats) of every document in one batch, and of each half of its window.

    `token_ids_per_document` are token id sequences of any lengths. A document of n tokens has n - 1 scored
    tokens (the first token has nothing before it); the first half is the first ceil((n - 1) / 2) of them and
    the second half the rest. A document with fewer than 2 tokens has no scored token and gets NaN, and a
    document with exactly 2 tokens has an empty second half, also NaN.

    Padding goes on the right, built here and not by the tokenizer (whose default side differs between
    models). The model is causal, so nothing to the right of a token can change its prediction and the
    result for a document does not depend on which other documents share its batch.
    """
    import torch
    document_count = len(token_ids_per_document)
    token_counts = torch.tensor([len(token_ids) for token_ids in token_ids_per_document], dtype=torch.long)
    longest_token_count = max(int(token_counts.max()), 1)
    input_ids = torch.zeros((document_count, longest_token_count), dtype=torch.long)
    for document_index, token_ids in enumerate(token_ids_per_document):
        input_ids[document_index, : len(token_ids)] = torch.as_tensor(token_ids, dtype=torch.long)
    is_real_token = torch.arange(longest_token_count)[None, :] < token_counts[:, None]
    input_ids, is_real_token, token_counts = input_ids.to(device), is_real_token.to(device), token_counts.to(device)

    transformer_body = causal_lm.base_model
    output_projection = causal_lm.get_output_embeddings()
    assert transformer_body is not causal_lm and output_projection is not None, (
        "this model does not expose its body and its output projection separately, so its logits cannot be chunked")
    with torch.inference_mode():
        hidden_states = transformer_body(
            input_ids=input_ids, attention_mask=is_real_token.long(), use_cache=False).last_hidden_state
        has_next_token = is_real_token[:, 1:]                         # position t predicts token t + 1
        hidden_states_of_scored_positions = hidden_states[:, :-1][has_next_token]
        next_token_ids = input_ids[:, 1:][has_next_token]
        token_losses = torch.empty(next_token_ids.shape[0], dtype=torch.float32, device=device)
        for chunk_start in range(0, next_token_ids.shape[0], logit_chunk_token_count):
            chunk = slice(chunk_start, chunk_start + logit_chunk_token_count)
            # float32 for the loss: in bfloat16 a log-softmax over 248,320 entries loses the small terms
            chunk_logits = output_projection(hidden_states_of_scored_positions[chunk]).float()
            token_losses[chunk] = torch.nn.functional.cross_entropy(chunk_logits, next_token_ids[chunk], reduction="none")

        position_grid = torch.arange(longest_token_count - 1, device=device)[None, :].expand(document_count, -1)
        document_grid = torch.arange(document_count, device=device)[:, None].expand(-1, longest_token_count - 1)
        document_of_scored_token = document_grid[has_next_token]
        scored_token_counts = (token_counts - 1).clamp(min=0)
        first_half_token_counts = (scored_token_counts + 1) // 2
        is_in_first_half = position_grid[has_next_token] < first_half_token_counts[document_of_scored_token]
        zeros = torch.zeros(document_count, dtype=torch.float64, device=device)
        loss_sums = zeros.index_add(0, document_of_scored_token, token_losses.double())
        first_half_loss_sums = zeros.index_add(0, document_of_scored_token, (token_losses * is_in_first_half).double())
        second_half_token_counts = scored_token_counts - first_half_token_counts
        # 0 / 0 gives NaN, which is the wanted value for a part with no scored token
        mean_losses = loss_sums / scored_token_counts
        first_half_mean_losses = first_half_loss_sums / first_half_token_counts
        second_half_mean_losses = (loss_sums - first_half_loss_sums) / second_half_token_counts
    return {
        "scored_token_count": scored_token_counts.cpu().numpy().astype(np.int32),
        "mean_loss_nats_per_token": mean_losses.cpu().numpy().astype(np.float32),
        "first_half_mean_loss_nats_per_token": first_half_mean_losses.cpu().numpy().astype(np.float32),
        "second_half_mean_loss_nats_per_token": second_half_mean_losses.cpu().numpy().astype(np.float32),
    }


def group_documents_into_length_batches(token_counts, padded_tokens_per_batch: int) -> list[np.ndarray]:
    """Return batches of document indices, shortest documents first, each within the padded-token budget.

    A batch is padded to its longest document, so its cost is documents x longest length. Sorting by length
    first keeps that close to the number of real tokens.
    """
    shortest_first_order = np.argsort(token_counts, kind="stable")
    batches, current_batch = [], []
    for document_index in shortest_first_order:
        # the candidate is the longest so far, so it sets the padded length of the batch
        padded_tokens_with_candidate = (len(current_batch) + 1) * max(int(token_counts[document_index]), 1)
        if current_batch and padded_tokens_with_candidate > padded_tokens_per_batch:
            batches.append(np.array(current_batch))
            current_batch = []
        current_batch.append(int(document_index))
    if current_batch:
        batches.append(np.array(current_batch))
    return batches


def write_score_table(output_path, score_rows: dict, score_settings: dict) -> None:
    """Write the scores gathered so far; a temporary file is renamed so a killed job never leaves half a file."""
    score_table = pa.table({
        "doc_id": pa.array(score_rows["doc_id"], type=pa.string()),
        "source": pa.array(score_rows["source"], type=pa.string()),
        "token_count": pa.array(score_rows["token_count"], type=pa.int32()),
        "scored_token_count": pa.array(score_rows["scored_token_count"], type=pa.int32()),
        "mean_loss_nats_per_token": pa.array(score_rows["mean_loss_nats_per_token"], type=pa.float32()),
        "first_half_mean_loss_nats_per_token": pa.array(score_rows["first_half_mean_loss_nats_per_token"], type=pa.float32()),
        "second_half_mean_loss_nats_per_token": pa.array(score_rows["second_half_mean_loss_nats_per_token"], type=pa.float32()),
    }).replace_schema_metadata({"settings_json": json.dumps(score_settings)})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(output_path) + ".writing")
    pq.write_table(score_table, temporary_path)
    os.replace(temporary_path, output_path)


def score_documents_with_causal_lm(model_path, pool_paths, output_path, max_tokens_per_document=1024, *, device,
                                   earlier_decision_paths=(),
                                   padded_tokens_per_batch=default_padded_tokens_per_batch,
                                   logit_chunk_token_count=default_logit_chunk_token_count,
                                   documents_per_read=4_096, documents_between_saves=20_000) -> None:
    """Stage 5: write the mean next-token loss of every surviving document under the model at `model_path`.

    Only the first `max_tokens_per_document` tokens of a document are scored. The stage is run once per
    model (`select` needs a student and a reference); it is by far the most expensive stage, so
      * it saves every `documents_between_saves` documents and, when `output_path` already exists, continues
        after the documents scored so far (the model and the window must match, which is checked);
      * it can be split over several jobs by giving each a part of the pool in `pool_paths` and its own
        `output_path`; `select` takes any number of score files.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    score_settings = {"model_path": str(model_path), "max_tokens_per_document": max_tokens_per_document}
    score_rows = {column_name: [] for column_name in score_column_names}
    if Path(output_path).exists():
        earlier_scores = read_decision_columns(output_path)
        assert earlier_scores["settings"] == score_settings, (
            f"{output_path} was written with {earlier_scores['settings']}, this run uses {score_settings}: "
            "delete the file or choose another output path")
        for column_name in score_column_names:
            score_rows[column_name] = earlier_scores[column_name].tolist()
        print(f"score: continuing after the {len(score_rows['doc_id']):,} documents already in {output_path}")
    documents_to_skip = read_doc_ids_dropped_by_earlier_stages(earlier_decision_paths) | set(score_rows["doc_id"])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # bfloat16 on a GPU is the dtype the Qwen3.5 weights are stored in; a CPU has no fast bfloat16
    model_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    causal_lm = AutoModelForCausalLM.from_pretrained(model_path, dtype=model_dtype).to(device).eval()

    start_seconds = time.perf_counter()
    scored_document_count_this_run, scored_token_count_this_run, document_count_at_last_save = 0, 0, 0
    character_limit = max_tokens_per_document * characters_kept_per_token_before_tokenising
    for record_batch in iterate_pool_record_batches(
            resolve_pool_parquet_paths(pool_paths), ("doc_id", "source", "token_count", "text"), documents_to_skip,
            documents_per_read):
        texts = [text[:character_limit] for text in record_batch.column("text").to_pylist()]
        token_ids_per_document = tokenizer(
            texts, add_special_tokens=False, truncation=True, max_length=max_tokens_per_document)["input_ids"]
        window_token_counts = np.array([len(token_ids) for token_ids in token_ids_per_document])
        losses_in_read_order = {
            column_name: np.empty(len(texts), dtype=np.int32 if column_name == "scored_token_count" else np.float32)
            for column_name in score_column_names[3:]}
        for batch_indices in group_documents_into_length_batches(window_token_counts, padded_tokens_per_batch):
            batch_losses = compute_document_losses_for_batch(
                causal_lm, [token_ids_per_document[index] for index in batch_indices], device, logit_chunk_token_count)
            for column_name, column_values in batch_losses.items():
                losses_in_read_order[column_name][batch_indices] = column_values
        score_rows["doc_id"].extend(record_batch.column("doc_id").to_pylist())
        score_rows["source"].extend(record_batch.column("source").to_pylist())
        score_rows["token_count"].extend(record_batch.column("token_count").to_pylist())
        for column_name, column_values in losses_in_read_order.items():
            score_rows[column_name].extend(column_values.tolist())

        scored_document_count_this_run += len(texts)
        scored_token_count_this_run += int(losses_in_read_order["scored_token_count"].sum())
        elapsed_seconds = time.perf_counter() - start_seconds
        print(f"score: {scored_document_count_this_run:,} documents, {scored_token_count_this_run / elapsed_seconds:,.0f} "
              f"tokens per second", flush=True)
        if scored_document_count_this_run - document_count_at_last_save >= documents_between_saves:
            write_score_table(output_path, score_rows, score_settings)
            document_count_at_last_save = scored_document_count_this_run
    assert score_rows["doc_id"], "no document reached the score stage"
    assert_doc_ids_are_unique(score_rows["doc_id"])
    write_score_table(output_path, score_rows, score_settings)
    unscored_document_count = int(np.isnan(np.array(score_rows["mean_loss_nats_per_token"], dtype=np.float32)).sum())
    print(f"score: {len(score_rows['doc_id']):,} documents -> {output_path} "
          f"({unscored_document_count:,} have fewer than 2 tokens and no loss)")


# ---------------------------------------------------------------------------
# stage 6, select: keep what the student cannot predict yet and a stronger reference can
# ---------------------------------------------------------------------------

# A document is "trivial" when the student's loss is in the lowest 10% of its source, and "noise" when the
# reference's loss is in the highest 10% of its source. The quantiles are taken per source because a loss of
# 2.0 nats is easy prose and hard code; they are starting values, not tuned.
default_trivial_quantile = 0.10
default_noise_quantile = 0.90


def read_score_tables(score_paths) -> dict:
    """Return the columns of one or several score files of the same model, concatenated."""
    assert len(score_paths) > 0, "no score file given"
    score_columns_per_file = [read_decision_columns(score_path) for score_path in score_paths]
    model_paths = sorted({columns["settings"].get("model_path", "") for columns in score_columns_per_file})
    assert len(model_paths) == 1, f"the score files {[str(path) for path in score_paths]} come from several models: {model_paths}"
    scores = {
        column_name: np.concatenate([columns[column_name] for columns in score_columns_per_file])
        for column_name in score_column_names}
    assert_doc_ids_are_unique(scores["doc_id"].tolist())
    scores["settings"] = score_columns_per_file[0]["settings"]
    return scores


def select_learnable_documents(student_score_paths, reference_score_paths, output_path,
                               trivial_quantile=default_trivial_quantile, noise_quantile=default_noise_quantile,
                               quantiles_by_source=None, token_budget=None, token_budget_by_source=None,
                               earlier_decision_paths=()) -> dict:
    """Stage 6: label every scored document trivial, noise or learnable, and keep the learnable ones.

    `excess_loss = student_loss - reference_loss` is how much of the student's loss a stronger model shows
    to be avoidable. Per source:
      * trivial   - the student's loss is below the `trivial_quantile` of that source: already learned;
      * noise     - the reference's loss is above the `noise_quantile` of that source: nobody can predict it
                    (tables of numbers, hashes, broken extraction); checked after trivial, so a document that
                    is both counts as trivial;
      * learnable - everything else.
    `quantiles_by_source` maps a source to its own (trivial, noise) pair. With `token_budget` (the whole
    selection) or `token_budget_by_source`, the learnable documents are ranked by excess loss, largest
    first, and kept until the budget is used; the rest are dropped as `over_token_budget`.

    A document that one of the two models did not score, or that has fewer than 2 tokens, is dropped as
    `not_scored` and counted. Returns the decision columns it wrote.
    """
    student_scores = read_score_tables(student_score_paths)
    reference_scores = read_score_tables(reference_score_paths)
    assert student_scores["settings"].get("max_tokens_per_document") == reference_scores["settings"].get("max_tokens_per_document"), (
        "student and reference were scored on windows of different length; their losses are not comparable")
    dropped_doc_ids = read_doc_ids_dropped_by_earlier_stages(earlier_decision_paths)

    reference_row_of_doc_id = {doc_id: row for row, doc_id in enumerate(reference_scores["doc_id"].tolist())}
    student_doc_ids = student_scores["doc_id"].tolist()
    doc_ids_only_in_reference = sorted(set(reference_row_of_doc_id) - set(student_doc_ids))
    doc_ids = np.array(
        [doc_id for doc_id in student_doc_ids + doc_ids_only_in_reference if doc_id not in dropped_doc_ids], dtype=object)
    assert len(doc_ids) > 0, "no document reached the select stage"
    student_row_of_doc_id = {doc_id: row for row, doc_id in enumerate(student_doc_ids)}
    student_rows = np.array([student_row_of_doc_id.get(doc_id, -1) for doc_id in doc_ids])
    reference_rows = np.array([reference_row_of_doc_id.get(doc_id, -1) for doc_id in doc_ids])
    student_losses = np.where(student_rows >= 0, student_scores["mean_loss_nats_per_token"][student_rows], np.nan)
    reference_losses = np.where(reference_rows >= 0, reference_scores["mean_loss_nats_per_token"][reference_rows], np.nan)
    sources = np.where(student_rows >= 0, student_scores["source"][student_rows], reference_scores["source"][reference_rows])
    token_counts = np.where(
        student_rows >= 0, student_scores["token_count"][student_rows], reference_scores["token_count"][reference_rows]).astype(np.int64)
    excess_losses = (student_losses - reference_losses).astype(np.float32)

    is_scored = np.isfinite(student_losses) & np.isfinite(reference_losses)
    if not is_scored.all():
        print(f"select: {int((~is_scored).sum()):,} documents lack a loss from one of the two models; dropped as not_scored")
    learnability_labels = np.where(is_scored, "learnable", "not_scored").astype(object)
    thresholds_by_source = {}
    for source in sorted(set(sources.tolist())):
        is_scored_in_source = is_scored & (sources == source)
        if not is_scored_in_source.any():
            continue
        source_trivial_quantile, source_noise_quantile = (quantiles_by_source or {}).get(source, (trivial_quantile, noise_quantile))
        trivial_loss_threshold = float(np.quantile(student_losses[is_scored_in_source], source_trivial_quantile))
        noise_loss_threshold = float(np.quantile(reference_losses[is_scored_in_source], source_noise_quantile))
        is_trivial = is_scored_in_source & (student_losses < trivial_loss_threshold)
        is_noise = is_scored_in_source & (reference_losses > noise_loss_threshold) & ~is_trivial
        learnability_labels[is_trivial] = "trivial"
        learnability_labels[is_noise] = "noise"
        thresholds_by_source[source] = {
            "trivial_quantile": source_trivial_quantile, "student_loss_below_which_trivial": trivial_loss_threshold,
            "noise_quantile": source_noise_quantile, "reference_loss_above_which_noise": noise_loss_threshold}

    drop_reasons = np.where(learnability_labels == "learnable", "", learnability_labels).astype(object)

    def drop_learnable_documents_beyond_budget(is_candidate: np.ndarray, budget_tokens: int) -> None:
        """Keep the candidates with the largest excess loss until their tokens reach the budget."""
        candidate_indices = np.flatnonzero(is_candidate & (drop_reasons == ""))
        largest_excess_first = candidate_indices[np.argsort(-excess_losses[candidate_indices], kind="stable")]
        is_over_budget = np.cumsum(token_counts[largest_excess_first]) > budget_tokens
        drop_reasons[largest_excess_first[is_over_budget]] = "over_token_budget"

    for source, source_token_budget in (token_budget_by_source or {}).items():
        assert source in thresholds_by_source, f"token budget given for {source!r}, which has no scored document"
        drop_learnable_documents_beyond_budget(sources == source, int(source_token_budget))
    if token_budget is not None:
        drop_learnable_documents_beyond_budget(np.ones(len(doc_ids), dtype=bool), int(token_budget))

    decision_columns = {
        "doc_id": doc_ids,
        "source": sources,
        "token_count": token_counts,
        "keep": drop_reasons == "",
        "drop_reason": drop_reasons,
        "learnability_label": learnability_labels,
        "student_loss_nats_per_token": student_losses.astype(np.float32),
        "reference_loss_nats_per_token": reference_losses.astype(np.float32),
        "excess_loss_nats_per_token": excess_losses,
    }
    stage_settings = {
        "student": student_scores["settings"], "reference": reference_scores["settings"],
        "thresholds_by_source": thresholds_by_source, "token_budget": token_budget,
        "token_budget_by_source": token_budget_by_source or {}}
    write_stage_decisions(output_path, "select", decision_columns, stage_settings)
    for source, label_token_shares in summarise_label_token_shares(decision_columns).items():
        shares_text = ", ".join(f"{label} {share:.1%}" for label, share in label_token_shares.items())
        print(f"  {source}: token share by label: {shares_text}")
    return decision_columns


def summarise_label_token_shares(select_decision_columns: dict) -> dict:
    """Return, per source, the share of its tokens under each learnability label."""
    token_counts = select_decision_columns["token_count"].astype(np.int64)
    label_token_shares_by_source = {}
    for source in sorted(set(select_decision_columns["source"].tolist())):
        is_in_source = select_decision_columns["source"] == source
        source_token_count = max(int(token_counts[is_in_source].sum()), 1)
        label_token_shares_by_source[source] = {
            label: float(token_counts[is_in_source & (select_decision_columns["learnability_label"] == label)].sum() / source_token_count)
            for label in ("trivial", "noise", "learnable", "not_scored")}
    return label_token_shares_by_source


# ---------------------------------------------------------------------------
# report: merge the stage files into results/funnel_report.json
# ---------------------------------------------------------------------------

def write_funnel_report(funnel_directory, report_path=default_funnel_report_path) -> dict:
    """Merge the decision files of the funnel into one JSON: counts and tokens per stage, source and reason.

    Stages whose file is missing are listed under `stages_not_run`. The files must chain: the documents that
    enter a stage are the documents the previous stage kept. If they do not, a stage was rerun without
    rerunning the later ones, and the report would add up numbers from different funnels; that stops here.
    """
    funnel_report = {"funnel_directory_name": Path(funnel_directory).name, "stages": {}, "stages_not_run": [],
                     "survivors_after_each_stage": []}
    previous_stage_name, previous_stage_kept = None, None
    for stage_name in decision_stage_names:
        decision_path = get_decision_path(funnel_directory, stage_name)
        if not decision_path.exists():
            funnel_report["stages_not_run"].append(stage_name)
            continue
        decision_columns = read_decision_columns(decision_path)
        stage_summary = summarise_stage_decisions(decision_columns)
        if previous_stage_kept is None:
            funnel_report["survivors_after_each_stage"].append({"stage": "pool", **stage_summary["entered"]})
        else:
            assert stage_summary["entered"] == previous_stage_kept, (
                f"`{stage_name}` saw {stage_summary['entered']} but `{previous_stage_name}` kept {previous_stage_kept}: "
                f"`{stage_name}` was computed from another version of the earlier stages and must be rerun")
        funnel_report["survivors_after_each_stage"].append({"stage": stage_name, **stage_summary["kept"]})
        funnel_report["stages"][stage_name] = {"settings": decision_columns["settings"], **stage_summary}
        if stage_name == "filter":
            funnel_report["stages"][stage_name]["token_share_by_script_tag"] = summarise_script_token_shares(decision_columns)
        if stage_name == "select":
            funnel_report["stages"][stage_name]["token_share_by_label"] = summarise_label_token_shares(decision_columns)
        previous_stage_name, previous_stage_kept = stage_name, stage_summary["kept"]
    assert funnel_report["stages"], f"no decision file found in {funnel_directory}"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(funnel_report, indent=2) + "\n")
    print(f"report -> {report_path}")
    for survivors in funnel_report["survivors_after_each_stage"]:
        print(f"  after {survivors['stage']:<9} {survivors['documents']:>12,} documents {survivors['tokens']:>16,} tokens")
    if funnel_report["stages_not_run"]:
        print(f"  stages without a decision file: {funnel_report['stages_not_run']}")
    return funnel_report


def summarise_script_token_shares(filter_decision_columns: dict) -> dict:
    """Return, per source, the share of its tokens in documents of each script tag."""
    token_counts = filter_decision_columns["token_count"].astype(np.int64)
    script_token_shares_by_source = {}
    for source in sorted(set(filter_decision_columns["source"].tolist())):
        is_in_source = filter_decision_columns["source"] == source
        source_token_count = max(int(token_counts[is_in_source].sum()), 1)
        script_token_shares_by_source[source] = {
            script_tag: float(token_counts[is_in_source & (filter_decision_columns["script_tag"] == script_tag)].sum() / source_token_count)
            for script_tag in ("latin", "cjk", "other")}
    return script_token_shares_by_source


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Return the parser with one subcommand per stage."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_stage_parser(stage_name: str, help_text: str, reads_pool: bool = True) -> argparse.ArgumentParser:
        stage_parser = subparsers.add_parser(stage_name, help=help_text)
        stage_parser.add_argument("--funnel-dir", type=Path, default=None,
                                  help="folder for the stage files (default: $MIDTRAIN_DATA/funnel)")
        if reads_pool:
            stage_parser.add_argument("--pool", nargs="*", default=None,
                                      help="parquet files, folders or globs (default: the whole $MIDTRAIN_DATA/pool)")
        if stage_name not in ("filter", "report"):
            stage_parser.add_argument("--earlier-decisions", nargs="*", type=Path, default=None,
                                      help="decision files to apply first (default: every earlier stage in --funnel-dir)")
        return stage_parser

    filter_parser = add_stage_parser("filter", "cheap heuristics with one reason per dropped document")
    for threshold_name, threshold_value in default_filter_thresholds.items():
        filter_parser.add_argument(f"--{threshold_name.replace('_', '-')}", type=type(threshold_value), default=threshold_value)
    filter_parser.add_argument("--source-threshold", nargs=3, action="append", default=[], metavar=("SOURCE", "NAME", "VALUE"),
                               help="one threshold for one source, e.g. math_web max_short_line_character_share 0.8; "
                                    "repeatable, added to default_filter_threshold_overrides_by_source")
    filter_parser.add_argument("--worker-count", type=int, default=get_usable_cpu_count())

    dedup_parser = add_stage_parser("dedup", "exact and near duplicates across sources")
    dedup_parser.add_argument("--source-priority", nargs="+", default=list(default_source_priority),
                              help="sources from the copy to keep to the copy to drop")
    dedup_parser.add_argument("--agreement-threshold", type=float, default=default_signature_agreement_threshold)
    dedup_parser.add_argument("--band-count", type=int, default=default_lsh_band_count)
    dedup_parser.add_argument("--rows-per-band", type=int, default=default_lsh_rows_per_band)
    dedup_parser.add_argument("--worker-count", type=int, default=get_usable_cpu_count())

    embed_parser = add_stage_parser("embed", "one vector per document")
    embed_parser.add_argument("--embedding-model", default=default_embedding_model)
    embed_parser.add_argument("--device", required=True, help="cuda, cuda:1 or cpu")

    semdedup_parser = add_stage_parser("semdedup", "semantic duplicates inside k-means clusters", reads_pool=False)
    semdedup_parser.add_argument("--similarity-threshold", type=float, default=default_semantic_similarity_threshold)
    semdedup_parser.add_argument("--documents-per-cluster", type=int, default=default_documents_per_cluster)
    semdedup_parser.add_argument("--source-priority", nargs="+", default=list(default_source_priority))
    semdedup_parser.add_argument("--sources-to-leave-out", nargs="*", default=[],
                                 help="sources that are kept without being compared, e.g. synthetic_rephrase")
    semdedup_parser.add_argument("--device", required=True, help="cuda, cuda:1 or cpu")

    score_parser = add_stage_parser("score", "mean next-token loss per document under a causal LM")
    score_parser.add_argument("--model-path", required=True)
    score_parser.add_argument("--score-name", default=None, help="student or reference; names scores_<name>.parquet")
    score_parser.add_argument("--output-path", type=Path, default=None, help="instead of --score-name, for split runs")
    score_parser.add_argument("--max-tokens-per-document", type=int, default=1024)
    score_parser.add_argument("--padded-tokens-per-batch", type=int, default=default_padded_tokens_per_batch)
    score_parser.add_argument("--logit-chunk-token-count", type=int, default=default_logit_chunk_token_count)
    score_parser.add_argument("--device", required=True, help="cuda, cuda:1 or cpu")

    select_parser = add_stage_parser("select", "learnability selection from student and reference losses", reads_pool=False)
    select_parser.add_argument("--student-scores", nargs="*", type=Path, default=None,
                               help="default: scores_student*.parquet in --funnel-dir")
    select_parser.add_argument("--reference-scores", nargs="*", type=Path, default=None,
                               help="default: scores_reference*.parquet in --funnel-dir")
    select_parser.add_argument("--trivial-quantile", type=float, default=default_trivial_quantile)
    select_parser.add_argument("--noise-quantile", type=float, default=default_noise_quantile)
    select_parser.add_argument("--source-quantiles", nargs="*", default=[], metavar="SOURCE=TRIVIAL,NOISE",
                               help="other quantiles for one source, e.g. wiki_en=0.05,0.95")
    select_parser.add_argument("--token-budget", type=float, default=None, help="tokens to keep in total")
    select_parser.add_argument("--source-token-budgets", nargs="*", default=[], metavar="SOURCE=TOKENS",
                               help="tokens to keep per source, e.g. replay_web=1.2e9")

    report_parser = add_stage_parser("report", "merge the stage files into one JSON", reads_pool=False)
    report_parser.add_argument("--report-path", type=Path, default=default_funnel_report_path)
    return parser


def parse_source_value_pairs(source_value_pairs: list[str]) -> dict:
    """Turn ["wiki_en=0.05,0.95", ...] into {"wiki_en": "0.05,0.95", ...}."""
    values_by_source = {}
    for source_value_pair in source_value_pairs:
        assert source_value_pair.count("=") == 1, f"expected SOURCE=VALUE, got {source_value_pair!r}"
        source, value = source_value_pair.split("=")
        values_by_source[source] = value
    return values_by_source


def main(argument_list=None) -> None:
    """Run one stage from the command line."""
    arguments = build_argument_parser().parse_args(argument_list)
    funnel_directory = arguments.funnel_dir or get_midtrain_data_directory() / "funnel"
    earlier_decision_paths = getattr(arguments, "earlier_decisions", None)
    if earlier_decision_paths is None and arguments.command not in ("filter", "report"):
        earlier_decision_paths = get_default_earlier_decision_paths(funnel_directory, arguments.command)

    if arguments.command == "filter":
        thresholds = {threshold_name: getattr(arguments, threshold_name) for threshold_name in default_filter_thresholds}
        threshold_overrides_by_source = {
            source: dict(overrides) for source, overrides in default_filter_threshold_overrides_by_source.items()}
        for source, threshold_name, threshold_value in arguments.source_threshold:
            assert threshold_name in default_filter_thresholds, f"unknown filter threshold {threshold_name!r}"
            threshold_type = type(default_filter_thresholds[threshold_name])
            threshold_overrides_by_source.setdefault(source, {})[threshold_name] = threshold_type(threshold_value)
        filter_documents(arguments.pool, get_decision_path(funnel_directory, "filter"), thresholds,
                         threshold_overrides_by_source, arguments.worker_count)
    elif arguments.command == "dedup":
        deduplicate_documents(arguments.pool, get_decision_path(funnel_directory, "dedup"), tuple(arguments.source_priority),
                              earlier_decision_paths, arguments.agreement_threshold, arguments.band_count,
                              arguments.rows_per_band, arguments.worker_count)
    elif arguments.command == "embed":
        embed_pool_documents(arguments.pool, funnel_directory / "embeddings_float16.npy",
                             funnel_directory / "embedding_doc_ids.parquet", arguments.embedding_model,
                             device=arguments.device, earlier_decision_paths=earlier_decision_paths)
    elif arguments.command == "semdedup":
        deduplicate_documents_semantically(
            funnel_directory / "embeddings_float16.npy", funnel_directory / "embedding_doc_ids.parquet",
            get_decision_path(funnel_directory, "semdedup"), arguments.similarity_threshold,
            arguments.documents_per_cluster, tuple(arguments.source_priority), device=arguments.device,
            earlier_decision_paths=earlier_decision_paths, sources_to_leave_out=tuple(arguments.sources_to_leave_out))
    elif arguments.command == "score":
        assert (arguments.score_name is None) != (arguments.output_path is None), "give --score-name or --output-path"
        output_path = arguments.output_path or funnel_directory / f"scores_{arguments.score_name}.parquet"
        score_documents_with_causal_lm(
            arguments.model_path, arguments.pool, output_path, arguments.max_tokens_per_document,
            device=arguments.device, earlier_decision_paths=earlier_decision_paths,
            padded_tokens_per_batch=arguments.padded_tokens_per_batch,
            logit_chunk_token_count=arguments.logit_chunk_token_count)
    elif arguments.command == "select":
        student_score_paths = arguments.student_scores or sorted(funnel_directory.glob("scores_student*.parquet"))
        reference_score_paths = arguments.reference_scores or sorted(funnel_directory.glob("scores_reference*.parquet"))
        assert student_score_paths and reference_score_paths, (
            f"no scores_student*.parquet or scores_reference*.parquet in {funnel_directory}: run `score` for both models")
        quantiles_by_source = {
            source: tuple(float(quantile) for quantile in quantile_pair.split(","))
            for source, quantile_pair in parse_source_value_pairs(arguments.source_quantiles).items()}
        token_budget_by_source = {
            source: int(float(budget)) for source, budget in parse_source_value_pairs(arguments.source_token_budgets).items()}
        select_learnable_documents(
            student_score_paths, reference_score_paths, get_decision_path(funnel_directory, "select"),
            arguments.trivial_quantile, arguments.noise_quantile, quantiles_by_source,
            None if arguments.token_budget is None else int(arguments.token_budget), token_budget_by_source,
            earlier_decision_paths)
    else:
        write_funnel_report(funnel_directory, arguments.report_path)


if __name__ == "__main__":
    main()
