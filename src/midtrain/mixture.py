"""Turn a recipe (mixture weights, a schedule, a long-context share) into the exact token sequences of a run.

Why this file exists. A mixture study is only as good as its control over what each run saw. Sampling a source at
random for every sequence gives the right shares on average, but a 20M-token run of 5,000 sequences then misses a 5%
source's share by several percent of itself, which is the size of the effects we want to measure. Here the shares are
exact: every sequence slot goes to the source that is furthest behind its target (smooth weighted round-robin), the
document order of each source is a seeded permutation, and two runs with the same recipe and seed see the same
tokens in the same order.

What a recipe controls (a YAML file, see configs/recipes/):
    components      token share of each pool source; they sum to 1
    subsets         optional: the curated subset of a source to draw from (a list of document positions)
    schedule        when the non-replay share is spent: `uniform`, `late` (ramps up as the learning rate anneals) or
                    `early`; the mean share over the run is the same in all three
    long_context    the share of tokens that sit in long sequences, their length, and the pack (document order) they
                    come from: related documents side by side (`semantic`) or shuffled (`random`), see
                    document_packing.py
    context_length  the length of an ordinary sequence
    token_budget    the tokens of the whole run
    seed

A sequence is a slice of one source's stream: its documents in the seeded order, an end-of-text token after each. A
document can continue into the next sequence of the same source, as in ordinary pretraining. A long sequence is the
first `sequence_length` tokens of one pack, so that its documents are the ones the pack put together.

No document that the decontamination gate flagged (data_hygiene.py, README §11) can reach a run: every stream leaves
flagged documents out, and a recipe that names a source the gate has not scanned stops with an error. The first gate
run flagged 6,030 of 2,686,237 documents (2026-11-20); most matches are number sequences that also occur in a
benchmark's examples, and all of them are dropped, because 0.2% of the pool costs nothing and a real leak would.

Commands (after `source scripts/env.sh`):
    $MIDTRAIN_PYTHON -m midtrain.mixture show configs/recipes/pilot_replay_only.yaml       # realized shares, no GPU
    $MIDTRAIN_PYTHON -m midtrain.mixture make-subset --source replay_web --name learnable \
        --decisions $MIDTRAIN_DATA/funnel/replay_web.select.parquet --keep-column keep
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from midtrain.data_sources import data_root, load_source_specifications, open_token_source, tokenizer_path

schedule_names = ("uniform", "late", "early")
# The non-replay share is multiplied by a factor that moves linearly between these two values over the run and has
# mean 1, so `late` and `early` spend exactly the tokens `uniform` spends. 0.4 to 1.6 keeps the replay share
# positive for every recipe with at least 37.5% replay (1 - 1.6 * 0.625 = 0).
schedule_factor_low, schedule_factor_high = 0.4, 1.6


def end_of_text_token_id() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert isinstance(token_id, int) and token_id >= 0, "the tokenizer has no <|endoftext|> token"
    return token_id


def role_of_source(source_name: str, source_specifications: dict) -> str:
    """replay, curated or hard for the public sources (configs/sources.yaml); the built sources by their name."""
    if source_name in source_specifications["sources"]:
        return source_specifications["sources"][source_name]["role"]
    if source_name.startswith("synthetic"):
        return "synthetic"
    if source_name.startswith("primitives"):
        return "primitives"
    raise ValueError(f"unknown source {source_name!r}: add it to configs/sources.yaml or name it synthetic_* / primitives*")


# ---------------------------------------------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------------------------------------------
def load_recipe(recipe_path: str | Path) -> dict:
    recipe = yaml.safe_load(Path(recipe_path).read_text())
    recipe.setdefault("name", Path(recipe_path).stem)
    recipe.setdefault("schedule", "uniform")
    recipe.setdefault("subsets", {})
    recipe.setdefault("context_length", 4096)
    recipe.setdefault("seed", 1)
    recipe.setdefault("long_context", {"token_share": 0.0})
    check_recipe(recipe)
    return recipe


def check_recipe(recipe: dict) -> None:
    weights = recipe["components"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6, f"{recipe['name']}: component shares sum to {sum(weights.values())}"
    assert all(weight >= 0 for weight in weights.values()), f"{recipe['name']}: negative share"
    assert recipe["schedule"] in schedule_names, f"{recipe['name']}: schedule must be one of {schedule_names}"
    long_share = recipe["long_context"].get("token_share", 0.0)
    assert 0.0 <= long_share < 1.0, f"{recipe['name']}: long-context share {long_share}"
    if long_share > 0:
        assert recipe["long_context"]["sequence_length"] % recipe["context_length"] == 0, (
            "a long sequence must be a whole number of ordinary sequences, so that a step holds the same tokens "
            "whether it draws long or short ones")
    if recipe["schedule"] != "uniform":
        source_specifications = load_source_specifications()
        replay_share = sum(weight for name, weight in weights.items()
                           if role_of_source(name, source_specifications) == "replay")
        assert 1.0 - schedule_factor_high * (1.0 - replay_share) >= -1e-9, (
            f"{recipe['name']}: a {recipe['schedule']} schedule needs at least "
            f"{1 - 1 / schedule_factor_high:.3f} replay; this recipe has {replay_share:.3f}")


def component_shares_at_progress(recipe: dict, progress: float, source_specifications: dict) -> dict[str, float]:
    """The token share of each source at a point of the run (0 = start, 1 = end)."""
    weights = recipe["components"]
    if recipe["schedule"] == "uniform":
        return dict(weights)
    span = schedule_factor_high - schedule_factor_low
    factor = (schedule_factor_low + span * progress if recipe["schedule"] == "late"
              else schedule_factor_high - span * progress)
    is_replay = {name: role_of_source(name, source_specifications) == "replay" for name in weights}
    replay_total = sum(weight for name, weight in weights.items() if is_replay[name])
    non_replay_total_now = factor * (1.0 - replay_total)
    shares = {}
    for name, weight in weights.items():
        if is_replay[name]:
            # the replay sources share what is left, in their own proportions
            shares[name] = weight / replay_total * (1.0 - non_replay_total_now) if replay_total > 0 else 0.0
        else:
            shares[name] = weight * factor
    return shares


# ---------------------------------------------------------------------------------------------------------------
# the decontamination gate: which documents may not be trained on
# ---------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_decontamination_gate() -> dict:
    """The gate's verdicts: `scanned_document_count` per source and the set of flagged document ids per source.

    MIDTRAIN_ALLOW_UNCHECKED_POOL=1 switches the gate off; it exists for tests on made-up text and must never be
    set for a run that is evaluated."""
    if os.environ.get("MIDTRAIN_ALLOW_UNCHECKED_POOL") == "1":
        return {"is_switched_off": True, "scanned_document_count": {}, "flagged_document_ids": {}}
    import pyarrow.parquet as pq

    from midtrain.data_sources import repository_root

    report_path = repository_root / "results" / "hygiene_decontamination.json"
    flagged_path = data_root() / "hygiene" / "decontamination_flagged.parquet"
    assert report_path.exists() and flagged_path.exists(), (
        "the pool has not passed the decontamination gate: run `python -m midtrain.data_hygiene decontaminate` first")
    report = json.loads(report_path.read_text())
    flagged = pq.read_table(flagged_path, columns=["doc_id", "source"]).to_pydict()
    flagged_document_ids: dict[str, set] = {}
    for document_id, source_name in zip(flagged["doc_id"], flagged["source"]):
        flagged_document_ids.setdefault(source_name, set()).add(document_id)
    return {"is_switched_off": False, "flagged_document_ids": flagged_document_ids,
            "scanned_document_count": {name: entry["documents"] for name, entry in report["scanned_by_source"].items()}}


def is_document_flagged(source_name: str, document_ids: list[str]) -> np.ndarray:
    """One boolean per train document of the source: True when the gate flagged it."""
    gate = load_decontamination_gate()
    if gate["is_switched_off"]:
        return np.zeros(len(document_ids), dtype=bool)
    assert source_name in gate["scanned_document_count"], (
        f"the decontamination gate has not scanned {source_name}: rerun `data_hygiene decontaminate` after adding a source")
    # the gate scans train and validation documents together, so it must have seen at least the train documents
    assert gate["scanned_document_count"][source_name] >= len(document_ids), (
        f"{source_name} has {len(document_ids):,} train documents but the gate scanned "
        f"{gate['scanned_document_count'][source_name]:,}: the pool changed after the gate ran, rerun it")
    flagged_ids = gate["flagged_document_ids"].get(source_name, set())
    return np.fromiter((document_id in flagged_ids for document_id in document_ids), dtype=bool, count=len(document_ids))


# ---------------------------------------------------------------------------------------------------------------
# streams: an endless, seeded walk through the documents of one source or of one pack
# ---------------------------------------------------------------------------------------------------------------
def seed_for(recipe_seed: int, stream_name: str, epoch_index: int) -> list[int]:
    return [int(recipe_seed), zlib.crc32(stream_name.encode("utf-8")), int(epoch_index)]


def subset_document_positions(source_name: str, subset_name: str | None, document_count: int) -> np.ndarray:
    if subset_name is None:
        return np.arange(document_count, dtype=np.int64)
    subset_path = data_root() / "subsets" / source_name / f"{subset_name}.document_positions.npy"
    assert subset_path.exists(), f"no subset {subset_name!r} for {source_name}: run `mixture make-subset` first"
    positions = np.load(subset_path)
    assert positions.max() < document_count, f"subset {subset_name} of {source_name} is from another build of the pool"
    return positions


def open_source_stream(source_name: str, subset_name: str | None, recipe_seed: int) -> dict:
    token_source = open_token_source(source_name, "train")
    document_count = len(token_source["document_offsets"]) - 1
    positions = subset_document_positions(source_name, subset_name, document_count)
    positions = positions[~is_document_flagged(source_name, token_source["document_ids"])[positions]]
    assert len(positions) > 0, f"{source_name}: no document is left after the decontamination gate"
    stream = {"name": source_name, "token_sources": [token_source], "source_of_reference": np.zeros(len(positions), np.int16),
              "document_of_reference": positions, "shuffle": True, "seed": recipe_seed, "epochs_started": 0,
              "reference_cursor": 0, "token_cursor_in_document": 0, "order": None}
    start_next_epoch(stream)
    return stream


def open_pack_stream(pack_name: str, recipe_seed: int, sequence_length: int) -> dict:
    """A pack lists documents of several sources in the order they should be read (document_packing.py).

    Checked here, because a silent miss would change what the experiment compares: every pack is at least one
    sequence long (a shorter one would be topped up with the next pack's unrelated documents), and no packed
    document is one the decontamination gate flagged."""
    pack_directory = data_root() / "packs" / pack_name
    assert pack_directory.exists(), f"no pack {pack_name!r}: run `document_packing build` first"
    source_names = json.loads((pack_directory / "source_names.json").read_text())
    stream = {"name": f"pack:{pack_name}", "token_sources": [open_token_source(name, "train") for name in source_names],
              "source_of_reference": np.load(pack_directory / "source_of_reference.npy"),
              "document_of_reference": np.load(pack_directory / "document_of_reference.npy"),
              "pack_start_references": np.load(pack_directory / "pack_start_references.npy"),
              "shuffle": False, "seed": recipe_seed, "epochs_started": 0, "reference_cursor": 0,
              "token_cursor_in_document": 0, "order": None}
    reference_count = len(stream["document_of_reference"])
    tokens_of_reference = np.zeros(reference_count, dtype=np.int64)
    for source_index, (source_name, token_source) in enumerate(zip(source_names, stream["token_sources"])):
        of_this_source = stream["source_of_reference"] == source_index
        document_positions = stream["document_of_reference"][of_this_source]
        assert not is_document_flagged(source_name, token_source["document_ids"])[document_positions].any(), (
            f"pack {pack_name} holds documents of {source_name} that the decontamination gate flagged: rebuild the pack")
        document_lengths = np.diff(token_source["document_offsets"])
        tokens_of_reference[of_this_source] = document_lengths[document_positions] + 1       # + the end-of-text token
    tokens_of_pack = np.add.reduceat(tokens_of_reference, stream["pack_start_references"])
    assert tokens_of_pack.min() >= sequence_length, (
        f"pack {pack_name}: the shortest pack has {tokens_of_pack.min():,} tokens, a long sequence needs {sequence_length:,}")
    start_next_epoch(stream)
    return stream


def start_next_epoch(stream: dict) -> None:
    """A source stream reshuffles its documents; a pack stream keeps each pack intact and reshuffles the packs."""
    random_generator = np.random.default_rng(seed_for(stream["seed"], stream["name"], stream["epochs_started"]))
    reference_count = len(stream["document_of_reference"])
    if stream["shuffle"]:
        stream["order"] = random_generator.permutation(reference_count)
    else:
        pack_starts = stream["pack_start_references"]
        pack_stops = np.append(pack_starts[1:], reference_count)
        pack_order = random_generator.permutation(len(pack_starts))
        stream["order"] = np.concatenate([np.arange(pack_starts[pack], pack_stops[pack]) for pack in pack_order])
        # where each pack begins in this epoch's order, for read_one_pack_as_sequence
        sizes_in_order = (pack_stops - pack_starts)[pack_order]
        stream["pack_start_cursors"] = np.concatenate([[0], np.cumsum(sizes_in_order)[:-1]])
        stream["pack_cursor"] = 0
    stream["epochs_started"] += 1
    stream["reference_cursor"] = 0
    stream["token_cursor_in_document"] = 0


def read_tokens_from_stream(stream: dict, token_count: int, end_of_text_id: int) -> np.ndarray:
    """The next `token_count` tokens of the stream: documents in order, an end-of-text token after each."""
    output = np.empty(token_count, dtype=np.int64)
    filled = 0
    while filled < token_count:
        if stream["reference_cursor"] >= len(stream["order"]):
            start_next_epoch(stream)
        reference = stream["order"][stream["reference_cursor"]]
        token_source = stream["token_sources"][stream["source_of_reference"][reference]]
        document_position = stream["document_of_reference"][reference]
        document_start = token_source["document_offsets"][document_position]
        document_length = token_source["document_offsets"][document_position + 1] - document_start
        cursor = stream["token_cursor_in_document"]
        if cursor < document_length:
            take = min(token_count - filled, document_length - cursor)
            output[filled:filled + take] = token_source["token_ids"][document_start + cursor:document_start + cursor + take]
            filled += take
            stream["token_cursor_in_document"] += take
        if stream["token_cursor_in_document"] >= document_length and filled < token_count:
            output[filled] = end_of_text_id
            filled += 1
            stream["reference_cursor"] += 1
            stream["token_cursor_in_document"] = 0
    return output


def read_one_pack_as_sequence(stream: dict, token_count: int, end_of_text_id: int) -> np.ndarray:
    """One long training sequence: the first `token_count` tokens of the next pack. What the pack holds beyond
    that is skipped, so that the next sequence starts at the next pack and never straddles two of them."""
    if stream["pack_cursor"] >= len(stream["pack_start_cursors"]):
        start_next_epoch(stream)
    stream["reference_cursor"] = int(stream["pack_start_cursors"][stream["pack_cursor"]])
    stream["token_cursor_in_document"] = 0
    stream["pack_cursor"] += 1
    return read_tokens_from_stream(stream, token_count, end_of_text_id)


# ---------------------------------------------------------------------------------------------------------------
# steps: which stream every sequence slot of the run reads from
# ---------------------------------------------------------------------------------------------------------------
def iterate_training_steps(recipe: dict, tokens_per_step: int) -> Iterator[dict]:
    """Yield one dict per optimizer step: `micro_batches` (int64 arrays of equal-length sequences), `progress`,
    and `tokens_by_stream`, the tokens this step took from each stream."""
    context_length = recipe["context_length"]
    assert tokens_per_step % context_length == 0, "a step must hold a whole number of sequences"
    source_specifications = load_source_specifications()
    end_of_text_id = end_of_text_token_id()
    streams = {name: open_source_stream(name, recipe["subsets"].get(name), recipe["seed"])
               for name, weight in recipe["components"].items() if weight > 0}
    long_share = recipe["long_context"].get("token_share", 0.0)
    long_length = recipe["long_context"].get("sequence_length", 0)
    if long_share > 0:
        streams["long_context"] = open_pack_stream(recipe["long_context"]["pack"], recipe["seed"], long_length)

    step_count = recipe["token_budget"] // tokens_per_step
    slots_per_step = tokens_per_step // context_length
    slots_per_long_sequence = long_length // context_length if long_share > 0 else 0
    credit_of_source = {name: 0.0 for name in recipe["components"]}
    credit_of_long_sequences = 0.0

    for step_index in range(step_count):
        progress = step_index / max(1, step_count - 1)
        tokens_by_stream: dict[str, int] = {}
        micro_batches = []

        # long sequences first: their share of the step's slots, carried over between steps so that it is exact
        long_sequences_this_step = 0
        if long_share > 0:
            credit_of_long_sequences += long_share * slots_per_step / slots_per_long_sequence
            long_sequences_this_step = min(int(credit_of_long_sequences), slots_per_step // slots_per_long_sequence)
            credit_of_long_sequences -= long_sequences_this_step
            if long_sequences_this_step:
                rows = [read_one_pack_as_sequence(streams["long_context"], long_length, end_of_text_id)
                        for _ in range(long_sequences_this_step)]
                micro_batches.append(np.stack(rows))
                tokens_by_stream["long_context"] = long_sequences_this_step * long_length

        # the other slots: each goes to the source that is furthest behind its share at this point of the run
        short_slots = slots_per_step - long_sequences_this_step * slots_per_long_sequence
        shares_now = component_shares_at_progress(recipe, progress, source_specifications)
        rows = []
        for _ in range(short_slots):
            for name, share in shares_now.items():
                credit_of_source[name] += share
            chosen = max(credit_of_source, key=lambda name: (credit_of_source[name], name))
            credit_of_source[chosen] -= 1.0
            rows.append(read_tokens_from_stream(streams[chosen], context_length, end_of_text_id))
            tokens_by_stream[chosen] = tokens_by_stream.get(chosen, 0) + context_length
        if rows:
            micro_batches.append(np.stack(rows))
        yield {"step_index": step_index, "step_count": step_count, "progress": progress,
               "micro_batches": micro_batches, "tokens_by_stream": tokens_by_stream,
               "epochs_started": {name: stream["epochs_started"] for name, stream in streams.items()}}


def iterate_validation_sequences(source_name: str, sequence_length: int, maximum_sequence_count: int) -> Iterator[np.ndarray]:
    """Fixed validation sequences of one source: its held-out documents in file order, end-of-text after each."""
    token_source = open_token_source(source_name, "validation")
    document_count = len(token_source["document_offsets"]) - 1
    stream = {"name": f"validation:{source_name}", "token_sources": [token_source],
              "source_of_reference": np.zeros(document_count, np.int16),
              "document_of_reference": np.arange(document_count, dtype=np.int64), "shuffle": True, "seed": 0,
              "epochs_started": 0, "reference_cursor": 0, "token_cursor_in_document": 0,
              "order": np.arange(document_count)}
    end_of_text_id = end_of_text_token_id()
    available_tokens = int(token_source["document_offsets"][-1]) + document_count
    for _ in range(min(maximum_sequence_count, available_tokens // sequence_length)):
        yield read_tokens_from_stream(stream, sequence_length, end_of_text_id)


# ---------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------
def show_realized_shares(recipe_path: str, tokens_per_step: int) -> dict:
    """Walk the whole recipe without a model and report what it would train on."""
    recipe = load_recipe(recipe_path)
    totals: dict[str, int] = {}
    first_third: dict[str, int] = {}
    last_third: dict[str, int] = {}
    epochs = {}
    for step in iterate_training_steps(recipe, tokens_per_step):
        for name, tokens in step["tokens_by_stream"].items():
            totals[name] = totals.get(name, 0) + tokens
            if step["progress"] < 1 / 3:
                first_third[name] = first_third.get(name, 0) + tokens
            if step["progress"] > 2 / 3:
                last_third[name] = last_third.get(name, 0) + tokens
        epochs = step["epochs_started"]
    all_tokens = sum(totals.values())
    report = {name: {"share": round(tokens / all_tokens, 4),
                     "share_in_first_third": round(first_third.get(name, 0) / max(1, sum(first_third.values())), 4),
                     "share_in_last_third": round(last_third.get(name, 0) / max(1, sum(last_third.values())), 4),
                     "passes_over_the_source": epochs.get(name, 0)} for name, tokens in sorted(totals.items())}
    print(json.dumps({"recipe": recipe["name"], "tokens": all_tokens, "streams": report}, indent=1))
    return report


def make_subset_from_decisions(source_name: str, subset_name: str, decisions_path: str, keep_column: str) -> None:
    """Store a curated subset as document positions in the train token file of the source."""
    import pyarrow.parquet as pq

    decisions = pq.read_table(decisions_path, columns=["doc_id", keep_column]).to_pydict()
    kept_document_ids = {document_id for document_id, keep in zip(decisions["doc_id"], decisions[keep_column]) if keep}
    document_ids = open_token_source(source_name, "train")["document_ids"]
    positions = np.array([position for position, document_id in enumerate(document_ids)
                          if document_id in kept_document_ids], dtype=np.int64)
    assert len(positions) > 0, f"the subset {subset_name} of {source_name} would be empty"
    subset_directory = data_root() / "subsets" / source_name
    subset_directory.mkdir(parents=True, exist_ok=True)
    np.save(subset_directory / f"{subset_name}.document_positions.npy", positions)
    print(f"{source_name}/{subset_name}: {len(positions):,} of {len(document_ids):,} train documents")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show")
    show.add_argument("recipe")
    show.add_argument("--tokens-per-step", type=int, default=32768)
    subset = commands.add_parser("make-subset")
    subset.add_argument("--source", required=True)
    subset.add_argument("--name", required=True)
    subset.add_argument("--decisions", required=True)
    subset.add_argument("--keep-column", default="keep")
    arguments = parser.parse_args()
    if arguments.command == "show":
        show_realized_shares(arguments.recipe, arguments.tokens_per_step)
    else:
        make_subset_from_decisions(arguments.source, arguments.name, arguments.decisions, arguments.keep_column)


if __name__ == "__main__":
    main()
