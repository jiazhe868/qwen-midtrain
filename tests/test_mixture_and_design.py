"""Tests of the mixture sampler (mixture.py) and of the experiment design (mixing_law.py); CPU only, seconds.

Why these tests exist. A mixture study compares runs that differ by a few percent of their tokens. A sampler that is
off by a few percent, a long sequence that mixes two packs, or a flagged document that slips into a stream would
not stop any run and would not show in any loss curve; the comparison would just be wrong. Each test checks one
such promise on a made-up pool in which every token tells where it came from: a token id is
`source_number * 1,000,000 + document_number * 1,000 + position`.

    source scripts/env.sh && $MIDTRAIN_PYTHON -m pytest tests/test_mixture_and_design.py -q
"""
import json

import numpy as np
import pytest

from midtrain import data_sources, mixing_law, mixture

source_numbers = {"replay_web": 1, "wiki_zh": 2, "math_web": 3, "code": 4, "wiki_en": 5}
end_of_text_id_in_tests = 999_999_999
context_length = 64
tokens_per_step = 512                     # 8 sequences per step
documents_per_source = 400
tokens_per_document = 50


def source_of_token(token_id: int) -> str:
    return {number: name for name, number in source_numbers.items()}[int(token_id) // 1_000_000]


def document_id_of(source_name: str, document_number: int) -> str:
    """An id whose first 12 hex digits are never divisible by the huge `validation_one_in` used below, so that
    every made-up document lands in the train split (data_sources.is_validation_document reads those digits)."""
    return f"{source_name}:{document_number + 1:012x}0000"


@pytest.fixture(scope="module", autouse=True)
def made_up_pool(tmp_path_factory):
    """Five sources of 400 documents with 50 tokens each, written by the real token-file writer."""
    patcher = pytest.MonkeyPatch()
    patcher.setenv("MIDTRAIN_DATA", str(tmp_path_factory.mktemp("midtrain_data")))
    patcher.setenv("MIDTRAIN_ALLOW_UNCHECKED_POOL", "1")
    patcher.setattr(mixture, "end_of_text_token_id", lambda: end_of_text_id_in_tests)
    mixture.load_decontamination_gate.cache_clear()
    for source_name, source_number in source_numbers.items():
        texts = [f"{source_name} document {document_number}" for document_number in range(documents_per_source)]
        document_ids = [document_id_of(source_name, document_number) for document_number in range(documents_per_source)]
        token_arrays = [np.arange(tokens_per_document, dtype=np.uint32) + source_number * 1_000_000 + document_number * 1_000
                        for document_number in range(documents_per_source)]
        data_sources.write_token_files(source_name, document_ids, texts, token_arrays, validation_one_in=10 ** 9)
    yield
    patcher.undo()
    mixture.load_decontamination_gate.cache_clear()


def make_recipe(**changes) -> dict:
    recipe = {"name": "test", "token_budget": 200 * tokens_per_step, "context_length": context_length,
              "components": {"replay_web": 0.425, "wiki_zh": 0.075, "math_web": 0.30, "code": 0.20},
              "schedule": "uniform", "subsets": {}, "seed": 1, "long_context": {"token_share": 0.0}}
    recipe.update(changes)
    mixture.check_recipe(recipe)
    return recipe


def sequences_of_run(recipe: dict) -> list[np.ndarray]:
    return [row for step in mixture.iterate_training_steps(recipe, tokens_per_step)
            for micro_batch in step["micro_batches"] for row in micro_batch]


def sequence_count_by_source(sequences: list[np.ndarray]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sequence in sequences:
        source_name = source_of_token(sequence[sequence != end_of_text_id_in_tests][0])
        counts[source_name] = counts.get(source_name, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# the sampler
# ---------------------------------------------------------------------------
def test_realised_shares_match_the_recipe_to_one_sequence():
    recipe = make_recipe()
    sequences = sequences_of_run(recipe)
    counts = sequence_count_by_source(sequences)
    for source_name, share in recipe["components"].items():
        assert abs(counts[source_name] - share * len(sequences)) <= 1, f"{source_name}: {counts[source_name]} of {len(sequences)}"


def test_a_sequence_holds_tokens_of_one_source_only():
    for sequence in sequences_of_run(make_recipe()):
        sources_in_sequence = {source_of_token(token_id) for token_id in sequence if token_id != end_of_text_id_in_tests}
        assert len(sources_in_sequence) == 1


def test_same_recipe_and_seed_give_the_same_tokens_and_another_seed_does_not():
    first = np.stack(sequences_of_run(make_recipe()))
    again = np.stack(sequences_of_run(make_recipe()))
    other_seed = np.stack(sequences_of_run(make_recipe(seed=2)))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_seed)


def test_late_and_early_spend_the_same_total_and_differ_in_time():
    non_replay_names = ("math_web", "code")
    totals, first_third = {}, {}
    for schedule in mixture.schedule_names:
        sequences = sequences_of_run(make_recipe(schedule=schedule))
        totals[schedule] = sum(sequence_count_by_source(sequences)[name] for name in non_replay_names)
        first_third[schedule] = sum(sequence_count_by_source(sequences[:len(sequences) // 3]).get(name, 0) for name in non_replay_names)
    assert max(totals.values()) - min(totals.values()) <= 4, f"the schedules spend different totals: {totals}"
    assert first_third["early"] > first_third["uniform"] > first_third["late"], first_third


def test_a_late_schedule_with_too_little_replay_is_refused():
    with pytest.raises(AssertionError, match="replay"):
        make_recipe(components={"replay_web": 0.30, "math_web": 0.40, "code": 0.30}, schedule="late")


# ---------------------------------------------------------------------------
# long sequences and packs
# ---------------------------------------------------------------------------
def write_pack_set(pack_name: str, documents_per_pack: int, pack_count: int) -> None:
    """Packs of consecutive wiki_en and math_web documents: pack p holds documents p*k .. p*k+k-1 of both."""
    pack_directory = data_sources.data_root() / "packs" / pack_name
    pack_directory.mkdir(parents=True, exist_ok=True)
    source_of_reference, document_of_reference, pack_starts = [], [], []
    for pack_index in range(pack_count):
        pack_starts.append(len(document_of_reference))
        for offset in range(documents_per_pack):
            source_of_reference.append(offset % 2)
            document_of_reference.append(pack_index * documents_per_pack + offset)
    (pack_directory / "source_names.json").write_text(json.dumps(["wiki_en", "math_web"]))
    np.save(pack_directory / "source_of_reference.npy", np.array(source_of_reference, dtype=np.int16))
    np.save(pack_directory / "document_of_reference.npy", np.array(document_of_reference, dtype=np.int64))
    np.save(pack_directory / "pack_start_references.npy", np.array(pack_starts, dtype=np.int64))


def test_a_long_sequence_reads_one_pack_from_its_start():
    documents_per_pack, long_length = 6, 256            # 6 x 51 = 306 tokens per pack, one sequence takes the first 256
    write_pack_set("test_packs", documents_per_pack, pack_count=40)
    recipe = make_recipe(long_context={"token_share": 0.5, "sequence_length": long_length, "pack": "test_packs"})
    long_sequences = [row for step in mixture.iterate_training_steps(recipe, tokens_per_step)
                      for micro_batch in step["micro_batches"] if micro_batch.shape[1] == long_length for row in micro_batch]
    assert len(long_sequences) == 200 * tokens_per_step // long_length // 2, "half of the tokens should sit in long sequences"
    for sequence in long_sequences:
        document_numbers = [(int(token_id) % 1_000_000) // 1_000 for token_id in sequence if token_id != end_of_text_id_in_tests]
        packs_in_sequence = {document_number // documents_per_pack for document_number in document_numbers}
        assert len(packs_in_sequence) == 1, f"a long sequence holds documents of packs {packs_in_sequence}"
        assert document_numbers[0] % documents_per_pack == 0 and int(sequence[0]) % 1_000 == 0, "it does not start at the pack's start"


def test_a_pack_shorter_than_a_sequence_is_refused():
    write_pack_set("short_packs", documents_per_pack=2, pack_count=10)          # 102 tokens per pack
    recipe = make_recipe(long_context={"token_share": 0.5, "sequence_length": 256, "pack": "short_packs"})
    with pytest.raises(AssertionError, match="shortest pack"):
        next(mixture.iterate_training_steps(recipe, tokens_per_step))


# ---------------------------------------------------------------------------
# the decontamination gate
# ---------------------------------------------------------------------------
def test_flagged_documents_never_reach_a_stream(monkeypatch):
    flagged = {document_id_of("math_web", document_number) for document_number in range(0, documents_per_source, 3)}
    gate = {"is_switched_off": False, "flagged_document_ids": {"math_web": flagged},
            "scanned_document_count": {name: documents_per_source for name in source_numbers}}
    monkeypatch.setattr(mixture, "load_decontamination_gate", lambda: gate)
    seen_math_documents = {(int(token_id) % 1_000_000) // 1_000 for sequence in sequences_of_run(make_recipe())
                           for token_id in sequence if token_id != end_of_text_id_in_tests and source_of_token(token_id) == "math_web"}
    assert seen_math_documents, "the run read no math document at all"
    assert not any(document_number % 3 == 0 for document_number in seen_math_documents)


def test_a_source_the_gate_has_not_scanned_is_refused(monkeypatch):
    gate = {"is_switched_off": False, "flagged_document_ids": {}, "scanned_document_count": {"replay_web": documents_per_source}}
    monkeypatch.setattr(mixture, "load_decontamination_gate", lambda: gate)
    with pytest.raises(AssertionError, match="has not scanned"):
        next(mixture.iterate_training_steps(make_recipe(), tokens_per_step))


# ---------------------------------------------------------------------------
# the design and the law
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def screening_design() -> dict:
    return mixing_law.load_design(data_sources.repository_root / "configs" / "designs" / "screen18.yaml")


def test_every_run_of_the_design_is_a_valid_recipe(screening_design):
    runs = mixing_law.run_table_of_design(screening_design)
    assert len(runs) == 18
    for run in runs:
        assert abs(sum(run["shares"].values()) - 1.0) < 1e-5
        recipe = dict(run["recipe"], subsets={})
        recipe.setdefault("long_context", {"token_share": 0.0})
        mixture.check_recipe(recipe)
        long_share = recipe["long_context"].get("token_share", 0.0)
        replay_of_all_tokens = (recipe["components"]["replay_web"] + recipe["components"]["wiki_zh"]) * (1.0 - long_share)
        assert abs(replay_of_all_tokens - run["levels"]["replay_share"]) < 1e-4, "the replay share of all tokens is not the factor's level"


def test_a_planted_law_is_recovered_exactly_without_noise(screening_design):
    runs = mixing_law.run_table_of_design(screening_design)
    design_matrix, names = mixing_law.regressors_of_runs(runs)
    planted = np.array([30.0, 8.0, 3.0, 12.0, 6.0, -4.0, 2.0, -0.3, 0.3, 0.8])
    law = mixing_law.fit_mixing_law(runs, design_matrix @ planted)
    assert np.allclose([law["coefficients"][name] for name in names], planted, atol=1e-8)
    assert law["degrees_of_freedom"] == 8


def test_the_best_mixture_stays_inside_the_tested_ranges_and_respects_the_guard(screening_design):
    runs = mixing_law.run_table_of_design(screening_design)
    design_matrix, names = mixing_law.regressors_of_runs(runs)
    target_law = mixing_law.fit_mixing_law(runs, design_matrix @ np.array([30.0, 8.0, 3.0, 12.0, 6.0, -4.0, 2.0, -0.3, 0.3, 0.8]))
    # the guard falls by 10 units per unit of math share: with a tolerance of 0.5, math may take at most 5% of the tokens
    guard_law = mixing_law.fit_mixing_law(runs, design_matrix @ np.array([50.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    unguarded = mixing_law.best_mixture_under_constraint(screening_design, target_law, None, 0.5)
    guarded = mixing_law.best_mixture_under_constraint(screening_design, target_law, guard_law, 0.5)
    for result in (unguarded, guarded):
        for factor_name, factor in screening_design["factors"].items():
            level = result["best"]["levels"][factor_name]
            if isinstance(level, float):
                assert min(factor["levels"]) - 1e-9 <= level <= max(factor["levels"]) + 1e-9
    assert guarded["best"]["shares"]["math"] <= 0.05 + 1e-6 < unguarded["best"]["shares"]["math"]
    assert guarded["best"]["predicted_target"] <= unguarded["best"]["predicted_target"]
    assert "replay_share" in unguarded["factors_at_the_edge_of_their_range"]


def test_rank_agreement_is_one_for_the_same_order_and_minus_one_for_the_reverse():
    values = np.arange(18, dtype=float)
    assert mixing_law.spearman_with_interval(values, 2 * values + 1)["spearman"] == 1.0
    assert mixing_law.spearman_with_interval(values, -values)["spearman"] == -1.0
