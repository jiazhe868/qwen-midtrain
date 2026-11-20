"""Tests of the two hygiene gates (data_hygiene.py); CPU only, seconds.

Why these tests exist. Both gates fail silently when they fail: a decontamination layer that misses a planted
question still reports a small, reassuring count, and a diversity check that cannot tell 30 topics from 3 still
passes every batch. The first measurement of English Wikipedia found 2 clusters with 91% of the documents labelled
noise (11-20), a check that could not fail; these tests pin the behaviour on data where the right answer is known.

    source scripts/env.sh && $MIDTRAIN_PYTHON -m pytest tests/test_data_hygiene.py -q
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from midtrain import data_hygiene
from midtrain.data_sources import pool_schema

filler_text = "The committee met on Tuesday to review the budget for the coming year and agreed to postpone the vote. " * 20


def topic_vectors(topic_count: int, document_count: int, seed: int) -> np.ndarray:
    """Unit vectors in 384 dimensions scattered around `topic_count` random centres, as a sentence encoder would
    place documents of that many topics."""
    random_generator = np.random.default_rng(seed)
    centres = random_generator.normal(size=(topic_count, 384))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    vectors = centres[random_generator.integers(0, topic_count, document_count)]
    vectors = vectors + 0.35 * random_generator.normal(size=vectors.shape) / np.sqrt(384)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------
def test_vendi_score_counts_identical_documents_once_and_orthogonal_ones_each():
    assert data_hygiene.vendi_score_of_unit_vectors(np.ones((50, 16))) == pytest.approx(1.0)
    assert data_hygiene.vendi_score_of_unit_vectors(np.eye(16)) == pytest.approx(16.0)


def test_clusters_and_vendi_score_separate_thirty_topics_from_three():
    diverse = data_hygiene.measure_embedding_statistics(topic_vectors(30, 6000, seed=0), 2000)
    collapsed = data_hygiene.measure_embedding_statistics(topic_vectors(3, 6000, seed=1), 2000)
    assert diverse["hdbscan_cluster_count"] == 30 and collapsed["hdbscan_cluster_count"] == 3
    assert diverse["embedding_vendi_score"] > 4 * collapsed["embedding_vendi_score"]


def test_the_gate_fails_a_collapsed_batch_on_the_vendi_score():
    diverse = data_hygiene.measure_embedding_statistics(topic_vectors(30, 6000, seed=0), 2000)
    collapsed = data_hygiene.measure_embedding_statistics(topic_vectors(3, 6000, seed=1), 2000)
    shared = {"sampled_token_count": 1000, "bigram_entropy_bits": 16.0, "template_repetition_rate": 0.02,
              "zstd_compression_ratio": 3.3}
    checks = data_hygiene.check_diversity_against_baseline({**shared, **collapsed}, {**shared, **diverse})
    verdict_of = {check["measurement"]: check["passed"] for check in checks}
    assert verdict_of["embedding_vendi_score"] is False
    assert verdict_of["hdbscan_clusters_per_10k_documents"] is False


# ---------------------------------------------------------------------------
# decontamination
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("matched_text, kind", [
    ("0 0 0 0 0 0 0 0 0 0 0 0 0", "numbers"),
    ("1 1 2 1 3 1 4 1 5 1 6 1 7", "numbers"),
    ("for 2 3 5 7 11 13 17 19 23 29 31 and", "mostly_numbers"),
    ("a b c d e f g h i j k l m", "alphabet"),
    ("is a natural number greater than 1 that is not a product of", "words"),
])
def test_matched_ngrams_are_sorted_into_kinds(matched_text, kind):
    assert data_hygiene.kind_of_matched_ngram(matched_text) == kind


def test_a_question_planted_in_an_unrelated_document_is_flagged_and_clean_text_is_not(tmp_path):
    question = ("A train leaves the station at nine in the morning and travels at sixty miles per hour toward a town "
                "that lies one hundred and eighty miles away. At what time does it arrive?")
    evaluation_items = [{"item_id": "made_up:test:0", "task": "made_up", "kind": "numeric", "question": question,
                         "answer_text": "12", "meta": {}}]
    texts = [filler_text + question + " " + filler_text,              # the whole question inside a document
             filler_text + question[len(question) // 3:] + filler_text,  # its last two thirds (more than 13 words)
             filler_text * 3]                                           # nothing of it
    document_ids = ["test:whole", "test:part", "test:clean"]
    pq.write_table(pa.table({"doc_id": document_ids, "source": ["test"] * 3, "text": texts,
                             "token_count": [len(text) // 4 for text in texts], "meta_json": ["{}"] * 3}, schema=pool_schema),
                   tmp_path / "part-00000.parquet")
    result = data_hygiene.flag_documents_by_ngram([tmp_path / "part-00000.parquet"], evaluation_items, worker_count=1)
    assert set(result["flagged_documents"]) == {"test:whole", "test:part"}
