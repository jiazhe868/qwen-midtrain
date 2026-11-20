"""Tests of the curation funnel on tiny synthetic pools; CPU only, no network, a few seconds.

Why these tests exist. Every stage of the funnel throws documents away, and a stage that throws away the
wrong ones still produces a tidy decision file. Each test therefore builds a pool in which the right answer
is known by construction (a copy with two words changed, a vector at a chosen cosine similarity, a loss
placed in a chosen quantile) and checks that the stage finds exactly that.

The models are stand-ins: a randomly initialised two-layer GPT-2 with a 64-word vocabulary for `score`, and a
bag-of-words random projection in place of the sentence-embedding model. One test loads the real embedding
model and is skipped when it is not in the local cache.

    source scripts/env.sh && $MIDTRAIN_PYTHON -m pytest tests/test_curation_funnel.py -q
"""

import hashlib
import json
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from midtrain import curation_funnel

# ---------------------------------------------------------------------------
# fixtures: tiny pools, random text, stand-in models
# ---------------------------------------------------------------------------

# every random draw in this file comes from a generator seeded with a number written at its call, so a failing
# test fails the same way on every run
pseudo_word_vocabulary = [f"w{word_index}" for word_index in range(5_000)]


def make_random_words(random_generator, word_count: int) -> list[str]:
    """Return `word_count` words drawn uniformly from 5,000 pseudo-words (two such texts share almost no 5-gram)."""
    return [pseudo_word_vocabulary[index] for index in random_generator.integers(0, len(pseudo_word_vocabulary), word_count)]


def make_doc_id(source: str, text: str) -> str:
    """Return the pool's doc_id for a text (docs/conventions.md)."""
    return f"{source}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"


def write_pool_file(pool_directory, source: str, texts: list[str], token_counts=None) -> list[str]:
    """Write one pool source as parquet and return its doc_ids. Token counts default to the number of words."""
    if token_counts is None:
        token_counts = [len(text.split()) for text in texts]
    doc_ids = [make_doc_id(source, text) for text in texts]
    pool_table = pa.table({
        "doc_id": doc_ids,
        "source": [source] * len(texts),
        "text": texts,
        "token_count": pa.array(token_counts, type=pa.int32()),
        "meta_json": ["{}"] * len(texts),
    })
    (pool_directory / source).mkdir(parents=True, exist_ok=True)
    pq.write_table(pool_table, pool_directory / source / "part-00000.parquet")
    return doc_ids


def read_decisions_by_doc_id(decision_path) -> dict:
    """Return {doc_id: row as a dict} for a decision file."""
    return {row["doc_id"]: row for row in pq.read_table(decision_path).to_pylist()}


word_vector_cache = {}


def embed_documents_with_random_projection(texts, embedding_model, device, documents_per_batch=64):
    """Stand-in for the sentence-embedding model: the normalised sum of one fixed random vector per word.

    Two texts that share most of their words get a cosine similarity near their word overlap, which is all
    the tests need from an embedding.
    """
    embeddings = np.zeros((len(texts), 64), dtype=np.float32)
    for text_index, text in enumerate(texts):
        for word in text.lower().split():
            if word not in word_vector_cache:
                word_vector_cache[word] = np.random.default_rng(zlib.crc32(word.encode("utf-8"))).standard_normal(64)
            embeddings[text_index] += word_vector_cache[word]
    return embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)


tiny_vocabulary_words = [f"w{word_index}" for word_index in range(60)]


def save_tiny_causal_lm(model_directory, seed: int) -> None:
    """Save a randomly initialised 2-layer GPT-2 and a 64-entry word-level tokenizer into one folder."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    vocabulary = {word: token_id for token_id, word in enumerate(["[UNK]", "[PAD]"] + tiny_vocabulary_words)}
    word_level_tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    word_level_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=word_level_tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(model_directory)
    torch.manual_seed(seed)
    tiny_config = GPT2Config(vocab_size=64, n_positions=128, n_embd=32, n_layer=2, n_head=2)
    GPT2LMHeadModel(tiny_config).save_pretrained(model_directory)


def make_unit_vector_at_cosine(base_vector: np.ndarray, cosine_similarity: float, random_generator) -> np.ndarray:
    """Return a unit vector whose cosine similarity to the unit vector `base_vector` is exactly the given value."""
    random_direction = random_generator.standard_normal(base_vector.size)
    orthogonal_direction = random_direction - (random_direction @ base_vector) * base_vector
    orthogonal_direction /= np.linalg.norm(orthogonal_direction)
    return cosine_similarity * base_vector + np.sqrt(1.0 - cosine_similarity**2) * orthogonal_direction


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

def test_filter_names_the_first_failed_check_of_every_dropped_document(tmp_path):
    random_generator = np.random.default_rng(1)
    prose = " ".join(make_random_words(random_generator, 200)) + "."
    texts_by_name = {
        "prose": prose,
        "stub": "only five words in here",
        "dump": " ".join(make_random_words(random_generator, 900)),
        "loop": "click here to read more " * 40,
        # the next two repeat nothing, so that only the check they are built for can fire
        "symbols": " ".join("".join(random_generator.choice(list("+-=|<>*#~/"), 6)) + word for word in make_random_words(random_generator, 80)),
        "menu": "\n".join([" ".join(make_random_words(random_generator, 2)) for _ in range(100)] + [prose[:300]]),
    }
    doc_id_by_name = dict(zip(texts_by_name, write_pool_file(tmp_path / "pool", "replay_web", list(texts_by_name.values()))))
    # a source file of 100 short lines: fine for code, and it would be a menu anywhere else
    code_doc_id = write_pool_file(tmp_path / "pool", "code", ["\n".join(["x = x + 1", "}", "return x"] * 34)])[0]

    curation_funnel.filter_documents(
        [tmp_path / "pool"], tmp_path / "filter_decisions.parquet", {"min_token_count": 64, "max_token_count": 500})
    decisions = read_decisions_by_doc_id(tmp_path / "filter_decisions.parquet")

    expected_drop_reasons = {
        "prose": "", "stub": "too_few_tokens", "dump": "too_many_tokens", "loop": "repeated_word_4gram",
        "symbols": "low_alphanumeric_share", "menu": "mostly_short_lines"}
    for name, expected_drop_reason in expected_drop_reasons.items():
        decision = decisions[doc_id_by_name[name]]
        assert decision["drop_reason"] == expected_drop_reason, name
        assert decision["keep"] == (expected_drop_reason == ""), name
    assert decisions[code_doc_id]["keep"], "code only gets the token bounds"
    # the raw measurements are stored, so that a threshold can be changed without measuring again
    assert decisions[doc_id_by_name["loop"]]["most_frequent_word_4gram_share"] > 0.15
    assert decisions[doc_id_by_name["prose"]]["alphanumeric_character_share"] > 0.95
    assert decisions[doc_id_by_name["menu"]]["short_line_character_share"] > 0.5


def test_script_tag_follows_the_letters():
    def tag(text: str) -> str:
        return curation_funnel.tag_script_from_character_class_counts(curation_funnel.count_character_classes(text))

    assert tag("The quick brown fox jumps over the lazy dog, twice: 2 times.") == "latin"
    assert tag("今天天气很好，我们一起去公园散步吧。") == "cjk"
    assert tag("これは日本語の文章です。カタカナも含みます。") == "cjk"
    assert tag("Съешь ещё этих мягких французских булок, да выпей чаю.") == "other"
    assert tag("12345 +-*/ 67890 !!!") == "other", "no letters at all"
    # a third of the letters being CJK already means most of the content is CJK
    assert tag("今天我们讨论 transformer 模型的 attention 机制和 training 方法") == "cjk"


def test_cjk_text_is_split_into_characters_so_that_it_has_shingles():
    words = curation_funnel.split_into_normalised_words("今天天气很好 hello World", "cjk")
    assert words == [character.encode("utf-8") for character in "今天天气很好"] + [b"hello", b"world"]
    assert curation_funnel.split_into_normalised_words("  Hello\n\tWORLD  ", "latin") == [b"hello", b"world"]
    assert curation_funnel.split_into_normalised_words("   ", "latin") == []


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

def replace_some_words(words: list[str], positions: list[int]) -> list[str]:
    """Return a copy of `words` with the words at `positions` replaced by words that occur nowhere else."""
    changed_words = list(words)
    for position in positions:
        changed_words[position] = f"changed{position}"
    return changed_words


def test_near_duplicate_pairs_are_found_and_unrelated_pairs_are_not(tmp_path):
    random_generator = np.random.default_rng(2)
    original_word_lists = [make_random_words(random_generator, 300) for _ in range(40)]
    # 2 changed words break 10 of the 296 five-word shingles: Jaccard 286 / 306 = 0.93
    near_copy_word_lists = [replace_some_words(words, [50, 200]) for words in original_word_lists[:5]]
    # 150 changed words leave no shingle of the second half intact: Jaccard about 0.33, a different document
    half_copy_word_lists = [replace_some_words(words, list(range(150, 300))) for words in original_word_lists[5:8]]
    original_doc_ids = write_pool_file(tmp_path / "pool", "replay_web", [" ".join(words) for words in original_word_lists])
    near_copy_doc_ids = write_pool_file(tmp_path / "pool", "papers", [" ".join(words) for words in near_copy_word_lists])
    half_copy_doc_ids = write_pool_file(tmp_path / "pool", "wiki_en", [" ".join(words) for words in half_copy_word_lists])

    curation_funnel.deduplicate_documents([tmp_path / "pool"], tmp_path / "dedup_decisions.parquet")
    decisions = read_decisions_by_doc_id(tmp_path / "dedup_decisions.parquet")

    for original_doc_id, near_copy_doc_id in zip(original_doc_ids[:5], near_copy_doc_ids):
        # `papers` ranks before `replay_web`, so the web original is the one that goes
        assert decisions[near_copy_doc_id]["keep"]
        assert decisions[original_doc_id]["drop_reason"] == "near_duplicate"
        assert decisions[original_doc_id]["duplicate_of_doc_id"] == near_copy_doc_id
        assert decisions[original_doc_id]["signature_agreement_with_kept"] >= 0.8
    for unrelated_doc_id in original_doc_ids[5:] + half_copy_doc_ids:
        assert decisions[unrelated_doc_id]["keep"], "an unrelated or half-rewritten document was dropped"
    assert sum(not decision["keep"] for decision in decisions.values()) == 5


def test_exact_duplicates_differ_only_in_case_and_spacing(tmp_path):
    random_generator = np.random.default_rng(3)
    words = make_random_words(random_generator, 80)
    original_text = " ".join(words)
    reformatted_text = "  " + "\n\n".join(word.upper() for word in words) + "\t"
    other_text = " ".join(make_random_words(random_generator, 80))
    short_text, same_short_text, other_short_text = "three words only", "Three  WORDS only", "three words here"
    doc_ids = write_pool_file(
        tmp_path / "pool", "replay_web",
        [original_text, reformatted_text, other_text, short_text, same_short_text, other_short_text])

    curation_funnel.deduplicate_documents([tmp_path / "pool"], tmp_path / "dedup_decisions.parquet")
    decisions = read_decisions_by_doc_id(tmp_path / "dedup_decisions.parquet")

    assert sorted(decisions[doc_id]["drop_reason"] for doc_id in doc_ids[:2]) == ["", "exact_duplicate"]
    assert decisions[doc_ids[2]]["keep"]
    # texts with fewer than 5 words have no 5-word shingle; they still deduplicate exactly, and only exactly
    assert sorted(decisions[doc_id]["drop_reason"] for doc_id in doc_ids[3:5]) == ["", "exact_duplicate"]
    assert decisions[doc_ids[5]]["keep"]


def test_cross_source_priority_keeps_the_curated_copy(tmp_path):
    random_generator = np.random.default_rng(4)
    article_words = make_random_words(random_generator, 300)
    curated_text = " ".join(article_words)
    # the web copy is longer (page chrome around the article), which would win inside one source
    web_text = " ".join(make_random_words(random_generator, 12) + article_words + make_random_words(random_generator, 12))
    exact_text = " ".join(make_random_words(random_generator, 100))
    curated_doc_ids = write_pool_file(tmp_path / "pool", "wiki_en", [curated_text, exact_text])
    web_doc_ids = write_pool_file(tmp_path / "pool", "replay_web", [web_text, exact_text.upper()])
    rephrase_doc_ids = write_pool_file(tmp_path / "pool", "synthetic_rephrase", [exact_text + " "])

    curation_funnel.deduplicate_documents([tmp_path / "pool"], tmp_path / "dedup_decisions.parquet")
    decisions = read_decisions_by_doc_id(tmp_path / "dedup_decisions.parquet")

    assert decisions[curated_doc_ids[0]]["keep"] and decisions[curated_doc_ids[1]]["keep"]
    assert decisions[web_doc_ids[0]]["drop_reason"] == "near_duplicate"
    assert decisions[web_doc_ids[0]]["duplicate_of_doc_id"] == curated_doc_ids[0]
    for exact_copy_doc_id in (web_doc_ids[1], rephrase_doc_ids[0]):
        assert decisions[exact_copy_doc_id]["drop_reason"] == "exact_duplicate"
        assert decisions[exact_copy_doc_id]["duplicate_of_doc_id"] == curated_doc_ids[1]

    # with the priority turned around, the web copy wins
    curation_funnel.deduplicate_documents(
        [tmp_path / "pool"], tmp_path / "dedup_decisions.parquet",
        source_priority=("replay_web", "synthetic_rephrase", "wiki_en"))
    decisions = read_decisions_by_doc_id(tmp_path / "dedup_decisions.parquet")
    assert decisions[web_doc_ids[0]]["keep"] and not decisions[curated_doc_ids[0]]["keep"]


def test_a_source_without_a_priority_stops_the_dedup(tmp_path):
    write_pool_file(tmp_path / "pool", "brand_new_source", [" ".join(make_random_words(np.random.default_rng(5), 50))])
    with pytest.raises(AssertionError, match="missing from the source priority list"):
        curation_funnel.deduplicate_documents([tmp_path / "pool"], tmp_path / "dedup_decisions.parquet")


def test_a_repeated_doc_id_stops_the_funnel(tmp_path):
    text = " ".join(make_random_words(np.random.default_rng(6), 80))
    write_pool_file(tmp_path / "pool", "replay_web", [text, text])
    with pytest.raises(AssertionError, match="repeat a doc_id"):
        curation_funnel.filter_documents([tmp_path / "pool"], tmp_path / "filter_decisions.parquet")


def test_chain_of_near_duplicates_keeps_both_ends():
    """A~B, B~C, C~D, but A is unlike C and D: the leader rule keeps A and C and drops B and D.

    Keeping one document per connected component would keep only A, and drop C and D although no kept
    document resembles them.
    """
    random_generator = np.random.default_rng(7)
    signature_a = random_generator.integers(0, 2**32, 128, dtype=np.uint32)
    signature_b, signature_c, signature_d = signature_a.copy(), None, None
    signature_b[0:16] += 1                      # 16 of 128 values differ: agreement 0.875 with A
    signature_c = signature_b.copy()
    signature_c[16:32] += 1                     # 0.875 with B, 0.75 with A
    signature_d = signature_c.copy()
    signature_d[32:48] += 1                     # 0.875 with C, 0.625 with A
    unrelated_signatures = random_generator.integers(0, 2**32, (20, 128), dtype=np.uint32)
    signatures = np.concatenate([np.stack([signature_a, signature_b, signature_c, signature_d]), unrelated_signatures])

    near_duplicates = curation_funnel.find_near_duplicates(signatures)

    assert near_duplicates["duplicate_of_index"].tolist() == [-1, 0, -1, 2] + [-1] * 20
    assert near_duplicates["agreement_with_kept"][1] == pytest.approx(0.875)
    assert near_duplicates["agreement_with_kept"][3] == pytest.approx(0.875)
    # A-C and B-D share 12 and 14 bands, so banding proposes them; they are just not confirmed
    assert near_duplicates["candidate_pair_count"] > near_duplicates["confirmed_pair_count"] == 3


def test_connected_components_carry_their_smallest_index():
    first_indices = np.array([5, 1, 2, 8, 6])
    second_indices = np.array([6, 2, 3, 9, 7])
    labels = curation_funnel.label_connected_components(10, first_indices, second_indices)
    assert labels.tolist() == [0, 1, 1, 1, 4, 5, 5, 5, 8, 8]
    # a long chain, given in an order that hides it, still settles
    chain_order = np.random.default_rng(8).permutation(999)
    labels = curation_funnel.label_connected_components(1_000, chain_order, chain_order + 1)
    assert (labels == 0).all()


def test_minhash_agreement_estimates_the_jaccard_similarity():
    random_generator = np.random.default_rng(9)
    shared_keys = random_generator.integers(0, 2**32, 600, dtype=np.uint64)
    only_first_keys = random_generator.integers(0, 2**32, 200, dtype=np.uint64)
    only_second_keys = random_generator.integers(0, 2**32, 200, dtype=np.uint64)
    signatures = curation_funnel.compute_minhash_signatures([
        np.concatenate([shared_keys, only_first_keys]), np.concatenate([shared_keys, only_second_keys]),
        only_first_keys, random_generator.integers(0, 2**32, 1, dtype=np.uint64)])
    assert signatures.shape == (4, 128) and signatures.dtype == np.uint32
    agreement = (signatures[0] == signatures[1]).mean()
    # Jaccard 600 / 1000 = 0.6; 128 hash functions give a standard deviation of 0.043
    assert abs(agreement - 0.6) < 0.13
    assert (signatures[0] == signatures[3]).mean() < 0.05
    # the same keys in another order and in another block give the same signature
    reordered_signatures = curation_funnel.compute_minhash_signatures([only_first_keys[::-1].copy()])
    assert (reordered_signatures[0] == signatures[2]).all()


def test_worker_processes_compute_the_same_decisions_as_one_process(tmp_path):
    """Guards the stable word hash: with Python's per-process `hash()` the workers would disagree."""
    random_generator = np.random.default_rng(10)
    word_lists = [make_random_words(random_generator, 120) for _ in range(30)]
    texts = [" ".join(words) for words in word_lists] + [" ".join(replace_some_words(words, [60])) for words in word_lists[:6]]
    write_pool_file(tmp_path / "pool", "replay_web", texts)

    curation_funnel.deduplicate_documents(
        [tmp_path / "pool"], tmp_path / "one_process.parquet", worker_count=1, documents_per_worker_batch=7)
    curation_funnel.deduplicate_documents(
        [tmp_path / "pool"], tmp_path / "two_processes.parquet", worker_count=2, documents_per_worker_batch=7)

    one_process_decisions = pq.read_table(tmp_path / "one_process.parquet")
    two_process_decisions = pq.read_table(tmp_path / "two_processes.parquet")
    for column_name in one_process_decisions.column_names:
        # numpy's comparison, because it counts two NaN (the agreement of a kept document) as equal
        np.testing.assert_array_equal(
            one_process_decisions.column(column_name).to_numpy(zero_copy_only=False),
            two_process_decisions.column(column_name).to_numpy(zero_copy_only=False), err_msg=column_name)
    assert one_process_decisions.column("keep").to_pylist().count(False) == 6


# ---------------------------------------------------------------------------
# embed and semdedup
# ---------------------------------------------------------------------------

def is_embedding_model_in_local_cache() -> bool:
    try:
        from transformers.utils import cached_file
        cached_file(curation_funnel.default_embedding_model, "config.json", local_files_only=True)
        cached_file(curation_funnel.default_embedding_model, "model.safetensors", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not is_embedding_model_in_local_cache(), reason="the embedding model is not downloaded")
def test_embed_documents_returns_unit_vectors_in_the_given_order():
    texts = [
        "The earthquake struck the coast at dawn and damaged hundreds of buildings.",
        "A long text. " * 400,
        "At dawn a quake hit the coastline, and several hundred buildings were damaged.",
        "To bake bread you need flour, water, salt and yeast.",
    ]
    embeddings = curation_funnel.embed_documents(texts, curation_funnel.default_embedding_model, "cpu")
    assert embeddings.shape == (4, 384) and embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert embeddings[0] @ embeddings[2] > 0.8 > embeddings[0] @ embeddings[3]
    # a text gets the same vector whatever it is batched with (the batch is reordered by length inside)
    alone = curation_funnel.embed_documents(texts[2:3], curation_funnel.default_embedding_model, "cpu")
    np.testing.assert_allclose(alone[0], embeddings[2], atol=1e-4)
    with pytest.raises(AssertionError, match="no pooling is recorded"):
        curation_funnel.embed_documents(texts, "some/unknown-embedding-model", "cpu")


def write_embeddings_for_semdedup(directory, embeddings: np.ndarray, sources: list[str]) -> list[str]:
    """Write vectors and their doc_id order the way the embed stage does; return the doc_ids."""
    doc_ids = [f"{source}:{row:016x}" for row, source in enumerate(sources)]
    np.save(directory / "embeddings_float16.npy", embeddings.astype(np.float16))
    pq.write_table(pa.table({
        "doc_id": doc_ids, "source": sources, "token_count": pa.array([100] * len(sources), type=pa.int32())}),
        directory / "embedding_doc_ids.parquet")
    return doc_ids


@pytest.mark.parametrize("documents_per_cluster", [1_500, 12])
def test_semdedup_drops_paraphrase_like_vectors_above_the_threshold_only(tmp_path, documents_per_cluster):
    random_generator = np.random.default_rng(11)
    base_vectors = random_generator.standard_normal((30, 64))
    base_vectors /= np.linalg.norm(base_vectors, axis=1, keepdims=True)
    # paraphrases of documents 0-9 at cosine 0.97 (above 0.93), of documents 10-19 at 0.85 (below)
    close_vectors = np.stack([make_unit_vector_at_cosine(base_vectors[row], 0.97, random_generator) for row in range(10)])
    distant_vectors = np.stack([make_unit_vector_at_cosine(base_vectors[row], 0.85, random_generator) for row in range(10, 20)])
    doc_ids = write_embeddings_for_semdedup(
        tmp_path, np.concatenate([close_vectors, distant_vectors, base_vectors]),
        ["replay_web"] * 20 + ["wiki_en"] * 30)
    close_doc_ids, distant_doc_ids, base_doc_ids = doc_ids[:10], doc_ids[10:20], doc_ids[20:]

    curation_funnel.deduplicate_documents_semantically(
        tmp_path / "embeddings_float16.npy", tmp_path / "embedding_doc_ids.parquet",
        tmp_path / "semdedup_decisions.parquet", similarity_threshold=0.93,
        documents_per_cluster=documents_per_cluster, device="cpu")
    decisions = read_decisions_by_doc_id(tmp_path / "semdedup_decisions.parquet")

    for row, close_doc_id in enumerate(close_doc_ids):
        assert decisions[close_doc_id]["drop_reason"] == "semantic_duplicate"
        assert decisions[close_doc_id]["nearest_kept_doc_id"] == base_doc_ids[row], "the curated document is the kept one"
        assert decisions[close_doc_id]["nearest_kept_cosine_similarity"] == pytest.approx(0.97, abs=2e-3)
    for kept_doc_id in distant_doc_ids + base_doc_ids:
        assert decisions[kept_doc_id]["keep"]
    assert sum(not decision["keep"] for decision in decisions.values()) == 10

    # a source that is left out is neither compared nor dropped
    curation_funnel.deduplicate_documents_semantically(
        tmp_path / "embeddings_float16.npy", tmp_path / "embedding_doc_ids.parquet",
        tmp_path / "semdedup_decisions.parquet", similarity_threshold=0.93,
        documents_per_cluster=documents_per_cluster, device="cpu", sources_to_leave_out=("replay_web",))
    assert all(decision["keep"] for decision in read_decisions_by_doc_id(tmp_path / "semdedup_decisions.parquet").values())


def find_semantic_duplicates_one_by_one(embeddings: np.ndarray, similarity_threshold: float) -> np.ndarray:
    """The rule of the semdedup stage written as plainly as possible, to check the blocked version against."""
    is_kept = np.zeros(len(embeddings), dtype=bool)
    for position in range(len(embeddings)):
        similarity_to_kept = embeddings[:position][is_kept[:position]] @ embeddings[position]
        is_kept[position] = not (similarity_to_kept >= similarity_threshold).any()
    return is_kept


def test_semantic_duplicates_are_compared_with_kept_documents_only():
    angles_degrees = np.array([0.0, 20.0, 40.0])        # cos 20 = 0.94, cos 40 = 0.77
    embeddings = np.stack([np.cos(np.radians(angles_degrees)), np.sin(np.radians(angles_degrees))], axis=1).astype(np.float32)
    for documents_per_block in (2_048, 2, 1):
        result = curation_funnel.find_semantic_duplicates_in_cluster(torch.from_numpy(embeddings), 0.93, documents_per_block)
        # the second document is dropped as a copy of the first; the third resembles only the dropped one
        assert result["is_kept"].tolist() == [True, False, True]
        assert result["nearest_kept_position"].tolist() == [-1, 0, 0]
        assert result["nearest_kept_similarity"][1:] == pytest.approx([0.9397, 0.7660], abs=1e-3)
        assert np.isnan(result["nearest_kept_similarity"][0])

    # many drops, block edges in every position: the blocked walk equals the one-by-one walk
    random_generator = np.random.default_rng(12)
    random_embeddings = random_generator.standard_normal((300, 6)).astype(np.float32)
    random_embeddings /= np.linalg.norm(random_embeddings, axis=1, keepdims=True)
    expected_is_kept = find_semantic_duplicates_one_by_one(random_embeddings, 0.6)
    assert 20 < expected_is_kept.sum() < 280
    for documents_per_block in (7, 64, 2_048):
        result = curation_funnel.find_semantic_duplicates_in_cluster(torch.from_numpy(random_embeddings), 0.6, documents_per_block)
        assert (result["is_kept"] == expected_is_kept).all()


def test_kmeans_separates_well_separated_groups():
    random_generator = np.random.default_rng(13)
    group_centres = np.eye(8, dtype=np.float32)[:4]
    embeddings = np.concatenate([centre + 0.05 * random_generator.standard_normal((50, 8)) for centre in group_centres]).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    cluster_indices = curation_funnel.cluster_embeddings_with_kmeans(torch.from_numpy(embeddings), cluster_count=4).numpy()
    # every cluster holds documents of one group only (a seed that starts two centroids in one group still
    # ends with pure clusters; purity is what semdedup needs)
    for cluster_index in range(4):
        groups_in_cluster = set((np.flatnonzero(cluster_indices == cluster_index) // 50).tolist())
        assert len(groups_in_cluster) <= 1
    assert curation_funnel.cluster_embeddings_with_kmeans(torch.from_numpy(embeddings), cluster_count=1).unique().tolist() == [0]


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def compute_losses_from_full_logits(causal_lm, token_ids: list[int]) -> tuple[float, float, float]:
    """Reference: mean loss, first-half mean and second-half mean of one document from the model's own logits."""
    input_ids = torch.tensor([token_ids])
    with torch.inference_mode():
        logits = causal_lm(input_ids=input_ids).logits[0]
    token_losses = torch.nn.functional.cross_entropy(logits[:-1], input_ids[0, 1:], reduction="none").double()
    first_half_token_count = (len(token_losses) + 1) // 2
    return (token_losses.mean().item(), token_losses[:first_half_token_count].mean().item(),
            token_losses[first_half_token_count:].mean().item())


def test_chunked_loss_equals_unchunked_loss(tmp_path):
    from transformers import AutoModelForCausalLM
    save_tiny_causal_lm(tmp_path, seed=14)
    causal_lm = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    random_generator = np.random.default_rng(14)
    token_ids_per_document = [random_generator.integers(2, 64, token_count).tolist() for token_count in (37, 5, 64, 2, 1, 18, 3)]

    # 17 positions per chunk cuts through documents: the 123 scored tokens of this batch take 8 chunks
    chunked = curation_funnel.compute_document_losses_for_batch(causal_lm, token_ids_per_document, "cpu", logit_chunk_token_count=17)
    unchunked = curation_funnel.compute_document_losses_for_batch(causal_lm, token_ids_per_document, "cpu", logit_chunk_token_count=10_000)

    assert chunked["scored_token_count"].tolist() == [36, 4, 63, 1, 0, 17, 2]
    for document_index, token_ids in enumerate(token_ids_per_document):
        if len(token_ids) < 2:
            assert np.isnan(chunked["mean_loss_nats_per_token"][document_index]), "one token has nothing to predict"
            continue
        expected_mean, expected_first_half, expected_second_half = compute_losses_from_full_logits(causal_lm, token_ids)
        for losses in (chunked, unchunked):
            assert losses["mean_loss_nats_per_token"][document_index] == pytest.approx(expected_mean, rel=1e-5)
            assert losses["first_half_mean_loss_nats_per_token"][document_index] == pytest.approx(expected_first_half, rel=1e-5)
            if len(token_ids) == 2:
                assert np.isnan(losses["second_half_mean_loss_nats_per_token"][document_index])
            else:
                assert losses["second_half_mean_loss_nats_per_token"][document_index] == pytest.approx(expected_second_half, rel=1e-5)
    # an untrained model is close to uniform over its 64 tokens: ln 64 = 4.16
    assert 3.5 < chunked["mean_loss_nats_per_token"][2] < 5.0


def test_length_batches_stay_within_the_padded_token_budget():
    token_counts = np.array([10, 1000, 12, 500, 11, 0, 480, 1024, 9])
    batches = curation_funnel.group_documents_into_length_batches(token_counts, padded_tokens_per_batch=1_024)
    assert sorted(index for batch in batches for index in batch) == list(range(9))
    for batch in batches:
        assert len(batch) * token_counts[batch].max() <= 1_024
    assert [sorted(batch.tolist()) for batch in batches] == [[0, 2, 4, 5, 8], [3, 6], [1], [7]]


def test_score_stage_writes_losses_and_continues_after_an_interruption(tmp_path, capsys):
    save_tiny_causal_lm(tmp_path / "model", seed=15)
    random_generator = np.random.default_rng(15)
    texts = [" ".join(random_generator.choice(tiny_vocabulary_words, word_count)) for word_count in (30, 12, 50, 7, 90, 25, 3, 41, 16)]
    doc_ids = write_pool_file(tmp_path / "pool", "wiki_en", texts)
    score_path = tmp_path / "scores_student.parquet"

    def run_score_stage(pool_paths, **overrides):
        settings = {"max_tokens_per_document": 40, "device": "cpu", "documents_per_read": 4, "documents_between_saves": 4,
                    "padded_tokens_per_batch": 100, "logit_chunk_token_count": 16, **overrides}
        curation_funnel.score_documents_with_causal_lm(tmp_path / "model", pool_paths, score_path, **settings)

    run_score_stage([tmp_path / "pool"])
    scores = read_decisions_by_doc_id(score_path)
    assert set(scores) == set(doc_ids)
    # the window is the first 40 tokens, so the 50- and 90-word documents have 39 scored tokens
    assert [scores[doc_id]["scored_token_count"] for doc_id in doc_ids] == [29, 11, 39, 6, 39, 24, 2, 39, 15]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    causal_lm = AutoModelForCausalLM.from_pretrained(tmp_path / "model").eval()
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "model")
    for doc_id, text in zip(doc_ids, texts):
        expected_mean, expected_first_half, expected_second_half = compute_losses_from_full_logits(
            causal_lm, tokenizer(text, add_special_tokens=False)["input_ids"][:40])
        assert scores[doc_id]["mean_loss_nats_per_token"] == pytest.approx(expected_mean, rel=1e-5)
        assert scores[doc_id]["first_half_mean_loss_nats_per_token"] == pytest.approx(expected_first_half, rel=1e-5)
        assert scores[doc_id]["second_half_mean_loss_nats_per_token"] == pytest.approx(expected_second_half, rel=1e-5)

    # a second pool file arrives: only its documents are scored, the earlier rows are kept as they are
    new_doc_ids = write_pool_file(tmp_path / "pool", "papers", [" ".join(random_generator.choice(tiny_vocabulary_words, 20)) for _ in range(3)])
    capsys.readouterr()
    run_score_stage([tmp_path / "pool"])
    assert "continuing after the 9 documents" in capsys.readouterr().out
    scores_after_continuing = read_decisions_by_doc_id(score_path)
    assert set(scores_after_continuing) == set(doc_ids + new_doc_ids)
    assert all(scores_after_continuing[doc_id] == scores[doc_id] for doc_id in doc_ids)
    # scores of another window length must not be mixed into the same file
    with pytest.raises(AssertionError, match="was written with"):
        run_score_stage([tmp_path / "pool"], max_tokens_per_document=20)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def write_constructed_scores(score_path, model_name: str, doc_ids, sources, token_counts, losses) -> None:
    curation_funnel.write_score_table(score_path, {
        "doc_id": list(doc_ids), "source": list(sources), "token_count": list(token_counts),
        "scored_token_count": [min(count, 1024) - 1 for count in token_counts],
        "mean_loss_nats_per_token": list(losses), "first_half_mean_loss_nats_per_token": list(losses),
        "second_half_mean_loss_nats_per_token": list(losses),
    }, {"model_path": model_name, "max_tokens_per_document": 1024})


def test_learnability_labels_land_where_constructed(tmp_path):
    """Two sources of 100 documents whose loss scales differ by a factor of 3, as prose and code do.

    In each source the student's loss is its rank (1..100 times the scale), so the 10 lowest ranks are
    trivial. The reference is half the student's loss, except for documents 60-69, where the reference is as
    lost as the student: those have the 10 highest reference losses of their source and are noise.
    """
    doc_ids, sources, student_losses, reference_losses = [], [], [], []
    for source, loss_scale in (("wiki_en", 1.0), ("code", 3.0)):
        for rank in range(1, 101):
            doc_ids.append(f"{source}:{rank:016x}")
            sources.append(source)
            student_losses.append(loss_scale * rank / 20)
            is_constructed_noise = 60 <= rank <= 69
            reference_losses.append(loss_scale * rank / 20 * (1.0 if is_constructed_noise else 0.5))
    token_counts = [200] * 200
    write_constructed_scores(tmp_path / "scores_student.parquet", "student", doc_ids, sources, token_counts, student_losses)
    # the reference file is in another order and lacks one document, which then counts as not scored
    reference_order = np.random.default_rng(16).permutation(199)
    write_constructed_scores(
        tmp_path / "scores_reference.parquet", "reference", [doc_ids[row] for row in reference_order],
        [sources[row] for row in reference_order], [200] * 199, [reference_losses[row] for row in reference_order])

    curation_funnel.select_learnable_documents(
        [tmp_path / "scores_student.parquet"], [tmp_path / "scores_reference.parquet"], tmp_path / "select_decisions.parquet")
    decisions = read_decisions_by_doc_id(tmp_path / "select_decisions.parquet")

    assert decisions["code:0000000000000064"]["learnability_label"] == "not_scored"      # rank 100 of code, the missing one
    assert not decisions["code:0000000000000064"]["keep"]
    for source, document_count in (("wiki_en", 100), ("code", 99)):
        labels_by_rank = {rank: decisions[f"{source}:{rank:016x}"]["learnability_label"] for rank in range(1, document_count + 1)}
        assert [rank for rank, label in labels_by_rank.items() if label == "trivial"] == list(range(1, 11))
        assert [rank for rank, label in labels_by_rank.items() if label == "noise"] == list(range(60, 70))
        assert all(decisions[f"{source}:{rank:016x}"]["keep"] == (labels_by_rank[rank] == "learnable") for rank in labels_by_rank)
    assert decisions["wiki_en:0000000000000032"]["excess_loss_nats_per_token"] == pytest.approx(1.25)      # rank 50: 2.5 - 1.25

    # a token budget keeps the learnable documents with the largest excess loss: ranks 100, 99, ... of wiki_en
    curation_funnel.select_learnable_documents(
        [tmp_path / "scores_student.parquet"], [tmp_path / "scores_reference.parquet"], tmp_path / "select_decisions.parquet",
        token_budget_by_source={"wiki_en": 1_000})
    decisions = read_decisions_by_doc_id(tmp_path / "select_decisions.parquet")
    kept_wiki_ranks = [rank for rank in range(1, 101) if decisions[f"wiki_en:{rank:016x}"]["keep"]]
    assert kept_wiki_ranks == [96, 97, 98, 99, 100]
    assert decisions["wiki_en:0000000000000050"]["drop_reason"] == "over_token_budget"            # rank 80
    assert sum(decision["keep"] for decision in decisions.values() if decision["source"] == "code") == 79

    # other quantiles for one source only
    curation_funnel.select_learnable_documents(
        [tmp_path / "scores_student.parquet"], [tmp_path / "scores_reference.parquet"], tmp_path / "select_decisions.parquet",
        quantiles_by_source={"wiki_en": (0.30, 1.0)})
    decisions = read_decisions_by_doc_id(tmp_path / "select_decisions.parquet")
    wiki_labels = [decisions[f"wiki_en:{rank:016x}"]["learnability_label"] for rank in range(1, 101)]
    assert wiki_labels.count("trivial") == 30 and wiki_labels.count("noise") == 0


# ---------------------------------------------------------------------------
# the whole funnel from the command line, and the report
# ---------------------------------------------------------------------------

def test_funnel_runs_end_to_end_and_the_report_adds_up(tmp_path, monkeypatch):
    random_generator = np.random.default_rng(17)
    pool_directory, funnel_directory = tmp_path / "pool", tmp_path / "funnel"

    def make_text(word_count: int) -> str:
        return " ".join(random_generator.choice(tiny_vocabulary_words, word_count))

    wiki_texts = [make_text(120) for _ in range(30)]
    web_texts = [make_text(120) for _ in range(40)]
    web_texts += ["too short"]                                                    # filter: too_few_tokens
    web_texts += [wiki_texts[0].upper()]                                          # dedup: exact copy of a wiki document
    web_texts += [" ".join(replace_some_words(wiki_texts[1].split(), [60]))]      # dedup: near copy of a wiki document
    # semdedup: the words of a wiki document in another order (no shared 5-gram, the same bag of words)
    web_texts += [" ".join(random_generator.permutation(wiki_texts[2].split()))]
    write_pool_file(pool_directory, "wiki_en", wiki_texts)
    write_pool_file(pool_directory, "replay_web", web_texts)
    save_tiny_causal_lm(tmp_path / "student_model", seed=18)
    save_tiny_causal_lm(tmp_path / "reference_model", seed=19)
    monkeypatch.setattr(curation_funnel, "embed_documents", embed_documents_with_random_projection)

    shared_arguments = ["--funnel-dir", str(funnel_directory)]
    pool_arguments = shared_arguments + ["--pool", str(pool_directory)]
    curation_funnel.main(["filter", *pool_arguments, "--min-token-count", "20", "--worker-count", "1"])
    curation_funnel.main(["dedup", *pool_arguments, "--worker-count", "1"])
    curation_funnel.main(["embed", *pool_arguments, "--device", "cpu"])
    curation_funnel.main(["semdedup", *shared_arguments, "--device", "cpu", "--similarity-threshold", "0.95"])
    for score_name in ("student", "reference"):
        curation_funnel.main(["score", *pool_arguments, "--score-name", score_name, "--device", "cpu",
                              "--model-path", str(tmp_path / f"{score_name}_model"), "--max-tokens-per-document", "64"])
    curation_funnel.main(["select", *shared_arguments])
    curation_funnel.main(["report", *shared_arguments, "--report-path", str(tmp_path / "funnel_report.json")])

    funnel_report = json.loads((tmp_path / "funnel_report.json").read_text())
    assert funnel_report["stages_not_run"] == []
    survivors = {entry["stage"]: entry["documents"] for entry in funnel_report["survivors_after_each_stage"]}
    assert survivors["pool"] == 74 and survivors["filter"] == 73 and survivors["dedup"] == 71 and survivors["semdedup"] == 70
    assert funnel_report["stages"]["filter"]["dropped_by_reason"] == {"too_few_tokens": {"documents": 1, "tokens": 2}}
    assert funnel_report["stages"]["dedup"]["by_source"]["replay_web"]["dropped_by_reason"] == {
        "exact_duplicate": {"documents": 1, "tokens": 120}, "near_duplicate": {"documents": 1, "tokens": 120}}
    assert funnel_report["stages"]["dedup"]["by_source"]["wiki_en"]["dropped_by_reason"] == {}
    assert funnel_report["stages"]["semdedup"]["dropped_by_reason"] == {"semantic_duplicate": {"documents": 1, "tokens": 120}}
    assert funnel_report["stages"]["filter"]["token_share_by_script_tag"]["wiki_en"]["latin"] == 1.0
    select_stage = funnel_report["stages"]["select"]
    assert select_stage["entered"]["documents"] == 70
    # 10% trivial and 10% noise per source: 3 + 3 of the 30 wiki documents (the web source has 40)
    assert select_stage["by_source"]["wiki_en"]["dropped_by_reason"]["trivial"]["documents"] == 3
    assert select_stage["token_share_by_label"]["wiki_en"]["learnable"] == pytest.approx(24 / 30)
    assert survivors["select"] == select_stage["kept"]["documents"] == 70 - sum(
        dropped["documents"] for dropped in select_stage["dropped_by_reason"].values())
    for stage_name in curation_funnel.decision_stage_names:
        stage = funnel_report["stages"][stage_name]
        for counted in ("documents", "tokens"):
            assert stage["entered"][counted] == sum(source["entered"][counted] for source in stage["by_source"].values())
            assert stage["entered"][counted] - stage["kept"][counted] == sum(
                dropped[counted] for dropped in stage["dropped_by_reason"].values())

    # rerunning an early stage with another threshold makes the later files stale, and the report says so
    curation_funnel.main(["filter", *pool_arguments, "--min-token-count", "121", "--worker-count", "1"])
    with pytest.raises(AssertionError, match="must be rerun"):
        curation_funnel.main(["report", *shared_arguments, "--report-path", str(tmp_path / "funnel_report.json")])


def test_a_missing_earlier_stage_stops_the_next_one(tmp_path):
    write_pool_file(tmp_path / "pool", "wiki_en", [" ".join(make_random_words(np.random.default_rng(20), 80))])
    with pytest.raises(AssertionError, match="filter_decisions.parquet is missing"):
        curation_funnel.main(["dedup", "--funnel-dir", str(tmp_path / "funnel"), "--pool", str(tmp_path / "pool")])
    # naming no earlier decision at all runs the stage on the whole pool
    curation_funnel.main(["dedup", "--funnel-dir", str(tmp_path / "funnel"), "--pool", str(tmp_path / "pool"),
                          "--earlier-decisions", "--worker-count", "1"])
    assert (tmp_path / "funnel" / "dedup_decisions.parquet").exists()
