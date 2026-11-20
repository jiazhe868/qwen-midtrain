"""Tests of document packing on a tiny pool with six obvious topics; CPU only, no network, under a minute.

Why these tests exist. A pack set that breaks a promise still trains. A pack that is too short is filled from the
next one, a document that sits in two packs is read twice, a pack with two editions of one text teaches copying,
and a random arm that lost a few documents still looks like a control; none of it shows in a loss curve. Each
test checks one promise of document_packing.py on data where the right answer is known by construction. Every
pool document is written with the words of one of six topics, so "related" has a ground truth that does not come
from the topic vectors the packs were built with, and some documents exist in two editions, so "near copy" has
one too.

The pool is written by the real writers of data_sources.py (pool parquet, token files, a held-out split) into a
temporary MIDTRAIN_DATA, and the three pack sets are built once for the whole file. A token is a word here; no
tokenizer is loaded.

    source scripts/env.sh && $MIDTRAIN_PYTHON -m pytest tests/test_document_packing.py -q
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse
from threadpoolctl import threadpool_limits

from midtrain import data_sources, document_packing

# ---------------------------------------------------------------------------
# the tiny pool: six topics, five sources of short documents, long papers, a gate file
# ---------------------------------------------------------------------------

topic_names = ("volcano", "sourdough", "sonata", "chess", "sailing", "orchid")
words_per_topic = 40
shared_word_count = 60
# Packs of 1,024 tokens from documents of 64 to 300 words: about six documents per pack and about 100 packs, enough
# for the statistics and small enough to build in a few seconds.
test_pack_token_length = 1_024
test_token_budget = 110_000
natural_pack_count = 6
test_seed = 1
# One document in seven is held out, so the pool holds documents that the train token files do not, as the real
# pool does; the texts have to be found by doc_id.
validation_one_in = 7
documents_per_topic_by_source = {"wiki_en": 56, "textbooks": 50, "forum_qa": 32, "math_web": 32, "code": 32}
# the first documents of every topic and source exist in a second edition: a near copy under another doc_id
editions_per_topic_and_source = 5
# documents on a topic of their own, related to nothing: what is left of them at the end of a build fills the
# weak packs that the build has to drop
lone_documents_per_source = 12
contract_file_names = ("source_names.json", "source_of_reference.npy", "document_of_reference.npy",
                       "pack_start_references.npy")


@pytest.fixture(scope="module", autouse=True)
def few_linear_algebra_threads():
    """Four threads for the matrix work of this file. The matrices are tiny: with the library's default of one
    thread per core the tests burned 9 minutes of processor time in 26 s, with four threads 46 s in 21 s."""
    with threadpool_limits(limits=4, user_api="blas"):
        yield


def make_topic_text(random_generator, topic_name: str, word_count: int) -> str:
    """Three words in four come from the topic's own 40 words, the others from 60 words that every topic shares."""
    words = []
    for is_topic_word in random_generator.random(word_count) < 0.75:
        if is_topic_word:
            words.append(f"{topic_name}{random_generator.integers(words_per_topic):02d}")
        else:
            words.append(f"shared{random_generator.integers(shared_word_count):02d}")
    return " ".join(words)


def make_second_edition(random_generator, text: str, topic_name: str) -> str:
    """The same document with its last 15% rewritten: 85% of its word 8-grams are those of the first edition."""
    words = text.split()
    kept_word_count = int(0.85 * len(words))
    rewritten_end = make_topic_text(random_generator, topic_name, len(words) - kept_word_count)
    return " ".join(words[:kept_word_count]) + " " + rewritten_end


def word_shingles_of_text(text: str) -> set:
    """Every run of 8 consecutive words, as the near-copy rule counts them."""
    words = text.lower().split()
    return {tuple(words[start:start + 8]) for start in range(len(words) - 7)}


def write_pool_source(source_name: str, texts: list[str]) -> list[str]:
    """Write one source as `data_sources build` does (pool rows, train and validation token files); return the doc_ids."""
    document_ids = [data_sources.document_id_for_text(source_name, text) for text in texts]
    assert len(set(document_ids)) == len(document_ids), "two fake documents have the same text"
    token_arrays = [np.arange(len(text.split()), dtype=np.uint32) for text in texts]
    data_sources.write_pool_rows(source_name, document_ids, texts, [len(tokens) for tokens in token_arrays],
                                 [{} for _ in texts])
    data_sources.write_token_files(source_name, document_ids, texts, token_arrays, validation_one_in)
    return document_ids


def write_gate_file(flagged_document_ids) -> None:
    gate_path = document_packing.flagged_document_ids_path()
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"doc_id": pa.array(sorted(flagged_document_ids), type=pa.string())}), gate_path)


def train_documents_of_source(source_name: str) -> tuple[list[str], np.ndarray]:
    token_source = data_sources.open_token_source(source_name, "train")
    return token_source["document_ids"], np.diff(token_source["document_offsets"])


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """The pool, the gate file and the three pack sets, built once; MIDTRAIN_DATA points at them for the whole file."""
    environment = pytest.MonkeyPatch()
    environment.setenv("MIDTRAIN_DATA", str(tmp_path_factory.mktemp("midtrain_data")))
    # every random draw of this file comes from a generator seeded with a number written at its call
    random_generator = np.random.default_rng(20261120)
    topic_of_document_id, text_of_document_id, edition_pairs = {}, {}, []
    for source_name, documents_per_topic in documents_per_topic_by_source.items():
        texts, topics = [], []
        for topic_name in topic_names:
            texts_of_topic = [make_topic_text(random_generator, topic_name, int(random_generator.integers(64, 301)))
                              for _ in range(documents_per_topic)]
            second_editions = [make_second_edition(random_generator, text, topic_name)
                               for text in texts_of_topic[:editions_per_topic_and_source]]
            edition_pairs.extend(
                (data_sources.document_id_for_text(source_name, first_edition),
                 data_sources.document_id_for_text(source_name, second_edition))
                for first_edition, second_edition in zip(texts_of_topic, second_editions))
            texts += texts_of_topic + second_editions
            topics += [topic_name] * (len(texts_of_topic) + len(second_editions))
        for lone_document_index in range(lone_documents_per_source):
            lone_topic_name = f"lone{source_name.replace('_', '')}{lone_document_index}x"
            texts.append(make_topic_text(random_generator, lone_topic_name, int(random_generator.integers(64, 301))))
            topics.append(lone_topic_name)
        # documents that no pack of related documents may hold: stubs, and documents longer than half a pack
        texts += [make_topic_text(random_generator, "volcano", 30) for _ in range(5)]
        texts += [make_topic_text(random_generator, "chess", 700) for _ in range(5)]
        if source_name == "textbooks":
            # three documents that are long enough for a natural pack
            texts += [make_topic_text(random_generator, "orchid", 1_500) for _ in range(3)]
        document_ids = write_pool_source(source_name, texts)
        topic_of_document_id.update(zip(document_ids, topics))
        text_of_document_id.update(zip(document_ids, texts))
    # ten papers that fill a natural pack and six that are too short for one
    paper_texts = [make_topic_text(random_generator, topic_names[paper_index % 6],
                                   int(random_generator.integers(1_100, 2_600))) for paper_index in range(10)]
    paper_texts += [make_topic_text(random_generator, "sonata", int(random_generator.integers(300, 900)))
                    for _ in range(6)]
    text_of_document_id.update(zip(write_pool_source("papers", paper_texts), paper_texts))

    # the gate flags ten wiki_en and three code documents that could be sampled, one long paper, and one doc_id
    # that is not in the pool at all
    flagged_document_ids = {"replay_web:0000000000000000"}
    flagged_candidates_by_source = {source_name: 0 for source_name in documents_per_topic_by_source}
    for source_name, flagged_count in (("wiki_en", 10), ("code", 3)):
        document_ids, token_counts = train_documents_of_source(source_name)
        candidates = [document_id for document_id, token_count in zip(document_ids, token_counts)
                      if 64 <= token_count <= test_pack_token_length // 2]
        flagged_document_ids.update(candidates[-flagged_count:])
        flagged_candidates_by_source[source_name] = flagged_count
    paper_ids, paper_token_counts = train_documents_of_source("papers")
    flagged_long_paper = next(document_id for document_id, token_count in zip(paper_ids, paper_token_counts)
                              if token_count >= test_pack_token_length)
    flagged_document_ids.add(flagged_long_paper)
    write_gate_file(flagged_document_ids)

    summaries = {
        "semantic": document_packing.build_semantic_packs("semantic_test", test_token_budget, test_pack_token_length,
                                                          test_seed, worker_count=1),
        "random": document_packing.build_random_packs("random_test", "semantic_test", test_pack_token_length,
                                                      test_seed),
        "natural": document_packing.build_natural_packs("natural_test", natural_pack_count * test_pack_token_length,
                                                        test_pack_token_length, test_seed)}
    yield {"topic_of_document_id": topic_of_document_id, "text_of_document_id": text_of_document_id,
           "edition_pairs": edition_pairs, "flagged_document_ids": flagged_document_ids,
           "flagged_candidates_by_source": flagged_candidates_by_source, "summaries": summaries}
    environment.undo()


def read_pack_set_files(pack_name: str) -> dict:
    """The four contract files as a reader loads them, and what the tests derive from them and the token files."""
    pack_directory = data_sources.data_root() / "packs" / pack_name
    source_names = json.loads((pack_directory / "source_names.json").read_text())
    source_of_reference = np.load(pack_directory / "source_of_reference.npy")
    document_of_reference = np.load(pack_directory / "document_of_reference.npy")
    pack_start_references = np.load(pack_directory / "pack_start_references.npy")
    train_documents = [train_documents_of_source(source_name) for source_name in source_names]
    return {
        "source_names": source_names, "source_of_reference": source_of_reference,
        "document_of_reference": document_of_reference, "pack_start_references": pack_start_references,
        "document_id_of_reference": [train_documents[source][0][document]
                                     for source, document in zip(source_of_reference, document_of_reference)],
        "token_count_of_reference": np.array([train_documents[source][1][document]
                                              for source, document in zip(source_of_reference, document_of_reference)]),
        "references_of_packs": np.split(np.arange(len(source_of_reference)), pack_start_references[1:])}


def read_contract_file_bytes(pack_name: str) -> dict:
    pack_directory = data_sources.data_root() / "packs" / pack_name
    return {file_name: (pack_directory / file_name).read_bytes() for file_name in contract_file_names}


# ---------------------------------------------------------------------------
# the control: both arms hold the same documents, cut by the same rule
# ---------------------------------------------------------------------------

def test_semantic_and_random_packs_hold_exactly_the_same_documents(workspace):
    semantic_pack_set = read_pack_set_files("semantic_test")
    random_pack_set = read_pack_set_files("random_test")
    assert semantic_pack_set["source_names"] == random_pack_set["source_names"]
    assert sorted(semantic_pack_set["document_id_of_reference"]) == sorted(random_pack_set["document_id_of_reference"])
    # the same set in another arrangement, or the control would control nothing
    assert semantic_pack_set["document_id_of_reference"] != random_pack_set["document_id_of_reference"]
    assert len(semantic_pack_set["document_id_of_reference"]) > 400


def test_every_pack_reaches_the_pack_length_and_closes_with_the_document_that_fills_it(workspace):
    for pack_name in ("semantic_test", "random_test"):
        pack_set = read_pack_set_files(pack_name)
        documents_after_the_filling_one = []
        for references in pack_set["references_of_packs"]:
            # the length a reader sees: every document followed by one end-of-text token
            length_after_each_document = np.cumsum(pack_set["token_count_of_reference"][references] + 1)
            assert length_after_each_document[-1] >= test_pack_token_length, pack_name
            filling_document = int(np.searchsorted(length_after_each_document, test_pack_token_length))
            documents_after_the_filling_one.append(len(references) - 1 - filling_document)
        assert len(documents_after_the_filling_one) > 80, "too few packs to test anything"
        assert not any(documents_after_the_filling_one[:-1]), f"{pack_name}: a pack went on after it was full"
        if pack_name == "semantic_test":
            assert documents_after_the_filling_one[-1] == 0
        else:
            # the documents that could not fill one more pack stay behind the last full pack
            assert (documents_after_the_filling_one[-1]
                    == workspace["summaries"]["random"]["statistics"]["documents_after_the_last_full_pack"])


def test_no_document_appears_twice_and_only_documents_of_allowed_length_are_packed(workspace):
    document_ids_by_pack_name = {}
    for pack_name in ("semantic_test", "random_test", "natural_test"):
        pack_set = read_pack_set_files(pack_name)
        document_ids = pack_set["document_id_of_reference"]
        assert len(set(document_ids)) == len(document_ids), f"{pack_name}: a document appears twice"
        document_ids_by_pack_name[pack_name] = set(document_ids)
        if pack_name == "natural_test":
            assert pack_set["token_count_of_reference"].min() >= test_pack_token_length
        else:
            assert pack_set["token_count_of_reference"].min() >= 64
            assert pack_set["token_count_of_reference"].max() <= test_pack_token_length // 2
    assert not document_ids_by_pack_name["natural_test"] & document_ids_by_pack_name["semantic_test"]


def test_cutting_a_document_order_keeps_the_last_documents_behind_the_last_full_pack():
    # six tokens per document with its end-of-text token, packs of 12: two documents fill a pack
    pack_starts, documents_after_last_full_pack = document_packing.cut_document_order_into_packs(
        np.array([5, 5, 5, 5, 5]), 12)
    assert pack_starts.tolist() == [0, 2] and documents_after_last_full_pack == 1
    pack_starts, documents_after_last_full_pack = document_packing.cut_document_order_into_packs(
        np.array([5, 5, 5, 5]), 12)
    assert pack_starts.tolist() == [0, 2] and documents_after_last_full_pack == 0
    with pytest.raises(AssertionError, match="do not fill one pack"):
        document_packing.cut_document_order_into_packs(np.array([5]), 12)


# ---------------------------------------------------------------------------
# relatedness: against the topics the documents were written from
# ---------------------------------------------------------------------------

def measure_topic_agreement_with_first_document(pack_set: dict, topic_of_document_id: dict) -> float:
    """Mean over packs of the share of later documents that have the topic of the pack's first document."""
    agreement_per_pack = []
    for references in pack_set["references_of_packs"]:
        topics = [topic_of_document_id[pack_set["document_id_of_reference"][reference]] for reference in references]
        agreement_per_pack.append(np.mean([topic == topics[0] for topic in topics[1:]]))
    return float(np.mean(agreement_per_pack))


def test_semantic_packs_are_more_related_than_random_packs(workspace):
    semantic_statistics = workspace["summaries"]["semantic"]["statistics"]
    random_statistics = workspace["summaries"]["random"]["statistics"]
    for statistic_name in ("similarity_of_consecutive_documents_mean", "similarity_to_first_document_mean"):
        assert semantic_statistics[statistic_name] > random_statistics[statistic_name] + 0.3, statistic_name
    # the same comparison without the topic vectors: six topics, so a random pack agrees about one time in six
    semantic_agreement = measure_topic_agreement_with_first_document(
        read_pack_set_files("semantic_test"), workspace["topic_of_document_id"])
    random_agreement = measure_topic_agreement_with_first_document(
        read_pack_set_files("random_test"), workspace["topic_of_document_id"])
    assert semantic_agreement > 0.8, semantic_agreement
    assert random_agreement < 0.40, random_agreement


# ---------------------------------------------------------------------------
# the greedy rule, replayed on made-up vectors
# ---------------------------------------------------------------------------

def make_clustered_documents(seed: int, document_count: int = 400) -> tuple:
    """Unit vectors around 12 centres in 16 dimensions, with token counts, sources and an order of first documents."""
    random_generator = np.random.default_rng(seed)
    centres = random_generator.standard_normal((12, 16))
    centre_of_document = random_generator.integers(12, size=document_count)
    vectors = centres[centre_of_document] + 0.6 * random_generator.standard_normal((document_count, 16))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    token_counts = random_generator.integers(64, 301, document_count)
    sources = random_generator.integers(3, size=document_count).astype(np.int16)
    return vectors.astype(np.float32), token_counts, sources, random_generator.permutation(document_count)


def make_shingle_matrix(shingle_ids_of_documents: list[set]) -> scipy.sparse.csr_matrix:
    """The matrix the module gets from the pool: one row per document, a 1 in the column of every shingle it holds."""
    first_shingle_of_document = np.concatenate([[0], np.cumsum([len(ids) for ids in shingle_ids_of_documents])])
    column_ids = np.array([shingle_id for ids in shingle_ids_of_documents for shingle_id in sorted(ids)], dtype=np.int32)
    return scipy.sparse.csr_matrix((np.ones(len(column_ids), dtype=np.float32), column_ids, first_shingle_of_document),
                                   shape=(len(shingle_ids_of_documents), 2 ** 30))


def check_packs_are_complete_and_start_in_order(token_counts, pack_token_length, order_of_first_documents,
                                                packing) -> None:
    """Every pack starts with the next unused document of the order, closes with the document that fills it, and
    what is left over could not fill another pack. Holds when no pack was given up."""
    assert packing["packs_given_up"] == 0
    lengths = token_counts + 1
    is_used = np.zeros(len(lengths), dtype=bool)
    place_in_order = 0
    for pack in np.split(packing["documents_in_pack_order"], packing["pack_starts"][1:]):
        while is_used[order_of_first_documents[place_in_order]]:
            place_in_order += 1
        assert pack[0] == order_of_first_documents[place_in_order], (
            "a pack does not start with the next unused document")
        assert not is_used[pack].any() and len(set(pack.tolist())) == len(pack), "a document is packed twice"
        assert lengths[pack].sum() >= pack_token_length, "a pack was closed before it was full"
        assert lengths[pack[:-1]].sum() < pack_token_length, "a document was added to a pack that was already full"
        is_used[pack] = True
    assert np.array_equal(packing["leftover_documents"], np.flatnonzero(~is_used))
    assert lengths[~is_used].sum() < pack_token_length, "the leftover documents could have filled another pack"


def check_every_choice_is_the_best_fit(topic_vectors, sources, shingle_ids_of_documents, packing,
                                       source_repeat_penalty, near_copy_shingle_share) -> None:
    """Replay a plain search (shortlist as large as the document count) in float64 and check every added document.

    It must not be a near copy of the pack, and no unused document that is not a near copy may fit the pack
    better. The replay knows nothing of blocks, shortlists or matrix products. A tolerance of 1e-5 absorbs the
    float32 rounding of the module; a choice that is worse than the best by more than that fails.
    """
    vectors = topic_vectors.astype(np.float64)
    is_used = np.zeros(len(vectors), dtype=bool)
    for pack in np.split(packing["documents_in_pack_order"], packing["pack_starts"][1:]):
        members = [int(pack[0])]
        is_used[pack[0]] = True
        shingles_in_pack = set(shingle_ids_of_documents[pack[0]])
        for added_document in pack[1:]:
            centroid = vectors[members].sum(axis=0)
            centroid /= np.linalg.norm(centroid)
            share_of_pack_from_source = np.bincount(sources[members], minlength=int(sources.max()) + 1) / len(members)
            fit_to_pack = vectors @ centroid - source_repeat_penalty * share_of_pack_from_source[sources]
            is_near_copy = np.array([len(shingle_ids & shingles_in_pack) > near_copy_shingle_share * len(shingle_ids)
                                     for shingle_ids in shingle_ids_of_documents])
            assert not is_near_copy[added_document], "a near copy of the pack was added"
            assert fit_to_pack[added_document] >= fit_to_pack[~is_used & ~is_near_copy].max() - 1e-5, (
                "a document was added although another one fitted the pack better")
            members.append(int(added_document))
            is_used[added_document] = True
            shingles_in_pack |= shingle_ids_of_documents[added_document]


def test_blocked_packing_makes_the_choices_of_a_plain_search_over_all_documents():
    vectors, token_counts, sources, order_of_first_documents = make_clustered_documents(seed=5)
    no_shingles = [set() for _ in range(len(vectors))]
    # a shortlist larger than the document count is the plain search; blocks of 7 make many packs share a block
    packing = document_packing.pack_related_documents(
        vectors, token_counts, sources, make_shingle_matrix(no_shingles), 1_024, order_of_first_documents,
        shortlist_document_count=10_000, first_documents_per_block=7, source_repeat_penalty=0.1)
    assert len(packing["pack_starts"]) > 50 and packing["near_copy_skip_count"] == 0
    check_packs_are_complete_and_start_in_order(token_counts, 1_024, order_of_first_documents, packing)
    check_every_choice_is_the_best_fit(vectors, sources, no_shingles, packing, source_repeat_penalty=0.1,
                                       near_copy_shingle_share=0.2)
    # the block size is only a way to compute and must not change a single choice
    packing_one_at_a_time = document_packing.pack_related_documents(
        vectors, token_counts, sources, make_shingle_matrix(no_shingles), 1_024, order_of_first_documents,
        shortlist_document_count=10_000, first_documents_per_block=1, source_repeat_penalty=0.1)
    assert np.array_equal(packing["documents_in_pack_order"], packing_one_at_a_time["documents_in_pack_order"])
    assert np.array_equal(packing["pack_starts"], packing_one_at_a_time["pack_starts"])


def test_a_shortlist_holds_the_documents_nearest_the_first_and_is_refilled_when_it_runs_out():
    vectors, token_counts, sources, order_of_first_documents = make_clustered_documents(seed=6)
    float64_vectors = vectors.astype(np.float64)
    # 40 is more than a pack ever takes; 4 is fewer, so a pack empties its shortlist and takes the next four
    for shortlist_document_count in (40, 4):
        packing = document_packing.pack_related_documents(
            vectors, token_counts, sources, make_shingle_matrix([set() for _ in range(len(vectors))]), 1_024,
            order_of_first_documents, shortlist_document_count=shortlist_document_count,
            first_documents_per_block=16, source_repeat_penalty=0.0)
        check_packs_are_complete_and_start_in_order(token_counts, 1_024, order_of_first_documents, packing)
        is_used = np.zeros(len(vectors), dtype=bool)
        refill_count = 0
        for pack in np.split(packing["documents_in_pack_order"], packing["pack_starts"][1:]):
            is_used[pack[0]] = True
            is_not_yet_shortlisted = ~is_used
            similarity_to_first = float64_vectors @ float64_vectors[pack[0]]
            added_documents = pack[1:]
            for shortlist_start in range(0, len(added_documents), shortlist_document_count):
                taken_from_this_shortlist = added_documents[shortlist_start:shortlist_start + shortlist_document_count]
                candidates = np.flatnonzero(is_not_yet_shortlisted)
                if len(candidates) > shortlist_document_count:
                    lowest_similarity_on_shortlist = np.sort(similarity_to_first[candidates])[-shortlist_document_count]
                    assert similarity_to_first[taken_from_this_shortlist].min() >= lowest_similarity_on_shortlist - 1e-5, (
                        "a document was added that is not among the nearest to the first document")
                is_not_yet_shortlisted[taken_from_this_shortlist] = False
                refill_count += shortlist_start > 0
            is_used[pack] = True
        assert (refill_count > 0) == (shortlist_document_count == 4)


def test_source_penalty_brings_a_second_source_into_a_pack_without_leaving_the_topic():
    # Two sources write about the same four topics, and each source adds its own house style to every document:
    # a document is its topic axis plus 0.3 of its source's style axis. Two documents on one topic then have a
    # cosine similarity of 1.00 inside a source and 0.92 across sources; other topics are at 0.08 or below.
    random_generator = np.random.default_rng(7)
    topics = np.repeat(np.arange(4), 12)
    sources = np.tile(np.repeat(np.arange(2), 6), 4).astype(np.int16)
    vectors = np.zeros((48, 6))
    vectors[np.arange(48), topics] = 1.0
    vectors[np.arange(48), 4 + sources] = 0.3
    vectors += 0.005 * random_generator.standard_normal(vectors.shape)       # no two documents exactly alike
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
    token_counts = np.full(48, 200)                                           # six documents fill a pack of 1,024
    order_of_first_documents = random_generator.permutation(48)

    sources_per_pack_by_penalty = {}
    for penalty in (0.0, 0.1):
        packing = document_packing.pack_related_documents(
            vectors, token_counts, sources, make_shingle_matrix([set() for _ in range(48)]), 1_024,
            order_of_first_documents, source_repeat_penalty=penalty)
        packs = np.split(packing["documents_in_pack_order"], packing["pack_starts"][1:])
        assert len(packs) == 8 and len(packing["leftover_documents"]) == 0
        assert all(len(set(topics[pack])) == 1 for pack in packs), f"penalty {penalty}: a pack left its topic"
        sources_per_pack_by_penalty[penalty] = [len(set(sources[pack])) for pack in packs]
    # without the penalty the 0.08 head start of the own source decides every choice
    assert sources_per_pack_by_penalty[0.0] == [1] * 8
    assert sources_per_pack_by_penalty[0.1] == [2] * 8


# ---------------------------------------------------------------------------
# near copies: two editions of one text never share a pack
# ---------------------------------------------------------------------------

def test_a_near_copy_is_passed_over_for_its_pack_and_packed_in_another():
    vectors, token_counts, sources, order_of_first_documents = make_clustered_documents(seed=8)
    # documents 340 to 399 are copies of documents 0 to 59: the same topic vector, the same 30 shingles; every
    # other document has 30 shingles of its own
    original_of_copy = dict(zip(range(340, 400), range(60)))
    vectors[340:400], token_counts[340:400] = vectors[0:60], token_counts[0:60]
    shingle_ids_of_documents = [set(range(1_000 * original_of_copy.get(document, document),
                                          1_000 * original_of_copy.get(document, document) + 30))
                                for document in range(400)]

    copies_packed_with_their_original = {}
    for near_copy_shingle_share in (0.2, 1.0):
        packing = document_packing.pack_related_documents(
            vectors, token_counts, sources, make_shingle_matrix(shingle_ids_of_documents), 1_024,
            order_of_first_documents, shortlist_document_count=10_000, first_documents_per_block=7,
            source_repeat_penalty=0.1, near_copy_shingle_share=near_copy_shingle_share)
        check_packs_are_complete_and_start_in_order(token_counts, 1_024, order_of_first_documents, packing)
        check_every_choice_is_the_best_fit(vectors, sources, shingle_ids_of_documents, packing,
                                           source_repeat_penalty=0.1, near_copy_shingle_share=near_copy_shingle_share)
        pack_of_document = np.full(400, -1)
        pack_of_document[packing["documents_in_pack_order"]] = np.repeat(
            np.arange(len(packing["pack_starts"])),
            np.diff(np.append(packing["pack_starts"], len(packing["documents_in_pack_order"]))))
        pairs_with_both_packed = [(copy, original) for copy, original in original_of_copy.items()
                                  if pack_of_document[copy] >= 0 and pack_of_document[original] >= 0]
        assert len(pairs_with_both_packed) >= 50, "a passed-over copy must stay free for other packs"
        copies_packed_with_their_original[near_copy_shingle_share] = sum(
            pack_of_document[copy] == pack_of_document[original] for copy, original in pairs_with_both_packed)
        assert (packing["near_copy_skip_count"] > 0) == (near_copy_shingle_share == 0.2)
    # a share of 1.0 switches the rule off: then a copy is the best fit of the pack that holds its original
    assert copies_packed_with_their_original[1.0] >= 40
    assert copies_packed_with_their_original[0.2] == 0


def test_a_pack_that_only_near_copies_could_fill_is_given_up():
    # six copies of one text, 600 tokens each: a pack of 1,024 needs two of them, and no two may share a pack
    vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1))
    packing = document_packing.pack_related_documents(
        vectors, np.full(6, 600), np.zeros(6, dtype=np.int16), make_shingle_matrix([set(range(30))] * 6), 1_024,
        np.arange(6))
    assert len(packing["pack_starts"]) == 0 and len(packing["documents_in_pack_order"]) == 0
    assert packing["packs_given_up"] == 5 and packing["leftover_documents"].tolist() == [0, 1, 2, 3, 4, 5]
    assert packing["near_copy_skip_count"] == 5 + 4 + 3 + 2 + 1


def test_no_pack_holds_a_near_copy_of_its_earlier_documents_and_second_editions_go_elsewhere(workspace):
    texts = workspace["text_of_document_id"]
    first_edition_id, second_edition_id = workspace["edition_pairs"][0]
    shingles_of_second_edition = word_shingles_of_text(texts[second_edition_id])
    assert (len(shingles_of_second_edition & word_shingles_of_text(texts[first_edition_id]))
            > 0.7 * len(shingles_of_second_edition)), "the planted editions are not near copies"

    share_seen_earlier_by_pack_name = {}
    for pack_name in ("semantic_test", "random_test"):
        pack_set = read_pack_set_files(pack_name)
        pack_of_document_id, seen_shingle_count, shingle_count = {}, 0, 0
        for pack_index, references in enumerate(pack_set["references_of_packs"]):
            shingles_in_pack = set()
            for reference in references:
                document_id = pack_set["document_id_of_reference"][reference]
                pack_of_document_id[document_id] = pack_index
                shingles_of_document = word_shingles_of_text(texts[document_id])
                shingles_seen_earlier = len(shingles_of_document & shingles_in_pack)
                if pack_name == "semantic_test":
                    assert (shingles_seen_earlier
                            <= document_packing.near_copy_shingle_share * len(shingles_of_document)), (
                        f"pack {pack_index} holds a near copy of one of its earlier documents")
                seen_shingle_count += shingles_seen_earlier
                shingle_count += len(shingles_of_document)
                shingles_in_pack |= shingles_of_document
        share_seen_earlier_by_pack_name[pack_name] = seen_shingle_count / shingle_count
        if pack_name == "semantic_test":
            packs_of_edition_pairs = [(pack_of_document_id[first], pack_of_document_id[second])
                                      for first, second in workspace["edition_pairs"]
                                      if first in pack_of_document_id and second in pack_of_document_id]
            assert len(packs_of_edition_pairs) >= 10, "too few planted editions were packed to test anything"
            assert all(pack_of_first != pack_of_second for pack_of_first, pack_of_second in packs_of_edition_pairs)
    # the summaries carry the same measure, computed from hashed shingles, for both arms
    for mode_name, pack_name in (("semantic", "semantic_test"), ("random", "random_test")):
        statistics = workspace["summaries"][mode_name]["statistics"]
        assert abs(statistics["share_of_shingles_seen_earlier_in_pack"] - share_seen_earlier_by_pack_name[pack_name]) < 1e-4
    semantic_statistics = workspace["summaries"]["semantic"]["statistics"]
    assert semantic_statistics["near_copy_skips"] > 0
    assert semantic_statistics["documents_past_the_near_copy_share"] == 0


# ---------------------------------------------------------------------------
# the tail: weak packs and the packs built last are dropped, from both arms
# ---------------------------------------------------------------------------

def test_choosing_packs_drops_those_below_the_similarity_floor_and_then_the_last_built():
    consecutive_similarity_of_pack = np.array([0.9, 0.4, 0.8, 0.7, 0.6, 0.3, 0.9])
    document_tokens_of_pack = np.full(7, 100)
    # the two packs below 0.5 go; of the other five, the first three reach a budget of 250
    is_pack_kept = document_packing.choose_packs_to_keep(consecutive_similarity_of_pack, document_tokens_of_pack, 250)
    assert is_pack_kept.tolist() == [True, False, True, True, False, False, False]
    # a budget that the packs above the floor cannot reach keeps all of them
    is_pack_kept = document_packing.choose_packs_to_keep(consecutive_similarity_of_pack, document_tokens_of_pack, 900)
    assert is_pack_kept.tolist() == [True, False, True, True, True, False, True]
    # a budget met exactly keeps the pack that meets it and nothing after
    is_pack_kept = document_packing.choose_packs_to_keep(consecutive_similarity_of_pack, document_tokens_of_pack, 300)
    assert is_pack_kept.tolist() == [True, False, True, True, False, False, False]


def test_the_tail_of_the_build_is_dropped_and_both_arms_hold_only_the_kept_documents(workspace):
    statistics = workspace["summaries"]["semantic"]["statistics"]
    semantic_pack_set = read_pack_set_files("semantic_test")
    # the sample holds 10% more than the budget; every source ends within one document of its share
    oversampled_budget = round(test_token_budget * (1 + document_packing.oversampling_share))
    assert oversampled_budget <= statistics["sampled_document_tokens"] < oversampled_budget + 5 * test_pack_token_length // 2
    assert statistics["packs_dropped_below_minimum_pack_similarity"] >= 1, "the fake pool should leave weak packs"
    assert statistics["packs_dropped_from_the_end_of_the_build"] >= 1, "the fake pool should leave packs to spare"
    assert (statistics["packs_built"] - statistics["packs_dropped_below_minimum_pack_similarity"]
            - statistics["packs_dropped_from_the_end_of_the_build"] == statistics["packs"])
    assert (statistics["sampled_documents"] - statistics["leftover_documents_dropped"]
            - statistics["documents_in_dropped_packs"] == statistics["documents"])
    # back at the budget: the kept packs reach it, and without the last of them they would not
    tokens_of_packs = [int(semantic_pack_set["token_count_of_reference"][references].sum())
                       for references in semantic_pack_set["references_of_packs"]]
    assert sum(tokens_of_packs) == statistics["document_tokens"] >= test_token_budget
    assert sum(tokens_of_packs[:-1]) < test_token_budget

    # no kept pack is below the floor, measured again from the stored topic vectors
    topic_vectors = np.load(data_sources.data_root() / "packs" / "semantic_test" / "topic_vector_of_reference.npy")
    for references in semantic_pack_set["references_of_packs"]:
        similarity_of_consecutive_documents = np.sum(topic_vectors[references[1:]] * topic_vectors[references[:-1]], axis=1)
        assert similarity_of_consecutive_documents.mean() >= document_packing.minimum_pack_similarity
    assert statistics["share_of_packs_below_minimum_pack_similarity"] == 0.0
    assert statistics["similarity_of_consecutive_documents_percentiles"]["1"] >= document_packing.minimum_pack_similarity
    # the documents of the dropped packs are in neither arm
    random_document_ids = read_pack_set_files("random_test")["document_id_of_reference"]
    assert sorted(random_document_ids) == sorted(semantic_pack_set["document_id_of_reference"])
    assert len(random_document_ids) == statistics["documents"]
    assert workspace["summaries"]["random"]["statistics"]["share_of_packs_below_minimum_pack_similarity"] > 0.9


# ---------------------------------------------------------------------------
# the contract with the reader, the natural packs, the gate
# ---------------------------------------------------------------------------

def test_contract_files_have_the_types_and_layout_the_reader_expects(workspace):
    for mode_name in ("semantic", "random", "natural"):
        pack_set = read_pack_set_files(f"{mode_name}_test")
        reference_count = len(pack_set["source_of_reference"])
        source_names, source_of_reference = pack_set["source_names"], pack_set["source_of_reference"]
        assert isinstance(source_names, list) and all(isinstance(name, str) for name in source_names)
        assert source_of_reference.dtype == np.int16 and source_of_reference.ndim == 1
        assert pack_set["document_of_reference"].dtype == np.int64
        assert pack_set["document_of_reference"].shape == (reference_count,)
        assert pack_set["pack_start_references"].dtype == np.int64 and pack_set["pack_start_references"].ndim == 1
        assert pack_set["pack_start_references"][0] == 0
        assert np.all(np.diff(pack_set["pack_start_references"]) > 0)
        assert pack_set["pack_start_references"][-1] < reference_count
        assert 0 <= source_of_reference.min() and source_of_reference.max() < len(source_names)
        pack_directory = data_sources.data_root() / "packs" / f"{mode_name}_test"
        summary = json.loads((pack_directory / "pack_summary.json").read_text())
        assert summary["mode"] == mode_name and summary["seed"] == test_seed
        assert summary["settings"]["pack_token_length"] == test_pack_token_length
        assert summary["statistics"]["packs"] == len(pack_set["pack_start_references"])
        assert summary["statistics"]["documents"] == reference_count


def test_natural_packs_hold_one_long_document_each(workspace):
    natural_pack_set = read_pack_set_files("natural_test")
    assert natural_pack_set["source_names"] == ["papers", "textbooks"]
    assert natural_pack_set["pack_start_references"].tolist() == list(range(natural_pack_count))
    assert len(natural_pack_set["source_of_reference"]) == natural_pack_count
    assert natural_pack_set["token_count_of_reference"].min() >= test_pack_token_length
    # the budget buys six packs; the other long documents that the gate did not flag are counted as left out
    long_unflagged_document_count = 0
    for source_name in ("papers", "textbooks"):
        document_ids, token_counts = train_documents_of_source(source_name)
        long_unflagged_document_count += sum(
            token_count >= test_pack_token_length and document_id not in workspace["flagged_document_ids"]
            for document_id, token_count in zip(document_ids, token_counts))
    assert long_unflagged_document_count > natural_pack_count, "the fake pool has too few long documents for this test"
    natural_statistics = workspace["summaries"]["natural"]["statistics"]
    assert natural_statistics["candidate_documents_left_out"] == long_unflagged_document_count - natural_pack_count
    assert natural_statistics["documents_per_pack_mean"] == 1.0


def test_flagged_documents_are_in_no_pack_and_their_removal_is_counted(workspace):
    for pack_name in ("semantic_test", "random_test", "natural_test"):
        packed_document_ids = set(read_pack_set_files(pack_name)["document_id_of_reference"])
        assert not packed_document_ids & workspace["flagged_document_ids"], pack_name
    assert (workspace["summaries"]["semantic"]["statistics"]["flagged_candidates_removed_by_source"]
            == workspace["flagged_candidates_by_source"])
    assert (workspace["summaries"]["natural"]["statistics"]["flagged_candidates_removed_by_source"]
            == {"papers": 1, "textbooks": 0})


def test_a_missing_gate_file_stops_every_build(workspace):
    gate_path = document_packing.flagged_document_ids_path()
    hidden_gate_path = gate_path.with_suffix(".hidden")
    gate_path.rename(hidden_gate_path)
    try:
        with pytest.raises(AssertionError, match="decontaminate"):
            document_packing.build_semantic_packs("semantic_without_gate", test_token_budget, test_pack_token_length,
                                                  test_seed, worker_count=1)
        with pytest.raises(AssertionError, match="decontaminate"):
            document_packing.build_random_packs("random_without_gate", "semantic_test", test_pack_token_length,
                                                test_seed)
        with pytest.raises(AssertionError, match="decontaminate"):
            document_packing.build_natural_packs("natural_without_gate", natural_pack_count * test_pack_token_length,
                                                 test_pack_token_length, test_seed)
    finally:
        hidden_gate_path.rename(gate_path)
    assert not (data_sources.data_root() / "packs" / "semantic_without_gate").exists()


def test_a_document_flagged_after_the_semantic_build_stops_the_random_build(workspace):
    # the gate is run again and now flags a document that the existing semantic pack set holds
    newly_flagged_document_id = read_pack_set_files("semantic_test")["document_id_of_reference"][3]
    write_gate_file(workspace["flagged_document_ids"] | {newly_flagged_document_id})
    try:
        with pytest.raises(AssertionError, match="flagged by the decontamination gate"):
            document_packing.build_random_packs("random_after_new_flag", "semantic_test", test_pack_token_length,
                                                test_seed)
    finally:
        write_gate_file(workspace["flagged_document_ids"])
    assert not (data_sources.data_root() / "packs" / "random_after_new_flag").exists()


# ---------------------------------------------------------------------------
# sampling, seeds, the command line
# ---------------------------------------------------------------------------

def test_sampling_gives_every_source_its_share_and_refuses_a_budget_the_pool_cannot_fill(workspace):
    sample = document_packing.sample_documents_for_token_budget(
        test_token_budget, test_pack_token_length, test_seed, workspace["flagged_document_ids"])
    for source_index, source_name in enumerate(sample["source_names"]):
        sampled_token_count = int(sample["token_count_of_document"][sample["source_of_document"] == source_index].sum())
        share_of_budget = document_packing.source_token_shares[source_name] * test_token_budget
        # the last document takes the source across its share, so the excess is less than one document
        assert share_of_budget <= sampled_token_count < share_of_budget + test_pack_token_length // 2, source_name
    assert not set(sample["document_ids"]) & workspace["flagged_document_ids"]
    with pytest.raises(AssertionError, match="lower --token-budget"):
        document_packing.sample_documents_for_token_budget(
            10 * test_token_budget, test_pack_token_length, test_seed, workspace["flagged_document_ids"])


def test_same_seed_gives_byte_identical_files_and_another_seed_gives_other_packs(workspace):
    for seed, name_suffix in ((test_seed, "again"), (2, "other_seed")):
        document_packing.build_semantic_packs(f"semantic_{name_suffix}", test_token_budget, test_pack_token_length,
                                              seed, worker_count=1)
        document_packing.build_random_packs(f"random_{name_suffix}", f"semantic_{name_suffix}",
                                            test_pack_token_length, seed)
        document_packing.build_natural_packs(f"natural_{name_suffix}", natural_pack_count * test_pack_token_length,
                                             test_pack_token_length, seed)
    for mode_name in ("semantic", "random", "natural"):
        first_build = read_contract_file_bytes(f"{mode_name}_test")
        assert read_contract_file_bytes(f"{mode_name}_again") == first_build, (
            f"{mode_name}: the same seed gave other files")
        other_seed_build = read_contract_file_bytes(f"{mode_name}_other_seed")
        assert other_seed_build["document_of_reference.npy"] != first_build["document_of_reference.npy"], mode_name
    # the topic vectors are part of what a seed fixes: the random start of the SVD comes from it
    packs_directory = data_sources.data_root() / "packs"
    assert ((packs_directory / "semantic_again" / "topic_vector_of_reference.npy").read_bytes()
            == (packs_directory / "semantic_test" / "topic_vector_of_reference.npy").read_bytes())
    # a random arm with its own seed still holds the documents of its semantic pack set
    document_packing.build_random_packs("random_seed_3", "semantic_test", test_pack_token_length, 3)
    assert (sorted(read_pack_set_files("random_seed_3")["document_id_of_reference"])
            == sorted(read_pack_set_files("semantic_test")["document_id_of_reference"]))
    assert read_contract_file_bytes("random_seed_3") != read_contract_file_bytes("random_test")


def test_random_build_refuses_another_pack_length_or_a_pack_set_that_is_not_semantic(workspace):
    with pytest.raises(AssertionError, match="same length rule"):
        document_packing.build_random_packs("random_other_length", "semantic_test", 2 * test_pack_token_length,
                                            test_seed)
    with pytest.raises(AssertionError, match="need a semantic one"):
        document_packing.build_random_packs("random_of_natural", "natural_test", test_pack_token_length, test_seed)
    with pytest.raises(AssertionError, match="no pack set"):
        document_packing.build_random_packs("random_of_nothing", "never_built", test_pack_token_length, test_seed)


def test_command_line_builds_the_same_packs_in_worker_processes_shows_one_and_merges_the_results(
        workspace, monkeypatch, tmp_path, capsys):
    results_path = tmp_path / "results" / "pack_summary.json"
    monkeypatch.setattr(document_packing, "results_summary_path", results_path)       # never the repository's file
    length_arguments = ["--pack-token-length", str(test_pack_token_length), "--seed", str(test_seed)]
    document_packing.main(["build", "--mode", "semantic", "--name", "semantic_cli", "--token-budget",
                           str(test_token_budget), "--worker-count", "2"] + length_arguments)
    document_packing.main(["build", "--mode", "random", "--name", "random_cli", "--same-documents-as", "semantic_cli",
                           "--worker-count", "2"] + length_arguments)
    # no --name: the default is the mode and the pack length, natural_1k here and natural_16k for 16,384
    document_packing.main(["build", "--mode", "natural", "--token-budget",
                           str(natural_pack_count * test_pack_token_length)] + length_arguments)

    # two worker processes read the pool files; the packs must be those of the build in one process
    assert read_contract_file_bytes("semantic_cli") == read_contract_file_bytes("semantic_test")
    assert read_contract_file_bytes("random_cli") == read_contract_file_bytes("random_test")
    assert read_contract_file_bytes("natural_1k") == read_contract_file_bytes("natural_test")
    results = json.loads(results_path.read_text())
    assert list(results) == ["semantic_cli", "random_cli", "natural_1k"]
    assert results["random_cli"]["settings"]["same_documents_as"] == "semantic_cli"
    assert results["semantic_cli"]["statistics"] == workspace["summaries"]["semantic"]["statistics"]
    assert results["random_cli"]["statistics"] == workspace["summaries"]["random"]["statistics"]
    # a rebuild replaces its own entry and keeps the others
    document_packing.main(["build", "--mode", "random", "--name", "random_cli", "--same-documents-as", "semantic_cli",
                           "--worker-count", "1"] + length_arguments)
    assert list(json.loads(results_path.read_text())) == ["semantic_cli", "random_cli", "natural_1k"]
    with pytest.raises(AssertionError, match="--token-budget has no effect"):
        document_packing.main(["build", "--mode", "random", "--name", "random_cli", "--token-budget", "1000"]
                              + length_arguments)

    capsys.readouterr()
    document_packing.main(["show", "--name", "semantic_cli", "--pack-index", "0"])
    shown_text = capsys.readouterr().out
    first_pack = read_pack_set_files("semantic_cli")
    for reference in first_pack["references_of_packs"][0]:
        document_id = first_pack["document_id_of_reference"][reference]
        assert document_id in shown_text
        assert workspace["text_of_document_id"][document_id][:120] in shown_text, "the start of a document is not shown"
    # a natural pack has no topic vectors and one document; it must show all the same
    document_packing.main(["show", "--name", "natural_1k", "--pack-index", str(natural_pack_count - 1)])
    shown_text = capsys.readouterr().out
    last_natural_document_id = read_pack_set_files("natural_1k")["document_id_of_reference"][-1]
    assert f"natural_1k, pack {natural_pack_count - 1} of {natural_pack_count}: 1 documents" in shown_text
    assert workspace["text_of_document_id"][last_natural_document_id][:120] in shown_text
    with pytest.raises(AssertionError, match="has packs 0 to"):
        document_packing.main(["show", "--name", "natural_1k", "--pack-index", str(natural_pack_count)])
