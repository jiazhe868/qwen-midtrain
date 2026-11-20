"""Group pool documents into packs that each fill one long training sequence: related, shuffled, or one long document.

Why this file exists. Part of a run's tokens is spent on long sequences of 16,384 tokens, four times the ordinary
4,096 (the long-context share of a recipe, mixture.py). Most documents are far shorter than that: the median is 651
tokens in wiki_en, 1,251 in textbooks, 469 in forum_qa, 888 in math_web and 488 in code, so a long sequence has to
hold many documents. The usual way to fill it is to concatenate documents at random. Such a sequence is long, but
nothing in its first 12,000 tokens helps to predict its last 4,000, so the model gets no practice at using distant
context. Semantic document packing puts related documents side by side: an encyclopedia article, a textbook section
and a forum question on the same topic. The experiment compares the two ways, so the two arms must hold exactly the
same documents and differ only in how the documents are grouped and ordered.

The three modes.
    semantic   packs of related documents from wiki_en, textbooks, forum_qa, math_web and code
    random     the documents of a semantic pack set, shuffled with a seed and cut into packs by the same length
               rule. The few documents after its last full pack stay at the end of that pack, where the reader
               does not reach them: dropping them would leave this arm with fewer documents than the other.
    natural    one document per pack that is itself at least one sequence long (papers and long textbooks); these
               documents are in neither of the other two modes
No mode packs a document that the decontamination gate flagged (data_hygiene.py); a build stops when the gate's
file is missing.

How the semantic packs are built.
  1. Sample. Every source gets a share (`source_token_shares`) of the token budget plus 10%. Its train documents
     of 64 to 8,192 tokens are drawn in a seeded random order until the share is reached.
  2. Describe. Each text is read from the pool by doc_id. Its first 20,000 characters become TF-IDF weights over
     hashed words, reduced to 128 dimensions with a truncated SVD and scaled to unit length: its topic vector.
     Its word 8-grams are hashed as well, for the near-copy test.
  3. Pack. The next unused document in a seeded random order opens a pack. The unused document that fits the pack
     best is added again and again: the one most similar to the centroid of the pack, with a small penalty for a
     source the pack already holds. A candidate that is a near copy of what the pack holds is passed over for
     this pack and stays free for others. The pack is closed as soon as it holds 16,384 tokens, counting one
     end-of-text token per document. Documents stay in the order they were added. Documents that are left when
     no further pack can be filled are dropped, and their count is printed.
  4. Drop the tail. Packs whose consecutive documents have a mean similarity below 0.5 are dropped, and then
     packs from the end of the build order until the token budget is met again. The 10% of step 1 pay for this.
The reader takes the first 16,384 tokens of a pack and drops the rest, so every pack is at least that long.

History (measured on the real pool, 2026-11-20, seed 1; results/pack_summary.json).
  * Result. 6,782 semantic packs of 16 documents on average, 120.0M document tokens. The cosine similarity of
    the topic vectors of consecutive documents is 0.81 in the semantic packs (1st percentile of the packs 0.58)
    and 0.15 in the random packs of the same documents (95th percentile 0.22).
  * Sources. With the plain most-similar rule 61% of the packs held a single source, because documents of one
    source resemble each other before they share any topic. Subtracting from every document the mean vector of
    its source made it worse (1.45 sources per pack against 1.55): the mean of math_web is mostly "mathematics",
    which is topic and not style, and removing it cut the link to the mathematics textbooks. A penalty on
    repeating a source worked; the numbers are at `source_repeat_penalty`.
  * Near copies. The same textbook section exists in several editions, and packing by similarity put the copies
    side by side: 1.9% of the word 8-grams of the semantic packs (4.9% of those in textbooks) already occurred
    earlier in the same pack, and 9% of the packs held a document that was more than half a repeat. The random
    packs had 0.00%. A model can lower its loss on such a pack by copying, which is another skill than using
    related text, so the two arms would have differed in more than relatedness. The near-copy test of step 3
    was added for this; the numbers are at `near_copy_shingle_share`.
  * The tail. Greedy packing serves the first packs best. Without step 4 the similarity of consecutive documents
    fell from 0.87 in the first tenth of the packs to 0.65 in the last tenth; 85 of 6,799 packs were below 0.5,
    72 of them among the last 300 built, and those read as collections of leftovers. With step 4, 7,444 packs
    are built, 111 are dropped for a similarity below 0.5 and 551 from the end; the last tenth is at 0.73.
    The dropped packs are not a fair sample of the sources: they hold 23% of the sampled math_web documents and
    6% of the forum_qa documents, so the kept tokens are 13.6% math_web and 16.0% forum_qa where the sample has
    15% of each.
  * Cost. The semantic build takes 100 to 120 s on 64 cores: 30 to 40 s to read the texts, count words and hash
    shingles in 29 worker processes, 33 to 50 s for the SVD, 31 s to pack on one thread, 6 s to check and
    measure. The random build takes 36 s, nearly all of it to read the texts again for the shingles. The
    natural build takes a second.

What is written to $MIDTRAIN_DATA/packs/<pack_name>/ (the first four files are the contract with
`open_pack_stream` in mixture.py; a reference is one document at one place in one pack):
    source_names.json               a JSON list; a source's index in it is its id
    source_of_reference.npy         int16, the source id of every reference
    document_of_reference.npy       int64, the document's position in that source's train token file
    pack_start_references.npy       int64, the index of the first reference of every pack, ascending from 0
    topic_vector_of_reference.npy   float32, the topic vector of every reference (semantic and random only)
    pack_summary.json               settings, seed, statistics, seconds per stage
The summary of every built pack set is also merged into results/pack_summary.json, one key per pack name.

Commands (from the repository root, after `source scripts/env.sh`; CPU only):
    $MIDTRAIN_PYTHON -m midtrain.document_packing build --mode semantic --name semantic_16k
    $MIDTRAIN_PYTHON -m midtrain.document_packing build --mode random --name random_16k --same-documents-as semantic_16k
    $MIDTRAIN_PYTHON -m midtrain.document_packing build --mode natural --name natural_16k
    $MIDTRAIN_PYTHON -m midtrain.document_packing show --name semantic_16k --pack-index 0
    $MIDTRAIN_PYTHON -m pytest tests/test_document_packing.py -q
`build` also takes --token-budget (default 120,000,000), --pack-token-length (default 16,384) and --seed (default 1).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from threadpoolctl import threadpool_limits

from midtrain.data_sources import data_root, open_token_source, repository_root

pack_mode_names = ("semantic", "random", "natural")
default_pack_token_length = 16_384
default_token_budget = 120_000_000
default_seed = 1
results_summary_path = repository_root / "results" / "pack_summary.json"

# Token shares of the sources in the semantic and random packs. Encyclopedia and textbook text lead because they
# are the long-form sources; question threads, math pages and code share the other 45% equally.
source_token_shares = {"wiki_en": 0.30, "textbooks": 0.25, "forum_qa": 0.15, "math_web": 0.15, "code": 0.15}
# The sources whose long documents fill the natural packs. Of the train documents that the gate did not flag,
# 19,517 papers and 864 textbook documents hold at least 16,384 tokens.
natural_source_names = ("papers", "textbooks")
# A pack holds documents of 64 tokens up to half a pack (8,192 for a 16,384 pack). Shorter ones are stubs. A longer
# one would dominate its pack: with two or more documents in every pack no document can fill more than half of the
# sequence the reader takes.
minimum_document_tokens = 64
# The topic of a document shows in its first pages. 20,000 characters are about 5,000 tokens: 2.3% of the sampled
# documents are longer, and the cut leaves out 3.3% of all characters.
maximum_characters_per_text = 20_000
# Words are hashed into 2**18 = 262,144 columns instead of collecting a vocabulary first: hashing needs no pass
# over all texts, so the files can be counted independently in worker processes.
hashed_word_column_count = 2 ** 18
# A word is two or more word characters and starts with a letter. Bare numbers are left out: two pages that share
# "12" and "2019" are not related by that.
word_pattern = r"(?u)\b[^\W\d_]\w+\b"
topic_dimension_count = 128
# Candidates of one pack: the unused documents most similar to its first document. A pack holds 16 documents on
# average, so 2,048 candidates leave the choice wide open. Measured with shortlists of 512, 2,048, 8,192 and 32,768
# documents: the similarity of consecutive documents is 0.798, 0.799, 0.799 and 0.799, and packing takes 19, 16,
# 31 and 118 s. So 2,048 loses nothing against a search that is 16 times wider. When every document of a
# shortlist is in the pack or was passed over as a near copy, the next 2,048 are taken.
shortlist_document_count = 2_048
# The similarities of this many first documents to all documents are computed in one matrix product. 256 rows of
# 125,000 float32 values are 128 MB.
first_documents_per_block = 256
# How much a candidate loses for coming from a source the pack already holds: this number times the share of the
# pack's documents from that source. Documents of one source share its house style (the layout of an encyclopedia
# article, the chrome of a math page): two random documents of one source have a similarity of 0.25, two of
# different sources 0.12. Without a penalty the nearest documents are therefore nearly always from the same source,
# and a pack rarely gets the article, the textbook section and the question on one topic. Measured, before the
# near-copy test and the tail drop existed:
#     penalty                                0      0.1    0.2    0.3
#     sources per pack                       1.55   1.96   2.25   2.48
#     packs with a single source             61%    36%    24%    17%
#     similarity of consecutive documents    0.808  0.799  0.778  0.753
# 0.1 is the step that costs almost nothing. 0 gives the plain rule: always the most similar document.
source_repeat_penalty = 0.1
# The near-copy test. A shingle is a run of 8 consecutive words of the whole text (lower case, every word
# counts). Its hash is one of 2**30 column numbers, so two different shingles of a pack almost never meet.
shingle_word_count = 8
hashed_shingle_column_count = 2 ** 30
# A candidate is a near copy, and is passed over for this pack, when more than this share of its shingles already
# occurs in the documents of the pack. Measured on the real pool, after step 4:
#     share                                              off      0.2      0.1      0.05
#     word 8-grams that occur earlier in their pack      2.07%    0.45%    0.27%    0.16%
#     similarity of consecutive documents                0.818    0.813    0.811    0.809
#     times a candidate was passed over                  0        58,887   104,078  151,969
# The random packs have 0.003%. Before the rule, the documents that repeated 10% to 20% of their text carried as
# much repeated text (0.18% of all tokens) as those between 20% and 30%; those between 5% and 10% carried 0.10%.
# What is left at 0.1 is the stock phrases of documents of one kind: "was listed on the National Register of
# Historic Places", "by the end of this section you will be able to", license badges. No share above zero
# removes those, and a share of zero would forbid a pack of documents of one kind.
near_copy_shingle_share = 0.1
# Step 4. The sample holds this much more than the token budget, so that the weakest packs can be dropped.
oversampling_share = 0.10
# A pack whose consecutive documents have a mean cosine similarity below this is dropped: 111 of 7,444 built
# packs. The packs below 0.5 read as collections of leftovers; the random packs have a median of 0.15.
minimum_pack_similarity = 0.5


def pack_directory_for(pack_name: str) -> Path:
    return data_root() / "packs" / pack_name


def random_generator_for(seed: int, purpose: str) -> np.random.Generator:
    """One generator per random choice, named by its purpose, all derived from the one `--seed`.

    A separate stream for each choice means that changing one of them (say, a source's share) leaves the draws of
    the others as they were.
    """
    return np.random.default_rng([int(seed), zlib.crc32(purpose.encode("utf-8"))])


# ---------------------------------------------------------------------------------------------------------------
# the decontamination gate: documents that no pack may hold
# ---------------------------------------------------------------------------------------------------------------
def flagged_document_ids_path() -> Path:
    return data_root() / "hygiene" / "decontamination_flagged.parquet"


def load_flagged_document_ids() -> set[str]:
    """The doc_ids flagged by the decontamination gate (data_hygiene.py): text that overlaps an evaluation item.

    Every build removes them from its candidates and checks again before it writes. A missing file stops the
    build: packs built before the gate ran would look exactly like clean ones afterwards.
    """
    flagged_path = flagged_document_ids_path()
    assert flagged_path.exists(), (
        f"missing {flagged_path}: run `data_hygiene decontaminate` first; no pack is built from a pool that the "
        f"decontamination gate has not checked")
    return set(pq.read_table(flagged_path, columns=["doc_id"]).column("doc_id").to_pylist())


def find_flagged_documents(document_ids: list[str], flagged_document_ids: set[str]) -> np.ndarray:
    """One boolean per document of a token file: is its doc_id flagged."""
    return np.fromiter((document_id in flagged_document_ids for document_id in document_ids), dtype=bool,
                       count=len(document_ids))


def document_ids_of_references(source_names: list[str], source_of_reference: np.ndarray,
                               document_of_reference: np.ndarray) -> list[str]:
    """The doc_id of every reference, looked up by its position in the train token file of its source."""
    document_id_of_reference = np.empty(len(source_of_reference), dtype=object)
    for source_index, source_name in enumerate(source_names):
        is_from_this_source = source_of_reference == source_index
        if is_from_this_source.any():
            document_ids_of_source = open_token_source(source_name, "train")["document_ids"]
            document_id_of_reference[is_from_this_source] = [
                document_ids_of_source[position] for position in document_of_reference[is_from_this_source]]
    return document_id_of_reference.tolist()


def assert_no_packed_document_is_flagged(source_names: list[str], source_of_reference: np.ndarray,
                                         document_of_reference: np.ndarray, flagged_document_ids: set[str]) -> None:
    """Look up the doc_id of every reference in the token files and stop if the gate flagged one of them.

    This goes from the written positions back to doc_ids, so it also catches a pack set that was built before
    the gate file changed (the random packs take their documents from an existing semantic pack set).
    """
    packed_document_ids = set(document_ids_of_references(source_names, source_of_reference, document_of_reference))
    flagged_and_packed = packed_document_ids & flagged_document_ids
    assert not flagged_and_packed, (
        f"{len(flagged_and_packed)} packed documents are flagged by the decontamination gate, for example "
        f"{sorted(flagged_and_packed)[:3]}: rebuild the pack set from the current gate file")


# ---------------------------------------------------------------------------------------------------------------
# sample: which documents go into the semantic packs
# ---------------------------------------------------------------------------------------------------------------
def sample_documents_for_token_budget(token_budget: int, pack_token_length: int, seed: int,
                                      flagged_document_ids: set[str]) -> dict:
    """Draw documents from every source, in a seeded random order, until the source holds its share of the budget.

    The candidates of a source are its train documents of allowed length that the decontamination gate did not
    flag. Returns arrays with one entry per sampled document, source after source: `source_of_document` (index
    into `source_names`), `position_of_document` (in the source's train token file), `token_count_of_document`,
    the list `document_ids`, and `flagged_candidates_removed_by_source`. A source ends with the document that
    takes it across its share, so every source is over its share by less than one document. The budget counts
    document tokens, without end-of-text tokens.
    """
    maximum_document_tokens = pack_token_length // 2
    source_names = list(source_token_shares)
    assert abs(sum(source_token_shares.values()) - 1.0) < 1e-9, "the source shares must sum to 1"
    source_of_document, position_of_document, token_count_of_document, document_ids = [], [], [], []
    flagged_candidates_removed_by_source = {}
    for source_index, source_name in enumerate(source_names):
        token_source = open_token_source(source_name, "train")
        token_count_of_every_document = np.diff(token_source["document_offsets"])
        has_allowed_length = ((token_count_of_every_document >= minimum_document_tokens)
                              & (token_count_of_every_document <= maximum_document_tokens))
        is_flagged = find_flagged_documents(token_source["document_ids"], flagged_document_ids)
        candidate_positions = np.flatnonzero(has_allowed_length & ~is_flagged)
        flagged_candidates_removed_by_source[source_name] = int((has_allowed_length & is_flagged).sum())
        wanted_token_count = source_token_shares[source_name] * token_budget
        shuffled_positions = random_generator_for(seed, f"sample of {source_name}").permutation(candidate_positions)
        cumulative_token_count = np.cumsum(token_count_of_every_document[shuffled_positions])
        assert len(cumulative_token_count) > 0 and cumulative_token_count[-1] >= wanted_token_count, (
            f"{source_name} holds {int(token_count_of_every_document[candidate_positions].sum()):,} tokens in "
            f"documents of {minimum_document_tokens} to {maximum_document_tokens} tokens, fewer than its share of "
            f"{wanted_token_count:,.0f}: lower --token-budget")
        # the shortest start of the shuffled order whose tokens reach the share
        sampled_document_count = int(np.searchsorted(cumulative_token_count, wanted_token_count)) + 1
        sampled_positions = shuffled_positions[:sampled_document_count]
        source_of_document.append(np.full(sampled_document_count, source_index, dtype=np.int16))
        position_of_document.append(sampled_positions.astype(np.int64))
        token_count_of_document.append(token_count_of_every_document[sampled_positions].astype(np.int64))
        document_ids.extend(token_source["document_ids"][position] for position in sampled_positions)
        print(f"  {source_name:10s} {sampled_document_count:>7,} of {len(candidate_positions):>7,} candidates, "
              f"{int(cumulative_token_count[sampled_document_count - 1]):>11,} tokens; "
              f"{flagged_candidates_removed_by_source[source_name]:,} flagged candidates removed", flush=True)
    return {"source_names": source_names, "source_of_document": np.concatenate(source_of_document),
            "position_of_document": np.concatenate(position_of_document),
            "token_count_of_document": np.concatenate(token_count_of_document), "document_ids": document_ids,
            "flagged_candidates_removed_by_source": flagged_candidates_removed_by_source}


# ---------------------------------------------------------------------------------------------------------------
# describe: read the texts from the pool; word counts for the topic vector, shingles for the near-copy test
# ---------------------------------------------------------------------------------------------------------------
def pool_file_paths_of_source(source_name: str) -> list[Path]:
    pool_file_paths = sorted((data_root() / "pool" / source_name).glob("part-*.parquet"))
    assert pool_file_paths, f"no pool rows for {source_name}: run `data_sources build` first"
    return pool_file_paths


def read_texts_from_pool_file(pool_file_path: str, wanted_document_ids: list[str],
                              maximum_characters: int | None) -> tuple[list[str], list[str]]:
    """The wanted documents that one pool file holds: their doc_ids and their texts.

    A text is cut at `maximum_characters` unless that is None. The token files know a document only by its
    position and doc_id, and the pool also holds the validation documents, so the text is found by doc_id. The
    doc_id column is read first; the much larger text column is read only when the file holds a wanted document.
    """
    document_ids_in_file = pq.read_table(pool_file_path, columns=["doc_id"]).column("doc_id").combine_chunks()
    is_wanted_row = pc.is_in(document_ids_in_file, value_set=pa.array(wanted_document_ids, type=pa.string()))
    if not pc.any(is_wanted_row).as_py():
        return [], []
    wanted_texts = pq.read_table(pool_file_path, columns=["text"]).column("text").filter(is_wanted_row)
    if maximum_characters is not None:
        wanted_texts = pc.utf8_slice_codeunits(wanted_texts, start=0, stop=maximum_characters)
    return document_ids_in_file.filter(is_wanted_row).to_pylist(), wanted_texts.to_pylist()


def describe_documents_in_pool_file(task: tuple[str, list[str]]
                                    ) -> tuple[list[str], scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """Runs in a worker process: the wanted documents of one pool file, their hashed word counts and their shingles.

    Both matrices have one row per found document. The word counts are taken from the first
    `maximum_characters_per_text` characters. The shingle matrix marks which hashed word 8-grams the whole
    text holds.
    """
    pool_file_path, wanted_document_ids = task
    found_document_ids, found_texts = read_texts_from_pool_file(pool_file_path, wanted_document_ids, None)
    if not found_texts:
        return ([], scipy.sparse.csr_matrix((0, hashed_word_column_count), dtype=np.float32),
                scipy.sparse.csr_matrix((0, hashed_shingle_column_count), dtype=np.float32))
    # common English words ("the", "of") say nothing about a topic; dropping them before counting, and not
    # leaving them to the weights, keeps the count matrix small
    word_counter = HashingVectorizer(n_features=hashed_word_column_count, alternate_sign=False, norm=None,
                                     stop_words="english", token_pattern=word_pattern, dtype=np.float32)
    shingle_hasher = HashingVectorizer(n_features=hashed_shingle_column_count, alternate_sign=False, norm=None,
                                       binary=True, ngram_range=(shingle_word_count, shingle_word_count),
                                       token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    word_counts = word_counter.transform([text[:maximum_characters_per_text] for text in found_texts])
    return found_document_ids, word_counts, shingle_hasher.transform(found_texts)


def describe_documents(source_names: list[str], source_of_document: np.ndarray, document_ids: list[str],
                       worker_count: int) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """Hashed word counts and shingles of every document, row i of both matrices for document i.

    One task per pool file. Splitting a text into words is Python code that holds the interpreter lock, so the
    files are read in worker processes; hashing needs no shared vocabulary, so the workers need not talk.
    """
    tasks = []
    for source_index, source_name in enumerate(source_names):
        documents_of_source = np.flatnonzero(source_of_document == source_index)
        wanted_document_ids = [document_ids[document] for document in documents_of_source]
        tasks.extend((str(pool_file_path), wanted_document_ids)
                     for pool_file_path in pool_file_paths_of_source(source_name))
    if worker_count <= 1:
        results = [describe_documents_in_pool_file(task) for task in tasks]
    else:
        # spawned and not forked: this process already runs the threads of the linear algebra library, and a
        # forked copy of a process with threads can wait forever for a lock that one of them held
        with multiprocessing.get_context("spawn").Pool(min(worker_count, len(tasks))) as worker_pool:
            results = worker_pool.map(describe_documents_in_pool_file, tasks)

    document_of_document_id = {document_id: document for document, document_id in enumerate(document_ids)}
    assert len(document_of_document_id) == len(document_ids), "a doc_id is wanted twice"
    document_of_found_row = np.array([document_of_document_id[document_id]
                                      for found_document_ids, _, _ in results for document_id in found_document_ids],
                                     dtype=np.int64)
    found_document_count = len(np.unique(document_of_found_row))
    assert len(document_of_found_row) == len(document_ids) and found_document_count == len(document_ids), (
        f"the pool holds {found_document_count:,} of the {len(document_ids):,} wanted documents "
        f"({len(document_of_found_row):,} rows): the pool and the token files are from different builds")
    found_row_of_document = np.argsort(document_of_found_row)
    word_counts = scipy.sparse.vstack([word_counts for _, word_counts, _ in results], format="csr")
    shingle_ids = scipy.sparse.vstack([shingle_ids for _, _, shingle_ids in results], format="csr")
    return word_counts[found_row_of_document], shingle_ids[found_row_of_document]


def scale_rows_to_unit_length(vectors: np.ndarray) -> np.ndarray:
    """Each row divided by its length, so that a dot product of two rows is their cosine similarity."""
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def compute_topic_vectors(word_counts: scipy.sparse.csr_matrix, seed: int) -> np.ndarray:
    """One unit vector per document; the dot product of two of them is how related the two documents are.

    TF-IDF gives a word weight by how rare it is across the documents (and the logarithm of its count inside the
    document, so that a word repeated 100 times does not count 100 times). The truncated SVD then keeps the
    directions in which words vary together, which makes two texts on one topic similar even when they share few
    exact words. `seed` fixes the random start of the SVD.
    """
    weighted_word_counts = TfidfTransformer(sublinear_tf=True).fit_transform(word_counts)
    # the SVD needs fewer dimensions than documents; only the tiny pools of the tests are that small
    dimension_count = min(topic_dimension_count, word_counts.shape[0] - 1)
    reduced_vectors = TruncatedSVD(n_components=dimension_count, random_state=seed).fit_transform(weighted_word_counts)
    return scale_rows_to_unit_length(reduced_vectors.astype(np.float32))


# ---------------------------------------------------------------------------------------------------------------
# pack: related documents (greedy), or a given order cut by the same length rule
# ---------------------------------------------------------------------------------------------------------------
def pack_related_documents(topic_vector_of_document: np.ndarray, token_count_of_document: np.ndarray,
                           source_of_document: np.ndarray, shingle_ids_of_document: scipy.sparse.csr_matrix,
                           pack_token_length: int, order_of_first_documents: np.ndarray,
                           shortlist_document_count: int = shortlist_document_count,
                           first_documents_per_block: int = first_documents_per_block,
                           source_repeat_penalty: float = source_repeat_penalty,
                           near_copy_shingle_share: float = near_copy_shingle_share) -> dict:
    """Greedy packing of related documents.

    Returns a dict: `documents_in_pack_order`, `pack_starts` (the start of every pack in that order),
    `leftover_documents`, `near_copy_skip_count` and `packs_given_up`.

    The next unused document in `order_of_first_documents` (all documents in a seeded random order) becomes the
    first document of a pack. Then the unused document that fits the pack so far best is added, again and again,
    until the pack holds `pack_token_length` tokens (one end-of-text token counted per document).

    How well a document fits: its cosine similarity to the centroid of the pack (the mean of the topic vectors
    in it), minus `source_repeat_penalty` times the share of the pack's documents that come from its source.
    The centroid is used because the end of a sequence should be predictable from everything before it, so a new
    document is compared with the whole pack. Comparing with the last document only lets a pack wander from
    topic to topic; comparing with the first document only ignores what the pack already holds. The penalty is
    explained at `source_repeat_penalty`.

    Near copies. The best-fitting candidate is passed over when more than `near_copy_shingle_share` of its
    shingles (row of `shingle_ids_of_document`) already occur in the pack: the model could copy it from the
    context, and the random packs offer nothing of the kind. It is passed over for this pack only. A pack can
    only gain shingles, so a document that was a near copy once stays one, and it is not looked at again.

    Cost. Searching all documents in every step is one product of the whole matrix per added document. Here the
    candidates of a pack are the `shortlist_document_count` unused documents most similar to its first document,
    and each step searches only those; when all of them are in the pack or passed over, the next most similar
    ones are taken. The similarities of the first documents to all documents come from one matrix product per
    block of `first_documents_per_block`. A document of a block that an earlier pack of the same block took is
    skipped, so the packs are the same as if the first documents were handled one at a time. With a shortlist as
    large as the document count this is the plain search over all documents.

    Leftover documents are those still unused when their tokens can no longer fill a pack, and the documents of
    a pack that was given up: one that could not be filled because every other unused document is a near copy
    of what it holds.
    """
    document_count = len(token_count_of_document)
    source_count = int(source_of_document.max()) + 1
    length_of_document = token_count_of_document.astype(np.int64) + 1          # with its end-of-text token
    assert np.array_equal(np.sort(order_of_first_documents), np.arange(document_count)), (
        "order_of_first_documents must hold every document once")
    assert shingle_ids_of_document.shape[0] == document_count, "one row of shingles per document"
    first_shingle_of_document = shingle_ids_of_document.indptr
    shingle_ids = shingle_ids_of_document.indices
    is_document_used = np.zeros(document_count, dtype=bool)
    unused_document_count = document_count
    unused_token_count = int(length_of_document.sum())
    documents_in_pack_order: list[int] = []
    pack_starts: list[int] = []
    documents_of_packs_given_up: list[int] = []
    near_copy_skip_count = 0
    packs_given_up = 0
    next_place_in_order = 0

    # One thread for the matrix products below. The products inside a pack are small (2,048 by 128 numbers, 23
    # microseconds), and between them the other threads of the linear algebra library spin: on 20,000 documents
    # its 64 default threads needed 2.1 s and burned 18.4 s of processor time, on a machine shared with others.
    with threadpool_limits(limits=1, user_api="blas"):
        while unused_token_count >= pack_token_length:
            # the next block: the first unused documents of the seeded order
            first_documents_of_block: list[int] = []
            while len(first_documents_of_block) < first_documents_per_block and next_place_in_order < document_count:
                candidate_first_document = int(order_of_first_documents[next_place_in_order])
                next_place_in_order += 1
                if not is_document_used[candidate_first_document]:
                    first_documents_of_block.append(candidate_first_document)
            similarity_of_block_to_all_documents = (
                topic_vector_of_document[first_documents_of_block] @ topic_vector_of_document.T)

            for row_in_block, first_document in enumerate(first_documents_of_block):
                if is_document_used[first_document]:
                    continue                                  # an earlier pack of this block took it
                if unused_token_count < pack_token_length:
                    break                                     # what is left cannot fill a pack: the leftover
                is_document_used[first_document] = True
                # minus infinity marks a document that cannot be put on a shortlist of this pack (any more)
                similarity_to_first_document = similarity_of_block_to_all_documents[row_in_block]
                similarity_to_first_document[is_document_used] = -np.inf
                documents_not_yet_shortlisted = unused_document_count - 1

                documents_of_pack = [first_document]
                tokens_in_pack = int(length_of_document[first_document])
                sum_of_pack_topic_vectors = topic_vector_of_document[first_document].copy()
                pack_documents_of_source = np.zeros(source_count, dtype=np.float32)
                pack_documents_of_source[source_of_document[first_document]] = 1
                shingles_in_pack = set(shingle_ids[first_shingle_of_document[first_document]:
                                                   first_shingle_of_document[first_document + 1]].tolist())
                shortlist = np.zeros(0, dtype=np.int64)
                is_shortlisted_open = np.zeros(0, dtype=bool)            # can this shortlisted document still be added
                while tokens_in_pack < pack_token_length:
                    if not is_shortlisted_open.any():
                        if documents_not_yet_shortlisted == 0:
                            break                             # nothing is left to offer: the pack is given up
                        if documents_not_yet_shortlisted > shortlist_document_count:
                            shortlist = np.argpartition(similarity_to_first_document,
                                                        -shortlist_document_count)[-shortlist_document_count:]
                        else:
                            shortlist = np.flatnonzero(similarity_to_first_document > -np.inf)
                        shortlist.sort()                      # equal scores then go to the lower document index
                        similarity_to_first_document[shortlist] = -np.inf
                        documents_not_yet_shortlisted -= len(shortlist)
                        topic_vector_of_shortlisted = topic_vector_of_document[shortlist]
                        source_of_shortlisted = source_of_document[shortlist]
                        is_shortlisted_open = np.ones(len(shortlist), dtype=bool)

                    length_of_sum = max(float(np.linalg.norm(sum_of_pack_topic_vectors)), 1e-12)
                    centroid_direction = sum_of_pack_topic_vectors / length_of_sum
                    share_of_pack_from_source = pack_documents_of_source / len(documents_of_pack)
                    fit_to_pack = (topic_vector_of_shortlisted @ centroid_direction
                                   - source_repeat_penalty * share_of_pack_from_source[source_of_shortlisted])
                    fit_to_pack[~is_shortlisted_open] = -np.inf
                    best_row_of_shortlist = int(np.argmax(fit_to_pack))
                    is_shortlisted_open[best_row_of_shortlist] = False
                    candidate_document = int(shortlist[best_row_of_shortlist])
                    shingles_of_candidate = shingle_ids[first_shingle_of_document[candidate_document]:
                                                        first_shingle_of_document[candidate_document + 1]].tolist()
                    shingles_already_in_pack = len(shingles_in_pack.intersection(shingles_of_candidate))
                    if shingles_already_in_pack > near_copy_shingle_share * len(shingles_of_candidate):
                        near_copy_skip_count += 1             # a near copy: not for this pack, free for others
                        continue
                    documents_of_pack.append(candidate_document)
                    tokens_in_pack += int(length_of_document[candidate_document])
                    sum_of_pack_topic_vectors += topic_vector_of_shortlisted[best_row_of_shortlist]
                    pack_documents_of_source[source_of_document[candidate_document]] += 1
                    shingles_in_pack.update(shingles_of_candidate)

                is_document_used[documents_of_pack] = True
                unused_document_count -= len(documents_of_pack)
                unused_token_count -= tokens_in_pack
                if tokens_in_pack < pack_token_length:
                    packs_given_up += 1
                    documents_of_packs_given_up.extend(documents_of_pack)
                else:
                    pack_starts.append(len(documents_in_pack_order))
                    documents_in_pack_order.extend(documents_of_pack)

    leftover_documents = np.sort(np.concatenate([np.flatnonzero(~is_document_used),
                                                 np.array(documents_of_packs_given_up, dtype=np.int64)]))
    return {"documents_in_pack_order": np.array(documents_in_pack_order, dtype=np.int64),
            "pack_starts": np.array(pack_starts, dtype=np.int64), "leftover_documents": leftover_documents,
            "near_copy_skip_count": near_copy_skip_count, "packs_given_up": packs_given_up}


def choose_packs_to_keep(consecutive_similarity_of_pack: np.ndarray, document_tokens_of_pack: np.ndarray,
                         token_budget: int) -> np.ndarray:
    """Which packs of a build stay, one boolean per pack in build order (step 4 of the file docstring).

    Greedy packing builds its last packs from what the earlier ones left. Two kinds of packs go: every pack whose
    consecutive documents have a mean similarity below `minimum_pack_similarity`, and then the packs built last,
    as many as the sample holds beyond the budget. What stays are the first packs at or above the floor whose
    document tokens reach `token_budget`. When the packs above the floor hold less than the budget, all of them
    stay.
    """
    is_pack_kept = consecutive_similarity_of_pack >= minimum_pack_similarity
    tokens_of_kept_packs_so_far = np.cumsum(np.where(is_pack_kept, document_tokens_of_pack, 0))
    if tokens_of_kept_packs_so_far[-1] > token_budget:
        # the count only grows at a kept pack, so the first pack at which it reaches the budget is a kept one
        last_pack_needed = int(np.searchsorted(tokens_of_kept_packs_so_far, token_budget))
        is_pack_kept[last_pack_needed + 1:] = False
    return is_pack_kept


def cut_document_order_into_packs(token_count_in_order: np.ndarray, pack_token_length: int) -> tuple[np.ndarray, int]:
    """Read documents in the given order and close a pack as soon as it is full: the same rule as the greedy packs.

    Returns (the start of every pack, the number of documents after the last full pack). Those last documents
    cannot fill a pack of their own. They stay at the end of the last pack, behind the tokens the reader takes,
    because dropping them would leave the random arm with fewer documents than the semantic arm.
    """
    pack_starts = [0]
    tokens_in_open_pack = 0
    for place_in_order, token_count in enumerate(token_count_in_order.tolist()):
        tokens_in_open_pack += token_count + 1                                  # with its end-of-text token
        if tokens_in_open_pack >= pack_token_length:
            pack_starts.append(place_in_order + 1)
            tokens_in_open_pack = 0
    # the last entry is the start of a pack that never became full, or one past the last document
    documents_after_last_full_pack = len(token_count_in_order) - pack_starts.pop()
    assert pack_starts, f"the documents do not fill one pack of {pack_token_length} tokens"
    return np.array(pack_starts, dtype=np.int64), documents_after_last_full_pack


# ---------------------------------------------------------------------------------------------------------------
# check, measure, write and load a pack set
# ---------------------------------------------------------------------------------------------------------------
def token_counts_of_references(source_names: list[str], source_of_reference: np.ndarray,
                               document_of_reference: np.ndarray) -> np.ndarray:
    """The token count of every reference, looked up in the train token files the reader will use."""
    token_count_of_reference = np.zeros(len(source_of_reference), dtype=np.int64)
    for source_index, source_name in enumerate(source_names):
        token_count_of_every_document = np.diff(open_token_source(source_name, "train")["document_offsets"])
        is_from_this_source = source_of_reference == source_index
        positions = document_of_reference[is_from_this_source]
        assert len(positions) == 0 or (positions.min() >= 0 and positions.max() < len(token_count_of_every_document)), (
            f"a reference points past the {len(token_count_of_every_document):,} train documents of {source_name}")
        token_count_of_reference[is_from_this_source] = token_count_of_every_document[positions]
    return token_count_of_reference


def check_pack_set_against_contract(source_names: list[str], source_of_reference: np.ndarray,
                                    document_of_reference: np.ndarray, pack_start_references: np.ndarray,
                                    pack_token_length: int, flagged_document_ids: set[str]) -> np.ndarray:
    """Stop with an error when the arrays break what the reader relies on; return the token count of every reference.

    The reader relies on: the dtypes, ascending pack starts from 0, every document at most once, every pack at
    least `pack_token_length` tokens long, and no document that the decontamination gate flagged.
    """
    reference_count = len(source_of_reference)
    assert source_of_reference.dtype == np.int16, "source_of_reference must be int16"
    assert document_of_reference.dtype == np.int64 and len(document_of_reference) == reference_count, (
        "document_of_reference must be int64 with one entry per reference")
    assert pack_start_references.dtype == np.int64 and len(pack_start_references) > 0, (
        "pack_start_references must be int64 and hold at least one pack")
    assert pack_start_references[0] == 0 and np.all(np.diff(pack_start_references) > 0), (
        "pack_start_references must start at 0 and ascend")
    assert pack_start_references[-1] < reference_count, "pack_start_references must stay inside the references"
    assert source_of_reference.min() >= 0 and source_of_reference.max() < len(source_names), "unknown source id"
    # one number per (source, document) pair, to count the distinct pairs
    documents_per_source_bound = int(document_of_reference.max()) + 1
    key_of_reference = source_of_reference.astype(np.int64) * documents_per_source_bound + document_of_reference
    assert len(np.unique(key_of_reference)) == reference_count, "a document appears twice in the pack set"
    token_count_of_reference = token_counts_of_references(source_names, source_of_reference, document_of_reference)
    tokens_per_pack = np.add.reduceat(token_count_of_reference + 1, pack_start_references)
    assert tokens_per_pack.min() >= pack_token_length, (
        f"a pack holds {int(tokens_per_pack.min()):,} tokens, fewer than the {pack_token_length:,} the reader takes")
    assert_no_packed_document_is_flagged(source_names, source_of_reference, document_of_reference, flagged_document_ids)
    return token_count_of_reference


def measure_similarity_inside_packs(topic_vector_of_reference: np.ndarray,
                                    pack_start_references: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two numbers per pack: the mean cosine similarity of the topic vectors of a document and the document before
    it, and the mean similarity of every later document to the first document of the pack.

    The second falls when a pack drifts away from the topic it began with.
    """
    reference_count = len(topic_vector_of_reference)
    pack_count = len(pack_start_references)
    documents_per_pack = np.diff(np.append(pack_start_references, reference_count))
    assert documents_per_pack.min() >= 2, "a pack with one document has no similarity inside it"
    pack_of_reference = np.repeat(np.arange(pack_count), documents_per_pack)
    is_in_pack_of_previous = pack_of_reference[1:] == pack_of_reference[:-1]
    similarity_to_previous = np.einsum("ij,ij->i", topic_vector_of_reference[1:], topic_vector_of_reference[:-1])
    first_reference_of_reference = pack_start_references[pack_of_reference]
    is_later_document = np.arange(reference_count) != first_reference_of_reference
    similarity_to_first = np.einsum("ij,ij->i", topic_vector_of_reference,
                                    topic_vector_of_reference[first_reference_of_reference])
    sum_of_consecutive_similarity = np.bincount(
        pack_of_reference[1:][is_in_pack_of_previous], weights=similarity_to_previous[is_in_pack_of_previous],
        minlength=pack_count)
    sum_of_first_similarity = np.bincount(
        pack_of_reference[is_later_document], weights=similarity_to_first[is_later_document], minlength=pack_count)
    return sum_of_consecutive_similarity / (documents_per_pack - 1), sum_of_first_similarity / (documents_per_pack - 1)


def count_shingles_seen_earlier_in_pack(shingle_ids_of_document: scipy.sparse.csr_matrix,
                                        documents_in_pack_order: np.ndarray,
                                        pack_start_references: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two numbers per reference: how many of its shingles occur in an earlier document of its pack, and how many
    shingles it has. Their ratio is the share of a document that a reader of the pack has already seen."""
    first_shingle_of_document = shingle_ids_of_document.indptr
    shingle_ids = shingle_ids_of_document.indices
    reference_count = len(documents_in_pack_order)
    seen_shingle_count_of_reference = np.zeros(reference_count, dtype=np.int64)
    shingle_count_of_reference = np.zeros(reference_count, dtype=np.int64)
    pack_stop_references = np.append(pack_start_references[1:], reference_count)
    for pack_start, pack_stop in zip(pack_start_references.tolist(), pack_stop_references.tolist()):
        shingles_in_pack: set[int] = set()
        for reference in range(pack_start, pack_stop):
            document = documents_in_pack_order[reference]
            shingles_of_document = shingle_ids[first_shingle_of_document[document]:
                                               first_shingle_of_document[document + 1]].tolist()
            shingle_count_of_reference[reference] = len(shingles_of_document)
            seen_shingle_count_of_reference[reference] = len(shingles_in_pack.intersection(shingles_of_document))
            shingles_in_pack.update(shingles_of_document)
    return seen_shingle_count_of_reference, shingle_count_of_reference


def measure_pack_statistics(source_names: list[str], source_of_reference: np.ndarray,
                            token_count_of_reference: np.ndarray, pack_start_references: np.ndarray,
                            pack_token_length: int, topic_vector_of_reference: np.ndarray | None = None,
                            seen_shingle_count_of_reference: np.ndarray | None = None,
                            shingle_count_of_reference: np.ndarray | None = None) -> dict:
    """The numbers that describe a pack set; the similarity numbers need the topic vectors, the last three the shingles.

    Every pack is one training sequence, so the similarity numbers are taken over packs, of the mean inside a
    pack (`measure_similarity_inside_packs`). The ten values `..._by_tenth_of_build_order` follow the packs in
    the order they are stored. `share_of_shingles_seen_earlier_in_pack` is the share of all word 8-grams that
    already occur in an earlier document of the same pack: text a model could copy from its context.
    """
    reference_count = len(source_of_reference)
    pack_count = len(pack_start_references)
    documents_per_pack = np.diff(np.append(pack_start_references, reference_count))
    pack_of_reference = np.repeat(np.arange(pack_count), documents_per_pack)
    distinct_sources_per_pack = np.zeros(pack_count, dtype=np.int64)
    for source_index in range(len(source_names)):
        references_per_pack = np.bincount(pack_of_reference[source_of_reference == source_index], minlength=pack_count)
        distinct_sources_per_pack += references_per_pack > 0
    document_token_count = int(token_count_of_reference.sum())
    tokens_the_reader_takes = pack_count * pack_token_length
    statistics = {
        "packs": pack_count,
        "documents": reference_count,
        "documents_per_pack_mean": round(float(documents_per_pack.mean()), 2),
        "documents_per_pack_median": float(np.median(documents_per_pack)),
        "distinct_sources_per_pack_mean": round(float(distinct_sources_per_pack.mean()), 2),
        "share_of_packs_from_one_source": round(float((distinct_sources_per_pack == 1).mean()), 4),
        "document_tokens": document_token_count,
        "tokens_the_reader_takes": tokens_the_reader_takes,
        # the end of the last document of a pack lies beyond the pack length and is never trained on
        "share_of_pack_tokens_beyond_the_pack_length": round(
            1.0 - tokens_the_reader_takes / (document_token_count + reference_count), 4),
        "documents_by_source": {
            name: int((source_of_reference == index).sum()) for index, name in enumerate(source_names)},
        "token_share_by_source": {
            name: round(float(token_count_of_reference[source_of_reference == index].sum()) / document_token_count, 4)
            for index, name in enumerate(source_names)},
    }
    if topic_vector_of_reference is not None:
        consecutive_similarity_of_pack, first_similarity_of_pack = measure_similarity_inside_packs(
            topic_vector_of_reference, pack_start_references)
        statistics["similarity_of_consecutive_documents_mean"] = round(float(consecutive_similarity_of_pack.mean()), 4)
        statistics["similarity_to_first_document_mean"] = round(float(first_similarity_of_pack.mean()), 4)
        statistics["similarity_of_consecutive_documents_percentiles"] = {
            str(percent): round(float(np.percentile(consecutive_similarity_of_pack, percent)), 4)
            for percent in (1, 5, 50, 95)}
        statistics["share_of_packs_below_minimum_pack_similarity"] = round(
            float((consecutive_similarity_of_pack < minimum_pack_similarity).mean()), 4)
        statistics["similarity_of_consecutive_documents_by_tenth_of_build_order"] = [
            round(float(tenth.mean()), 4) for tenth in np.array_split(consecutive_similarity_of_pack, 10) if len(tenth)]
    if seen_shingle_count_of_reference is not None:
        is_past_near_copy_share = seen_shingle_count_of_reference > near_copy_shingle_share * shingle_count_of_reference
        statistics["share_of_shingles_seen_earlier_in_pack"] = round(
            float(seen_shingle_count_of_reference.sum()) / max(1, int(shingle_count_of_reference.sum())), 5)
        statistics["documents_past_the_near_copy_share"] = int(is_past_near_copy_share.sum())
        statistics["share_of_packs_with_a_document_past_the_near_copy_share"] = round(
            float((np.bincount(pack_of_reference[is_past_near_copy_share], minlength=pack_count) > 0).mean()), 4)
    return statistics


def print_pack_statistics(pack_name: str, statistics: dict) -> None:
    print(f"{pack_name}: {statistics['packs']:,} packs, {statistics['documents']:,} documents "
          f"({statistics['documents_per_pack_mean']} per pack, median {statistics['documents_per_pack_median']:g}), "
          f"{statistics['distinct_sources_per_pack_mean']} sources per pack "
          f"({100 * statistics['share_of_packs_from_one_source']:.1f}% of the packs hold one source)")
    print(f"  {statistics['document_tokens']:,} document tokens; the reader takes "
          f"{statistics['tokens_the_reader_takes']:,}, "
          f"{100 * statistics['share_of_pack_tokens_beyond_the_pack_length']:.1f}% lie beyond the pack length")
    token_shares = ", ".join(f"{name} {share:.3f}" for name, share in statistics["token_share_by_source"].items())
    print(f"  token share by source: {token_shares}")
    if "similarity_of_consecutive_documents_mean" in statistics:
        print(f"  similarity of consecutive documents {statistics['similarity_of_consecutive_documents_mean']:.3f}, "
              f"to the first document of the pack {statistics['similarity_to_first_document_mean']:.3f}")
        print(f"  consecutive similarity, percentiles of the packs: "
              f"{statistics['similarity_of_consecutive_documents_percentiles']}; "
              f"{100 * statistics['share_of_packs_below_minimum_pack_similarity']:.1f}% of the packs are below "
              f"{minimum_pack_similarity}")
        print(f"  consecutive similarity by tenth of the build order: "
              f"{statistics['similarity_of_consecutive_documents_by_tenth_of_build_order']}")
    if "share_of_shingles_seen_earlier_in_pack" in statistics:
        print(f"  {100 * statistics['share_of_shingles_seen_earlier_in_pack']:.3f}% of the word {shingle_word_count}-grams "
              f"already occur earlier in the same pack; {statistics['documents_past_the_near_copy_share']:,} documents "
              f"repeat more than {near_copy_shingle_share:.0%} of theirs")


def write_pack_set(pack_name: str, source_names: list[str], source_of_reference: np.ndarray,
                   document_of_reference: np.ndarray, pack_start_references: np.ndarray, summary: dict,
                   topic_vector_of_reference: np.ndarray | None = None) -> None:
    pack_directory = pack_directory_for(pack_name)
    pack_directory.mkdir(parents=True, exist_ok=True)
    (pack_directory / "source_names.json").write_text(json.dumps(list(source_names)) + "\n")
    np.save(pack_directory / "source_of_reference.npy", source_of_reference)
    np.save(pack_directory / "document_of_reference.npy", document_of_reference)
    np.save(pack_directory / "pack_start_references.npy", pack_start_references)
    topic_vector_path = pack_directory / "topic_vector_of_reference.npy"
    if topic_vector_of_reference is not None:
        np.save(topic_vector_path, topic_vector_of_reference.astype(np.float32))
    elif topic_vector_path.exists():
        topic_vector_path.unlink()               # a rebuild under an old name must not keep the old vectors
    (pack_directory / "pack_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {pack_directory}", flush=True)


def load_pack_set(pack_name: str) -> dict:
    """The arrays of a built pack set, the token count of every reference, and its summary."""
    pack_directory = pack_directory_for(pack_name)
    assert (pack_directory / "pack_start_references.npy").exists(), (
        f"no pack set {pack_name!r} under {pack_directory.parent}: run `document_packing build` for it first")
    source_names = json.loads((pack_directory / "source_names.json").read_text())
    source_of_reference = np.load(pack_directory / "source_of_reference.npy")
    document_of_reference = np.load(pack_directory / "document_of_reference.npy")
    topic_vector_path = pack_directory / "topic_vector_of_reference.npy"
    return {"source_names": source_names, "source_of_reference": source_of_reference,
            "document_of_reference": document_of_reference,
            "pack_start_references": np.load(pack_directory / "pack_start_references.npy"),
            "token_count_of_reference": token_counts_of_references(
                source_names, source_of_reference, document_of_reference),
            "topic_vector_of_reference": np.load(topic_vector_path) if topic_vector_path.exists() else None,
            "summary": json.loads((pack_directory / "pack_summary.json").read_text())}


def merge_summary_into_results(summary: dict) -> None:
    """Add or replace this pack set's entry in results/pack_summary.json; the other entries stay as they are."""
    summaries = json.loads(results_summary_path.read_text()) if results_summary_path.exists() else {}
    summaries[summary["pack_name"]] = summary
    results_summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_summary_path.write_text(json.dumps(summaries, indent=1) + "\n")
    print(f"merged {summary['pack_name']} into {results_summary_path}")


# ---------------------------------------------------------------------------------------------------------------
# the three builds
# ---------------------------------------------------------------------------------------------------------------
def build_semantic_packs(pack_name: str, token_budget: int, pack_token_length: int, seed: int,
                         worker_count: int) -> dict:
    """Sample, describe, pack, drop the tail, check, write; returns the summary that is also written."""
    seconds = {}
    stage_started = build_started = time.time()
    sampled_token_budget = int(round(token_budget * (1.0 + oversampling_share)))
    print(f"{pack_name}: sampling {sampled_token_budget:,} tokens ({token_budget:,} and {oversampling_share:.0%} "
          f"more) of documents with {minimum_document_tokens} to {pack_token_length // 2:,} tokens (seed {seed})",
          flush=True)
    flagged_document_ids = load_flagged_document_ids()
    sample = sample_documents_for_token_budget(sampled_token_budget, pack_token_length, seed, flagged_document_ids)
    source_names = sample["source_names"]
    source_of_document = sample["source_of_document"]
    token_count_of_document = sample["token_count_of_document"]
    seconds["sample"] = round(time.time() - stage_started, 1)

    stage_started = time.time()
    word_counts, shingle_ids_of_document = describe_documents(
        source_names, source_of_document, sample["document_ids"], worker_count)
    documents_without_words = int((np.diff(word_counts.indptr) == 0).sum())
    seconds["read_texts_count_words_hash_shingles"] = round(time.time() - stage_started, 1)
    print(f"  {word_counts.shape[0]:,} documents: {word_counts.nnz:,} word counts, {shingle_ids_of_document.nnz:,} "
          f"shingles, {documents_without_words} documents without a countable word "
          f"({seconds['read_texts_count_words_hash_shingles']} s)", flush=True)

    stage_started = time.time()
    topic_vector_of_document = compute_topic_vectors(word_counts, seed)
    seconds["compute_topic_vectors"] = round(time.time() - stage_started, 1)
    print(f"  topic vectors with {topic_vector_of_document.shape[1]} dimensions "
          f"({seconds['compute_topic_vectors']} s)", flush=True)

    stage_started = time.time()
    order_of_first_documents = random_generator_for(seed, "order of first documents").permutation(
        len(token_count_of_document))
    packing = pack_related_documents(topic_vector_of_document, token_count_of_document, source_of_document,
                                     shingle_ids_of_document, pack_token_length, order_of_first_documents)
    built_documents, built_pack_starts = packing["documents_in_pack_order"], packing["pack_starts"]
    leftover_token_count = int(token_count_of_document[packing["leftover_documents"]].sum())
    seconds["pack"] = round(time.time() - stage_started, 1)
    print(f"  {len(built_pack_starts):,} packs built ({seconds['pack']} s); {packing['near_copy_skip_count']:,} times a "
          f"near copy was passed over; dropped {len(packing['leftover_documents'])} leftover documents with "
          f"{leftover_token_count:,} tokens that could not fill a last pack ({packing['packs_given_up']} packs were "
          f"given up)", flush=True)

    # step 4: drop the packs below the similarity floor, then packs from the end, back to the token budget
    consecutive_similarity_of_built_pack, _ = measure_similarity_inside_packs(
        topic_vector_of_document[built_documents], built_pack_starts)
    is_pack_kept = choose_packs_to_keep(consecutive_similarity_of_built_pack,
                                        np.add.reduceat(token_count_of_document[built_documents], built_pack_starts),
                                        token_budget)
    is_pack_below_floor = consecutive_similarity_of_built_pack < minimum_pack_similarity
    documents_per_built_pack = np.diff(np.append(built_pack_starts, len(built_documents)))
    documents_in_pack_order = built_documents[np.repeat(is_pack_kept, documents_per_built_pack)]
    documents_per_kept_pack = documents_per_built_pack[is_pack_kept]
    pack_start_references = np.concatenate([[0], np.cumsum(documents_per_kept_pack)[:-1]]).astype(np.int64)
    dropped_document_count = len(built_documents) - len(documents_in_pack_order)
    kept_token_count = int(token_count_of_document[documents_in_pack_order].sum())
    print(f"  dropped {int(is_pack_below_floor.sum()):,} packs below a consecutive similarity of "
          f"{minimum_pack_similarity} and {int((~is_pack_kept & ~is_pack_below_floor).sum()):,} packs from the end of "
          f"the build; {int(is_pack_kept.sum()):,} packs with {kept_token_count:,} document tokens stay", flush=True)
    if kept_token_count < token_budget:
        print(f"  NOTE: the kept packs hold fewer tokens than the budget of {token_budget:,}: more than "
              f"{oversampling_share:.0%} of the sample was dropped", flush=True)

    stage_started = time.time()
    source_of_reference = source_of_document[documents_in_pack_order]
    document_of_reference = sample["position_of_document"][documents_in_pack_order]
    topic_vector_of_reference = topic_vector_of_document[documents_in_pack_order]
    token_count_of_reference = check_pack_set_against_contract(
        source_names, source_of_reference, document_of_reference, pack_start_references, pack_token_length,
        flagged_document_ids)
    assert np.array_equal(token_count_of_reference, token_count_of_document[documents_in_pack_order]), (
        "the token files changed while the packs were built")
    seen_shingle_count_of_reference, shingle_count_of_reference = count_shingles_seen_earlier_in_pack(
        shingle_ids_of_document, documents_in_pack_order, pack_start_references)
    assert np.all(seen_shingle_count_of_reference <= near_copy_shingle_share * shingle_count_of_reference), (
        "a pack holds a near copy of one of its earlier documents")
    statistics = measure_pack_statistics(
        source_names, source_of_reference, token_count_of_reference, pack_start_references, pack_token_length,
        topic_vector_of_reference, seen_shingle_count_of_reference, shingle_count_of_reference)
    statistics["sampled_documents"] = len(token_count_of_document)
    statistics["sampled_document_tokens"] = int(token_count_of_document.sum())
    statistics["near_copy_skips"] = packing["near_copy_skip_count"]
    statistics["packs_given_up_for_near_copies"] = packing["packs_given_up"]
    statistics["leftover_documents_dropped"] = len(packing["leftover_documents"])
    statistics["leftover_tokens_dropped"] = leftover_token_count
    statistics["packs_built"] = len(built_pack_starts)
    statistics["packs_dropped_below_minimum_pack_similarity"] = int(is_pack_below_floor.sum())
    statistics["packs_dropped_from_the_end_of_the_build"] = int((~is_pack_kept & ~is_pack_below_floor).sum())
    statistics["documents_in_dropped_packs"] = dropped_document_count
    statistics["documents_without_a_countable_word"] = documents_without_words
    statistics["flagged_candidates_removed_by_source"] = sample["flagged_candidates_removed_by_source"]
    seconds["check_and_measure"] = round(time.time() - stage_started, 1)
    seconds["total"] = round(time.time() - build_started, 1)
    summary = {"pack_name": pack_name, "mode": "semantic", "seed": seed,
               "settings": {"pack_token_length": pack_token_length, "token_budget": token_budget,
                            "oversampling_share": oversampling_share,
                            "source_token_shares": source_token_shares,
                            "minimum_document_tokens": minimum_document_tokens,
                            "maximum_document_tokens": pack_token_length // 2,
                            "maximum_characters_per_text": maximum_characters_per_text,
                            "hashed_word_column_count": hashed_word_column_count,
                            "topic_dimension_count": int(topic_vector_of_document.shape[1]),
                            "shortlist_document_count": shortlist_document_count,
                            "source_repeat_penalty": source_repeat_penalty,
                            "shingle_word_count": shingle_word_count,
                            "near_copy_shingle_share": near_copy_shingle_share,
                            "minimum_pack_similarity": minimum_pack_similarity},
               "statistics": statistics, "seconds": seconds}
    write_pack_set(pack_name, source_names, source_of_reference, document_of_reference, pack_start_references,
                   summary, topic_vector_of_reference)
    print_pack_statistics(pack_name, statistics)
    print(f"  built in {seconds['total']} s", flush=True)
    return summary


def build_random_packs(pack_name: str, same_documents_as: str, pack_token_length: int, seed: int,
                       worker_count: int = 1) -> dict:
    """The control arm: the documents of a semantic pack set in a seeded random order, cut by the same length rule.

    The texts are read once more, only to measure how much of a pack repeats its earlier documents; the random
    packs get no near-copy rule, because a random pack almost never holds two copies.
    """
    build_started = time.time()
    flagged_document_ids = load_flagged_document_ids()
    semantic_pack_set = load_pack_set(same_documents_as)
    semantic_summary = semantic_pack_set["summary"]
    assert semantic_summary["mode"] == "semantic", (
        f"{same_documents_as} is a {semantic_summary['mode']} pack set; the random packs need a semantic one")
    assert semantic_summary["settings"]["pack_token_length"] == pack_token_length, (
        f"{same_documents_as} was built with packs of {semantic_summary['settings']['pack_token_length']:,} tokens, "
        f"not {pack_token_length:,}: the two arms must be cut by the same length rule")
    source_names = semantic_pack_set["source_names"]
    reference_count = len(semantic_pack_set["source_of_reference"])
    random_order = random_generator_for(seed, "random order of documents").permutation(reference_count)
    source_of_reference = semantic_pack_set["source_of_reference"][random_order]
    document_of_reference = semantic_pack_set["document_of_reference"][random_order]
    pack_start_references, documents_after_last_full_pack = cut_document_order_into_packs(
        semantic_pack_set["token_count_of_reference"][random_order], pack_token_length)

    token_count_of_reference = check_pack_set_against_contract(
        source_names, source_of_reference, document_of_reference, pack_start_references, pack_token_length,
        flagged_document_ids)
    # the point of the control: the same documents, each once, in both arms
    sorted_semantic_references = np.lexsort(
        (semantic_pack_set["document_of_reference"], semantic_pack_set["source_of_reference"]))
    sorted_random_references = np.lexsort((document_of_reference, source_of_reference))
    assert (np.array_equal(semantic_pack_set["source_of_reference"][sorted_semantic_references],
                           source_of_reference[sorted_random_references])
            and np.array_equal(semantic_pack_set["document_of_reference"][sorted_semantic_references],
                               document_of_reference[sorted_random_references])), (
        f"{pack_name} and {same_documents_as} do not hold the same documents")

    assert semantic_pack_set["topic_vector_of_reference"] is not None, (
        f"{same_documents_as} has no topic_vector_of_reference.npy: rebuild it")
    topic_vector_of_reference = semantic_pack_set["topic_vector_of_reference"][random_order]
    # the shingles of the documents in the order of the semantic pack set; `random_order` points into that order
    _, shingle_ids_of_semantic_reference = describe_documents(
        source_names, semantic_pack_set["source_of_reference"],
        document_ids_of_references(source_names, semantic_pack_set["source_of_reference"],
                                   semantic_pack_set["document_of_reference"]), worker_count)
    seen_shingle_count_of_reference, shingle_count_of_reference = count_shingles_seen_earlier_in_pack(
        shingle_ids_of_semantic_reference, random_order, pack_start_references)
    statistics = measure_pack_statistics(
        source_names, source_of_reference, token_count_of_reference, pack_start_references, pack_token_length,
        topic_vector_of_reference, seen_shingle_count_of_reference, shingle_count_of_reference)
    statistics["documents_after_the_last_full_pack"] = documents_after_last_full_pack
    seconds = {"total": round(time.time() - build_started, 1)}
    summary = {"pack_name": pack_name, "mode": "random", "seed": seed,
               "settings": {"pack_token_length": pack_token_length, "same_documents_as": same_documents_as,
                            "shingle_word_count": shingle_word_count,
                            "near_copy_shingle_share": near_copy_shingle_share,
                            "minimum_pack_similarity": minimum_pack_similarity},
               "statistics": statistics, "seconds": seconds}
    write_pack_set(pack_name, source_names, source_of_reference, document_of_reference, pack_start_references,
                   summary, topic_vector_of_reference)
    print_pack_statistics(pack_name, statistics)
    print(f"  the same {reference_count:,} documents as {same_documents_as}; the {documents_after_last_full_pack} "
          f"documents after the last full pack stay at its end, behind the tokens the reader takes")
    print(f"  built in {seconds['total']} s", flush=True)
    return summary


def build_natural_packs(pack_name: str, token_budget: int, pack_token_length: int, seed: int) -> dict:
    """One document of at least `pack_token_length` tokens per pack: text that is long and coherent by itself.

    The reader takes `pack_token_length` tokens of every pack, so the budget buys `token_budget // pack_token_length`
    packs. They are drawn in a seeded random order from all long documents of the natural sources together, which
    keeps the sources in the proportion in which long documents occur. When the sources hold fewer long documents
    than the budget buys, every one of them becomes a pack.
    """
    build_started = time.time()
    flagged_document_ids = load_flagged_document_ids()
    source_names = list(natural_source_names)
    source_of_candidate, position_of_candidate = [], []
    flagged_candidates_removed_by_source = {}
    for source_index, source_name in enumerate(source_names):
        token_source = open_token_source(source_name, "train")
        is_long_document = np.diff(token_source["document_offsets"]) >= pack_token_length
        is_flagged = find_flagged_documents(token_source["document_ids"], flagged_document_ids)
        long_document_positions = np.flatnonzero(is_long_document & ~is_flagged)
        flagged_candidates_removed_by_source[source_name] = int((is_long_document & is_flagged).sum())
        source_of_candidate.append(np.full(len(long_document_positions), source_index, dtype=np.int16))
        position_of_candidate.append(long_document_positions.astype(np.int64))
    source_of_candidate = np.concatenate(source_of_candidate)
    position_of_candidate = np.concatenate(position_of_candidate)
    candidate_count = len(source_of_candidate)
    assert candidate_count > 0, f"no document of {source_names} holds {pack_token_length:,} tokens"
    wanted_pack_count = max(1, token_budget // pack_token_length)
    chosen_candidates = random_generator_for(seed, "natural documents").permutation(candidate_count)[:wanted_pack_count]

    source_of_reference = source_of_candidate[chosen_candidates]
    document_of_reference = position_of_candidate[chosen_candidates]
    pack_start_references = np.arange(len(chosen_candidates), dtype=np.int64)
    token_count_of_reference = check_pack_set_against_contract(
        source_names, source_of_reference, document_of_reference, pack_start_references, pack_token_length,
        flagged_document_ids)
    assert token_count_of_reference.min() >= pack_token_length, "a natural pack holds a document shorter than the pack"
    statistics = measure_pack_statistics(source_names, source_of_reference, token_count_of_reference,
                                         pack_start_references, pack_token_length)
    statistics["candidate_documents_by_source"] = {
        name: int((source_of_candidate == index).sum()) for index, name in enumerate(source_names)}
    statistics["candidate_documents_left_out"] = candidate_count - len(chosen_candidates)
    statistics["flagged_candidates_removed_by_source"] = flagged_candidates_removed_by_source
    statistics["document_tokens_median"] = float(np.median(token_count_of_reference))
    seconds = {"total": round(time.time() - build_started, 1)}
    summary = {"pack_name": pack_name, "mode": "natural", "seed": seed,
               "settings": {"pack_token_length": pack_token_length, "token_budget": token_budget,
                            "natural_source_names": source_names},
               "statistics": statistics, "seconds": seconds}
    write_pack_set(pack_name, source_names, source_of_reference, document_of_reference, pack_start_references, summary)
    print_pack_statistics(pack_name, statistics)
    print(f"  {candidate_count:,} unflagged documents hold at least {pack_token_length:,} tokens "
          f"({statistics['candidate_documents_by_source']}); flagged candidates removed: "
          f"{flagged_candidates_removed_by_source}")
    print(f"  the budget of {token_budget:,} tokens buys {wanted_pack_count:,} packs; {len(chosen_candidates):,} were "
          f"built and {statistics['candidate_documents_left_out']:,} candidates are left out")
    print(f"  built in {seconds['total']} s", flush=True)
    return summary


# ---------------------------------------------------------------------------------------------------------------
# show: read one pack
# ---------------------------------------------------------------------------------------------------------------
def show_pack(pack_name: str, pack_index: int, preview_character_count: int = 300) -> None:
    """Print the start of every document of one pack, with its similarity to the document before it and to the first."""
    pack_set = load_pack_set(pack_name)
    source_names = pack_set["source_names"]
    source_of_reference = pack_set["source_of_reference"]
    pack_start_references = pack_set["pack_start_references"]
    pack_count = len(pack_start_references)
    assert 0 <= pack_index < pack_count, f"{pack_name} has packs 0 to {pack_count - 1}, not {pack_index}"
    first_reference = int(pack_start_references[pack_index])
    is_last_pack = pack_index + 1 == pack_count
    stop_reference = len(source_of_reference) if is_last_pack else int(pack_start_references[pack_index + 1])
    references = list(range(first_reference, stop_reference))

    # the doc_id of every reference, then the texts by doc_id from the pool files of the sources in this pack
    document_id_of_reference = dict(zip(references, document_ids_of_references(
        source_names, source_of_reference[first_reference:stop_reference],
        pack_set["document_of_reference"][first_reference:stop_reference])))
    read_tasks = []
    for source_index, source_name in enumerate(source_names):
        wanted_document_ids = [document_id_of_reference[reference] for reference in references
                               if source_of_reference[reference] == source_index]
        if wanted_document_ids:
            # four times the preview: runs of spaces and line breaks are folded below, which shortens the text
            read_tasks.extend((str(pool_file_path), wanted_document_ids, 4 * preview_character_count)
                              for pool_file_path in pool_file_paths_of_source(source_name))
    # threads are enough here: reading parquet releases the interpreter lock, and no words are counted
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as thread_pool:
        read_results = list(thread_pool.map(lambda task: read_texts_from_pool_file(*task), read_tasks))
    text_of_document_id = {document_id: text for found_document_ids, found_texts in read_results
                           for document_id, text in zip(found_document_ids, found_texts)}

    token_count_of_reference = pack_set["token_count_of_reference"]
    topic_vector_of_reference = pack_set["topic_vector_of_reference"]
    tokens_in_pack = int(token_count_of_reference[first_reference:stop_reference].sum()) + len(references)
    print(f"{pack_name}, pack {pack_index} of {pack_count:,}: {len(references)} documents, "
          f"{tokens_in_pack:,} tokens with end-of-text tokens")
    for reference in references:
        document_id = document_id_of_reference[reference]
        similarity_note = ""
        if topic_vector_of_reference is not None and reference > first_reference:
            topic_vector = topic_vector_of_reference[reference]
            similarity_to_previous = float(topic_vector @ topic_vector_of_reference[reference - 1])
            similarity_to_first = float(topic_vector @ topic_vector_of_reference[first_reference])
            similarity_note = f"   similarity to previous {similarity_to_previous:.2f}, to first {similarity_to_first:.2f}"
        print(f"\n[{reference - first_reference}] {source_names[source_of_reference[reference]]}, "
              f"{int(token_count_of_reference[reference]):,} tokens, {document_id}{similarity_note}")
        print("    " + " ".join(text_of_document_id[document_id].split())[:preview_character_count])


# ---------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------
def main(argument_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build one pack set")
    build.add_argument("--mode", required=True, choices=pack_mode_names)
    build.add_argument("--name", default=None, help="pack name; default <mode>_<pack length in K>k, e.g. semantic_16k")
    build.add_argument("--token-budget", type=int, default=None,
                       help=f"semantic: document tokens the pack set holds; natural: tokens the reader will take; "
                            f"default {default_token_budget:,}")
    build.add_argument("--pack-token-length", type=int, default=default_pack_token_length)
    build.add_argument("--seed", type=int, default=default_seed)
    build.add_argument("--same-documents-as", default=None,
                       help="random only: the semantic pack set whose documents are shuffled; "
                            "default semantic_<pack length in K>k")
    build.add_argument("--worker-count", type=int, default=min(32, os.cpu_count() or 8),
                       help="semantic and random: worker processes that read the pool files")
    show = commands.add_parser("show", help="print the start of every document of one pack")
    show.add_argument("--name", required=True)
    show.add_argument("--pack-index", type=int, default=0)
    arguments = parser.parse_args(argument_list)

    if arguments.command == "show":
        show_pack(arguments.name, arguments.pack_index)
        return
    length_label = f"{arguments.pack_token_length // 1024}k"
    pack_name = arguments.name or f"{arguments.mode}_{length_label}"
    if arguments.mode == "random":
        assert arguments.token_budget is None, (
            "the random packs take their documents from the semantic pack set, so --token-budget has no effect")
        summary = build_random_packs(pack_name, arguments.same_documents_as or f"semantic_{length_label}",
                                     arguments.pack_token_length, arguments.seed, arguments.worker_count)
    else:
        assert arguments.same_documents_as is None, "--same-documents-as is for --mode random only"
        token_budget = arguments.token_budget or default_token_budget
        if arguments.mode == "semantic":
            summary = build_semantic_packs(pack_name, token_budget, arguments.pack_token_length, arguments.seed,
                                           arguments.worker_count)
        else:
            summary = build_natural_packs(pack_name, token_budget, arguments.pack_token_length, arguments.seed)
    merge_summary_into_results(summary)


if __name__ == "__main__":
    main()
