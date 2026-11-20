# Failure atlas

[English](PITFALLS.md) | [中文](PITFALLS.zh-CN.md)

How this pipeline silently fails, how each failure was detected, and what now prevents it from recurring.

Every entry is a mistake actually made here while building the continued-pretraining study, caught by measurement,
corrected, and where possible given a permanent regression test. Nothing below is a generic best-practice list.

**The organising observation: seven of the eight incidents produced something plausible that pointed the wrong way (a
pack set that looked related, a diversity check that passed, a contamination count, a shift of the loss, a sentence
about where data came from), and none of them stopped the pipeline.** The one incident that stopped it (C1) was the cheapest to fix.

Companion document: [README.md](README.md) explains how the pipeline works and what the experiments showed.

---

## Index

| id | incident | category | cost if uncaught |
|---|---|---|---|
| A1 | Semantic packs put editions of one text side by side | data integrity | the long-context comparison would have measured copying, not the use of related text |
| A2 | The last packs built were leftovers | data integrity | the semantic arm would have carried a tail of random packs |
| B1 | Density clustering found nothing in 384 dimensions | metric validity | the diversity gate's cluster check could not fail |
| B2 | Two thirds of the contamination count were runs of numbers | metric validity | a contamination figure more than four times too high, attributed to the wrong benchmarks |
| B3 | Validation too small for the effects it was meant to show | metric validity | noise of ±0.4% read as an effect of the mixture |
| C1 | The serving engine refused the trained checkpoint | systems | every probe would have stopped; no wrong number |
| D1 | Data provenance written from memory | selection & reporting | a wrong statement about who wrote the probe's training data |
| E1 | The stop marker was invisible to the server | eval protocol | every probe score graded on answers from turns the model invented |

Each incident uses the same template: **Symptom → The false story → Root cause → Fix → Permanent regression test →
Invalidated artifacts → Generalisable lesson.**

---

# A · Data integrity

## A1 — Semantic packs put editions of one text side by side

**Symptom.** In the first build of the semantic packs, 1.9% of the word 8-grams of a pack had already occurred
earlier in the same pack (4.9% for textbooks), and 9% of the packs held a document that was more than half repeated
from an earlier one. The random packs, built from the same documents, had 0.00%.

**The false story.** "Semantic packing teaches the model to use distant context: the loss on the end of long
documents falls more after semantic packs than after random packs." The model would have learnt to copy from the
context, a different and much easier skill, and the random arm offered almost no chance to do so.

**Root cause.** The same textbook section is published in several editions, and open course material is copied
between sites. For the similarity search a second edition is the most related document there is, so it was always
chosen next.

**Fix.** A candidate is skipped for a pack when more than 10% of its word 8-grams already occur in the pack; it stays
available for other packs (`document_packing.py`, `near_copy_shingle_share`). Repeated 8-grams fell to 0.27%, and
what remains are stock phrases ("by the end of this section you will be able to").

**Permanent regression test.** `tests/test_document_packing.py::test_no_pack_holds_a_near_copy_of_its_earlier_documents_and_second_editions_go_elsewhere`,
`::test_a_near_copy_is_passed_over_for_its_pack_and_packed_in_another`.

**Invalidated artifacts.** The first build of `semantic_16k` and `random_16k`; nothing had been trained on it.

**Generalisable lesson.** Grouping documents by similarity also groups their duplicates: measure repetition inside
each group, not only across the corpus.

## A2 — The last packs built were leftovers

**Symptom.** The mean similarity of neighbouring documents fell from 0.87 in the first tenth of packs to 0.65 in the
last tenth; 72 of the 85 packs below 0.5 were among the last 300 built. One pack read near the end held 50 short,
unrelated documents.

**The false story.** "Semantic packing helps a little." A tail of unrelated packs in the semantic arm moves it toward
the random arm and shrinks any difference between them.

**Root cause.** Greedy packing takes the best matches first. The documents left at the end are the ones that matched
nothing, and they still have to go somewhere.

**Fix.** 10% more documents are sampled than the budget needs; packs below 0.5 similarity and the last packs built
are dropped, and their documents leave both arms, so that the random arm still holds exactly the semantic arm's
documents. No pack is below 0.5 now; the first percentile is 0.577.

**Permanent regression test.** `tests/test_document_packing.py::test_the_tail_of_the_build_is_dropped_and_both_arms_hold_only_the_kept_documents`,
`::test_choosing_packs_drops_those_below_the_similarity_floor_and_then_the_last_built`.

**Invalidated artifacts.** The first build of both pack sets.

**Generalisable lesson.** A greedy procedure is good at its start and bad at its end; look at the quality of what it
produced last before reporting the average.

---

# B · Metric validity

## B1 — Density clustering found nothing in 384 dimensions

**Symptom.** The first diversity measurement of English Wikipedia reported 2 HDBSCAN clusters among 20,000
documents, with 91% of them labelled noise.

**The false story.** "The rewritten text passes the diversity gate: it shows at least half as many clusters as real
text." With a baseline of 1 cluster per 10,000 documents, half of it is 0.5, and any batch passes.

**Root cause.** Density-based clustering needs dense regions. In 384 dimensions the distances between random pairs
of documents are all about the same (the curse of dimensionality), so almost no point has enough close neighbours to
start a cluster.

**Fix.** HDBSCAN now runs on the first 16 principal directions of each sample. The gate also checks the Vendi score
(the exponential of the entropy of the eigenvalues of the similarity matrix), which needs no clustering setting:
on test vectors, 30 topics give 30 clusters and a score of 51, 3 topics give 3 clusters and a score of 7
(`data_hygiene.py`).

**Permanent regression test.** `tests/test_data_hygiene.py::test_clusters_and_vendi_score_separate_thirty_topics_from_three`,
`::test_the_gate_fails_a_collapsed_batch_on_the_vendi_score`.

**Invalidated artifacts.** The first Wikipedia baseline, deleted before any batch was compared with it.

**Generalisable lesson.** Before a check guards anything, show that it fails on an input that should fail.

## B2 — Two thirds of the contamination count were runs of numbers

**Symptom.** The 13-gram layer flagged 6,030 of 2,686,237 documents, among them 2,093 math pages that "matched
HumanEval".

**The false story.** "The web pool is contaminated with HumanEval and MBPP; code scores of models trained on it are
inflated."

**Root cause.** HumanEval and MBPP test cases contain long lists of numbers, and so does every table in a paper or a
math page. Any 13 consecutive numbers that happen to agree count as a match. Reading the matches: 54% are numbers
only, 12% mostly numbers, 10% runs of single letters ("a b c d e f g …"), and 24% words, where the real leaks are.

**Fix.** The gate now writes the breakdown by kind next to the count (`data_hygiene.py explain-flags`,
`results/hygiene_decontamination.json`). It still drops all flagged documents: that costs 0.2% of the pool.

**Permanent regression test.** `tests/test_data_hygiene.py::test_matched_ngrams_are_sorted_into_kinds`,
`::test_a_question_planted_in_an_unrelated_document_is_flagged_and_clean_text_is_not`.

**Invalidated artifacts.** None; the count is now reported with its breakdown.

**Generalisable lesson.** An n-gram overlap count measures overlap, not leakage; read a sample of the matches before
reporting the count.

## B3 — Validation too small for the effects it was meant to show

**Symptom.** After a pilot run in which 15% of the tokens were English Wikipedia, the loss on held-out Wikipedia was
0.4% worse.

**The false story.** "Training on Wikipedia makes the model worse on Wikipedia": a surprising finding that would
have invited an explanation.

**Root cause.** The pilot measured each source on 6 held-out sequences, 24,576 tokens; the loss moves by about 0.4%
between such samples. The effects the design has to resolve are of the same size.

**Fix.** 32 sequences per source, 131,072 tokens, the same for every run (`train_continue.py`,
`validation_sequences_per_source`).

**Permanent regression test.** None; the README reports the noise level next to every loss table.

**Invalidated artifacts.** None withdrawn; the pilot table states that differences below 0.5% are noise.

**Generalisable lesson.** Size the evaluation for the smallest effect you intend to report, before the first run.

---

# C · Systems

## C1 — The serving engine refused the trained checkpoint

**Symptom.** The inference server stopped at load time: "Invalid type of HuggingFace config … Qwen3_5TextConfig".

**The false story.** "Continued pretraining broke the checkpoint."

**Root cause.** Training loads only the text model, and saving it wrote a text-only configuration with the text
model's tensor names. The server's loader for this architecture expects the base checkpoint's full layout: the text
weights under `model.language_model.*`, plus the vision and multi-token-prediction weights and the full
configuration.

**Fix.** `save_checkpoint` writes the trained tensors under the base checkpoint's names and copies every other
tensor and file from the base. Verified by identical weights after reloading and by a served answer.

**Permanent regression test.** None automated; every probe starts the server on a saved checkpoint, so a regression
stops the first probe.

**Invalidated artifacts.** None; it was caught at the first probe.

**Generalisable lesson.** Save checkpoints in the format of the checkpoint you started from, not the format of the
class you trained.

---

# D · Selection & reporting

## D1 — Data provenance written from memory

**Symptom.** The probe's documentation said that Qwen models wrote its instruction data. The dataset card shows that
StarCoder2 wrote the code instructions and Llama 3.1 the everyday conversations.

**The false story.** "All model-written data in the probe comes from the same family as the model under test."

**Root cause.** The sentence was written from memory, before the dataset card was read.

**Fix.** The text now names the models the card names (`sft_probe.py`).

**Permanent regression test.** None; provenance statements are now checked against the source before they are
committed.

**Invalidated artifacts.** None.

**Generalisable lesson.** Write provenance with the dataset card open.

---

# E · Eval protocol

## E1 — The stop marker was invisible to the server

**Symptom.** After the fixed fine-tune, the base 0.8B model solved 11.0% of GSM8K questions per sample. The same
model without any fine-tune, asked with few-shot examples, solved 18.1%. Its probe answers were also 7 times longer
than its few-shot answers (about 1,540 characters against a median of 222).

**The false story.** "The fine-tune hurts single-sample accuracy and widens the spread of answers; pass@1 is low
and pass@64 is high, so the model has a lot of room for RL." The published table showed exactly that shape.

**Root cause.** The fine-tune teaches the model to end its turn with the special token `<|im_end|>`, and the probe
asked the server to stop at the text `<|im_end|>`. The server matches stop strings against the decoded text, and the
decoded text leaves special tokens out, so the stop never fired. The model's end-of-sequence token is
`<|endoftext|>`, which it rarely writes after a chat turn, so generation ran on into turns the model invented until
the 512-token limit, and the grader read the last `\boxed{}` it found there.

**Fix.** Generations stop at the token ids of `<|im_end|>` and `<|endoftext|>` (`sft_probe.end_token_ids`), and every
task records the share of answers cut at the token limit. The first base probe was withdrawn and rerun.

**Permanent regression test.** `tests/test_sft_probe.py::test_sampling_stops_at_the_end_tokens_by_id`,
`::test_the_fine_tune_teaches_the_end_of_turn_as_one_special_token_with_loss`.

**Invalidated artifacts.** The first probe of the base checkpoint (`results/probe/base_0.8B.json` of 11-20 17:21,
withdrawn from the README the same evening). No other probe had run.

**Generalisable lesson.** When a fine-tune makes a model worse at the thing it was fine-tuned on, suspect the
harness before the model: read the raw answers and their lengths.
