# Failure atlas

[English](PITFALLS.md) | [中文](PITFALLS.zh-CN.md)

How this pipeline silently fails, how each failure was detected, and what now prevents it from recurring.

Every entry is a mistake actually made here while building the continued-pretraining study, caught by measurement,
corrected, and where possible given a permanent regression test. Nothing below is a generic best-practice list.

**The organising observation: ten of the twelve incidents produced something plausible that pointed the wrong way,
or could have (a pack set that looked related, a diversity check that passed, a contamination count, a shift of the
loss, rewrites that read well, training data believed clean, a sentence about where data came from), and none of them
stopped the pipeline.** The two incidents that stopped it (C1, C2) were the cheapest to fix.

Companion document: [README.md](README.md) explains how the pipeline works and what the experiments showed.

---

## Index

| id | incident | category | cost if uncaught |
|---|---|---|---|
| A1 | Semantic packs put editions of one text side by side | data integrity | the long-context comparison would have measured copying, not the use of related text |
| A2 | The last packs built were leftovers | data integrity | the semantic arm would have carried a tail of random packs |
| A3 | Rewrites pointed at a passage their reader never sees | data integrity | 12,000 rewrites in a register no real text has, or a diversity gate loosened to let them in |
| A4 | Solutions to training problems were never checked against the test sets | data integrity | a gain on GSM8K or MATH-500 that came partly from test items |
| B1 | Density clustering found nothing in 384 dimensions | metric validity | the diversity gate's cluster check could not fail |
| B2 | Two thirds of the contamination count were runs of numbers | metric validity | a contamination figure more than four times too high, attributed to the wrong benchmarks |
| B3 | Validation too small for the effects it was meant to show | metric validity | noise of ±0.4% read as an effect of the mixture |
| B4 | A letter format reported a knowledge loss that was not there | metric validity | the mixture search would have avoided a mixture that cost nothing |
| C1 | The serving engine refused the trained checkpoint | systems | every probe would have stopped; no wrong number |
| C2 | The probe's server would not start on a shared GPU | systems | every 2B probe would have stopped; no wrong number |
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

## A3 — Rewrites pointed at a passage their reader never sees

**Symptom.** Two batches of teacher rewrites failed the diversity gate on the Vendi score of their embeddings: v1 at
0.70 and 0.59 of the real passages' score at full size, v2 at 0.63 and 0.59 on a sample of 300 rewrites per source,
against a limit of 0.80. Each rewrite read well on its own.

**The false story.** "One teacher writing for one reader narrows the text; vary the reader, the voice, the format."
v2 drew a reader and a writer's voice for every request, and it scored lower than v1.

**Root cause.** The topics had survived: with each set's mean direction removed, the rewrites were 92% as spread as
their parents. What they shared was one direction, the register of talking about a text, and its most frequent mark
was a reference to the source itself: 83 of 144 rewrites of the plain request mention "the passage", "the text" or
"the author" in their first 400 words. The request presented the source as "Passage:", and the teacher kept pointing
at it. The instruction not to mention the passage was checked only against the first words of a rewrite.

**Fix.** The v3 request asks the teacher to write as the author of the material, never referring to a source, and to
keep the material's own names, places, dates and numbers. A rewrite that still points at a hidden source ("The text
states that...") is dropped (`hidden_source_reference` in `synthetic_rephrase.py`). On the same 150 passages the
Vendi ratio rose from 0.73 to 0.86, and the v3 sample passed the gate on both sources (README §7.3).

**Permanent regression test.** `tests/test_synthetic_rephrase.py::test_rewrites_that_point_at_a_hidden_source_are_dropped`,
`::test_ordinary_uses_of_text_passage_and_author_are_kept`.

**Invalidated artifacts.** The v1 and v2 batches (12,000 and 600 rewrites); neither entered a training run.

**Generalisable lesson.** When generated text fails a diversity check, find the direction the documents share before
adding variety: one repeated phrase can outweigh every variation added on top of it.

---

## A4 — Solutions to training problems were never checked against the test sets

**Symptom.** None in the scores. When two new test sets were gated on 11-23, the teacher's worked solutions were
checked against every evaluated item for the first time, after the first runs had trained on them: 433 of the 45,177
solutions share a 13-gram with a test item.

**The false story.** "The solutions answer training problems, and the solving step never reads a test split, so they
cannot hold a test item." The publishing step was built on that belief and wrote the solutions into the pool without
the gate that every other source passes.

**Root cause.** A training split is not clean against its own test split. Three GSM8K test problems have a near-copy
among the GSM8K training problems (16 solutions matched them). MATH problems of one kind share fixed wording across
splits ("convert the point to polar coordinates; enter your answer in the form"): 377 solutions matched 51 MATH-500
items that way. Another 44 matched MMLU-Pro, HumanEval or MBPP items through runs of numbers or common phrases.

**Fix.** The solutions were gated after the fact (`data_hygiene decontaminate --pool pool/synthetic_solutions`), and
every claim about them was recomputed without the matched test items. On the 1,316 GSM8K problems that no solution
matches, the gain of the distillation run is unchanged, +6.8 [+4.3, +9.2] (README §1.2). The flagged solutions stay
in the source, so that the 0.8B, 2B, 4B and 9B runs of the recipe see the same data. The second problem set
(NuminaMath-1.5) passes the gate, by 13-grams and by embeddings, before the teacher sees a problem
(`synthetic_rephrase.py select-problems`).

**Permanent regression test.** `select_numina_problems` runs `decontaminate_pool` before it writes the problems and
records the counts in `results/synthetic_solutions_numina_selection.json`. The first set has no test of its own; its
gate result is in README §1.2.

**Invalidated artifacts.** None. The GSM8K gain stands on the unmatched problems. The MATH-500 scores of the
distillation runs include the 51 matched items; their gain is within noise with or without them.

**Generalisable lesson.** Gate every source that enters training, the ones made from training splits included: "train
split" says where a problem came from, not what it overlaps.

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

## B4 — A letter format reported a knowledge loss that was not there

**Symptom.** After the first design run, zero-shot lettered HellaSwag fell 6.4 points [−9.3, −3.4] against the base
model, all of it on the WikiHow-style items (49.7 → 40.2; the ActivityNet items stayed at 51.2).

**The false story.** "This mixture costs commonsense knowledge; the replay share has to go up." The mixture search
would have pushed away from the run.

**Root cause.** Small models answer lettered questions partly by preferring some letters, and that preference moves
when the model is trained on anything. The same questions asked with five examples moved by −0.5 points, and scored
by cloze, the likelihood of each answer's own text with no letters, by +0.1: 38 of 1,000 choices changed, half each
way. The 0.8B base already picked "B" for 33 of 80 ARC questions after five examples, against 27 without.

**Fix.** The regression suite also scores every multiple-choice set by cloze (`base_regression.py`), and the guard of
the mixture search is the cloze mean (`mixing_law.py`). Lettered scores are still reported.

**Permanent regression test.** None automated; the README reports both formats next to each other.

**Invalidated artifacts.** The guard definition chosen earlier the same day (the mean of both lettered formats); no
analysis had used it.

**Generalisable lesson.** For a small model, a multiple-choice score can move with the answer format alone: score
the answer text, and treat a drop that appears in one format only as a format effect until shown otherwise.

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

## C2 — The probe's server would not start on a shared GPU

**Symptom.** The 2B smoke run trained normally, then its probe stopped: "Free memory on device cuda:0 (30.86/39.49
GiB) on startup is less than desired GPU memory utilization (0.8, 31.59 GiB)". The same probe had run on the 0.8B
model many times.

**The false story.** "The 2B model is too large for the probe's server on a 40 GB GPU."

**Root cause.** The server's memory share is a share of the whole GPU, and all of it must be free when the server
starts. Another user's job held 3 GiB of the GPU, and the probe's own process still held about 5 GiB after the
fine-tune: a reference to the output layer, the last batch's hidden states and the allocator's cache. The 0.8B
model's leftovers are smaller, so its probe fitted by a small margin.

**Fix.** The fine-tune drops every reference to the model and empties the allocator's cache before the server
starts. When the server still finds too little free memory, it is started again with the free share less 1 GiB
(`start_inference_server` in `sft_probe.py`).

**Permanent regression test.** `test_server_starts_again_with_the_free_share_and_then_with_the_block_limit` feeds
the server's two start-up complaints to the start function and checks the shares and limits it retries with.

**Invalidated artifacts.** None; it was caught in the smoke run, before any 2B design run.

**Generalisable lesson.** On a GPU that others share, size a memory share from what is free at start, and free your
own leftovers before you hand the GPU to a second process.

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
