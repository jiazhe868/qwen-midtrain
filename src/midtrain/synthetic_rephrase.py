"""Write synthetic training text with a teacher model: structured rephrasings and question-answer derivations of
real, high-value passages.

Why this file exists. Good public web text is finite, and its reasoning density has stopped rising. Generating
text from nothing with a language model is the obvious way out and the dangerous one: a model trained on a model's
free output drifts toward a narrower distribution (model collapse). The safer middle, used by WRAP, Nemotron-CC and
REWIRE, keeps a real passage as the source of facts and asks the teacher only to change its form. This file does
that in six forms, so that the same content is seen from several sides:
    explain_from_first_principles   a derivation that connects the passage's claims
    structured_notes                definitions, relations, a procedure, common mistakes
    question_answer_derivation      questions that need reasoning over the passage, with worked answers
    worked_problem                  exercises made from the content, solved step by step
    socratic_dialogue               a teacher and a student reach the conclusions, with one wrong turn corrected
    what_if_analysis                variations of the situation, reasoned through
The first two and the dialogue go to the pool source `synthetic_rephrase`, the other three to `synthetic_qa`. Outputs are plain
text: no chat tokens, no "here is the rewritten text" (chat special tokens in mid-training data took one published
model's GSM8K score from 49 to 0, arXiv 2512.13961).

Checks on every output, with the reason counted when one is dropped: length, a refusal or meta-comment, and the
share of its 8-word sequences copied from the source (a copy is not a rephrasing; a real passage would be cheaper).
The whole batch then has to pass the diversity gate of data_hygiene.py before it may enter a mixture.

History. The first batch (v1, 11-20: 12,000 rewrites at temperature 0.8, one or two styles per passage, every request
worded the same way) failed the diversity gate on two measures. The Vendi score of its embeddings (the effective
number of distinct documents) fell to 22.1 (rephrase) and 18.6 (qa) against 31.4 for the real passages, below the
limit of 25.1; its bigram entropy fell to 15.97 and 15.64 bits against 16.54, below the limit of 16.04. One rewrite
per parent still gave 22.2 and 18.7, so the narrowing comes from the rewriting itself: one teacher writes every
passage for the same reader in the same voice. The second batch (v2) changes the request: temperature 1.0, a reader
and a writer's voice drawn at random for each request (12 of each), and one style per passage, so that every rewrite
has a parent of its own. A 600-rewrite sample has to pass the gate against its own parents (`check-sample`) before
the full batch is written. Each batch lives in its own folder, synthetic/<batch>.

The v2 sample failed by more than v1 (Vendi ratio 0.63 and 0.60 at 300 rewrites per source). Two measurements found
why. With each set's mean direction removed, the rewrites are 92% as spread as their parents, so the topics survive;
what they share is one direction (the mean unit embedding grows from 0.70 to 0.76), and the words along it are those of
talking about a text: "passage", "text", "explicitly", "therefore". 58% of the rewrites of a 150-passage test
referred to "the passage", "the text" or "the author" in their first 400 words, a passage their reader never sees.
Of seven request wordings tried on the same 150 passages, asking the teacher to write as the author of the material,
never referring to a source, and to keep the material's own names and numbers, worked best (Vendi ratio 0.86 against
0.73 for the plain request; references to the source in 23 of 146 rewrites against 83 of 144). That is v3, and the
remaining references ("The text states that...") are now a reason to drop a rewrite.

Commands (after `source scripts/env.sh`; the teacher is served by any OpenAI-compatible server; --batch defaults to
the current batch, v3):
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase select --passage-count 14000
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase serve --gpu 1 --port 8201 &         # one teacher per GPU
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase generate --endpoint http://127.0.0.1:8201 [--endpoint ...] \
        --maximum-rewrites 600
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase check-sample --documents-per-source 300 --device cpu
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase publish            # outputs → pool rows of the two sources
    # worked solutions by the teacher: the training problems of GSM8K and MATH, then a second, wider problem set
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase solve --endpoint http://127.0.0.1:8202 --samples-per-problem 8
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase publish-solutions --maximum-per-problem 4
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase select-problems      # NuminaMath-1.5, through the gate
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase solve --problem-set numina --endpoint ... --samples-per-problem 2
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase publish-solutions --problem-set numina --maximum-per-problem 2
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from midtrain.data_sources import (data_root, document_id_for_text, pool_schema, repository_root, tokenizer_path,
                                   write_pool_rows)

# share of the passages taken from each real source: encyclopedic and textbook content first, as the project asks
passage_share_by_source = {"wiki_en": 0.30, "textbooks": 0.25, "papers": 0.15, "math_web": 0.15, "forum_qa": 0.15}
passage_token_range = (350, 1200)        # long enough to hold an argument, short enough for a 700-token rewrite
# Three styles restate the content (rephrase), three turn it into questions with worked answers (qa). An even split
# matters for cost: the screening design draws equal tokens from the two sources (up to 4.5M each in its run with the
# most rewritten text), and two teachers on shared GPUs write about 3,000 rewrites an hour (11-20).
pool_source_of_style = {"explain_from_first_principles": "synthetic_rephrase", "structured_notes": "synthetic_rephrase",
                        "socratic_dialogue": "synthetic_rephrase", "question_answer_derivation": "synthetic_qa",
                        "worked_problem": "synthetic_qa", "what_if_analysis": "synthetic_qa"}
shared_instruction = (
    "Write in plain text. Do not mention that you were given a passage, and do not address the reader about the task. "
    "Start directly with the content. Keep every fact consistent with the passage and add no facts of your own beyond "
    "common knowledge needed for the reasoning. Aim for 450 to 800 words.")
style_instructions = {
    "explain_from_first_principles":
        "Rewrite the passage as a clear explanation that builds its main claims step by step from what a careful "
        "student already knows. Make each step of reasoning explicit: say why one statement follows from another.",
    "structured_notes":
        "Turn the passage into dense study notes: the key definitions, the relations between the quantities or ideas, "
        "one procedure or argument written as numbered steps, and two mistakes a learner is likely to make, each with "
        "the reason it is wrong.",
    "question_answer_derivation":
        "Write four questions that can only be answered by reasoning over the passage, not by copying a sentence from "
        "it. After each question give a worked answer that shows the steps and ends with the conclusion.",
    "worked_problem":
        "Create two exercises from the content of the passage, one easier and one harder. State each exercise fully, "
        "then solve it step by step, checking the result at the end. For content without numbers, make the exercises "
        "case analyses with a justified conclusion.",
    "socratic_dialogue":
        "Write a dialogue in which a teacher leads a student to the passage's conclusions by asking questions. The "
        "student should make one plausible mistake on the way; the teacher asks a question that exposes it, and the "
        "student corrects it and explains what went wrong.",
    "what_if_analysis":
        "Pose three what-if variations of the situation in the passage (a changed condition, a removed assumption, an "
        "extreme case) and reason through what would follow in each, step by step, consistent with the passage.",
}
# Three styles reworded for v3 (11-21). At 600 rewrites per source the qa source missed the bigram-entropy limit by
# 0.01 bits. Per style, against each style's own parents on 100,000 tokens, the dialogue lost 0.61 bits, the questions
# 0.62 and the worked problems 0.57, against 0.09 to 0.26 for the other three (`check-styles`,
# results/synthetic_style_check_v3_rewrites.json). Their fixed labels ("Teacher:", "Question", "Step 1", "Solution",
# "Check") repeat in every rewrite. The reworded styles ask for the same reasoning without fixed labels. The questions
# improved most (0.45 to 0.10 bits on 45,000 tokens); the named-role dialogue did not (Vendi ratio 0.66) and is left
# out of v3 with the original dialogue.
style_instructions.update({
    "dialogue_with_named_roles":
        "Write a dialogue in which a person who knows this material leads a learner to its conclusions by asking "
        "questions. Give both speakers names and roles that fit the material, and write their lines as natural speech. "
        "The learner makes one plausible mistake on the way; a question exposes it, and the learner corrects it and "
        "explains what went wrong.",
    "questions_answered_in_prose":
        "Pose four questions that can only be answered by reasoning about this material, not by recalling one sentence "
        "of it, and answer each in connected prose that shows the steps and ends with the conclusion. Vary how the "
        "questions are introduced, and do not use labels such as 'Question', 'Answer' or 'Step'. Each answer states the "
        "facts it uses.",
    "worked_problems_in_prose":
        "Create two exercises from this material, one easier and one harder. State each exercise fully, then solve it in "
        "connected prose and check the result at the end. For material without numbers, make the exercises case "
        "analyses with a justified conclusion. Do not use labels such as 'Step 1', 'Solution' or 'Check'; state every "
        "fact the solution uses.",
})
pool_source_of_style.update({"dialogue_with_named_roles": "synthetic_rephrase",
                             "questions_answered_in_prose": "synthetic_qa", "worked_problems_in_prose": "synthetic_qa"})
style_names = list(style_instructions)
# One reader and one writer's voice are drawn for every request (v2). The first batch wrote every rewrite for the same
# reader in the same voice and failed the diversity gate (see the history above). Both lists keep to readers and
# voices for which an explanation stays faithful to the passage.
reader_descriptions = [
    "a curious high-school student", "a first-year university student", "a graduate student from a neighbouring field",
    "a working engineer who wants the practical point", "a teacher preparing a lesson",
    "a reader of a popular-science magazine", "a student revising for an exam",
    "a programmer who likes precise definitions", "a retired professional with time and curiosity",
    "an olympiad competitor who wants rigour", "a journalist checking the facts",
    "a researcher who needs a quick refresher"]
writer_voices = [
    "a patient tutor", "a textbook author", "a laboratory scientist", "a historian of ideas", "a practical engineer",
    "a puzzle setter", "a careful reviewer who checks every step", "a popular-science columnist",
    "a mathematician who values rigour", "a technical writer", "a coach who explains with everyday examples",
    "a lecturer giving an evening class"]
# v3: the teacher writes as the author of the material, never pointing at a source, and keeps the material's own
# specifics (see the history above). v2 batches keep their reader and voice; v1 batches have neither.
own_voice_instruction = (
    "Write as the author of a document that presents this material in your own words. Never refer to a passage, a "
    "text, a source, an excerpt or an author. Keep the material's own names, places, dates, numbers and examples, and "
    "open with them.")
current_batch_name = "v3"
current_framing = "own_voice_specific"
# Phrases that point the reader at a source they never see. "The text states that..." survived the v3 wording in 12
# of 146 test rewrites, mostly in worked answers; such a rewrite is dropped. "The passage of" is left alone: the
# passage of a law or of time.
hidden_source_reference = re.compile(
    r"\b(?:the|this) (?:passage|excerpt)\b(?! of\b)|\bthe (?:given|provided|original|source) text\b|\bthe source material\b|"
    r"\bthe text (?:states|mentions|notes|says|explicitly|suggests|lists|confirms|specifies|emphasizes|presents|"
    r"describes|indicates|shows|provides|gives|reports|implies|clarifies|highlights|defines)\b|"
    r"\b(?:in|from|by) the text\b", re.IGNORECASE)
refusal_or_meta_patterns = re.compile(
    r"^(here is|here's|sure[,!]|certainly|i cannot|i can't|i'm sorry|as an ai|below is|the passage|this passage|"
    r"rewritten|okay[,.]|of course)", re.IGNORECASE)
maximum_copied_eight_gram_share = 0.5


def synthetic_directory(batch_name: str) -> Path:
    """synthetic/<batch>: the passages, the rewrites and the logs of one batch."""
    return data_root() / "synthetic" / batch_name


def read_json_lines(file_path: Path) -> list[dict]:
    """One record per line, splitting at newlines only. str.splitlines() also splits at U+2028, U+2029 and U+0085,
    which web text contains and json.dumps(ensure_ascii=False) leaves unescaped: the passages file of 60,000 lines
    came apart into 60,073 pieces and the first generation stopped on a broken record (11-20)."""
    with open(file_path, encoding="utf-8") as json_lines_file:
        return [json.loads(line) for line in json_lines_file if line.strip()]


# ---------------------------------------------------------------------------------------------------------------
# select: which real passages get rewritten, and in which styles
# ---------------------------------------------------------------------------------------------------------------
def cut_passage_at_paragraphs(text: str, token_count: int) -> str | None:
    """A window of whole paragraphs from the start of a document whose length suits a rewrite, or None."""
    characters_per_token = len(text) / max(1, token_count)
    low_characters = passage_token_range[0] * characters_per_token
    high_characters = passage_token_range[1] * characters_per_token
    if len(text) < low_characters:
        return None
    paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
    window = ""
    for paragraph in paragraphs:
        if len(window) + len(paragraph) > high_characters:
            break
        window += paragraph + "\n\n"
    window = window.strip()
    if len(window) < low_characters:           # one huge paragraph: cut it at a sentence end instead
        window = text[:int(high_characters)]
        last_sentence_end = max(window.rfind(". "), window.rfind(".\n"))
        window = window[:last_sentence_end + 1] if last_sentence_end > low_characters else window
    return window if len(window) >= low_characters else None


def select_passages(batch_name: str, passage_count: int, seed: int, style_subset: list[str] | None = None,
                    should_append: bool = False) -> None:
    """Pick the passages and give each one style and the current framing of the request; writes
    synthetic/<batch>/passages.jsonl.

    The styles are dealt in turn over the shuffled passages, so each style gets an equal share and, with all six, the
    two pool sources get equal numbers of rewrites. (The first batch gave half of the passages two styles; with one
    style per passage every rewrite has a parent of its own, which matters for diversity, see the history above.)

    `should_append` adds new passages to an existing batch, none of them a parent already in it, dealt among
    `style_subset` only. It exists because the styles do not keep the same share of their rewrites: in the v3 sample
    the question-answer styles lost 37% of theirs to the hidden-source check and the rephrase styles 12%, so the qa
    source needs more passages for the same number of tokens.
    """
    from midtrain.mixture import load_decontamination_gate

    random_generator = np.random.default_rng(seed)
    synthetic_directory(batch_name).mkdir(parents=True, exist_ok=True)
    # A rewrite of a passage that the decontamination gate flagged would carry the evaluation item along in other
    # words, where no n-gram check finds it again. (The first batch chose its passages before this check existed.)
    flagged_document_ids = load_decontamination_gate()["flagged_document_ids"]
    passages_path = synthetic_directory(batch_name) / "passages.jsonl"
    existing_passages = read_json_lines(passages_path) if should_append else []
    parents_already_taken = {passage["parent_doc_id"] for passage in existing_passages}
    styles_to_deal = style_subset or style_names[:6]
    assert all(style_name in style_instructions for style_name in styles_to_deal), f"unknown style in {styles_to_deal}"
    passages = []
    for source_name, share in passage_share_by_source.items():
        wanted = int(round(share * passage_count))
        flagged_in_source = flagged_document_ids.get(source_name, set())
        pool_files = sorted((data_root() / "pool" / source_name).glob("part-*.parquet"))
        rows = []
        for pool_file in pool_files:
            table = pq.read_table(pool_file, columns=["doc_id", "text", "token_count"])
            rows.extend(zip(table.column("doc_id").to_pylist(), table.column("text").to_pylist(),
                            table.column("token_count").to_pylist()))
            if len(rows) >= 6 * wanted:
                break
        order = random_generator.permutation(len(rows))
        kept = 0
        for position in order:
            document_id, text, token_count = rows[position]
            if document_id in flagged_in_source or document_id in parents_already_taken:
                continue
            passage_text = cut_passage_at_paragraphs(text, token_count)
            if passage_text is None:
                continue
            passages.append({"parent_doc_id": document_id, "parent_source": source_name, "passage": passage_text})
            kept += 1
            if kept >= wanted:
                break
        print(f"{source_name}: {kept} passages", flush=True)
    random_generator.shuffle(passages)
    for passage_index, passage in enumerate(passages):
        passage["styles"] = [styles_to_deal[passage_index % len(styles_to_deal)]]
        passage["framing"] = current_framing
    with passages_path.open("a" if should_append else "w") as passage_file:
        for passage in passages:
            passage_file.write(json.dumps(passage, ensure_ascii=False) + "\n")
    print(f"{len(passages)} passages {'added' if should_append else 'selected'}, "
          f"{len(existing_passages) + len(passages)} in the batch")


# ---------------------------------------------------------------------------------------------------------------
# serve: one teacher per GPU
# ---------------------------------------------------------------------------------------------------------------
def serve_teacher(gpu_index: int, port: int) -> None:
    """Run the teacher on one GPU until killed. The batch limits are the ones a 27B teacher needs on a 40 GB GPU:
    at the server's defaults its warm-up pass alone ran out of memory (11-19)."""
    teacher_path = os.environ["MIDTRAIN_TEACHER"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", teacher_path, "--served-model-name", "teacher",
               "--port", str(port), "--max-model-len", "6144", "--gpu-memory-utilization", "0.85",
               "--max-num-batched-tokens", "2048", "--max-num-seqs", "160", "--mamba-ssm-cache-dtype", "float32",
               "--gdn-prefill-backend", "triton", "--language-model-only", "--no-enable-prefix-caching", "--seed", "0"]
    os.execvpe(command[0], command, environment)


# ---------------------------------------------------------------------------------------------------------------
# generate: ask the teacher, check each output, append to a results file that a restart can continue
# ---------------------------------------------------------------------------------------------------------------
def rewrite_key(parent_document_id: str, style_name: str) -> str:
    return f"{parent_document_id}|{style_name}"


def copied_eight_gram_share(output_text: str, source_text: str) -> float:
    """Share of the output's 8-word sequences that also occur in the source passage."""
    def eight_grams(text: str) -> set:
        words = re.findall(r"\w+", text.lower())
        return {" ".join(words[index:index + 8]) for index in range(len(words) - 7)}
    output_grams = eight_grams(output_text)
    return len(output_grams & eight_grams(source_text)) / max(1, len(output_grams))


def reason_to_drop(output_text: str, source_text: str, finish_reason: str) -> str | None:
    word_count = len(output_text.split())
    if finish_reason != "stop":
        return "cut_off"
    if word_count < 200:
        return "too_short"
    if refusal_or_meta_patterns.match(output_text.strip()):
        return "refusal_or_meta_comment"
    if "<|im_" in output_text or "<think>" in output_text:
        return "chat_tokens"
    if copied_eight_gram_share(output_text, source_text) > maximum_copied_eight_gram_share:
        return "mostly_copied"
    if hidden_source_reference.search(output_text):
        return "refers_to_hidden_source"
    return None


def build_request_text(passage: dict, style_name: str) -> str:
    """The user message of one rewrite: the framing of its batch, the style, the shared rules, then the passage.
    v3 passages carry framing "own_voice_specific"; v2 passages a reader and a voice; v1 passages neither."""
    framing = ""
    if passage.get("framing") == "own_voice_specific":
        framing = own_voice_instruction + " "
    elif passage.get("reader"):
        framing = f"Write for {passage['reader']}, in the voice of {passage['writer_voice']}. "
    return f"{framing}{style_instructions[style_name]} {shared_instruction}\n\nPassage:\n{passage['passage']}"


def ask_teacher(endpoint: str, passage: dict, style_name: str, temperature: float) -> tuple[str, str]:
    request_body = {
        "model": "teacher",
        "messages": [{"role": "user", "content": build_request_text(passage, style_name)}],
        "max_tokens": 1800, "temperature": temperature, "top_p": 0.95,   # 1,500 cut 9% of a 600-rewrite sample, mostly dialogues
        "chat_template_kwargs": {"enable_thinking": False},      # the rewrite needs no hidden reasoning; 3x faster
    }
    request = urllib.request.Request(f"{endpoint}/v1/chat/completions", json.dumps(request_body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    response = json.load(urllib.request.urlopen(request, timeout=1800))
    choice = response["choices"][0]
    return (choice["message"].get("content") or "").strip(), choice.get("finish_reason") or ""


def generate_rewrites(batch_name: str, endpoints: list[str], requests_in_flight_per_endpoint: int,
                      stop_after_hours: float, temperature: float, maximum_rewrites: int | None,
                      style_filter: list[str] | None = None, results_name: str = "rewrites.jsonl") -> None:
    """Ask the teachers for every planned rewrite of the batch not yet in its results file, and append each result.

    `style_filter` limits the work to some styles. `results_name` writes to another file of the batch folder: the
    data folder is on NFS, where two processes appending to one file can overwrite each other's lines, so a second
    generator writes its own file, and the files are joined when neither is running.
    """
    passages = read_json_lines(synthetic_directory(batch_name) / "passages.jsonl")
    results_path = synthetic_directory(batch_name) / results_name
    finished_keys = set()
    if results_path.exists():
        finished_keys = {record["key"] for record in read_json_lines(results_path)}
    todo = [(passage, style_name) for passage in passages for style_name in passage["styles"]
            if rewrite_key(passage["parent_doc_id"], style_name) not in finished_keys
            and (style_filter is None or style_name in style_filter)]
    if maximum_rewrites is not None:
        # the passages are in a seeded random order, so the first ones are a fair sample of all of them
        todo = todo[:max(0, maximum_rewrites - len(finished_keys))]
    print(f"{len(finished_keys)} rewrites already done, {len(todo)} to do on {len(endpoints)} teacher(s)", flush=True)

    write_lock = threading.Lock()
    deadline = time.time() + stop_after_hours * 3600
    counts = {"kept": 0, "dropped": 0, "failed": 0, "output_words": 0}
    started = time.time()

    def do_one(job_index: int) -> None:
        if time.time() > deadline:
            return
        passage, style_name = todo[job_index]
        endpoint = endpoints[job_index % len(endpoints)]
        try:
            output_text, finish_reason = ask_teacher(endpoint, passage, style_name, temperature)
        except OSError:
            with write_lock:
                counts["failed"] += 1
            return
        drop_reason = reason_to_drop(output_text, passage["passage"], finish_reason)
        record = {"key": rewrite_key(passage["parent_doc_id"], style_name), "parent_doc_id": passage["parent_doc_id"],
                  "parent_source": passage["parent_source"], "style": style_name, "framing": passage.get("framing"),
                  "reader": passage.get("reader"), "writer_voice": passage.get("writer_voice"), "temperature": temperature,
                  "drop_reason": drop_reason,
                  "copied_eight_gram_share": round(copied_eight_gram_share(output_text, passage["passage"]), 3),
                  "text": output_text}
        with write_lock:
            with results_path.open("a") as results_file:
                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["dropped" if drop_reason else "kept"] += 1
            counts["output_words"] += len(output_text.split())
            done = counts["kept"] + counts["dropped"]
            if done % 500 == 0:
                minutes = (time.time() - started) / 60
                print(f"{done} rewrites in {minutes:.0f} min ({counts['kept']} kept, {counts['dropped']} dropped, "
                      f"{counts['failed']} failed requests, {counts['output_words'] / max(1e-9, minutes):.0f} words/min)",
                      flush=True)

    with concurrent.futures.ThreadPoolExecutor(requests_in_flight_per_endpoint * len(endpoints)) as request_pool:
        list(request_pool.map(do_one, range(len(todo))))
    print(json.dumps(counts), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# publish: kept rewrites become pool rows of the two synthetic sources
# ---------------------------------------------------------------------------------------------------------------
def publish_rewrites_to_pool(batch_name: str, styles_left_out: list[str] | None = None) -> None:
    """Write the kept rewrites of one batch as the pool rows of the two sources, replacing whatever batch was there.
    `styles_left_out` names styles whose rewrites stay out (v3: the three styles that were reworded)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    records = [record for record in read_json_lines(synthetic_directory(batch_name) / "rewrites.jsonl")
               if record["style"] not in (styles_left_out or [])]
    drop_counts: dict[str, int] = {}
    rows_by_source: dict[str, list] = {"synthetic_rephrase": [], "synthetic_qa": []}
    seen_text_hashes = set()
    for record in records:
        if record["drop_reason"]:
            drop_counts[record["drop_reason"]] = drop_counts.get(record["drop_reason"], 0) + 1
            continue
        text_hash = hashlib.sha1(record["text"].encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            drop_counts["exact_duplicate"] = drop_counts.get("exact_duplicate", 0) + 1
            continue
        seen_text_hashes.add(text_hash)
        rows_by_source[pool_source_of_style[record["style"]]].append(record)
    summary = {"batch": batch_name, "styles_left_out": styles_left_out or [], "rewrites": len(records),
               "dropped_by_reason": drop_counts, "sources": {}}
    for source_name, source_records in rows_by_source.items():
        texts = [record["text"] for record in source_records]
        token_counts = [len(token_ids) for token_ids in tokenizer(texts, add_special_tokens=False)["input_ids"]] if texts else []
        metas = [{"style": record["style"], "parent_doc_id": record["parent_doc_id"], "parent_source": record["parent_source"],
                  "copied_eight_gram_share": record["copied_eight_gram_share"], "batch": batch_name,
                  "framing": record.get("framing"), "reader": record.get("reader"), "writer_voice": record.get("writer_voice"),
                  "generator": Path(os.environ.get("MIDTRAIN_TEACHER", "teacher")).name} for record in source_records]
        write_pool_rows(source_name, [document_id_for_text(source_name, text) for text in texts], texts, token_counts, metas)
        by_style: dict[str, int] = {}
        for record in source_records:
            by_style[record["style"]] = by_style.get(record["style"], 0) + 1
        summary["sources"][source_name] = {"documents": len(texts), "tokens": int(sum(token_counts)), "by_style": by_style}
    (repository_root / "results" / f"synthetic_summary_{batch_name}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print("next: `python -m midtrain.data_sources tokenize --source synthetic_rephrase synthetic_qa`, then the hygiene gate")


# ---------------------------------------------------------------------------------------------------------------
# solve: worked solutions by the teacher for training-split math problems, kept only when the answer is right
# ---------------------------------------------------------------------------------------------------------------
# Why. Rewrites keep a passage's facts and change its form; they did not make the 0.8B model better at mathematics
# at 300M tokens (README §1.2). A stronger model's worked solutions to problems with known answers carry the skill
# itself: the teacher (Qwen3.8-27B) solves the training problems of GSM8K and MATH (both MIT), several times each, and
# only solutions whose final answer the project's graders accept are kept. The test splits are never read.
solution_instruction = (
    "Solve the problem step by step. Write each step as a short sentence or an equation, check the result, and end "
    "with the sentence 'The final answer is \\boxed{...}.'")
math_train_subjects = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                       "prealgebra", "precalculus")
# The second problem set (11-23). The first covers two distributions, GSM8K's and MATH's, and at 0.8B its gain was
# largest on GSM8K (README §1.2). Word problems from three other sources of NuminaMath-1.5 (Apache-2.0) widen it:
# orca_math (grade-school word problems written by a model), cn_k12 (Chinese school exercises, in English) and
# synthetic_math (new competition-style problems). Only problems that NuminaMath marks valid, with a valid solution and
# a short final answer, are drawn in a seeded order, and each passes the decontamination gate (13-grams and
# embeddings against every evaluation item) before the teacher sees it. Two attempts per problem and at most two kept
# solutions per problem: more problems and fewer copies of each than in the first set.
numina_revision = "1b05109f9e5c1ad06c0663519502416c30b300f8"
numina_problem_count_by_source = {"orca_math": 7000, "cn_k12": 6000, "synthetic_math": 7000}
solution_problem_sets = {"training_splits": {"folder": "solutions", "pool_source": "synthetic_solutions",
                                             "summary": "synthetic_solutions_summary.json"},
                         "numina": {"folder": "solutions_numina", "pool_source": "synthetic_solutions_numina",
                                    "summary": "synthetic_solutions_numina_summary.json"}}


def load_training_problems() -> list[dict]:
    """GSM8K and MATH training problems with their reference answers; figures in Asymptote code are left out."""
    from datasets import load_dataset

    from midtrain.eval_tasks import find_last_boxed_content

    problems = []
    for row_index, row in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
        problems.append({"problem_id": f"gsm8k_train:{row_index}", "task": "gsm8k", "problem": row["question"].strip(),
                         "reference_answer": row["answer"].partition("####")[2].strip().replace(",", "")})
    for subject in math_train_subjects:
        for row_index, row in enumerate(load_dataset("EleutherAI/hendrycks_math", subject, split="train")):
            reference = find_last_boxed_content(row["solution"])
            if reference and "[asy]" not in row["problem"]:
                problems.append({"problem_id": f"math_train:{subject}:{row_index}", "task": "math",
                                 "problem": row["problem"].strip(), "reference_answer": reference.strip(),
                                 "level": row["level"]})
    return problems


def numina_answer_is_checkable(answer_text: str) -> bool:
    """Whether a NuminaMath final answer is one short expression the graders can compare: not a list of values, not
    two cases joined by "or", not a sentence. Thousands separators (1,250) are allowed."""
    answer_text = (answer_text or "").strip()
    if not answer_text or len(answer_text) > 40 or "\n" in answer_text:
        return False
    if re.search(r";|\bor\b|\band\b|\\text\{\s*or|\\quad|\\pm", answer_text):
        return False
    if "," in answer_text and not re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", answer_text):
        return False
    return True


def select_numina_problems(seed: int) -> dict:
    """Draw the second problem set and pass it through the decontamination gate.

    Writes synthetic/solutions_numina/problems.jsonl and results/synthetic_solutions_numina_selection.json. 10% more
    problems than needed are drawn from each source, so that the gate can remove some and the counts still hold.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from midtrain import data_hygiene

    download_folder = data_root() / "raw" / "numinamath_1_5"
    snapshot_download("AI-MO/NuminaMath-1.5", repo_type="dataset", revision=numina_revision, local_dir=download_folder,
                      allow_patterns=["data/*.parquet"])
    columns = ["problem", "answer", "question_type", "problem_is_valid", "solution_is_valid", "source"]
    rows = [row for parquet_path in sorted((download_folder / "data").glob("*.parquet"))
            for row in pq.read_table(parquet_path, columns=columns).to_pylist()]
    candidates_by_source = {source_name: [] for source_name in numina_problem_count_by_source}
    for row_index, row in enumerate(rows):
        is_usable = (row["source"] in candidates_by_source and row["question_type"] == "math-word-problem"
                     and row["problem_is_valid"] == "Yes" and row["solution_is_valid"] == "Yes"
                     and numina_answer_is_checkable(row["answer"]) and "[asy]" not in row["problem"]
                     and len(row["problem"]) <= 1500)
        if is_usable:
            candidates_by_source[row["source"]].append({
                "problem_id": f"numina:{row['source']}:{row_index}", "task": row["source"],
                "problem": row["problem"].strip(), "reference_answer": row["answer"].strip()})
    random_generator = np.random.default_rng(seed)
    drawn_by_source = {}
    for source_name, wanted_count in numina_problem_count_by_source.items():
        candidates = candidates_by_source[source_name]
        drawn_by_source[source_name] = [candidates[index] for index in
                                        random_generator.permutation(len(candidates))[:int(wanted_count * 1.1)]]
    drawn = [problem for problems in drawn_by_source.values() for problem in problems]

    # the gate, exactly as for any pool source: the problems as one parquet file, then decontaminate_pool
    gate_folder = data_root() / "hygiene" / "solutions_numina_gate"
    (gate_folder / "pool").mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    texts = [problem["problem"] for problem in drawn]
    token_counts = [len(token_ids) for token_ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    pq.write_table(pa.table({"doc_id": [problem["problem_id"] for problem in drawn],
                             "source": ["numina_problems"] * len(drawn), "text": texts, "token_count": token_counts,
                             "meta_json": ["{}"] * len(drawn)}, schema=pool_schema),
                   gate_folder / "pool" / "part-00000.parquet")
    data_hygiene.decontaminate_pool([gate_folder / "pool"], data_root() / "eval", gate_folder / "flagged.parquet",
                                    gate_folder / "top_matches.jsonl", report_path=gate_folder / "report.json",
                                    methods=("ngram", "embedding"), device="cpu", worker_count=8)
    flagged_ids = set(pq.read_table(gate_folder / "flagged.parquet", columns=["doc_id"]).column("doc_id").to_pylist())

    kept = []
    selection = {"revision": numina_revision, "seed": seed, "sources": {}}
    for source_name, wanted_count in numina_problem_count_by_source.items():
        passing = [problem for problem in drawn_by_source[source_name] if problem["problem_id"] not in flagged_ids]
        assert len(passing) >= wanted_count, f"{source_name}: only {len(passing)} problems passed the gate"
        kept.extend(passing[:wanted_count])
        selection["sources"][source_name] = {
            "usable_candidates": len(candidates_by_source[source_name]), "drawn": len(drawn_by_source[source_name]),
            "flagged_by_the_gate": len(drawn_by_source[source_name]) - len(passing), "kept": wanted_count}
    output_folder = synthetic_directory(solution_problem_sets["numina"]["folder"])
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "problems.jsonl").open("w") as problems_file:
        for problem in kept:
            problems_file.write(json.dumps(problem, ensure_ascii=False) + "\n")
    (repository_root / "results" / "synthetic_solutions_numina_selection.json").write_text(
        json.dumps(selection, indent=1) + "\n")
    print(json.dumps(selection, indent=1))
    return selection


def ask_teacher_for_solutions(endpoint: str, problem_text: str, sample_count: int, temperature: float,
                              maximum_new_tokens: int) -> list[tuple[str, str]]:
    """`sample_count` independent solutions of one problem in one request: (text, finish reason) each."""
    request_body = {
        "model": "teacher", "n": sample_count, "max_tokens": maximum_new_tokens, "temperature": temperature,
        "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": f"{solution_instruction}\n\nProblem:\n{problem_text}"}]}
    request = urllib.request.Request(f"{endpoint}/v1/chat/completions", json.dumps(request_body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    response = json.load(urllib.request.urlopen(request, timeout=3600))
    return [((choice["message"].get("content") or "").strip(), choice.get("finish_reason") or "")
            for choice in response["choices"]]


def solve_training_problems(endpoints: list[str], samples_per_problem: int, temperature: float,
                            maximum_new_tokens: int, requests_in_flight_per_endpoint: int,
                            problem_set: str = "training_splits") -> None:
    """Append one record per solution to synthetic/<folder of the set>/solutions.jsonl; a restart skips solved
    problems. The first set is read from the hub, the second from the problems.jsonl that select-problems wrote."""
    from midtrain.eval_tasks import find_last_boxed_content, is_math_answer_equivalent

    folder = synthetic_directory(solution_problem_sets[problem_set]["folder"])
    problems = load_training_problems() if problem_set == "training_splits" else read_json_lines(folder / "problems.jsonl")
    output_path = folder / "solutions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finished_ids = {record["problem_id"] for record in read_json_lines(output_path)} if output_path.exists() else set()
    todo = [problem for problem in problems if problem["problem_id"] not in finished_ids]
    print(f"{len(problems)} problems, {len(finished_ids)} already solved, {len(todo)} to do on {len(endpoints)} teacher(s)",
          flush=True)
    write_lock = threading.Lock()
    counts = {"problems": 0, "solutions": 0, "correct": 0, "failed_requests": 0}
    started = time.time()

    def do_one(job_index: int) -> None:
        problem = todo[job_index]
        try:
            solutions = ask_teacher_for_solutions(endpoints[job_index % len(endpoints)], problem["problem"],
                                                  samples_per_problem, temperature, maximum_new_tokens)
        except OSError:
            with write_lock:
                counts["failed_requests"] += 1
            return
        records = []
        for sample_index, (solution_text, finish_reason) in enumerate(solutions):
            predicted = find_last_boxed_content(solution_text)
            is_correct = bool(finish_reason == "stop" and predicted is not None
                              and is_math_answer_equivalent(predicted, problem["reference_answer"]))
            records.append({**problem, "sample_index": sample_index, "solution": solution_text,
                            "finish_reason": finish_reason, "predicted_answer": predicted, "is_correct": is_correct})
        with write_lock:
            with output_path.open("a") as output_file:
                for record in records:
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["problems"] += 1
            counts["solutions"] += len(records)
            counts["correct"] += sum(record["is_correct"] for record in records)
            if counts["problems"] % 500 == 0:
                print(f"{counts['problems']} problems in {(time.time() - started) / 60:.0f} min: {counts['correct']} of "
                      f"{counts['solutions']} solutions correct, {counts['failed_requests']} failed requests", flush=True)

    with concurrent.futures.ThreadPoolExecutor(requests_in_flight_per_endpoint * len(endpoints)) as request_pool:
        list(request_pool.map(do_one, range(len(todo))))
    print(json.dumps(counts), flush=True)


def publish_solutions_to_pool(maximum_per_problem: int, problem_set: str = "training_splits") -> None:
    """Correct, finished and distinct solutions, at most `maximum_per_problem` per problem, become the pool source of
    the set (`synthetic_solutions`, `synthetic_solutions_numina`), as plain text: the problem, then the worked
    solution."""
    from transformers import AutoTokenizer

    problem_set_settings = solution_problem_sets[problem_set]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    records = read_json_lines(synthetic_directory(problem_set_settings["folder"]) / "solutions.jsonl")
    kept_by_problem: dict[str, list] = {}
    seen_texts = set()
    for record in records:
        if not record["is_correct"] or record["solution"] in seen_texts:
            continue
        kept = kept_by_problem.setdefault(record["problem_id"], [])
        if len(kept) < maximum_per_problem:
            kept.append(record)
            seen_texts.add(record["solution"])
    kept_records = [record for kept in kept_by_problem.values() for record in kept]
    texts = [f"Question: {record['problem']}\n\nAnswer: {record['solution']}" for record in kept_records]
    token_counts = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]] if texts else []
    metas = [{"problem_id": record["problem_id"], "task": record["task"], "sample_index": record["sample_index"],
              "generator": Path(os.environ.get("MIDTRAIN_TEACHER", "teacher")).name} for record in kept_records]
    pool_source = problem_set_settings["pool_source"]
    write_pool_rows(pool_source, [document_id_for_text(pool_source, text) for text in texts], texts, token_counts, metas)
    problem_count = len({record["problem_id"] for record in records})
    summary = {"problems": problem_count, "solutions": len(records),
               "correct": sum(record["is_correct"] for record in records),
               "problems_with_a_correct_solution": len(kept_by_problem), "published_solutions": len(kept_records),
               "published_tokens": int(sum(token_counts)), "maximum_per_problem": maximum_per_problem,
               "correct_share_by_task": {task: round(sum(r["is_correct"] for r in records if r["task"] == task)
                                                     / max(1, sum(1 for r in records if r["task"] == task)), 4)
                                         for task in sorted({record["task"] for record in records})}}
    (repository_root / "results" / problem_set_settings["summary"]).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# check-sample: the diversity gate on a batch's rewrites against the passages they were written from
# ---------------------------------------------------------------------------------------------------------------
def check_sample_against_parents(batch_name: str, documents_per_source: int, device: str, seed: int,
                                 styles_left_out: list[str] | None = None) -> int:
    """Run gate 2 of data_hygiene.py on the kept rewrites of a batch against their own parent passages; return the
    number of pool sources that failed.

    One rewrite per parent, at most `documents_per_source` per pool source, drawn in a seeded random order. Both sides
    are measured on the same documents count and the same number of tokens (95% of the smaller side's tokens, at most
    the gate's usual 1M), because every diversity number grows with the sample. The parents are the fair baseline: same
    topics, same count, written by people. Writes results/synthetic_sample_check_<batch>_<documents_per_source>.json.
    """
    from midtrain import data_hygiene

    passages_by_parent = {passage["parent_doc_id"]: passage
                          for passage in read_json_lines(synthetic_directory(batch_name) / "passages.jsonl")}
    records = [record for record in read_json_lines(synthetic_directory(batch_name) / "rewrites.jsonl")
               if record["style"] not in (styles_left_out or [])]
    random_generator = np.random.default_rng(seed)
    records_by_source: dict[str, list] = {"synthetic_rephrase": [], "synthetic_qa": []}
    parents_taken = set()
    for record_index in random_generator.permutation(len(records)):
        record = records[record_index]
        if record["drop_reason"] or record["parent_doc_id"] in parents_taken:
            continue
        source_records = records_by_source[pool_source_of_style[record["style"]]]
        if len(source_records) < documents_per_source:
            source_records.append(record)
            parents_taken.add(record["parent_doc_id"])
    tokenize_texts = data_hygiene.load_qwen_tokenize_function()
    output_root = data_root() / "hygiene" / f"sample_check_{batch_name}"
    report = {"batch": batch_name, "documents_per_source": documents_per_source, "seed": seed,
              "styles_left_out": styles_left_out or [],
              "kept_rewrites": sum(1 for record in records if not record["drop_reason"]), "rewrites": len(records),
              "sources": {}}
    failed_source_count = 0
    for source_name, source_records in records_by_source.items():
        texts_by_side = {"rewrites": [record["text"] for record in source_records],
                         "parents": [passages_by_parent[record["parent_doc_id"]]["passage"] for record in source_records]}
        parquet_path_by_side, token_total_by_side = {}, {}
        for side_name, texts in texts_by_side.items():
            token_counts = [len(token_ids) for token_ids in tokenize_texts(texts)]
            token_total_by_side[side_name] = int(sum(token_counts))
            side_directory = output_root / f"{source_name}_{side_name}"
            side_directory.mkdir(parents=True, exist_ok=True)
            parquet_path_by_side[side_name] = side_directory / "part-00000.parquet"
            pq.write_table(pa.table({"doc_id": [document_id_for_text(side_name, text) for text in texts],
                                     "source": [side_name] * len(texts), "text": texts, "token_count": token_counts,
                                     "meta_json": ["{}"] * len(texts)}, schema=pool_schema), parquet_path_by_side[side_name])
        token_sample_size = min(1_000_000, int(0.95 * min(token_total_by_side.values())))
        diversity_by_side = {side_name: data_hygiene.measure_diversity(
            [parquet_path_by_side[side_name]], tokenize_texts, data_hygiene.default_embedding_model, device,
            token_sample_size=token_sample_size, nearest_neighbour_sample_document_count=len(source_records),
            cluster_sample_document_count=len(source_records), seed=seed) for side_name in texts_by_side}
        checks = data_hygiene.check_diversity_against_baseline(diversity_by_side["rewrites"], diversity_by_side["parents"])
        has_passed = all(check["passed"] for check in checks)
        failed_source_count += 0 if has_passed else 1
        report["sources"][source_name] = {"documents": len(source_records), "token_sample_size": token_sample_size,
                                          "tokens": token_total_by_side, "passed": has_passed, "checks": checks,
                                          "diversity": diversity_by_side}
        print(f"\n{source_name}: {len(source_records)} rewrites against their parents, {token_sample_size:,} tokens a side: "
              f"{'PASSED' if has_passed else 'FAILED'}")
        for check in checks:
            print(f"  {check['measurement']:36s} rewrites {check['batch']:9.4g}  parents {check['baseline']:9.4g}  "
                  f"needs {check['requirement']:16s} {'ok' if check['passed'] else 'FAIL'}")
    output_path = repository_root / "results" / f"synthetic_sample_check_{batch_name}_{documents_per_source}.json"
    output_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {output_path}")
    return failed_source_count


def check_styles_against_parents(batch_name: str, results_name: str, token_sample_size: int, device: str) -> dict:
    """For every style of a results file: the share of its rewrites kept, why the others were dropped, and against
    exactly its own parents the bigram-entropy drop (on `token_sample_size` tokens a side), the Vendi ratio and the
    length of the mean unit embedding. It shows which style holds a source back (11-21: the three styles with fixed
    labels lost 0.53 to 0.63 bits, the others 0.09 to 0.28). Writes results/synthetic_style_check_<batch>_<file>.json.
    """
    from midtrain import data_hygiene
    from midtrain.curation_funnel import embed_documents

    passages_by_parent = {passage["parent_doc_id"]: passage
                          for passage in read_json_lines(synthetic_directory(batch_name) / "passages.jsonl")}
    records = read_json_lines(synthetic_directory(batch_name) / results_name)
    tokenize_texts = data_hygiene.load_qwen_tokenize_function()
    report = {"batch": batch_name, "results_name": results_name, "token_sample_size": token_sample_size, "styles": {}}
    for style_name in sorted({record["style"] for record in records}):
        style_records = [record for record in records if record["style"] == style_name]
        kept = [record for record in style_records if not record["drop_reason"]]
        drop_counts: dict[str, int] = {}
        for record in style_records:
            if record["drop_reason"]:
                drop_counts[record["drop_reason"]] = drop_counts.get(record["drop_reason"], 0) + 1
        entry = {"rewrites": len(style_records), "kept": len(kept), "dropped_by_reason": drop_counts}
        texts = [record["text"] for record in kept]
        parent_texts = [passages_by_parent[record["parent_doc_id"]]["passage"] for record in kept]
        token_ids, parent_token_ids = tokenize_texts(texts), tokenize_texts(parent_texts)
        if min(sum(map(len, token_ids)), sum(map(len, parent_token_ids))) >= token_sample_size:
            entry["bigram_entropy_drop_bits"] = (
                data_hygiene.measure_token_statistics(parent_token_ids, token_sample_size)["bigram_entropy_bits"]
                - data_hygiene.measure_token_statistics(token_ids, token_sample_size)["bigram_entropy_bits"])
        embeddings = embed_documents(texts, data_hygiene.default_embedding_model, device)
        parent_embeddings = embed_documents(parent_texts, data_hygiene.default_embedding_model, device)
        entry["vendi_ratio"] = (data_hygiene.vendi_score_of_unit_vectors(embeddings)
                                / data_hygiene.vendi_score_of_unit_vectors(parent_embeddings))
        entry["mean_vector_length"] = float(np.linalg.norm(embeddings.mean(axis=0)))
        entry["parent_mean_vector_length"] = float(np.linalg.norm(parent_embeddings.mean(axis=0)))
        report["styles"][style_name] = entry
        bigram_text = f"{entry['bigram_entropy_drop_bits']:.2f}" if "bigram_entropy_drop_bits" in entry else "too few tokens"
        print(f"{style_name:32s} kept {len(kept):4d} of {len(style_records):4d}  bigram drop {bigram_text}  "
              f"vendi ratio {entry['vendi_ratio']:.3f}  mean-vector length {entry['mean_vector_length']:.3f} "
              f"(parents {entry['parent_mean_vector_length']:.3f})  dropped {drop_counts}", flush=True)
    output_path = repository_root / "results" / f"synthetic_style_check_{batch_name}_{Path(results_name).stem}.json"
    output_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--passage-count", type=int, default=14000,
                        help="one rewrite per passage; the screening design needs about 12,000 kept rewrites")
    select.add_argument("--seed", type=int, default=20261122)
    select.add_argument("--append", action="store_true", help="add new passages to the batch instead of replacing it")
    select.add_argument("--styles", nargs="+", default=None, help="deal only these styles (default: all six)")
    serve = commands.add_parser("serve")
    serve.add_argument("--gpu", type=int, required=True)
    serve.add_argument("--port", type=int, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--endpoint", action="append", required=True)
    generate.add_argument("--requests-in-flight-per-endpoint", type=int, default=128)
    generate.add_argument("--stop-after-hours", type=float, default=24.0)
    generate.add_argument("--maximum-rewrites", type=int, default=None,
                          help="stop when this many rewrites exist; the screening design needs about 12,000 "
                               "(its largest synthetic share is 9M of 40M tokens, at about 800 tokens per rewrite)")
    generate.add_argument("--temperature", type=float, default=1.0,
                          help="1.0 from v2 on: the v1 batch at 0.8 failed the diversity gate")
    generate.add_argument("--styles", nargs="+", default=None, help="only these styles")
    generate.add_argument("--results-name", default="rewrites.jsonl", help="results file in the batch folder")
    check_sample = commands.add_parser("check-sample")
    check_sample.add_argument("--documents-per-source", type=int, default=300)
    check_sample.add_argument("--device", default="cpu", help="for the embedding model: cpu, cuda or cuda:1")
    check_sample.add_argument("--seed", type=int, default=0)
    check_styles = commands.add_parser("check-styles")
    check_styles.add_argument("--results-name", default="rewrites.jsonl")
    check_styles.add_argument("--token-sample-size", type=int, default=100_000)
    check_styles.add_argument("--device", default="cpu")
    select_problems = commands.add_parser("select-problems", help="draw the second problem set and gate it")
    select_problems.add_argument("--seed", type=int, default=20261123)
    solve = commands.add_parser("solve")
    solve.add_argument("--endpoint", action="append", required=True)
    solve.add_argument("--samples-per-problem", type=int, default=8)
    solve.add_argument("--temperature", type=float, default=1.0)
    solve.add_argument("--maximum-new-tokens", type=int, default=1024)
    solve.add_argument("--requests-in-flight-per-endpoint", type=int, default=24)
    publish_solutions = commands.add_parser("publish-solutions")
    publish_solutions.add_argument("--maximum-per-problem", type=int, default=4)
    for command_parser in (solve, publish_solutions):
        command_parser.add_argument("--problem-set", choices=list(solution_problem_sets), default="training_splits")
    publish = commands.add_parser("publish")
    for command_parser in (select, generate, check_sample, check_styles, publish):
        command_parser.add_argument("--batch", default=current_batch_name, help="folder synthetic/<batch>")
    for command_parser in (check_sample, publish):
        command_parser.add_argument("--leave-out-styles", nargs="+", default=None, help="styles whose rewrites stay out")
    arguments = parser.parse_args()
    if arguments.command == "select":
        select_passages(arguments.batch, arguments.passage_count, arguments.seed, arguments.styles, arguments.append)
    elif arguments.command == "serve":
        serve_teacher(arguments.gpu, arguments.port)
    elif arguments.command == "generate":
        generate_rewrites(arguments.batch, arguments.endpoint, arguments.requests_in_flight_per_endpoint,
                          arguments.stop_after_hours, arguments.temperature, arguments.maximum_rewrites, arguments.styles,
                          arguments.results_name)
    elif arguments.command == "select-problems":
        select_numina_problems(arguments.seed)
    elif arguments.command == "solve":
        solve_training_problems(arguments.endpoint, arguments.samples_per_problem, arguments.temperature,
                                arguments.maximum_new_tokens, arguments.requests_in_flight_per_endpoint,
                                arguments.problem_set)
    elif arguments.command == "publish-solutions":
        publish_solutions_to_pool(arguments.maximum_per_problem, arguments.problem_set)
    elif arguments.command == "check-styles":
        check_styles_against_parents(arguments.batch, arguments.results_name, arguments.token_sample_size, arguments.device)
    elif arguments.command == "check-sample":
        sys.exit(1 if check_sample_against_parents(arguments.batch, arguments.documents_per_source, arguments.device,
                                                   arguments.seed, arguments.leave_out_styles) else 0)
    else:
        publish_rewrites_to_pool(arguments.batch, arguments.leave_out_styles)


if __name__ == "__main__":
    main()
