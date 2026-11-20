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

Commands (after `source scripts/env.sh`; the teacher is served by any OpenAI-compatible server):
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase select --passage-count 50000
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase serve --gpu 1 --port 8201 &         # one teacher per GPU
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase generate --endpoint http://127.0.0.1:8201 [--endpoint ...] \
        --stop-after-hours 6
    $MIDTRAIN_PYTHON -m midtrain.synthetic_rephrase publish            # outputs → pool rows of the two sources
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from midtrain.data_sources import data_root, document_id_for_text, repository_root, tokenizer_path, write_pool_rows

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
style_names = list(style_instructions)
refusal_or_meta_patterns = re.compile(
    r"^(here is|here's|sure[,!]|certainly|i cannot|i can't|i'm sorry|as an ai|below is|the passage|this passage|"
    r"rewritten|okay[,.]|of course)", re.IGNORECASE)
maximum_copied_eight_gram_share = 0.5


def synthetic_directory() -> Path:
    return data_root() / "synthetic"


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


def select_passages(passage_count: int, seed: int) -> None:
    """Pick the passages and assign each one or two styles; writes synthetic/passages.jsonl."""
    random_generator = np.random.default_rng(seed)
    synthetic_directory().mkdir(parents=True, exist_ok=True)
    passages = []
    for source_name, share in passage_share_by_source.items():
        wanted = int(round(share * passage_count))
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
            passage_text = cut_passage_at_paragraphs(text, token_count)
            if passage_text is None:
                continue
            # two different styles for half of the passages: the same content from two sides
            style_count = 2 if random_generator.random() < 0.5 else 1
            styles = [style_names[index] for index in random_generator.choice(len(style_names), style_count, replace=False)]
            passages.append({"parent_doc_id": document_id, "parent_source": source_name, "passage": passage_text,
                             "styles": styles})
            kept += 1
            if kept >= wanted:
                break
        print(f"{source_name}: {kept} passages", flush=True)
    random_generator.shuffle(passages)
    with (synthetic_directory() / "passages.jsonl").open("w") as passage_file:
        for passage in passages:
            passage_file.write(json.dumps(passage, ensure_ascii=False) + "\n")
    print(f"{len(passages)} passages, {sum(len(passage['styles']) for passage in passages)} rewrites planned")


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
    return None


def ask_teacher(endpoint: str, passage_text: str, style_name: str, temperature: float) -> tuple[str, str]:
    request_body = {
        "model": "teacher",
        "messages": [{"role": "user", "content": f"{style_instructions[style_name]} {shared_instruction}\n\nPassage:\n{passage_text}"}],
        "max_tokens": 1800, "temperature": temperature, "top_p": 0.95,   # 1,500 cut 9% of a 600-rewrite sample, mostly dialogues
        "chat_template_kwargs": {"enable_thinking": False},      # the rewrite needs no hidden reasoning; 3x faster
    }
    request = urllib.request.Request(f"{endpoint}/v1/chat/completions", json.dumps(request_body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    response = json.load(urllib.request.urlopen(request, timeout=1800))
    choice = response["choices"][0]
    return (choice["message"].get("content") or "").strip(), choice.get("finish_reason") or ""


def generate_rewrites(endpoints: list[str], requests_in_flight_per_endpoint: int, stop_after_hours: float,
                      temperature: float, maximum_rewrites: int | None) -> None:
    passages = read_json_lines(synthetic_directory() / "passages.jsonl")
    results_path = synthetic_directory() / "rewrites.jsonl"
    finished_keys = set()
    if results_path.exists():
        finished_keys = {record["key"] for record in read_json_lines(results_path)}
    todo = [(passage, style_name) for passage in passages for style_name in passage["styles"]
            if rewrite_key(passage["parent_doc_id"], style_name) not in finished_keys]
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
            output_text, finish_reason = ask_teacher(endpoint, passage["passage"], style_name, temperature)
        except OSError:
            with write_lock:
                counts["failed"] += 1
            return
        drop_reason = reason_to_drop(output_text, passage["passage"], finish_reason)
        record = {"key": rewrite_key(passage["parent_doc_id"], style_name), "parent_doc_id": passage["parent_doc_id"],
                  "parent_source": passage["parent_source"], "style": style_name, "drop_reason": drop_reason,
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
                      f"{counts['failed']} failed requests, {counts['output_words'] / max(1e-9, minutes) :.0f} words/min)",
                      flush=True)

    with concurrent.futures.ThreadPoolExecutor(requests_in_flight_per_endpoint * len(endpoints)) as request_pool:
        list(request_pool.map(do_one, range(len(todo))))
    print(json.dumps(counts), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# publish: kept rewrites become pool rows of the two synthetic sources
# ---------------------------------------------------------------------------------------------------------------
def publish_rewrites_to_pool() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    records = read_json_lines(synthetic_directory() / "rewrites.jsonl")
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
    summary = {"rewrites": len(records), "dropped_by_reason": drop_counts, "sources": {}}
    for source_name, source_records in rows_by_source.items():
        texts = [record["text"] for record in source_records]
        token_counts = [len(token_ids) for token_ids in tokenizer(texts, add_special_tokens=False)["input_ids"]] if texts else []
        metas = [{"style": record["style"], "parent_doc_id": record["parent_doc_id"], "parent_source": record["parent_source"],
                  "copied_eight_gram_share": record["copied_eight_gram_share"],
                  "generator": Path(os.environ.get("MIDTRAIN_TEACHER", "teacher")).name} for record in source_records]
        write_pool_rows(source_name, [document_id_for_text(source_name, text) for text in texts], texts, token_counts, metas)
        by_style: dict[str, int] = {}
        for record in source_records:
            by_style[record["style"]] = by_style.get(record["style"], 0) + 1
        summary["sources"][source_name] = {"documents": len(texts), "tokens": int(sum(token_counts)), "by_style": by_style}
    (repository_root / "results" / "synthetic_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print("next: `python -m midtrain.data_sources tokenize --source synthetic_rephrase synthetic_qa`, then the hygiene gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--passage-count", type=int, default=50000)
    select.add_argument("--seed", type=int, default=20261120)
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
    generate.add_argument("--temperature", type=float, default=0.8,
                          help="0.8: below it the six styles converge on the same phrasing; above it facts start to drift")
    commands.add_parser("publish")
    arguments = parser.parse_args()
    if arguments.command == "select":
        select_passages(arguments.passage_count, arguments.seed)
    elif arguments.command == "serve":
        serve_teacher(arguments.gpu, arguments.port)
    elif arguments.command == "generate":
        generate_rewrites(arguments.endpoint, arguments.requests_in_flight_per_endpoint, arguments.stop_after_hours,
                          arguments.temperature, arguments.maximum_rewrites)
    else:
        publish_rewrites_to_pool()


if __name__ == "__main__":
    main()
