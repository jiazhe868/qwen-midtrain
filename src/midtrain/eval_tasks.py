r"""Build the public evaluation sets, render their prompts and grade model answers (code runs in a sandbox).

A candidate checkpoint of the continued pretraining is judged twice: as a base model on a regression suite (multiple
choice by the log-likelihood of the answer letter; few-shot generation for math, code and trivia), and after a short
fixed fine-tune by pass@k on math and code with chat prompts. Two checkpoints can only be compared when both judges
show them the same questions, the same prompt text and the same definition of "correct". The usual shortcut is an
evaluation harness, but a harness changes its prompts and answer extraction between releases, and it runs model-written
code with the rights of whoever starts it. So this one file owns all of it, in the order the stages run:

  1. build     download twelve public sets at pinned revisions, normalise them to the item schema of docs/conventions.md,
               write $MIDTRAIN_DATA/eval/<task>.jsonl, <task>_fewshot.jsonl and manifest.json
  2. prompts   render_base_prompt, render_chat_messages, multiple_choice_continuations, base_completion_stop_sequences
  3. graders   grade_completion, estimate_pass_at_k, grade_code_completions_in_parallel
  4. sandbox   run_python_program_in_sandbox

How a caller uses it:

    items = load_evaluation_items("gsm8k")
    few_shot_items = load_few_shot_items("gsm8k")
    prompt_text = render_base_prompt(items[0], few_shot_items)               # or render_chat_messages(items[0])
    is_correct = grade_completion(items[0], completion_text, "base")         # or "chat"

Pass the completion exactly as the model wrote it. A HumanEval completion is the indented function body, so stripping
its leading spaces breaks the program.

What was found while writing it (2026-11-20), and what each finding changed:

  - haonan-li/cmmlu now redirects to lmlmcat/cmmlu, which holds a loading script and one zip. datasets 5.1 refuses
    loading scripts, so the build reads the csv files out of the zip itself (in memory, nothing is unpacked).
  - HuggingFaceH4/MATH-500 has only a test split. Its shots come from the train split of EleutherAI/hendrycks_math,
    the full MATH set (hendrycks/competition_math is disabled on the hub). None of the 500 problems occurs there.
  - HumanEval has only a test split and is completed from its own prompt, so humaneval_fewshot.jsonl is empty.
  - TriviaQA rc.nocontext validation has 17,944 rows but 9,960 questions: a question is listed once per evidence
    source. The build keeps one row per question before it samples.
  - Six ARC-Challenge train questions have the same question text as a test question. Shots are therefore drawn only
    from questions whose text occurs nowhere in the evaluated split, and render_base_prompt refuses a shot that is the
    evaluated question.
  - MMLU-Pro validation pads 21 of its 70 rows with "N/A" options; they are dropped, as the official script does.
  - Two hub cards carry no licence field (MATH-500, HellaSwag) and one says "unknown" (TriviaQA). The manifest states
    where each licence was read, and the build stops when a card's licence field differs from the task table.
  - `python -I` ignores PYTHONHASHSEED (hash("a") differed between two runs), so a program that iterates over a set
    of strings could pass on one run and fail on the next. The sandbox starts Python with -s and a fully specified
    environment instead, which keeps the fixed hash seed.

Commands, from the repository root:

    source scripts/env.sh
    env -u HF_HUB_OFFLINE $MIDTRAIN_PYTHON -m midtrain.eval_tasks build      # needs the network; rebuilds are identical
    $MIDTRAIN_PYTHON -m midtrain.eval_tasks check                            # offline: gold answers and reference
                                                                             # solutions through graders and sandbox
    $MIDTRAIN_PYTHON -m midtrain.eval_tasks show --task gsm8k --style base   # print one rendered prompt
    $MIDTRAIN_PYTHON -m pytest tests/test_eval_tasks.py -q
"""

import argparse
import ast
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# stage 1a: the task table (what to download, how much of it to keep, under which licence)
# ---------------------------------------------------------------------------

# One seed for every subsample. A subsample is the items whose sha256(seed, purpose, item_id) sorts first, so it
# depends only on the item ids: not on the row order of a download and not on the version of Python's random module.
subsample_seed = 20261120

# A shot longer than this many characters (question, options and answer as rendered) is not used. Shots only show the
# format, and long ones cost context for every evaluated item: with this limit the longest base prompt of every task
# stays far below the context length of the smallest model (`check` prints the longest prompt of each task).
few_shot_item_character_limit = 600

# MMLU-Pro has up to ten options, so the letters run from A to J.
choice_letters = "ABCDEFGHIJ"

# Every revision is the commit that was current on 2026-11-20. Pinning it makes a rebuild write the same bytes (the
# manifest stores the sha256 of each file). `licence_on_hub_card` is the licence field of the card at that revision;
# `licence` is what the manifest reports, and `licence_note` says where it comes from when the card field is silent.
# Shot counts: 8 for GSM8K is the usual chain-of-thought prompt size; 5 for the other sets keeps prompts short; MBPP
# uses the 7 problems its authors set aside for prompting; trivia answers are a few words, so 8 shots cost little.
evaluation_tasks = {
    "gsm8k": {
        "kind": "numeric",
        "hub_id": "openai/gsm8k",
        "hub_config": "main",
        "hub_revision": "740312add88f781978c0658806c59bc2815b9866",
        "evaluated_split": "test",
        "evaluated_split_item_count": 1319,
        "evaluated_item_count": 1319,
        "stratified_by_meta_key": None,
        "few_shot_split": "train",
        "few_shot_item_count": 8,
        "few_shot_distinct_meta_key": None,
        "licence": "mit",
        "licence_on_hub_card": "mit",
        "licence_note": "",
    },
    "math500": {
        "kind": "numeric",
        "hub_id": "HuggingFaceH4/MATH-500",
        "hub_config": None,
        "hub_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "evaluated_split": "test",
        "evaluated_split_item_count": 500,
        "evaluated_item_count": 500,
        "stratified_by_meta_key": None,
        # MATH-500 has no train split: the shots come from the train split of the full MATH set
        "few_shot_hub_id": "EleutherAI/hendrycks_math",
        "few_shot_hub_configs": ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                                 "number_theory", "prealgebra", "precalculus"],
        "few_shot_hub_revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
        "few_shot_split": "train",
        "few_shot_item_count": 5,
        "few_shot_distinct_meta_key": "subject",
        "licence": "mit",
        "licence_on_hub_card": None,
        "licence_note": ("the hub card of MATH-500 has no licence field; MIT is the licence of the MATH set it is a "
                         "500-problem subset of (card of EleutherAI/hendrycks_math)"),
    },
    "humaneval": {
        "kind": "code",
        "hub_id": "openai/openai_humaneval",
        "hub_config": None,
        "hub_revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "evaluated_split": "test",
        "evaluated_split_item_count": 164,
        "evaluated_item_count": 164,
        "stratified_by_meta_key": None,
        "few_shot_split": None,        # test is the only split, and the prompt is the function's own signature
        "few_shot_item_count": 0,
        "few_shot_distinct_meta_key": None,
        "licence": "mit",
        "licence_on_hub_card": "mit",
        "licence_note": "",
    },
    "mbpp": {
        "kind": "code",
        "hub_id": "google-research-datasets/mbpp",
        "hub_config": "sanitized",
        "hub_revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "evaluated_split": "test",
        "evaluated_split_item_count": 257,
        "evaluated_item_count": 257,
        "stratified_by_meta_key": None,
        "few_shot_split": "prompt",
        "few_shot_item_count": 7,
        "few_shot_distinct_meta_key": None,
        "licence": "cc-by-4.0",
        "licence_on_hub_card": "cc-by-4.0",
        "licence_note": "",
    },
    "mmlu_pro": {
        "kind": "multiple_choice",
        "hub_id": "TIGER-Lab/MMLU-Pro",
        "hub_config": None,
        "hub_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "evaluated_split": "test",
        "evaluated_split_item_count": 12032,
        "evaluated_item_count": 1200,
        "stratified_by_meta_key": "category",
        "few_shot_split": "validation",
        "few_shot_item_count": 5,
        "few_shot_distinct_meta_key": "category",
        "licence": "mit",
        "licence_on_hub_card": "mit",
        "licence_note": "",
    },
    "arc_challenge": {
        "kind": "multiple_choice",
        "hub_id": "allenai/ai2_arc",
        "hub_config": "ARC-Challenge",
        "hub_revision": "210d026faf9955653af8916fad021475a3f00453",
        "evaluated_split": "test",
        "evaluated_split_item_count": 1172,
        "evaluated_item_count": 1172,
        "stratified_by_meta_key": None,
        "few_shot_split": "train",
        "few_shot_item_count": 5,
        "few_shot_distinct_meta_key": None,
        "licence": "cc-by-sa-4.0",
        "licence_on_hub_card": "cc-by-sa-4.0",
        "licence_note": "",
    },
    "hellaswag": {
        "kind": "multiple_choice",
        "hub_id": "Rowan/hellaswag",
        "hub_config": None,
        "hub_revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "evaluated_split": "validation",
        "evaluated_split_item_count": 10042,
        "evaluated_item_count": 1000,
        "stratified_by_meta_key": None,
        "few_shot_split": "train",
        "few_shot_item_count": 5,
        "few_shot_distinct_meta_key": None,
        "licence": "mit",
        "licence_on_hub_card": None,
        "licence_note": "the card has no licence field; its text says MIT and links the LICENSE file of the authors",
    },
    "commonsenseqa": {
        "kind": "multiple_choice",
        "hub_id": "tau/commonsense_qa",
        "hub_config": None,
        "hub_revision": "94630fe30dad47192a8546eb75f094926d47e155",
        "evaluated_split": "validation",
        "evaluated_split_item_count": 1221,
        "evaluated_item_count": 1221,
        "stratified_by_meta_key": None,
        "few_shot_split": "train",
        "few_shot_item_count": 5,
        "few_shot_distinct_meta_key": None,
        "licence": "mit",
        "licence_on_hub_card": "mit",
        "licence_note": "",
    },
    "triviaqa": {
        "kind": "short_answer",
        "hub_id": "mandarjoshi/trivia_qa",
        "hub_config": "rc.nocontext",
        "hub_revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        # the evidence columns are empty in this configuration; leaving them out makes 138,384 train rows quick to read
        "hub_column_names": ["question", "question_id", "question_source", "answer"],
        "evaluated_split": "validation",
        "evaluated_split_item_count": 9960,       # distinct questions; the split has 17,944 rows
        "evaluated_item_count": 1000,
        "stratified_by_meta_key": None,
        "few_shot_split": "train",
        "few_shot_item_count": 8,
        "few_shot_distinct_meta_key": None,
        "licence": "unknown",
        "licence_on_hub_card": "unknown",
        "licence_note": "the licence field of the hub card says 'unknown'",
    },
    "cmmlu": {
        "kind": "multiple_choice",
        "hub_id": "lmlmcat/cmmlu",                # haonan-li/cmmlu redirects here
        "hub_config": None,
        "hub_revision": "efcc940752ea4a1ea94d2727f11f83858d64fc8e",
        "hub_archive_file_name": "cmmlu_v1_0_1.zip",
        "evaluated_split": "test",
        "evaluated_split_item_count": 11582,
        "evaluated_item_count": 1000,
        "stratified_by_meta_key": "subject",
        "few_shot_split": "dev",
        "few_shot_item_count": 5,
        "few_shot_distinct_meta_key": "subject",
        "licence": "cc-by-nc-4.0",
        "licence_on_hub_card": "cc-by-nc-4.0",
        "licence_note": "the card text says CC BY-NC-SA 4.0 while its licence field says cc-by-nc-4.0; "
                        "non-commercial either way",
    },
    # Two sets of grade-school word problems from sources other than GSM8K (added 11-23, README §5.3). The
    # distillation data are worked solutions to the training problems of GSM8K and MATH, so a gain on GSM8K may be a
    # gain on its own distribution; these two sets ask whether it carries to problems written elsewhere. Both are
    # answered from the eight GSM8K shots, so the prompt stays the same and only the problems change. The Calc-X
    # copies (MU-NLPC) hold every item in one split with a numeric result; SVAMP writes thousands as 1_414.
    "svamp": {
        "kind": "numeric",
        "hub_id": "MU-NLPC/Calc-svamp",
        "hub_config": "default",
        "hub_revision": "1895931ce53425d46db1ffeb055d74ca1d912f17",
        "evaluated_split": "test",
        "evaluated_split_item_count": 1000,
        "evaluated_item_count": 1000,
        "stratified_by_meta_key": None,
        "few_shot_hub_id": "openai/gsm8k",
        "few_shot_hub_config": "main",
        "few_shot_hub_revision": "740312add88f781978c0658806c59bc2815b9866",
        "few_shot_split": "train",
        "few_shot_item_count": 8,
        "few_shot_distinct_meta_key": None,
        "licence": "mit",
        "licence_on_hub_card": "mit",
        "licence_note": "",
    },
    "asdiv_a": {
        "kind": "numeric",
        "hub_id": "MU-NLPC/Calc-asdiv_a",
        "hub_config": "default",
        "hub_revision": "520a6910e097ee287ecd2bb9104f7f45805f9df9",
        "evaluated_split": "test",
        "evaluated_split_item_count": 1218,
        "evaluated_item_count": 1218,
        "stratified_by_meta_key": None,
        "few_shot_hub_id": "openai/gsm8k",
        "few_shot_hub_config": "main",
        "few_shot_hub_revision": "740312add88f781978c0658806c59bc2815b9866",
        "few_shot_split": "train",
        "few_shot_item_count": 8,
        "few_shot_distinct_meta_key": None,
        "licence": "cc-by-nc-4.0",
        "licence_on_hub_card": None,
        "licence_note": "the hub card has no licence field; CC BY-NC 4.0 is the licence of ASDiv, which this set copies",
    },
}


# ---------------------------------------------------------------------------
# stage 1b: files and downloads
# ---------------------------------------------------------------------------

def data_directory() -> Path:
    """Return $MIDTRAIN_DATA, or ./midtrain_data when it is not set (docs/conventions.md)."""
    return Path(os.environ.get("MIDTRAIN_DATA", "midtrain_data"))


def evaluation_directory() -> Path:
    """Return the folder that holds <task>.jsonl, <task>_fewshot.jsonl and manifest.json."""
    return data_directory() / "eval"


def download_cache_directory() -> Path:
    """Return the folder for downloaded sets, next to the other large data instead of the home directory."""
    return data_directory() / "cache" / "eval_task_downloads"


def load_hub_rows(hub_id: str, hub_config: str | None, split_name: str, hub_revision: str,
                  column_names: list[str] | None = None) -> list[dict]:
    """Download one split of a hub dataset at a pinned revision and return its rows as dictionaries."""
    # imported here and not at the top: grading on a compute node needs neither this library nor its import time
    import datasets

    datasets.disable_progress_bars()
    dataset = datasets.load_dataset(hub_id, hub_config, split=split_name, revision=hub_revision,
                                    cache_dir=str(download_cache_directory()))
    if column_names is not None:
        dataset = dataset.select_columns(column_names)
    return list(dataset)


def read_cmmlu_rows_from_archive(task: dict, folder_name: str) -> list[tuple[str, int, dict]]:
    """Return (subject, row index, row) for every csv row under `folder_name` ("test" or "dev") of the CMMLU zip.

    The hub repository holds a loading script and this zip, and datasets 5.1 refuses loading scripts. The csv files
    are read straight from the zip in memory: nothing is unpacked to disk, so a crafted member path cannot write
    outside a folder. The csv module keeps every cell as the string that was typed; the loading script's pandas
    reader would turn an option column of numbers into floats.
    """
    from huggingface_hub import hf_hub_download

    archive_path = hf_hub_download(task["hub_id"], task["hub_archive_file_name"], repo_type="dataset",
                                   revision=task["hub_revision"], cache_dir=str(download_cache_directory() / "hub"))
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        member_names = sorted(member_name for member_name in archive.namelist()
                              if member_name.startswith(folder_name + "/") and member_name.endswith(".csv"))
        assert len(member_names) == 67, f"CMMLU should have 67 subjects under {folder_name}/, found {len(member_names)}"
        for member_name in member_names:
            subject_name = member_name[len(folder_name) + 1:-len(".csv")]
            csv_text = archive.read(member_name).decode("utf-8-sig")
            for row_index, row in enumerate(csv.DictReader(io.StringIO(csv_text))):
                rows.append((subject_name, row_index, row))
    return rows


def read_licence_from_hub_card(hub_id: str, hub_revision: str) -> str | None:
    """Return the licence field of a dataset card at a pinned revision, or None when the card has none."""
    from huggingface_hub import HfApi

    card_data = HfApi().dataset_info(hub_id, revision=hub_revision).card_data
    licence = getattr(card_data, "license", None) if card_data is not None else None
    if isinstance(licence, list):
        licence = ", ".join(licence)
    return licence


def write_json_lines(file_path: Path, items: list[dict]) -> str:
    """Write one JSON object per line and return the sha256 of the bytes written."""
    lines = []
    for item in items:
        line = json.dumps(item, ensure_ascii=False)       # Chinese and LaTeX stay readable by eye
        # these three characters are legal inside a JSON string, but str.splitlines() would cut the line at them
        for line_separator in ("\u2028", "\u2029", "\u0085"):
            line = line.replace(line_separator, "\\u%04x" % ord(line_separator))
        lines.append(line + "\n")
    file_bytes = "".join(lines).encode("utf-8")
    partial_path = file_path.with_name(file_path.name + ".partial")
    partial_path.write_bytes(file_bytes)
    os.replace(partial_path, file_path)                    # a reader never sees a half-written file
    return hashlib.sha256(file_bytes).hexdigest()


def read_json_lines(file_path: Path) -> list[dict]:
    """Read a file with one JSON object per line."""
    with open(file_path, encoding="utf-8") as json_lines_file:
        return [json.loads(line) for line in json_lines_file if line.strip()]


def load_evaluation_items(task_name: str) -> list[dict]:
    """Return the evaluated items of one task, as written by `build`."""
    assert task_name in evaluation_tasks, f"unknown task {task_name!r}; known tasks: {sorted(evaluation_tasks)}"
    file_path = evaluation_directory() / f"{task_name}.jsonl"
    assert file_path.exists(), f"{file_path} is missing: run `python -m midtrain.eval_tasks build` first"
    return read_json_lines(file_path)


def load_few_shot_items(task_name: str) -> list[dict]:
    """Return the shots of one task, in the order they are meant to appear (an empty list for HumanEval)."""
    assert task_name in evaluation_tasks, f"unknown task {task_name!r}; known tasks: {sorted(evaluation_tasks)}"
    file_path = evaluation_directory() / f"{task_name}_fewshot.jsonl"
    assert file_path.exists(), f"{file_path} is missing: run `python -m midtrain.eval_tasks build` first"
    return read_json_lines(file_path)


# ---------------------------------------------------------------------------
# stage 1c: one normaliser per set, from a row as published to an item of docs/conventions.md
# ---------------------------------------------------------------------------

def assemble_item(task_name: str, split_name: str, source_identifier, kind: str, question: str,
                  answer_fields: dict, meta: dict) -> dict:
    """Return an item with the keys in the order of docs/conventions.md; `meta` also records the split."""
    item = {"item_id": f"{task_name}:{split_name}:{source_identifier}", "task": task_name, "kind": kind,
            "question": question}
    item.update(answer_fields)
    item["meta"] = {"split": split_name, **meta}
    return item


def collapse_whitespace(text: str) -> str:
    """Return the text with every run of spaces, tabs and line breaks replaced by one space."""
    return " ".join(text.split())


def clean_gsm8k_solution(solution_text: str) -> str:
    """Return a GSM8K solution as one paragraph, without the calculator annotations such as <<48/2=24>>.

    Many solution lines end without a full stop; one is added so that the joined lines stay separate sentences.
    """
    solution_without_annotations = re.sub(r"<<[^>]*>>", "", solution_text)
    sentences = []
    for line in solution_without_annotations.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[-1] not in ".!?:;,":
            line += "."
        sentences.append(line)
    return " ".join(sentences)


def normalise_gsm8k_row(row: dict, split_name: str, row_index: int) -> dict:
    """GSM8K: the answer field is a worked solution, then `#### ` and the final number (93 of 8,792 with commas)."""
    assert row["answer"].count("####") == 1, f"gsm8k {split_name} row {row_index}: expected exactly one ####"
    solution_text, final_answer_text = row["answer"].split("####")
    answer_text = final_answer_text.strip().replace(",", "")
    assert is_plain_number_text(answer_text), f"gsm8k {split_name} row {row_index}: answer {answer_text!r}"
    meta = {"reference_solution": row["answer"], "worked_solution": clean_gsm8k_solution(solution_text)}
    return assemble_item("gsm8k", split_name, row_index, "numeric", row["question"].strip(),
                         {"answer_text": answer_text}, meta)


def normalise_calc_x_row(row: dict, task_name: str, split_name: str, row_index: int) -> dict:
    """SVAMP or ASDiv-A in the Calc-X copy: a question and its result, an integer, a decimal or (once) a fraction.

    SVAMP writes 68 of its 1,000 results with Python's thousands separator (1_414); it is removed. The reference
    solution is the set's own equation, where it has one, followed by the answer sentence of the base prompts.
    """
    answer_text = str(row["result"]).replace("_", "").strip()
    assert re.fullmatch(r"-?\d+(\.\d+)?(/\d+)?", answer_text), f"{task_name} row {row_index}: result {row['result']!r}"
    equation_text = (row.get("equation") or "").strip()
    reference_solution = (f"{equation_text} = {answer_text}. " if equation_text else "") + f"The answer is {answer_text}."
    meta = {"source_id": row["id"], "reference_solution": reference_solution}
    return assemble_item(task_name, split_name, row_index, "numeric", collapse_whitespace(row["question"]),
                         {"answer_text": answer_text}, meta)


def normalise_math500_row(row: dict, split_name: str, row_index: int) -> dict:
    """MATH-500: the answer is a LaTeX expression; 184 of the 500 are not plain numbers."""
    meta = {"subject": row["subject"], "level": row["level"], "reference_solution": row["solution"],
            "worked_solution": collapse_whitespace(replace_boxed_with_content(row["solution"]))}
    return assemble_item("math500", split_name, row["unique_id"], "numeric", row["problem"].strip(),
                         {"answer_text": row["answer"].strip()}, meta)


def normalise_math_train_row(row: dict, subject_config: str, row_index: int) -> dict | None:
    """A problem of the MATH train split as a shot candidate, or None when it cannot serve as one.

    A candidate needs exactly one boxed answer (it becomes `answer_text`), a short answer and no Asymptote figure:
    a figure is source code that a text model cannot see drawn.
    """
    boxed_answer_text = find_last_boxed_content(row["solution"])
    has_figure = "[asy]" in row["problem"] or "[asy]" in row["solution"]
    if boxed_answer_text is None or row["solution"].count("\\boxed") != 1 or has_figure or len(boxed_answer_text) > 20:
        return None
    meta = {"subject": row["type"], "level": row["level"], "reference_solution": row["solution"],
            "worked_solution": collapse_whitespace(replace_boxed_with_content(row["solution"]))}
    return assemble_item("math500", "train", f"{subject_config}/{row_index}", "numeric", row["problem"].strip(),
                         {"answer_text": boxed_answer_text.strip()}, meta)


def normalise_humaneval_row(row: dict, split_name: str, row_index: int) -> dict:
    """HumanEval: the question is the signature and docstring; the tests define check(candidate) and must be called."""
    assert "def check(candidate)" in row["test"], f"{row['task_id']}: the tests do not define check(candidate)"
    test_code = row["test"].strip("\n") + f"\n\ncheck({row['entry_point']})\n"
    meta = {"question_form": "function_stub", "reference_solution": row["canonical_solution"]}
    return assemble_item("humaneval", split_name, row["task_id"], "code", row["prompt"],
                         {"test_code": test_code, "entry_point": row["entry_point"]}, meta)


def find_function_called_by_tests(reference_code: str, test_list: list[str]) -> str:
    """Return the one top-level function of an MBPP reference solution that its tests call.

    12 of the 427 sanitized solutions define helper functions too, so "the first def" is not always the entry point.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)    # three solutions write "\w" in a plain string
        syntax_tree = ast.parse(reference_code)
    function_names = [node.name for node in syntax_tree.body if isinstance(node, ast.FunctionDef)]
    tests_text = "\n".join(test_list)
    called_function_names = [function_name for function_name in function_names
                             if re.search(rf"\b{re.escape(function_name)}\s*\(", tests_text)]
    assert len(called_function_names) == 1, (
        f"expected the tests to call exactly one of {function_names}, they call {called_function_names}")
    return called_function_names[0]


def normalise_mbpp_row(row: dict, split_name: str, row_index: int) -> dict:
    """MBPP (sanitized): a one-sentence task, a reference solution and three or more assert lines."""
    test_list = list(row["test_list"])
    test_imports = list(row["test_imports"])
    assert all(test_line.startswith("assert") for test_line in test_list), f"mbpp {row['task_id']}: non-assert test"
    assert '"""' not in row["prompt"] and not any('"""' in test_line for test_line in test_list), (
        f"mbpp {row['task_id']}: a triple quote would end the docstring that the base prompt puts this text in")
    meta = {"question_form": "task_description", "test_list": test_list, "test_imports": test_imports,
            "reference_solution": row["code"]}
    answer_fields = {"test_code": "\n".join(test_imports + test_list) + "\n",
                     "entry_point": find_function_called_by_tests(row["code"], test_list)}
    return assemble_item("mbpp", split_name, row["task_id"], "code", row["prompt"].strip(), answer_fields, meta)


def normalise_mmlu_pro_row(row: dict, split_name: str, row_index: int) -> dict:
    """MMLU-Pro: up to ten options. Padding options "N/A" are dropped, as the official evaluation script does."""
    answer_index_as_published = row["answer_index"]
    assert choice_letters[answer_index_as_published] == row["answer"], f"mmlu_pro {row['question_id']}: letter/index"
    assert row["options"][answer_index_as_published].strip() != "N/A", f"mmlu_pro {row['question_id']}: answer is N/A"
    choices = [option.strip() for option in row["options"] if option.strip() != "N/A"]
    dropped_before_answer = sum(option.strip() == "N/A" for option in row["options"][:answer_index_as_published])
    meta = {"category": row["category"], "source": row["src"]}
    return assemble_item("mmlu_pro", split_name, row["question_id"], "multiple_choice", row["question"].strip(),
                         {"choices": choices, "answer_index": answer_index_as_published - dropped_before_answer}, meta)


def normalise_arc_row(row: dict, split_name: str, row_index: int) -> dict:
    """ARC-Challenge: 3 to 5 options whose labels are letters or, in 22 of the 1,172 test rows, digits."""
    answer_index = row["choices"]["label"].index(row["answerKey"])
    choices = [choice_text.strip() for choice_text in row["choices"]["text"]]
    return assemble_item("arc_challenge", split_name, row["id"], "multiple_choice", row["question"].strip(),
                         {"choices": choices, "answer_index": answer_index}, {})


def clean_hellaswag_text(text: str) -> str:
    """Remove the WikiHow markers ([header], [title], [step], ...) from a HellaSwag context or ending.

    Two thirds of the rows come from WikiHow and carry these markers. " [title]" ends a headline, so it becomes a
    full stop unless the text before it already ends a sentence (the usual harness rule always adds one and so
    writes "breaking.. Look"); the other markers are dropped.
    """
    text = re.sub(r"([.!?:;,])? \[title\]", lambda marker: marker.group(1) or ".", text.strip())
    text = re.sub(r"\[.*?\]", "", text)
    return collapse_whitespace(text)


def normalise_hellaswag_row(row: dict, split_name: str, row_index: int) -> dict:
    """HellaSwag: a passage and four endings. The question names the task, so an item reads the same in any prompt.

    `ind` repeats between the two sources of the set, so the row index is the identifier.
    """
    passage = clean_hellaswag_text(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())
    choices = [clean_hellaswag_text(ending) for ending in row["endings"]]
    assert len(choices) == 4, f"hellaswag {split_name} row {row_index}: expected four endings"
    meta = {"passage": passage, "source_id": row["source_id"], "ind": row["ind"], "split_type": row["split_type"]}
    question = "Which ending continues the passage most plausibly?\n" + passage
    return assemble_item("hellaswag", split_name, row_index, "multiple_choice", question,
                         {"choices": choices, "answer_index": int(row["label"])}, meta)


def normalise_commonsenseqa_row(row: dict, split_name: str, row_index: int) -> dict:
    """CommonsenseQA: five options labelled A to E."""
    answer_index = row["choices"]["label"].index(row["answerKey"])
    choices = [choice_text.strip() for choice_text in row["choices"]["text"]]
    meta = {"question_concept": row["question_concept"]}
    return assemble_item("commonsenseqa", split_name, row["id"], "multiple_choice", row["question"].strip(),
                         {"choices": choices, "answer_index": answer_index}, meta)


def normalise_triviaqa_row(row: dict, split_name: str, row_index: int) -> dict:
    """TriviaQA: the canonical answer comes first, then every alias whose normalised form is new (up to 230 a row).

    An alias that normalises to nothing (for example "The") is dropped: it would match an empty answer.
    """
    answer = row["answer"]
    answer_aliases = []
    normalised_forms_seen = set()
    for alias in [answer["value"]] + list(answer["aliases"]) + list(answer["normalized_aliases"]):
        normalised_alias = normalise_short_answer_text(alias)
        if normalised_alias and normalised_alias not in normalised_forms_seen:
            normalised_forms_seen.add(normalised_alias)
            answer_aliases.append(alias)
    meta = {"question_source": row["question_source"], "answer_type": answer["type"]}
    return assemble_item("triviaqa", split_name, row["question_id"], "short_answer", row["question"].strip(),
                         {"answer_aliases": answer_aliases}, meta)


def normalise_cmmlu_row(row: dict, split_name: str, subject_name: str, row_index: int) -> dict:
    """CMMLU: a Chinese question with the options in columns A to D and the answer letter in `Answer`."""
    choices = [row[letter].strip() for letter in "ABCD"]
    meta = {"subject": subject_name, "language": "zh"}
    return assemble_item("cmmlu", split_name, f"{subject_name}/{row_index}", "multiple_choice", row["Question"].strip(),
                         {"choices": choices, "answer_index": "ABCD".index(row["Answer"].strip())}, meta)


# the sets that load one hub split for the evaluated items and one for the shots, row by row
row_normaliser_of_task = {
    "gsm8k": normalise_gsm8k_row,
    "mbpp": normalise_mbpp_row,
    "mmlu_pro": normalise_mmlu_pro_row,
    "arc_challenge": normalise_arc_row,
    "hellaswag": normalise_hellaswag_row,
    "commonsenseqa": normalise_commonsenseqa_row,
    "triviaqa": normalise_triviaqa_row,
}


def keep_first_item_of_each_id(items: list[dict]) -> list[dict]:
    """Return the items without those whose item_id already occurred (TriviaQA lists a question once per source)."""
    item_ids_seen = set()
    distinct_items = []
    for item in items:
        if item["item_id"] not in item_ids_seen:
            item_ids_seen.add(item["item_id"])
            distinct_items.append(item)
    return distinct_items


def build_items_of_task(task_name: str) -> tuple[list[dict], list[dict]]:
    """Download one set and return (every item of the evaluated split, every shot candidate), all normalised."""
    task = evaluation_tasks[task_name]
    if task_name == "cmmlu":
        evaluated_split_items = [normalise_cmmlu_row(row, "test", subject_name, row_index)
                                 for subject_name, row_index, row in read_cmmlu_rows_from_archive(task, "test")]
        few_shot_candidate_items = [normalise_cmmlu_row(row, "dev", subject_name, row_index)
                                    for subject_name, row_index, row in read_cmmlu_rows_from_archive(task, "dev")]
    elif task_name == "math500":
        evaluated_rows = load_hub_rows(task["hub_id"], task["hub_config"], "test", task["hub_revision"])
        evaluated_split_items = [normalise_math500_row(row, "test", row_index)
                                 for row_index, row in enumerate(evaluated_rows)]
        few_shot_candidate_items = []
        unusable_train_row_count = 0
        for subject_config in task["few_shot_hub_configs"]:
            train_rows = load_hub_rows(task["few_shot_hub_id"], subject_config, "train", task["few_shot_hub_revision"])
            for row_index, row in enumerate(train_rows):
                candidate_item = normalise_math_train_row(row, subject_config, row_index)
                if candidate_item is None:
                    unusable_train_row_count += 1
                else:
                    few_shot_candidate_items.append(candidate_item)
        print(f"  math500: {unusable_train_row_count:,} MATH train problems cannot be shots (a figure, a long answer "
              f"or not exactly one boxed answer); {len(few_shot_candidate_items):,} can")
    elif task_name in ("svamp", "asdiv_a"):
        evaluated_rows = load_hub_rows(task["hub_id"], task["hub_config"], task["evaluated_split"], task["hub_revision"])
        evaluated_split_items = [normalise_calc_x_row(row, task_name, task["evaluated_split"], row_index)
                                 for row_index, row in enumerate(evaluated_rows)]
        # the shot candidates are the GSM8K training problems, ranked by the same seeded hash as for GSM8K itself,
        # so the prompt carries the same eight worked solutions (check_built_task compares the two shot files)
        few_shot_rows = load_hub_rows(task["few_shot_hub_id"], task["few_shot_hub_config"], task["few_shot_split"],
                                      task["few_shot_hub_revision"])
        few_shot_candidate_items = [normalise_gsm8k_row(row, task["few_shot_split"], row_index)
                                    for row_index, row in enumerate(few_shot_rows)]
    elif task_name == "humaneval":
        evaluated_rows = load_hub_rows(task["hub_id"], task["hub_config"], "test", task["hub_revision"])
        evaluated_split_items = [normalise_humaneval_row(row, "test", row_index)
                                 for row_index, row in enumerate(evaluated_rows)]
        few_shot_candidate_items = []
    else:
        normalise_row = row_normaliser_of_task[task_name]
        evaluated_rows = load_hub_rows(task["hub_id"], task["hub_config"], task["evaluated_split"],
                                       task["hub_revision"], task.get("hub_column_names"))
        evaluated_split_items = [normalise_row(row, task["evaluated_split"], row_index)
                                 for row_index, row in enumerate(evaluated_rows)]
        few_shot_rows = load_hub_rows(task["hub_id"], task["hub_config"], task["few_shot_split"],
                                      task["hub_revision"], task.get("hub_column_names"))
        few_shot_candidate_items = [normalise_row(row, task["few_shot_split"], row_index)
                                    for row_index, row in enumerate(few_shot_rows)]
    if task_name == "triviaqa":
        evaluated_split_items = keep_first_item_of_each_id(evaluated_split_items)
        few_shot_candidate_items = keep_first_item_of_each_id(few_shot_candidate_items)
    return evaluated_split_items, few_shot_candidate_items


# ---------------------------------------------------------------------------
# stage 1d: subsamples and shots (seeded, and never a shot from the evaluated split)
# ---------------------------------------------------------------------------

def rank_items_by_seeded_hash(items: list[dict], purpose_text: str) -> list[dict]:
    """Return the items in a pseudo-random order that depends only on the seed, the purpose and each item_id."""
    def seeded_hash_of_item(item: dict) -> str:
        return hashlib.sha256(f"{subsample_seed}:{purpose_text}:{item['item_id']}".encode("utf-8")).hexdigest()
    return sorted(items, key=seeded_hash_of_item)


def allocate_proportional_quotas(size_of_group: dict[str, int], sample_count: int) -> dict[str, int]:
    """Split `sample_count` over the groups in proportion to their sizes, so that the quotas sum to it exactly.

    Every group first gets the rounded-down share; the places left over go to the groups with the largest remainders
    (ties go to the name that sorts first). Integer arithmetic keeps the result the same on every machine.
    """
    total_size = sum(size_of_group.values())
    assert 0 < sample_count <= total_size, f"cannot draw {sample_count} items from {total_size}"
    quota_of_group = {group_name: sample_count * size // total_size for group_name, size in size_of_group.items()}
    remainder_of_group = {group_name: sample_count * size % total_size for group_name, size in size_of_group.items()}
    places_left = sample_count - sum(quota_of_group.values())
    groups_by_largest_remainder = sorted(size_of_group, key=lambda group_name: (-remainder_of_group[group_name],
                                                                                 group_name))
    for group_name in groups_by_largest_remainder[:places_left]:
        quota_of_group[group_name] += 1
    return quota_of_group


def sample_evaluated_items(items: list[dict], sample_count: int, stratified_by_meta_key: str | None) -> list[dict]:
    """Return `sample_count` of the items, in their original order.

    Without a stratifying key the sample is the items with the smallest seeded hash. With one, every group (an
    MMLU-Pro category, a CMMLU subject) gets a share proportional to its size, so that the accuracy on the sample
    estimates the accuracy on the whole split, and the smallest hashes are taken inside each group.
    """
    if sample_count == len(items):
        return items
    if stratified_by_meta_key is None:
        chosen_items = rank_items_by_seeded_hash(items, "evaluated_subsample")[:sample_count]
    else:
        items_of_group = {}
        for item in items:
            items_of_group.setdefault(item["meta"][stratified_by_meta_key], []).append(item)
        quota_of_group = allocate_proportional_quotas(
            {group_name: len(group_items) for group_name, group_items in items_of_group.items()}, sample_count)
        chosen_items = []
        for group_name in sorted(items_of_group):
            ranked_group_items = rank_items_by_seeded_hash(items_of_group[group_name], "evaluated_subsample")
            chosen_items.extend(ranked_group_items[:quota_of_group[group_name]])
    chosen_item_ids = {item["item_id"] for item in chosen_items}
    return [item for item in items if item["item_id"] in chosen_item_ids]


def normalise_question_for_comparison(question_text: str) -> str:
    """Lowercase a question and collapse its whitespace, to find the same question in two splits."""
    return collapse_whitespace(question_text).lower()


def is_same_question(first_item: dict, second_item: dict) -> bool:
    """Whether two items are the same question: the same id, or the same question text with the same options.

    The options are part of the test because stems such as "Which of the following is true?" repeat between
    different questions (231 MMLU-Pro test rows repeat the question text of an earlier row).
    """
    if first_item["item_id"] == second_item["item_id"]:
        return True
    has_same_text = (normalise_question_for_comparison(first_item["question"])
                     == normalise_question_for_comparison(second_item["question"]))
    return has_same_text and first_item.get("choices") == second_item.get("choices")


def is_plain_number_text(answer_text: str) -> bool:
    """Whether an answer is written as a plain integer or decimal (every GSM8K answer, 316 of the 500 in MATH-500)."""
    return re.fullmatch(r"-?\d+(\.\d+)?", answer_text) is not None


def answer_class_of_item(item: dict):
    """Return what the shots of a prompt should vary: the answer letter, or number against expression for math."""
    if item["kind"] == "multiple_choice":
        return item["answer_index"]
    if item["kind"] == "numeric":
        return "number" if is_plain_number_text(item["answer_text"]) else "expression"
    return None


def select_few_shot_items(candidate_items: list[dict], few_shot_item_count: int,
                          distinct_meta_key: str | None) -> list[dict]:
    """Pick the shots from the candidates, in seeded order, spreading groups and answer classes.

    If all shots of a multiple-choice prompt had the same answer letter, the log-likelihood of that letter would rise
    for every evaluated item; if all math shots ended in a plain number, the model would never see how an expression
    is written as an answer. So each pick is the first candidate, in seeded order, whose group (an MMLU-Pro category,
    a CMMLU subject, a MATH subject) has no shot yet and whose answer class has been used least. Once every group
    has a shot, groups may repeat.
    """
    ranked_candidate_items = rank_items_by_seeded_hash(candidate_items, "few_shot")
    selected_items = []
    while len(selected_items) < few_shot_item_count:
        selected_item_ids = {selected_item["item_id"] for selected_item in selected_items}
        remaining_items = [candidate_item for candidate_item in ranked_candidate_items
                           if candidate_item["item_id"] not in selected_item_ids]
        assert remaining_items, f"only {len(selected_items)} shot candidates for {few_shot_item_count} shots"
        if distinct_meta_key is not None:
            used_groups = {selected_item["meta"][distinct_meta_key] for selected_item in selected_items}
            items_of_new_groups = [candidate_item for candidate_item in remaining_items
                                   if candidate_item["meta"][distinct_meta_key] not in used_groups]
            remaining_items = items_of_new_groups or remaining_items
        use_count_of_class = collections.Counter(answer_class_of_item(selected_item)
                                                 for selected_item in selected_items)
        smallest_use_count = min(use_count_of_class[answer_class_of_item(candidate_item)]
                                 for candidate_item in remaining_items)
        selected_items.append(next(candidate_item for candidate_item in remaining_items
                                   if use_count_of_class[answer_class_of_item(candidate_item)] == smallest_use_count))
    return selected_items


def assert_item_follows_schema(item: dict) -> None:
    """Fail loudly when an item does not follow docs/conventions.md. Every item passes through here before writing."""
    for key in ("item_id", "task", "kind", "question", "meta"):
        assert key in item, f"item {item.get('item_id')!r} has no {key!r}"
    item_id = item["item_id"]
    assert isinstance(item["question"], str) and item["question"].strip(), f"{item_id}: empty question"
    assert isinstance(item["meta"], dict), f"{item_id}: meta must be a dictionary"
    kind = item["kind"]
    if kind == "multiple_choice":
        choices = item["choices"]
        assert 2 <= len(choices) <= len(choice_letters), f"{item_id}: {len(choices)} options"
        assert all(isinstance(choice, str) and choice.strip() for choice in choices), f"{item_id}: an empty option"
        assert isinstance(item["answer_index"], int) and 0 <= item["answer_index"] < len(choices), (
            f"{item_id}: answer_index {item['answer_index']!r} is outside the {len(choices)} options")
    elif kind == "numeric":
        assert isinstance(item["answer_text"], str) and item["answer_text"].strip(), f"{item_id}: empty answer_text"
    elif kind == "short_answer":
        assert item["answer_aliases"], f"{item_id}: no answer_aliases"
        assert all(normalise_short_answer_text(alias) for alias in item["answer_aliases"]), (
            f"{item_id}: an alias that normalises to nothing would match an empty answer")
    elif kind == "code":
        assert item["test_code"].strip() and item["entry_point"].strip(), f"{item_id}: empty test_code or entry_point"
        assert item["meta"].get("question_form") in ("function_stub", "task_description"), (
            f"{item_id}: meta.question_form must say whether the question is a function stub or a task description")
    else:
        raise AssertionError(f"{item_id}: unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# stage 1e: build the files and the manifest
# ---------------------------------------------------------------------------

def build_evaluation_files_of_task(task_name: str) -> dict:
    """Download, normalise, subsample and write one task; return its manifest entry."""
    task = evaluation_tasks[task_name]
    licence_on_hub_card = read_licence_from_hub_card(task["hub_id"], task["hub_revision"])
    assert licence_on_hub_card == task["licence_on_hub_card"], (
        f"{task_name}: the hub card says licence {licence_on_hub_card!r}, the task table expects "
        f"{task['licence_on_hub_card']!r}: read the card again and update the table")

    evaluated_split_items, few_shot_candidate_items = build_items_of_task(task_name)
    for items in (evaluated_split_items, few_shot_candidate_items):
        assert len({item["item_id"] for item in items}) == len(items), f"{task_name}: an item_id occurs twice"
    assert len(evaluated_split_items) == task["evaluated_split_item_count"], (
        f"{task_name}: the evaluated split has {len(evaluated_split_items):,} items, the task table expects "
        f"{task['evaluated_split_item_count']:,}")
    evaluated_items = sample_evaluated_items(evaluated_split_items, task["evaluated_item_count"],
                                             task["stratified_by_meta_key"])
    assert len(evaluated_items) == task["evaluated_item_count"]

    # A shot must not be a question of the evaluated split: compare with the whole split, not only with the sample,
    # so that the shots stay valid if the sample is ever enlarged. Six ARC train questions are removed here.
    evaluated_question_texts = {normalise_question_for_comparison(item["question"]) for item in evaluated_split_items}
    eligible_candidate_items = [
        candidate_item for candidate_item in few_shot_candidate_items
        if normalise_question_for_comparison(candidate_item["question"]) not in evaluated_question_texts
        and len(render_base_block(candidate_item, with_answer=True)) <= few_shot_item_character_limit]
    few_shot_items = select_few_shot_items(eligible_candidate_items, task["few_shot_item_count"],
                                           task["few_shot_distinct_meta_key"])
    for item in evaluated_items + few_shot_items:
        assert_item_follows_schema(item)
        assert item["kind"] == task["kind"], f"{item['item_id']}: kind {item['kind']!r}, expected {task['kind']!r}"
    for few_shot_item in few_shot_items:
        assert few_shot_item["meta"]["split"] != task["evaluated_split"], (
            f"{few_shot_item['item_id']}: a shot from the evaluated split")
        assert not any(is_same_question(few_shot_item, evaluated_item) for evaluated_item in evaluated_split_items), (
            f"{few_shot_item['item_id']} is also an evaluated question")

    evaluation_directory().mkdir(parents=True, exist_ok=True)
    evaluated_file_name = f"{task_name}.jsonl"
    few_shot_file_name = f"{task_name}_fewshot.jsonl"
    evaluated_file_sha256 = write_json_lines(evaluation_directory() / evaluated_file_name, evaluated_items)
    few_shot_file_sha256 = write_json_lines(evaluation_directory() / few_shot_file_name, few_shot_items)

    if len(evaluated_items) == len(evaluated_split_items):
        subsample_description = "all items of the split"
    elif task["stratified_by_meta_key"] is None:
        subsample_description = f"the {len(evaluated_items):,} items with the smallest seeded hash"
    else:
        subsample_description = (f"{len(evaluated_items):,} items, each {task['stratified_by_meta_key']} in "
                                 f"proportion to its size, smallest seeded hash inside each")
    return {
        "kind": task["kind"],
        "hub_id": task["hub_id"],
        "hub_config": task["hub_config"],
        "hub_revision": task["hub_revision"],
        "evaluated_split": task["evaluated_split"],
        "evaluated_split_item_count": len(evaluated_split_items),
        "item_count": len(evaluated_items),
        "subsample": subsample_description,
        "few_shot_hub_id": task.get("few_shot_hub_id", task["hub_id"]),
        "few_shot_hub_revision": task.get("few_shot_hub_revision", task["hub_revision"]),
        "few_shot_split": task["few_shot_split"],
        "few_shot_item_count": len(few_shot_items),
        "licence": task["licence"],
        "licence_note": task["licence_note"],
        "files": {
            evaluated_file_name: {"item_count": len(evaluated_items), "sha256": evaluated_file_sha256},
            few_shot_file_name: {"item_count": len(few_shot_items), "sha256": few_shot_file_sha256},
        },
    }


def build_evaluation_files(task_names: list[str]) -> int:
    """Build the named tasks and write manifest.json; return how many tasks failed.

    A set that cannot be downloaded must not stop the others and must not be replaced by something else: its error
    is printed, recorded in the manifest under `tasks_not_built`, and the command ends with a non-zero status.
    Entries of tasks that are not rebuilt in this call are kept, so one task can be rebuilt alone.
    """
    import datasets

    manifest_path = evaluation_directory() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    task_entries = manifest.get("tasks", {})
    tasks_not_built = manifest.get("tasks_not_built", {})
    for task_name in task_names:
        print(f"building {task_name} ...", flush=True)
        try:
            task_entries[task_name] = build_evaluation_files_of_task(task_name)
            tasks_not_built.pop(task_name, None)
        except Exception as error:
            traceback.print_exc()
            tasks_not_built[task_name] = f"{type(error).__name__}: {error}"
            print(f"  {task_name} was NOT built: {type(error).__name__}: {error}", flush=True)
            continue
        entry = task_entries[task_name]
        print(f"  {task_name}: {entry['item_count']:,} items of {entry['evaluated_split_item_count']:,} in "
              f"{entry['evaluated_split']}, {entry['few_shot_item_count']} shots, licence {entry['licence']}",
              flush=True)
    manifest = {
        "written_by": "python -m midtrain.eval_tasks build",
        "written_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets_library_version": datasets.__version__,
        "subsample_seed": subsample_seed,
        "tasks": {task_name: task_entries[task_name] for task_name in evaluation_tasks if task_name in task_entries},
        "tasks_not_built": tasks_not_built,
    }
    evaluation_directory().mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} ({len(manifest['tasks'])} tasks built, {len(tasks_not_built)} not built)")
    return len([task_name for task_name in task_names if task_name in tasks_not_built])


# ---------------------------------------------------------------------------
# stage 2: prompts for a base model (few-shot completion) and for a chat-tuned model (messages)
# ---------------------------------------------------------------------------

# Where the text of a base model's answer ends. The caller may pass these to its generator to stop early; the grader
# cuts at them in any case, so the verdict is the same either way.
# A HumanEval completion is a function body, so the first line that starts a new top-level statement ends it.
function_stub_stop_sequences = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\nassert ", "\n#", "\n@", "\n```"]
# An MBPP completion is a whole solution and may define helper functions (5 of the 257 reference solutions do), so
# "\ndef " must not end it; the docstring of the next task does.
task_description_stop_sequences = ['\n"""', "\nclass ", "\nif __name__", "\nprint(", "\nassert ", "\n```"]


def is_chinese_item(item: dict) -> bool:
    """Whether the prompt around this item should be written in Chinese (CMMLU)."""
    return item["meta"].get("language") == "zh"


def render_lettered_choices(choices: list[str]) -> str:
    """Return the options one per line as `A. text`."""
    return "\n".join(f"{choice_letters[choice_index]}. {choice}" for choice_index, choice in enumerate(choices))


def multiple_choice_continuations(item: dict) -> list[str]:
    """Return the continuation of the base prompt for each option, in option order, for log-likelihood scoring.

    The caller scores each continuation after render_base_prompt(item, shots) and takes the most likely one; it is
    correct when its position equals item["answer_index"]. An English prompt ends in `Answer:`, so the letter carries
    its leading space; a Chinese prompt ends in a full-width colon, after which no space is written. The shots are
    rendered with the same strings, so the scored text always looks like the answers the model has just read.
    """
    assert item["kind"] == "multiple_choice", f"{item['item_id']} is not a multiple-choice item"
    letter_prefix = "" if is_chinese_item(item) else " "
    return [letter_prefix + choice_letters[choice_index] for choice_index in range(len(item["choices"]))]


def render_final_answer_sentence(answer_text: str) -> str:
    r"""Return the sentence that ends a worked solution: `The answer is 72.` or `The answer is $\frac{3}{4}$.`

    A plain number stays bare, as in GSM8K prompts; anything else is LaTeX and is put in math mode.
    """
    if is_plain_number_text(answer_text):
        return f"The answer is {answer_text}."
    return f"The answer is ${answer_text}$."


def render_base_block(item: dict, with_answer: bool) -> str:
    """Return one question of a base prompt: a shot when `with_answer`, otherwise the open question at the end."""
    kind = item["kind"]
    if kind == "multiple_choice":
        question_label, answer_label = ("题目：", "答案是：") if is_chinese_item(item) else ("", "Answer:")
        block = f"{question_label}{item['question']}\n{render_lettered_choices(item['choices'])}\n{answer_label}"
        if with_answer:
            block += multiple_choice_continuations(item)[item["answer_index"]]
        return block
    if kind == "numeric":
        block = f"Question: {item['question']}\nAnswer:"
        if with_answer:
            assert "worked_solution" in item["meta"], f"{item['item_id']}: a numeric shot needs meta.worked_solution"
            block += f" {item['meta']['worked_solution']} {render_final_answer_sentence(item['answer_text'])}"
        return block
    if kind == "short_answer":
        block = f"Question: {item['question']}\nAnswer:"
        if with_answer:
            block += f" {item['answer_aliases'][0]}"
        return block
    assert kind == "code" and item["meta"]["question_form"] == "task_description", (
        f"{item['item_id']}: only code items with a task description are rendered as blocks")
    # the task and its tests in a docstring, then the solution: the tests name the function and fix its arguments
    block = '"""\n' + item["question"] + "\n" + "\n".join(item["meta"]["test_list"]) + '\n"""\n'
    if with_answer:
        reference_lines = item["meta"]["reference_solution"].strip("\n").splitlines()
        block += "\n".join(reference_line.rstrip() for reference_line in reference_lines)
    return block


def render_base_prompt(item: dict, few_shot_items: list[dict]) -> str:
    """Return a completion-style few-shot prompt for a base model; the model's text continues it directly.

    multiple choice   a header line, then per question: the question, lettered options, `Answer:` and the letter
    numeric           `Question:` ... `Answer:` a short worked solution ending in `The answer is X.`
    short answer      `Question:` ... `Answer:` the answer
    code, stub        HumanEval's own prompt (signature and docstring); shots are not allowed
    code, task        per task a docstring with the task and its tests, then the solution

    A shot that is the evaluated question itself would give the answer away, so it is refused, not skipped: dropping
    it quietly would leave this one item with a shorter prompt than the others.
    """
    for few_shot_item in few_shot_items:
        assert not is_same_question(item, few_shot_item), (
            f"{item['item_id']} is among its own shots ({few_shot_item['item_id']}): take shots from another split")
        assert few_shot_item["kind"] == item["kind"], (
            f"shot {few_shot_item['item_id']} is {few_shot_item['kind']}, the item {item['item_id']} is {item['kind']}")
    if item["kind"] == "code" and item["meta"]["question_form"] == "function_stub":
        assert not few_shot_items, f"{item['item_id']}: a function stub is completed from its own prompt, without shots"
        return item["question"]
    blocks = [render_base_block(few_shot_item, with_answer=True) for few_shot_item in few_shot_items]
    blocks.append(render_base_block(item, with_answer=False))
    if item["kind"] == "code":
        return "\n\n\n".join(blocks)
    header = ""
    if item["kind"] == "multiple_choice":
        header = ("以下是单项选择题，请直接给出正确答案的选项。\n\n" if is_chinese_item(item)
                  else "The following are multiple choice questions (with answers).\n\n")
    return header + "\n\n".join(blocks)


def base_completion_stop_sequences(item: dict) -> list[str]:
    """Return the strings that end a base model's answer to this item (see the comment on the two code lists)."""
    kind = item["kind"]
    if kind == "code":
        is_function_stub = item["meta"]["question_form"] == "function_stub"
        return function_stub_stop_sequences if is_function_stub else task_description_stop_sequences
    if kind == "numeric":
        return ["\nQuestion:"]         # the model goes on to invent the next question; its answer must not be read
    return ["\n"]                      # a letter or a short answer is one line


def render_chat_messages(item: dict) -> list[dict]:
    r"""Return the `system` and `user` messages for a chat-tuned model.

    Math asks for step-by-step work and the final answer in \boxed{}; multiple choice and short answers ask for the
    letter or the answer in \boxed{}; code asks for a complete Python function in one fenced block.
    """
    kind = item["kind"]
    if kind == "numeric":
        system_text = "You are a careful mathematician."
        user_text = (f"{item['question']}\n\nSolve the problem step by step. Then give the final answer, without "
                     "units, in \\boxed{}.")
    elif kind == "multiple_choice" and is_chinese_item(item):
        system_text = "你是一位知识渊博的助手。"
        user_text = (f"{item['question']}\n{render_lettered_choices(item['choices'])}\n\n"
                     "请选出正确的选项，并把它的字母写在 \\boxed{} 中。")
    elif kind == "multiple_choice":
        system_text = "You are a knowledgeable assistant."
        user_text = (f"{item['question']}\n{render_lettered_choices(item['choices'])}\n\n"
                     "Choose the correct option and give its letter in \\boxed{}.")
    elif kind == "short_answer":
        system_text = "You are a knowledgeable assistant."
        user_text = f"{item['question']}\n\nGive the answer as a name or a short phrase in \\boxed{{}}."
    elif item["meta"]["question_form"] == "function_stub":
        system_text = "You are an expert Python programmer."
        user_text = ("Complete the following Python function.\n\n```python\n" + item["question"].rstrip("\n")
                     + "\n```\n\nReply with the complete function, together with the imports it needs, in one fenced "
                     "```python block.")
    else:
        system_text = "You are an expert Python programmer."
        shown_test_lines = item["meta"]["test_imports"] + item["meta"]["test_list"]
        user_text = (f"{item['question']}\n\nYour code must pass these tests:\n\n```python\n"
                     + "\n".join(shown_test_lines)
                     + "\n```\n\nReply with a complete Python function in one fenced ```python block.")
    return [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]


# ---------------------------------------------------------------------------
# stage 3a: reading the answer out of a completion
# ---------------------------------------------------------------------------

def select_answer_region(item: dict, completion_text: str, prompt_style: str) -> str:
    """Return the part of a completion that holds the answer.

    chat: the text after the last </think>, so that a boxed draft inside the reasoning is not taken for the answer.
    base: the text up to the first stop sequence, so that a question the model went on to invent is not graded.
    """
    if prompt_style == "chat":
        return completion_text.rsplit("</think>", 1)[-1]
    return cut_text_at_first_stop_sequence(completion_text.lstrip(), base_completion_stop_sequences(item))


def cut_text_at_first_stop_sequence(text: str, stop_sequences: list[str]) -> str:
    """Return the text up to the earliest occurrence of any stop sequence (all of it when none occurs)."""
    stop_indexes = [text.find(stop_sequence) for stop_sequence in stop_sequences]
    found_stop_indexes = [stop_index for stop_index in stop_indexes if stop_index >= 0]
    return text[:min(found_stop_indexes)] if found_stop_indexes else text


def find_matching_brace(text: str, open_brace_index: int) -> int:
    r"""Return the index of the brace that closes the one at `open_brace_index`, or -1 when the text ends first.

    A backslash escapes the next character, so the set braces of \{1,2\} are not counted.
    """
    depth = 0
    index = open_brace_index
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


boxed_command_pattern = re.compile(r"\\(?:boxed|fbox)\s*\{")


def find_last_boxed_content(text: str) -> str | None:
    r"""Return what the last complete \boxed{...} holds, or None. A box cut off by the token limit is skipped."""
    for boxed_match in reversed(list(boxed_command_pattern.finditer(text))):
        close_brace_index = find_matching_brace(text, boxed_match.end() - 1)
        if close_brace_index >= 0:
            return text[boxed_match.end():close_brace_index]
    return None


def replace_boxed_with_content(text: str) -> str:
    r"""Return the text with every \boxed{X} replaced by X (shots state the answer in a sentence instead)."""
    while True:
        boxed_match = boxed_command_pattern.search(text)
        if boxed_match is None:
            return text
        close_brace_index = find_matching_brace(text, boxed_match.end() - 1)
        if close_brace_index < 0:
            return text
        text = text[:boxed_match.start()] + text[boxed_match.end():close_brace_index] + text[close_brace_index + 1:]


# a number as it appears in running text; the minus sign counts only when it does not follow a digit, a word or a
# closing bracket, so that "10-5" ends in 5 and not in -5
number_in_text_pattern = re.compile(r"(?:(?<![\w.)])-)?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+)?")
answer_phrase_pattern = re.compile(r"(?i)\banswer\s*(?:is\b|:)[\s:*]*")
words_that_join_two_answers = ("and", "or", "to")


def extract_final_answer_text(answer_region: str) -> str | None:
    r"""Return the final answer a completion states, or None when it states none.

    In order: the last \boxed{...}; else the last `#### x`; else what follows the last `answer is`; else the last
    number in the text.
    """
    boxed_content = find_last_boxed_content(answer_region)
    if boxed_content is not None:
        return boxed_content
    hash_mark_answers = re.findall(r"####[ \t]*([^\n]+)", answer_region)
    if hash_mark_answers:
        return hash_mark_answers[-1]
    answer_phrase_matches = list(answer_phrase_pattern.finditer(answer_region))
    if answer_phrase_matches:
        answer_tail = answer_region[answer_phrase_matches[-1].end():].split("\n", 1)[0].replace("**", "").strip()
        # "18 because she sold ..." or "18 dollars a day": the number is the answer, the words after it are not.
        # "5 and 7" is two answers and is left whole, so it does not match a gold answer of 5.
        number_then_word = re.match(r"(\$?\s*-?\d[\d,]*(?:\.\d+)?\s*%?)\s+([A-Za-z]{2,})", answer_tail)
        if number_then_word and number_then_word.group(2).lower() not in words_that_join_two_answers:
            return number_then_word.group(1)
        end_of_sentence = re.search(r"\.(?=\s|$)", answer_tail)       # a full stop, but not a decimal point
        if end_of_sentence:
            answer_tail = answer_tail[:end_of_sentence.start()]
        if answer_tail.strip():
            return answer_tail.strip()
    numbers_in_text = number_in_text_pattern.findall(answer_region)
    if numbers_in_text:
        return numbers_in_text[-1].rstrip(",")
    return None


# ---------------------------------------------------------------------------
# stage 3b: comparing math answers (numbers numerically, everything else as normalised LaTeX)
# ---------------------------------------------------------------------------

latex_text_command_pattern = re.compile(
    r"\\(?:text|textbf|textit|textrm|textsf|mathrm|mathbf|mathit|mbox|operatorname)\s*\{([^{}]*)\}")


def replace_latex_text_commands(text: str, should_drop_units: bool) -> str:
    r"""Return the text with \text{...} and its relatives replaced by their content.

    With `should_drop_units` (math answers): words after a quantity are a unit and are removed together with an
    exponent on them, so `864 \mbox{ inches}^2` becomes `864`; `\text{ and }` between two answers becomes a comma.
    A wrapper around a whole answer, as in `\text{east}`, always keeps its content.
    """
    while True:
        command_match = latex_text_command_pattern.search(text)
        if command_match is None:
            return text
        content = command_match.group(1)
        text_before = text[:command_match.start()]
        text_after = text[command_match.end():]
        replacement = content
        if should_drop_units:
            stripped_before = text_before.rstrip()
            follows_a_quantity = bool(stripped_before) and (stripped_before[-1].isdigit()
                                                            or stripped_before[-1] in "})")
            if content.strip().lower() in ("and", "or"):
                replacement = ","
            elif follows_a_quantity and re.fullmatch(r"[A-Za-z\s./\-]*", content):
                replacement = ""
                text_after = re.sub(r"^\s*\^\s*(?:\{\s*\d\s*\}|\d)", "", text_after)
        text = text_before + replacement + text_after


def read_latex_argument(text: str, start_index: int) -> tuple[str | None, int]:
    r"""Return (argument, index after it) for the LaTeX argument at `start_index`, or (None, start_index).

    An argument is a braced group, a command such as \pi, or a single character: \frac12 means \frac{1}{2}.
    """
    if start_index >= len(text):
        return None, start_index
    if text[start_index] == "{":
        close_brace_index = find_matching_brace(text, start_index)
        if close_brace_index < 0:
            return None, start_index
        return text[start_index + 1:close_brace_index], close_brace_index + 1
    command_match = re.compile(r"\\[A-Za-z]+").match(text, start_index)
    if command_match:
        return command_match.group(0), command_match.end()
    return text[start_index], start_index + 1


def add_braces_to_shorthand_fractions(text: str) -> str:
    r"""Return the text with every fraction written as \frac{a}{b}: \frac43, \frac9{19} and \frac{270}7 all occur."""
    pieces = []
    index = 0
    while True:
        command_index = text.find("\\frac", index)
        if command_index < 0:
            pieces.append(text[index:])
            break
        pieces.append(text[index:command_index])
        numerator, index_after_numerator = read_latex_argument(text, command_index + len("\\frac"))
        denominator, index_after_denominator = (None, index_after_numerator)
        if numerator is not None:
            denominator, index_after_denominator = read_latex_argument(text, index_after_numerator)
        if denominator is None:              # a fraction cut off mid-way: keep the rest as it was written
            pieces.append(text[command_index:])
            break
        pieces.append("\\frac{" + add_braces_to_shorthand_fractions(numerator) + "}{"
                      + add_braces_to_shorthand_fractions(denominator) + "}")
        index = index_after_denominator
    return "".join(pieces)


def normalise_math_answer_text(answer_text: str) -> str:
    r"""Return a math answer in a canonical spelling, so that two ways of writing one answer compare equal.

    The rules follow the answers that occur in MATH-500 and GSM8K (the unit tests list one case per rule):
      thousands separators   1,234   10,\!080   10{,}080             -> 1234 10080 (only when the whole answer is one
                             number: the commas of (1,2) and of the list 3, 5, 7 stay)
      currency and percent   $12   \$18.90   50\%                    -> 12 18.90 50
      units and degrees      5.4 \text{ cents}   864 \mbox{ inches}^2   90^\circ   18 dollars   -> 5.4 864 90 18
      text answers           \text{east}   \text{(C)}   East         -> east c east
      LaTeX spelling         \dfrac \tfrac \left \right \! \,   \frac43   \sqrt2   4210_{5}   .5   x=5   trailing "."
    Numbers are compared by value afterwards (is_math_answer_equivalent), so 0.75 and \frac{3}{4} need no rule here.
    """
    text = answer_text.strip().replace("\u2212", "-")                # the Unicode minus sign
    boxed_content = find_last_boxed_content(text)
    if boxed_content is not None and text.startswith("\\boxed"):
        text = boxed_content
    text = text.replace(",\\!", "").replace("{,}", "")              # thousands separators the LaTeX way
    text = re.sub(r"\\(?:!|,|;|:)|\\q?quad(?![A-Za-z])|(?<!\\)\\ |~", " ", text)      # spacing commands
    text = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", text)        # not \leftarrow or \rightarrow
    text = re.sub(r"\\[dt]frac(?![A-Za-z])", r"\\frac", text)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ(?![A-Za-z])|\\degree(?![A-Za-z])|°", "", text)
    text = text.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    text = re.sub(r"(?<!\\)\\[(\[)\]]", "", text)                    # the math-mode brackets \( \) \[ \]
    text = replace_latex_text_commands(text, should_drop_units=True)
    text = re.sub(r"\.+$", "", text.strip()).strip()                 # a full stop that ended the sentence

    # "x = 5" and "y = 2x + 3" are answers to "find x" and "find the line": the name on the left is dropped when
    # it is at most two characters (the rule of the MATH reference grader); "x \in [-2,7]" loses its "x \in"
    sides_of_equation = text.split("=")
    if len(sides_of_equation) == 2 and 0 < len(sides_of_equation[0].strip()) <= 2:
        text = sides_of_equation[1].strip()
    text = re.sub(r"^[A-Za-z]\s*\\in(?![A-Za-z])\s*", "", text)

    # "18 dollars" and "5 square units": a number, a space and then only words is a quantity with its unit.
    # One-letter words are kept (they are variables, as in "2 x") and so are "and", "or", "to" (two answers).
    number_then_unit_words = re.fullmatch(r"(-?[\d,]*\.?\d+(?:\s*/\s*\d+)?)\s+((?:[A-Za-z]{2,}\.?\s*)+)", text)
    if number_then_unit_words:
        unit_words = number_then_unit_words.group(2).lower().split()
        if not any(unit_word.rstrip(".") in words_that_join_two_answers for unit_word in unit_words):
            text = number_then_unit_words.group(1)
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")

    text = re.sub(r"\s+", "", text)
    text = add_braces_to_shorthand_fractions(text)
    text = re.sub(r"\\sqrt([A-Za-z0-9])", r"\\sqrt{\1}", text)
    text = re.sub(r"([_^])\{([A-Za-z0-9])\}", r"\1\2", text)        # 4210_{5} and 4210_5
    text = re.sub(r"^(-?)\.(\d)", r"\g<1>0.\2", text)               # .35625 and 0.35625
    text = re.sub(r"^\(([A-Za-z])\)$", r"\1", text)                 # the option letter (C)
    if text.isalpha():
        text = text.lower()
    return text


def parse_normalised_math_answer_as_number(normalised_text: str) -> float | None:
    r"""Return the value of a normalised answer that is a number, or None when it is an expression.

    Understood: decimals, a/b, \frac{a}{b} and mixed numbers such as 1\frac{4}{5} (which LaTeX answers write without
    a plus sign).
    """
    decimal_pattern = r"(?:\d+\.?\d*|\.\d+)"
    if re.fullmatch(rf"[-+]?{decimal_pattern}", normalised_text):
        return float(normalised_text)
    simple_fraction = re.fullmatch(rf"([-+]?{decimal_pattern})/({decimal_pattern})", normalised_text)
    if simple_fraction and float(simple_fraction.group(2)) != 0:
        return float(simple_fraction.group(1)) / float(simple_fraction.group(2))
    latex_fraction = re.fullmatch(rf"([-+]?)(\d*)\\frac\{{([-+]?{decimal_pattern})\}}\{{({decimal_pattern})\}}",
                                  normalised_text)
    if latex_fraction and float(latex_fraction.group(4)) != 0:
        sign = -1.0 if latex_fraction.group(1) == "-" else 1.0
        whole_part = float(latex_fraction.group(2)) if latex_fraction.group(2) else 0.0
        return sign * (whole_part + float(latex_fraction.group(3)) / float(latex_fraction.group(4)))
    return None


def is_math_answer_equivalent(predicted_answer_text: str, gold_answer_text: str) -> bool:
    r"""Whether a predicted math answer equals the gold one.

    When both are numbers they are compared by value with a relative tolerance of 1e-6, so 0.75, 3/4 and
    \frac{3}{4} agree and a rounded 4.67 does not match \frac{14}{3}. Otherwise the normalised strings must be
    equal, with \frac{a}{b} of two integers spelled a/b so that a vector may use either form.
    """
    normalised_prediction = normalise_math_answer_text(predicted_answer_text)
    normalised_gold = normalise_math_answer_text(gold_answer_text)
    predicted_number = parse_normalised_math_answer_as_number(normalised_prediction)
    gold_number = parse_normalised_math_answer_as_number(normalised_gold)
    if predicted_number is not None and gold_number is not None:
        return math.isclose(predicted_number, gold_number, rel_tol=1e-6)
    integer_fraction_pattern = r"(?<!\d)\\frac\{(-?\d+)\}\{(\d+)\}"     # not after a digit: 1\frac{4}{5} is 1 + 4/5
    spelled_prediction = re.sub(integer_fraction_pattern, r"\1/\2", normalised_prediction)
    spelled_gold = re.sub(integer_fraction_pattern, r"\1/\2", normalised_gold)
    return bool(spelled_gold) and spelled_prediction == spelled_gold


# ---------------------------------------------------------------------------
# stage 3c: multiple choice, short answers, code programs
# ---------------------------------------------------------------------------

def normalise_short_answer_text(answer_text: str) -> str:
    """Normalise a short answer the way the TriviaQA evaluation script does.

    Lowercase, underscores and punctuation become spaces, the articles a, an and the are dropped, whitespace is
    collapsed. "The Beatles!" and "beatles" then agree.
    """
    punctuation_characters = set(string.punctuation + "‘’´`")
    text = answer_text.lower().replace("_", " ")
    text = "".join(" " if character in punctuation_characters else character for character in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_choice_index(answer_region: str, item: dict, prompt_style: str) -> int | None:
    r"""Return the index of the option a completion chooses, or None when it chooses none.

    base: the completion continues `Answer:`, so the letter is the first thing in it.
    chat: the last \boxed{...}; when it holds the option's text instead of its letter, the text is matched against
    the options. Without a box, the last `answer is X`, or a completion that is nothing but a letter.
    """
    valid_letters = choice_letters[:len(item["choices"])]
    chosen_letter = None
    if prompt_style == "base":
        letter_at_start = re.match(r"\s*[(（]?([A-Z])[)）]?(?![A-Za-z])", answer_region)
        if letter_at_start:
            chosen_letter = letter_at_start.group(1)
    else:
        boxed_content = find_last_boxed_content(answer_region)
        if boxed_content is not None:
            boxed_text = replace_latex_text_commands(boxed_content, should_drop_units=False).strip()
            letter_in_box = (re.fullmatch(r"[(（]?([A-Za-z])[)）]?[.:]?", boxed_text)
                             or re.match(r"[(（]?([A-Z])[)）]?[.:]\s", boxed_text))
            if letter_in_box:
                chosen_letter = letter_in_box.group(1).upper()
            else:
                normalised_box = normalise_short_answer_text(boxed_text)
                for choice_index, choice in enumerate(item["choices"]):
                    if normalised_box and normalise_short_answer_text(choice) == normalised_box:
                        return choice_index
        else:
            stated_letters = re.findall(
                r"(?:[Aa]nswer\s*(?:is|:)|答案\s*(?:是|为)?\s*[:：]?)[\s*(（\[]*([A-J])(?![A-Za-z])", answer_region)
            only_a_letter = re.fullmatch(r"[\s*(（]*([A-J])[\s*)）.:]*", answer_region)
            if stated_letters:
                chosen_letter = stated_letters[-1]
            elif only_a_letter:
                chosen_letter = only_a_letter.group(1)
    if chosen_letter is None or chosen_letter not in valid_letters:
        return None
    return choice_letters.index(chosen_letter)


def extract_short_answer_text(answer_region: str, prompt_style: str) -> str:
    r"""Return the short answer of a completion: the last \boxed{...} of a chat answer, else its first line."""
    if prompt_style == "chat":
        boxed_content = find_last_boxed_content(answer_region)
        if boxed_content is not None:
            return replace_latex_text_commands(boxed_content, should_drop_units=False)
    return answer_region.strip().split("\n", 1)[0]


fenced_block_pattern = re.compile(r"```[ \t]*([A-Za-z0-9_+\-]*)[^\n]*\n(.*?)```", re.DOTALL)


def extract_first_fenced_python_block(text: str) -> str:
    """Return the code of the first fenced Python block of a chat answer.

    In order: the first block tagged python, py or python3; else the first block without a tag; else, when a fence
    was opened and never closed (the answer hit the token limit), the text after it; else the whole text, so that
    an answer that is bare correct code is not failed for its formatting.
    """
    fenced_blocks = fenced_block_pattern.findall(text)
    python_blocks = [code for language, code in fenced_blocks if language.lower() in ("python", "py", "python3")]
    if python_blocks:
        return python_blocks[0]
    untagged_blocks = [code for language, code in fenced_blocks if not language]
    if untagged_blocks:
        return untagged_blocks[0]
    unclosed_block = re.search(r"```[^\n]*\n(.*)\Z", text, re.DOTALL)
    if not fenced_blocks and unclosed_block:
        return unclosed_block.group(1)
    return text


def build_code_program(item: dict, completion_text: str, prompt_style: str) -> str:
    """Return the program that tests a code completion: the solution, then the item's tests.

    base: the completion up to its first stop sequence. For a function stub (HumanEval) the item's prompt comes
    first, because the completion is only the function body.
    chat: the first fenced Python block. For a function stub the prompt still comes first: it carries the imports
    and helper functions that the model was shown, and a complete function in the block simply redefines the stub.
    """
    is_function_stub = item["meta"]["question_form"] == "function_stub"
    if prompt_style == "base":
        solution_text = cut_text_at_first_stop_sequence(completion_text, base_completion_stop_sequences(item))
        if is_function_stub:
            solution_text = item["question"] + solution_text
    else:
        solution_text = extract_first_fenced_python_block(completion_text.rsplit("</think>", 1)[-1])
        if is_function_stub:
            solution_text = item["question"] + "\n" + solution_text
    return solution_text.rstrip("\n") + "\n\n\n" + item["test_code"]


# ---------------------------------------------------------------------------
# stage 3d: the graders
# ---------------------------------------------------------------------------

def grade_completion(item: dict, completion_text: str, prompt_style: str, timeout_seconds: float = 10) -> bool:
    r"""Whether a completion answers the item correctly. `prompt_style` is "base" or "chat".

    numeric          the last \boxed{...}, else `#### x`, else `The answer is x`, else the last number; compared by
                     value when both sides are numbers, otherwise as normalised LaTeX
    multiple_choice  the letter
    short_answer     the normalised answer equals a normalised alias (the TriviaQA exact match)
    code             the program of build_code_program passes in the sandbox within `timeout_seconds`
    """
    assert prompt_style in ("base", "chat"), f"prompt_style must be 'base' or 'chat', not {prompt_style!r}"
    kind = item["kind"]
    if kind == "code":
        program_text = build_code_program(item, completion_text, prompt_style)
        return run_python_program_in_sandbox(program_text, timeout_seconds)["passed"]
    answer_region = select_answer_region(item, completion_text, prompt_style)
    if kind == "numeric":
        predicted_answer_text = extract_final_answer_text(answer_region)
        return predicted_answer_text is not None and is_math_answer_equivalent(predicted_answer_text,
                                                                               item["answer_text"])
    if kind == "multiple_choice":
        return extract_choice_index(answer_region, item, prompt_style) == item["answer_index"]
    assert kind == "short_answer", f"{item['item_id']}: unknown kind {kind!r}"
    normalised_prediction = normalise_short_answer_text(extract_short_answer_text(answer_region, prompt_style))
    normalised_aliases = {normalise_short_answer_text(alias) for alias in item["answer_aliases"]}
    return bool(normalised_prediction) and normalised_prediction in normalised_aliases


def estimate_pass_at_k(correct_sample_count: int, sample_count: int, k: int) -> float:
    """Return the unbiased estimate of pass@k from n samples of which c are correct: 1 - C(n-c, k) / C(n, k).

    It is the probability that k samples drawn without replacement from the n include a correct one. Taking the best
    of the first k samples instead would be unbiased too but far noisier; 1 - (1 - c/n)^k is biased. With fewer
    than k wrong samples every draw of k contains a correct one, and math.comb returns 0 for that case.
    """
    assert 0 <= correct_sample_count <= sample_count, (
        f"{correct_sample_count} correct samples out of {sample_count}")
    assert 1 <= k <= sample_count, f"pass@{k} needs at least {k} samples per problem, got {sample_count}"
    return 1.0 - math.comb(sample_count - correct_sample_count, k) / math.comb(sample_count, k)


def count_usable_processors() -> int:
    """Return how many processors this process may use (a batch job often gets fewer than the machine has)."""
    return len(os.sched_getaffinity(0))


def grade_code_completions_in_parallel(items: list[dict], completion_texts: list[str], prompt_style: str,
                                       timeout_seconds: float = 10, worker_count: int | None = None) -> list[dict]:
    """Grade many code completions at once; return {"passed", "reason"} for each, in the order given.

    `items[i]` is the item that `completion_texts[i]` answers; an item appears once per sample for pass@k. Each
    program runs in its own sandbox process, so the workers are threads that only wait. The default is half the
    usable processors: the timeout is wall-clock time, and with one worker per processor a slow but correct program
    could time out whenever the machine is busy with something else.
    """
    assert len(items) == len(completion_texts), f"{len(items)} items for {len(completion_texts)} completions"
    if worker_count is None:
        worker_count = max(1, count_usable_processors() // 2)
    program_texts = [build_code_program(item, completion_text, prompt_style)
                     for item, completion_text in zip(items, completion_texts)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as worker_pool:
        verdicts = list(worker_pool.map(lambda program_text: run_python_program_in_sandbox(program_text,
                                                                                           timeout_seconds),
                                        program_texts))
    unsandboxed_verdict_count = sum(verdict["reason"] == "no_sandbox" for verdict in verdicts)
    if unsandboxed_verdict_count:
        print(f"WARNING: {unsandboxed_verdict_count:,} of {len(verdicts):,} programs were counted as failed because "
              "no sandbox works on this machine; these are not model failures", file=sys.stderr, flush=True)
    return verdicts


# ---------------------------------------------------------------------------
# stage 4: the sandbox for model-written programs
# ---------------------------------------------------------------------------
# A model-written program must not reach the network and must not read or write a file outside its own temporary
# folder: it runs next to the checkpoints and the training data, with the rights of the person who evaluates.
#
#   bubblewrap   when `bwrap` is on PATH. The program sees the system folders and the Python installation
#                read-only, an empty /tmp and its own folder; no home folder, no other file, no network, no other
#                process; it is killed together with its parent.
#   apptainer    when $MIDTRAIN_SANDBOX_IMAGE names an image: the same view from `--containall`.
#   none         every program counts as failed with reason "no_sandbox" (fail closed), unless
#                MIDTRAIN_ALLOW_UNSANDBOXED=1 allows running it directly with the resource limits only.
#
# The first backend that runs a one-line probe program wins. What the sandbox does not do: it does not cap the
# number of processes or the total size of the files a program writes within its timeout, and its verdict line is
# safe against accidents (an early exit, printed text), not against a program written to attack this harness.

sandbox_memory_limit_bytes = 2 * 1024 ** 3            # address space; BLAS is held to one thread so numpy imports
sandbox_written_file_limit_bytes = 16 * 1024 ** 2     # per file; a loop that writes forever is stopped here
sandbox_open_file_limit = 256
sandbox_probe_timeout_seconds = 60.0                  # the first container start on a busy node can take long
# A probe that passed is kept for the life of the process. A probe that failed is repeated after this long: in a
# sister project a failed probe was cached for good, and one bad moment on one node failed every code answer that
# was graded there for the rest of a run.
sandbox_probe_retry_seconds = 60.0
sandbox_probe_state = {"backend_name": None, "last_failed_probe_time": None}
sandbox_probe_lock = threading.Lock()

# The program that runs inside the sandbox. It applies the limits with `resource` before any model-written line
# runs (a preexec_fn in the parent is not safe with threads), sends the program's own output to /dev/null so a
# print loop cannot fill a pipe or a disk, and reports on a private copy of stdout. The verdict line starts with a
# nonce that arrives on stdin, so nothing the program prints can pass for it; it is written only after the whole
# program, tests included, ran. An exit() or os._exit() in a solution therefore never looks like a pass. The
# program runs under the module name "sandboxed_program", not "__main__": a demo under `if __name__ == "__main__":`
# is not part of the solution and is skipped.
sandbox_runner_source = r'''
import os
import resource
import sys
import types


def run_program_and_report():
    program_path = sys.argv[1]
    processor_seconds, memory_bytes, written_file_bytes, open_file_count = (int(value) for value in sys.argv[2:6])
    verdict_nonce = sys.stdin.readline().strip()
    verdict_stream = os.fdopen(os.dup(1), "w")
    null_descriptor = os.open(os.devnull, os.O_RDWR)
    for standard_descriptor in (0, 1, 2):
        os.dup2(null_descriptor, standard_descriptor)
    resource.setrlimit(resource.RLIMIT_CPU, (processor_seconds, processor_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (written_file_bytes, written_file_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_count, open_file_count))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    program_module = types.ModuleType("sandboxed_program")
    program_module.__file__ = program_path
    sys.modules["sandboxed_program"] = program_module
    sys.argv = [program_path]
    try:
        with open(program_path, encoding="utf-8") as program_file:
            program_source = program_file.read()
        exec(compile(program_source, program_path, "exec"), program_module.__dict__)
        verdict = "passed"
    except BaseException as error:
        verdict = "failed:" + type(error).__name__
    verdict_stream.write(verdict_nonce + " " + verdict + "\n")
    verdict_stream.flush()
    os._exit(0)


run_program_and_report()
'''


def find_python_for_sandbox() -> tuple[str, list[str]]:
    """Return (the interpreter to start inside the sandbox, the folders it needs to see read-only).

    A virtual environment needs its own folder and the installation it was created from. The interpreter is named
    by the path that started this process when that path lies inside one of them, so that the environment's
    packages are found; otherwise by the file it resolves to.
    """
    python_prefixes = sorted({os.path.realpath(prefix) for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
                             | {sys.prefix, sys.base_prefix})
    python_path = sys.executable
    is_inside_a_prefix = any(python_path.startswith(prefix.rstrip("/") + "/") for prefix in python_prefixes)
    if not is_inside_a_prefix:
        python_path = os.path.realpath(sys.executable)
        python_prefixes = sorted(set(python_prefixes) | {str(Path(python_path).parents[1])})
    return python_path, python_prefixes


def build_bubblewrap_command(bubblewrap_path: str, sandbox_directory: str, python_prefixes: list[str],
                             environment: dict[str, str]) -> list[str]:
    """Return the bubblewrap prefix that shows a program the system, Python, an empty /tmp and its own folder.

    The tmpfs on /tmp is mounted before the binds: a later mount hides an earlier one, and the program's folder
    usually lies under /tmp. /bin, /lib and their relatives are symlinks into /usr on current systems and real
    folders on older ones; both are reproduced. /etc and the home folder are not shown at all. The root that holds
    these mount points is made read-only at the end: without that a write to any other path succeeds (measured: a
    file "in the home folder" could be created; it lived in the sandbox's memory and never reached the disk, but a
    write outside the program's folder should fail, not vanish).
    """
    command = [bubblewrap_path, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--ro-bind", "/usr", "/usr"]
    for system_directory in ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32"):
        if os.path.islink(system_directory):
            command += ["--symlink", os.readlink(system_directory), system_directory]
        elif os.path.isdir(system_directory):
            command += ["--ro-bind", system_directory, system_directory]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for python_prefix in python_prefixes:
        command += ["--ro-bind", python_prefix, python_prefix]
    command += ["--bind", sandbox_directory, sandbox_directory, "--remount-ro", "/", "--chdir", sandbox_directory]
    for variable_name, variable_value in sorted(environment.items()):
        command += ["--setenv", variable_name, variable_value]
    return command


def build_apptainer_command(apptainer_path: str, image_path: str, sandbox_directory: str,
                            python_prefixes: list[str], environment: dict[str, str]) -> list[str]:
    """Return the Apptainer prefix with the same view: `--containall` hides the host, `--network none` the network.

    `--cleanenv` drops the caller's environment, so every variable is passed with `--env`.
    """
    command = [apptainer_path, "exec", "--containall", "--no-home", "--cleanenv", "--net", "--network", "none",
               "--bind", f"{sandbox_directory}:{sandbox_directory}", "--pwd", sandbox_directory]
    for python_prefix in python_prefixes:
        command += ["--bind", f"{python_prefix}:{python_prefix}:ro"]
    command += [f"--env={variable_name}={variable_value}" for variable_name, variable_value in
                sorted(environment.items())]
    return command + [image_path]


def run_program_with_sandbox_backend(program_text: str, timeout_seconds: float, backend_name: str) -> dict:
    """Run one program with the named backend and return {"passed": bool, "reason": str}.

    Reasons: "passed"; "failed:<ExceptionName>" (the program raised, AssertionError for a failed test);
    "timeout"; "no_verdict:return_code=<n>" (the process ended without reporting: a resource limit killed it, it
    called os._exit, or the sandbox command itself failed, in which case its last error line is appended).
    """
    sandbox_directory = tempfile.mkdtemp(prefix="midtrain_sandbox_")
    try:
        program_path = os.path.join(sandbox_directory, "program.py")
        runner_path = os.path.join(sandbox_directory, "sandbox_runner.py")
        with open(program_path, "w", encoding="utf-8") as program_file:
            program_file.write(program_text)
        with open(runner_path, "w", encoding="utf-8") as runner_file:
            runner_file.write(sandbox_runner_source)
        python_path, python_prefixes = find_python_for_sandbox()
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": sandbox_directory,
            "LANG": "C.UTF-8",
            "PYTHONHASHSEED": "0",              # the same completion gets the same verdict on every run
            "PYTHONSAFEPATH": "1",              # the program's folder is not put on the import path
            "PYTHONDONTWRITEBYTECODE": "1",
            # One thread for the numeric libraries: OpenBLAS otherwise reserves a buffer per processor when numpy
            # is imported, which on a 64-processor machine does not fit in the address-space limit.
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        }
        if backend_name == "bubblewrap":
            command_prefix = build_bubblewrap_command(shutil.which("bwrap"), sandbox_directory, python_prefixes,
                                                      environment)
        elif backend_name == "apptainer":
            environment["TMPDIR"] = sandbox_directory
            apptainer_path = shutil.which("apptainer") or shutil.which("singularity")
            command_prefix = build_apptainer_command(apptainer_path, os.environ["MIDTRAIN_SANDBOX_IMAGE"],
                                                     sandbox_directory, python_prefixes, environment)
        else:
            assert backend_name == "unsandboxed", f"unknown sandbox backend {backend_name!r}"
            environment["TMPDIR"] = sandbox_directory
            command_prefix = []
        # -s and the explicit environment instead of -I: isolated mode would also ignore PYTHONHASHSEED
        processor_seconds = int(math.ceil(timeout_seconds)) + 2      # a backstop behind the wall-clock timeout
        command = command_prefix + [python_path, "-s", runner_path, program_path, str(processor_seconds),
                                    str(sandbox_memory_limit_bytes), str(sandbox_written_file_limit_bytes),
                                    str(sandbox_open_file_limit)]
        verdict_nonce = secrets.token_hex(16)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   cwd=sandbox_directory, env=environment,
                                   start_new_session=True)          # its own process group, killed as a whole
        try:
            output_bytes, error_bytes = process.communicate(input=(verdict_nonce + "\n").encode("utf-8"),
                                                            timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)   # bubblewrap dies, and with it every process inside
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:            # a survivor still holds the pipe: stop waiting for it
                process.kill()
            return {"passed": False, "reason": "timeout"}
        for output_line in reversed(output_bytes.decode("utf-8", "replace").splitlines()):
            if output_line.startswith(verdict_nonce + " "):
                verdict = output_line[len(verdict_nonce) + 1:]
                return {"passed": verdict == "passed", "reason": verdict}
        reason = f"no_verdict:return_code={process.returncode}"
        error_lines = error_bytes.decode("utf-8", "replace").strip().splitlines()
        if error_lines:
            reason += f":{error_lines[-1][:200]}"
        return {"passed": False, "reason": reason}
    finally:
        shutil.rmtree(sandbox_directory, ignore_errors=True)


def list_candidate_sandbox_backends() -> list[str]:
    """Return the backends this machine and environment offer, in the order they are tried."""
    candidate_backend_names = []
    if shutil.which("bwrap"):
        candidate_backend_names.append("bubblewrap")
    has_apptainer = shutil.which("apptainer") or shutil.which("singularity")
    if os.environ.get("MIDTRAIN_SANDBOX_IMAGE") and has_apptainer:
        candidate_backend_names.append("apptainer")
    if os.environ.get("MIDTRAIN_ALLOW_UNSANDBOXED") == "1":
        candidate_backend_names.append("unsandboxed")
    return candidate_backend_names


def find_working_sandbox_backend(clock=time.monotonic) -> str | None:
    """Return the name of the first backend that runs a probe program, or None when none does.

    A backend that passed is kept for the life of the process; after a failure the probe is repeated once
    `sandbox_probe_retry_seconds` have passed (see the comment on that value). The lock makes the workers of a
    thread pool wait for one probe instead of each starting its own.
    """
    with sandbox_probe_lock:
        if sandbox_probe_state["backend_name"] is not None:
            return sandbox_probe_state["backend_name"]
        last_failed_probe_time = sandbox_probe_state["last_failed_probe_time"]
        if last_failed_probe_time is not None and clock() - last_failed_probe_time < sandbox_probe_retry_seconds:
            return None
        probe_reports = []
        for backend_name in list_candidate_sandbox_backends():
            try:
                verdict = run_program_with_sandbox_backend("assert 6 * 7 == 42\n", sandbox_probe_timeout_seconds,
                                                           backend_name)
            except Exception as error:               # a missing program or a full disk must not stop the grading
                verdict = {"passed": False, "reason": f"{type(error).__name__}: {error}"}
            if verdict["passed"]:
                if backend_name == "unsandboxed":
                    print("WARNING: MIDTRAIN_ALLOW_UNSANDBOXED=1 and no sandbox works: model-written programs run "
                          "directly on this machine, with resource limits only", file=sys.stderr, flush=True)
                sandbox_probe_state["backend_name"] = backend_name
                return backend_name
            probe_reports.append(f"{backend_name}: {verdict['reason']}")
        sandbox_probe_state["last_failed_probe_time"] = clock()
        print("WARNING: no working sandbox (" + ("; ".join(probe_reports) or "no bwrap on PATH, no "
              "$MIDTRAIN_SANDBOX_IMAGE") + f"). Every program counts as failed with reason no_sandbox; the probe is "
              f"repeated in {sandbox_probe_retry_seconds:.0f} s.", file=sys.stderr, flush=True)
        return None


def run_python_program_in_sandbox(program_text: str, timeout_seconds: float = 10) -> dict:
    """Run a model-written Python program in the sandbox and return {"passed": bool, "reason": str}.

    The program passes when it runs to its last line without raising, within `timeout_seconds` of wall-clock time.
    Without a working sandbox it is not run at all and the reason is "no_sandbox" (see the comment that opens this
    stage); run_program_with_sandbox_backend lists the other reasons.
    """
    backend_name = find_working_sandbox_backend()
    if backend_name is None:
        return {"passed": False, "reason": "no_sandbox"}
    return run_program_with_sandbox_backend(program_text, timeout_seconds, backend_name)


# ---------------------------------------------------------------------------
# commands: build, check, show
# ---------------------------------------------------------------------------

def check_built_task(task_name: str, manifest_entry: dict) -> list[str]:
    """Check one built task end to end and return the problems found (an empty list when all is well).

    The files must match the manifest; every prompt must render; every grader must accept the gold answer in the
    forms a model would write it and reject a wrong one; every reference solution of a code task must pass in the
    sandbox and a solution that only raises must fail.
    """
    problems = []
    task = evaluation_tasks[task_name]
    for file_name, file_record in manifest_entry["files"].items():
        file_sha256 = hashlib.sha256((evaluation_directory() / file_name).read_bytes()).hexdigest()
        if file_sha256 != file_record["sha256"]:
            problems.append(f"{file_name} does not have the sha256 recorded in the manifest")
    items = load_evaluation_items(task_name)
    few_shot_items = load_few_shot_items(task_name)
    if len(items) != task["evaluated_item_count"] or len(few_shot_items) != task["few_shot_item_count"]:
        problems.append(f"{len(items)} items and {len(few_shot_items)} shots, the task table expects "
                        f"{task['evaluated_item_count']} and {task['few_shot_item_count']}")
    longest_base_prompt_characters = 0
    for item in items:
        assert_item_follows_schema(item)
        longest_base_prompt_characters = max(longest_base_prompt_characters,
                                             len(render_base_prompt(item, few_shot_items)))
        chat_messages = render_chat_messages(item)
        assert [message["role"] for message in chat_messages] == ["system", "user"]

    if task.get("few_shot_hub_id") == evaluation_tasks["gsm8k"]["hub_id"]:
        gsm8k_shot_ids = [shot["item_id"] for shot in load_few_shot_items("gsm8k")]
        if [shot["item_id"] for shot in few_shot_items] != gsm8k_shot_ids:
            problems.append("the shots differ from the GSM8K shots, so the prompt is not the GSM8K prompt")
    wrongly_graded_item_ids = []
    sandbox_report = ""
    if task["kind"] == "numeric":
        for item in items:
            gold_sentence = " " + render_final_answer_sentence(item["answer_text"])
            is_graded_well = (grade_completion(item, gold_sentence, "base")
                              and grade_completion(item, "\\boxed{" + item["answer_text"] + "}", "chat")
                              and grade_completion(item, item["meta"]["reference_solution"], "chat")
                              and not grade_completion(item, "\\boxed{" + item["answer_text"] + "1}", "chat"))
            if not is_graded_well:
                wrongly_graded_item_ids.append(item["item_id"])
    elif task["kind"] == "multiple_choice":
        for item in items:
            continuations = multiple_choice_continuations(item)
            wrong_index = (item["answer_index"] + 1) % len(item["choices"])
            is_graded_well = (grade_completion(item, continuations[item["answer_index"]], "base")
                              and not grade_completion(item, continuations[wrong_index], "base")
                              and grade_completion(item, f"\\boxed{{{choice_letters[item['answer_index']]}}}", "chat")
                              and not grade_completion(item, f"\\boxed{{{choice_letters[wrong_index]}}}", "chat"))
            if not is_graded_well:
                wrongly_graded_item_ids.append(item["item_id"])
    elif task["kind"] == "short_answer":
        for item in items:
            is_graded_well = (all(grade_completion(item, " " + alias, "base") for alias in item["answer_aliases"])
                              and grade_completion(item, "\\boxed{" + item["answer_aliases"][0] + "}", "chat")
                              and not grade_completion(item, " no such answer", "base"))
            if not is_graded_well:
                wrongly_graded_item_ids.append(item["item_id"])
    else:
        reference_base_completions, reference_chat_completions, broken_base_completions = [], [], []
        for item in items:
            reference_solution = item["meta"]["reference_solution"]
            if item["meta"]["question_form"] == "function_stub":
                reference_base_completions.append(reference_solution)
                reference_chat_completions.append(f"```python\n{item['question']}{reference_solution}\n```")
                broken_base_completions.append("    raise RuntimeError('deliberately broken')\n")
            else:
                reference_base_completions.append(reference_solution)
                reference_chat_completions.append(f"Here is the function.\n\n```python\n{reference_solution}\n```")
                broken_base_completions.append(f"def {item['entry_point']}(*arguments, **keyword_arguments):\n"
                                               "    raise RuntimeError('deliberately broken')\n")
        start_time = time.monotonic()
        base_verdicts = grade_code_completions_in_parallel(items, reference_base_completions, "base")
        chat_verdicts = grade_code_completions_in_parallel(items, reference_chat_completions, "chat")
        broken_verdicts = grade_code_completions_in_parallel(items, broken_base_completions, "base")
        elapsed_seconds = time.monotonic() - start_time
        for item, base_verdict, chat_verdict, broken_verdict in zip(items, base_verdicts, chat_verdicts,
                                                                    broken_verdicts):
            if not base_verdict["passed"] or not chat_verdict["passed"]:
                problems.append(f"the reference solution of {item['item_id']} does not pass: base "
                                f"{base_verdict['reason']}, chat {chat_verdict['reason']}")
            if broken_verdict["reason"] != "failed:RuntimeError":
                problems.append(f"a solution of {item['item_id']} that only raises gives {broken_verdict['reason']}")
        sandbox_report = (f", {3 * len(items)} programs in {elapsed_seconds:.1f} s with the "
                          f"{sandbox_probe_state['backend_name']} sandbox")
    if wrongly_graded_item_ids:
        problems.append(f"{len(wrongly_graded_item_ids)} items whose gold answer is not graded as expected, first: "
                        f"{wrongly_graded_item_ids[:5]}")
    print(f"{task_name}: {len(items):,} items, {len(few_shot_items)} shots, longest base prompt "
          f"{longest_base_prompt_characters:,} characters{sandbox_report}, {len(problems)} problems", flush=True)
    return problems


def check_built_files(task_names: list[str]) -> int:
    """Check the named tasks and return the number of problems (the command's exit status is non-zero if any)."""
    manifest_path = evaluation_directory() / "manifest.json"
    assert manifest_path.exists(), f"{manifest_path} is missing: run `python -m midtrain.eval_tasks build` first"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_problems = []
    for task_name in task_names:
        if task_name not in manifest["tasks"]:
            all_problems.append(f"{task_name} is not in the manifest: "
                                f"{manifest.get('tasks_not_built', {}).get(task_name, 'it was never built')}")
            continue
        all_problems.extend(f"{task_name}: {problem}" for problem in check_built_task(task_name,
                                                                                    manifest["tasks"][task_name]))
    for problem in all_problems:
        print("PROBLEM:", problem)
    print(f"{len(all_problems)} problems in {len(task_names)} tasks")
    return len(all_problems)


def show_rendered_prompt(task_name: str, item_index: int, prompt_style: str) -> None:
    """Print one item as the model would see it, then its gold answer, to read a prompt by eye."""
    item = load_evaluation_items(task_name)[item_index]
    if prompt_style == "base":
        print(render_base_prompt(item, load_few_shot_items(task_name)))
        if item["kind"] == "multiple_choice":
            print("\n--- continuations:", multiple_choice_continuations(item))
        print("--- stop sequences:", base_completion_stop_sequences(item))
    else:
        for message in render_chat_messages(item):
            print(f"--- {message['role']}\n{message['content']}")
    gold_answer = {key: item[key] for key in ("answer_text", "answer_index", "answer_aliases", "entry_point")
                   if key in item}
    print(f"--- item_id {item['item_id']}, gold {json.dumps(gold_answer, ensure_ascii=False)}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Build, check or show the public evaluation sets.")
    command_parsers = argument_parser.add_subparsers(dest="command", required=True)
    build_parser = command_parsers.add_parser("build", help="download, normalise and write the evaluation files")
    build_parser.add_argument("--tasks", nargs="+", choices=list(evaluation_tasks), default=list(evaluation_tasks))
    check_parser = command_parsers.add_parser("check", help="check the written files, the graders and the sandbox")
    check_parser.add_argument("--tasks", nargs="+", choices=list(evaluation_tasks), default=list(evaluation_tasks))
    show_parser = command_parsers.add_parser("show", help="print one rendered prompt and its gold answer")
    show_parser.add_argument("--task", required=True, choices=list(evaluation_tasks))
    show_parser.add_argument("--item-index", type=int, default=0)
    show_parser.add_argument("--style", choices=["base", "chat"], default="base")
    arguments = argument_parser.parse_args()
    if arguments.command == "build":
        sys.exit(1 if build_evaluation_files(arguments.tasks) else 0)
    if arguments.command == "check":
        sys.exit(1 if check_built_files(arguments.tasks) else 0)
    show_rendered_prompt(arguments.task, arguments.item_index, arguments.style)


if __name__ == "__main__":
    main()
