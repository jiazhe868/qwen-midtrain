r"""Tests for midtrain.eval_tasks: the graders, the pass@k estimator, the sandbox, the prompts and the build helpers.

Every checkpoint is compared through these functions, so a quiet defect here moves every reported number at once: a
grader that accepts a wrong answer, a prompt that shows the evaluated question among its own shots, or a sandbox that
lets model-written code reach the network or the user's files. The cases are written by hand and need neither the
network nor a GPU. The tests in the last section read the files that `build` wrote and are skipped until it has run.

Each sandbox test names the incident or the escape it guards against. The isolation tests run a real program in the
real sandbox of this machine; they are skipped only where no sandbox exists at all, and fail where one is configured
but broken.

Run from the repository root:

    source scripts/env.sh && $MIDTRAIN_PYTHON -m pytest tests/test_eval_tasks.py -q
"""

import hashlib
import itertools
import json
import os
import socket
import tempfile
import time
from pathlib import Path

import pytest

from midtrain import eval_tasks


# ---------------------------------------------------------------------------
# hand-written items, one maker per kind
# ---------------------------------------------------------------------------

def make_numeric_item(answer_text: str, question: str = "What is the value?", item_id: str = "hand:test:numeric",
                      worked_solution: str = "Work it out.") -> dict:
    return {"item_id": item_id, "task": "hand", "kind": "numeric", "question": question, "answer_text": answer_text,
            "meta": {"split": "test", "worked_solution": worked_solution}}


def make_multiple_choice_item(choices: list[str], answer_index: int, question: str = "Which one is right?",
                              item_id: str = "hand:test:choice", language: str | None = None) -> dict:
    meta = {"split": "test"}
    if language is not None:
        meta["language"] = language
    return {"item_id": item_id, "task": "hand", "kind": "multiple_choice", "question": question, "choices": choices,
            "answer_index": answer_index, "meta": meta}


def make_short_answer_item(answer_aliases: list[str], question: str = "Who wrote it?",
                           item_id: str = "hand:test:short") -> dict:
    return {"item_id": item_id, "task": "hand", "kind": "short_answer", "question": question,
            "answer_aliases": answer_aliases, "meta": {"split": "test"}}


def make_function_stub_item() -> dict:
    """A code item in the form of HumanEval: the question is a signature with a docstring."""
    question = ('from typing import List\n\n\ndef add_all(numbers: List[int]) -> int:\n'
                '    """ Return the sum of the numbers.\n    >>> add_all([1, 2])\n    3\n    """\n')
    test_code = ("def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n    assert candidate([]) == 0\n\n"
                 "check(add_all)\n")
    return {"item_id": "hand:test:stub", "task": "hand", "kind": "code", "question": question, "test_code": test_code,
            "entry_point": "add_all", "meta": {"split": "test", "question_form": "function_stub",
                                               "reference_solution": "    return sum(numbers)\n"}}


def make_task_description_item(item_id: str = "hand:test:task") -> dict:
    """A code item in the form of MBPP: the question is a sentence, the tests are assert lines."""
    test_list = ["assert double_all([1, 2]) == [2, 4]", "assert double_all([]) == []"]
    return {"item_id": item_id, "task": "hand", "kind": "code",
            "question": "Write a function to double every number of a list.",
            "test_code": "\n".join(test_list) + "\n", "entry_point": "double_all",
            "meta": {"split": "test", "question_form": "task_description", "test_list": test_list, "test_imports": [],
                     "reference_solution": "def double_all(numbers):\n    return [2 * number for number in numbers]"}}


# ---------------------------------------------------------------------------
# numeric answers
# ---------------------------------------------------------------------------

numeric_cases_that_are_correct = [
    # thousands separators
    ("1234", "She has 1,234 marbles. The answer is 1,234.", "base"),
    ("1234", r"So the total is \boxed{1,234}.", "chat"),
    ("1234", "#### 1,234", "base"),
    ("58500", r"\boxed{58{,}500}", "chat"),
    (r"10,\!080", r"\boxed{10080}", "chat"),
    (r"\$32,\!348", "The answer is $32,348.", "base"),
    # fractions against decimals, in both directions
    ("0.75", r"\boxed{\frac{3}{4}}", "chat"),
    (r"\frac{3}{4}", "The answer is 0.75.", "base"),
    ("0.75", r"The answer is $\dfrac{3}{4}$.", "base"),
    ("0.75", "The answer is 3/4.", "base"),
    (r"\frac43", r"\boxed{\frac{4}{3}}", "chat"),
    (r"\frac{14}{3}", r"\boxed{\dfrac{14}{3}}", "chat"),
    (r"1\frac{4}{5}", r"\boxed{\frac{9}{5}}", "chat"),
    (r"1\frac{4}{5}", r"\boxed{1.8}", "chat"),
    (r"-\frac{3}{8}", r"\boxed{-0.375}", "chat"),
    (".35625", r"\boxed{0.35625}", "chat"),
    # percent signs and currency
    ("50", r"\boxed{50\%}", "chat"),
    (r"50\%", "The answer is 50.", "base"),
    ("50", "The answer is 50%.", "base"),
    ("12", "The answer is $12.", "base"),
    ("12", r"\boxed{\$12}", "chat"),
    ("12", "The answer is $12.00.", "base"),
    # units and degrees
    ("12", "The answer is 12 dollars.", "base"),
    ("18", "The answer is 18 because 9 * 2 = 18 dollars a day", "base"),
    (r"5.4 \text{ cents}", r"\boxed{5.4}", "chat"),
    (r"864 \mbox{ inches}^2", r"\boxed{864}", "chat"),
    (r"90^\circ", r"\boxed{90}", "chat"),
    (r"90^\circ", r"\boxed{90^{\circ}}", "chat"),
    ("90", "The answer is 90 degrees.", "base"),
    # negative numbers, including the Unicode minus sign
    ("-5", "The answer is -5.", "base"),
    ("-5", r"\boxed{-5}", "chat"),
    ("-5", "The answer is −5.", "base"),
    # boxed text answers
    (r"\text{east}", r"\boxed{\text{east}}", "chat"),
    (r"\text{east}", r"\boxed{east}", "chat"),
    (r"\text{east}", r"\boxed{\text{East}}", "chat"),
    (r"\text{(C)}", r"\boxed{C}", "chat"),
    # LaTeX spelling
    (r"11\sqrt2", r"\boxed{11\sqrt{2}}", "chat"),
    (r"4210_{5}", r"\boxed{4210_5}", "chat"),
    ("x=5", r"\boxed{5}", "chat"),
    ("5", r"\boxed{x = 5}", "chat"),
    (r"y = 2x + 3", r"\boxed{2x+3}", "chat"),
    (r"x \in [-2,7]", r"\boxed{[-2,7]}", "chat"),
    (r"\left( 3, \frac{\pi}{2} \right)", r"\boxed{(3,\frac{\pi}{2})}", "chat"),
    ("3, 5, 7", r"\boxed{3,5,7}", "chat"),
    (r"\begin{pmatrix} -1/3 \\ 2/3 \end{pmatrix}", r"\boxed{\begin{pmatrix} -\frac{1}{3} \\ \frac{2}{3} \end{pmatrix}}",
     "chat"),
    # the fallbacks when nothing is boxed
    ("18", "She sells 9 eggs at $2 each, so she makes 9 * 2 = $18 every day", "base"),
    ("72", "The answer is: **72**", "chat"),
    ("7", "The answer is 7", "base"),
]

numeric_cases_that_are_wrong = [
    ("5", "The answer is 5 and 7.", "base"),                     # two answers are not the answer 5
    ("10", "The answer is 10 or 12.", "base"),
    ("12", r"\boxed{(1,2)}", "chat"),                            # the comma of a pair is not a thousands separator
    ("357", r"\boxed{3, 5, 7}", "chat"),
    ("12", "The answer is 1,2.", "base"),
    ("0.75", r"\boxed{0.76}", "chat"),
    (r"\frac{14}{3}", r"\boxed{4.67}", "chat"),                  # rounded, not equal
    ("-5", r"\boxed{5}", "chat"),
    ("50", r"\boxed{0.5}", "chat"),
    ("2", r"\boxed{2x}", "chat"),                                # 2x is an expression, not 2 with a unit
    (r"\text{east}", r"\boxed{\text{west}}", "chat"),
    ("(3,4]", r"\boxed{[3,4]}", "chat"),
    ("18", "I do not know.", "base"),
    ("18", "", "base"),
]


@pytest.mark.parametrize("gold_answer_text, completion_text, prompt_style", numeric_cases_that_are_correct)
def test_numeric_grader_accepts_other_spellings_of_the_gold_answer(gold_answer_text, completion_text, prompt_style):
    assert eval_tasks.grade_completion(make_numeric_item(gold_answer_text), completion_text, prompt_style)


@pytest.mark.parametrize("gold_answer_text, completion_text, prompt_style", numeric_cases_that_are_wrong)
def test_numeric_grader_rejects_wrong_answers(gold_answer_text, completion_text, prompt_style):
    assert not eval_tasks.grade_completion(make_numeric_item(gold_answer_text), completion_text, prompt_style)


def test_calc_x_results_lose_the_thousands_separator_and_keep_fractions():
    """SVAMP writes 68 of its results as 1_414; read as written, the gold answer would match no model answer."""
    row = {"id": "svamp__chal-9", "question": "Jack had  1000 marbles.\nHe found 414 more. How many now?",
           "result": "1_414", "equation": "( 1000.0 + 414.0 )"}
    item = eval_tasks.normalise_calc_x_row(row, "svamp", "test", 9)
    assert item["answer_text"] == "1414" and item["item_id"] == "svamp:test:9"
    assert item["question"] == "Jack had 1000 marbles. He found 414 more. How many now?"
    assert eval_tasks.grade_completion(item, " 1000 + 414 = 1414. The answer is 1414.", "base")
    assert eval_tasks.grade_completion(item, item["meta"]["reference_solution"], "chat")
    fraction_row = {"id": "asdiv_a__x", "question": "Share 10 cakes among 3.", "result": "10/3"}
    fraction_item = eval_tasks.normalise_calc_x_row(fraction_row, "asdiv_a", "test", 0)
    assert fraction_item["meta"]["reference_solution"] == "The answer is 10/3."
    assert eval_tasks.grade_completion(fraction_item, "\\boxed{\\frac{10}{3}}", "chat")


def test_final_answer_is_read_in_the_documented_order():
    r"""The last \boxed{}, else `#### x`, else `The answer is x`, else the last number."""
    extract = eval_tasks.extract_final_answer_text
    assert extract(r"A draft \boxed{3}. #### 5" + "\nThe answer is 7. Then 9 more.") == "3"
    assert extract("#### 5\nThe answer is 7. Then 9 more.") == "5"
    assert extract("The answer is 7. Then 9 more.") == "7"
    assert extract("There are 3 apples and 4 pears, 7 in total") == "7"
    assert extract("Nothing to see here.") is None


def test_boxed_answers_with_nested_and_escaped_braces():
    extract = eval_tasks.extract_final_answer_text
    assert extract(r"First \boxed{1}, then \boxed{2}.") == "2"
    assert extract(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract(r"\boxed{\{1,2\}}") == r"\{1,2\}"
    # the last box was cut off by the token limit: the last complete one is used
    assert extract(r"It is \boxed{4}. Checking again gives \boxed{\frac{8}{") == "4"


def test_last_number_does_not_take_a_minus_sign_from_a_subtraction():
    extract = eval_tasks.extract_final_answer_text
    assert extract("so 10-5") == "5"
    assert extract("the result = -5") == "-5"
    assert extract("which makes 1,234.") == "1,234"


def test_base_style_numeric_answer_ignores_the_question_the_model_invents_next():
    """A base model keeps writing after its answer; the answer of its invented question must not be graded."""
    completion_text = (" 3 + 4 = 7. The answer is 7.\n\nQuestion: What is 2 + 2?\nAnswer: 2 + 2 = 4. The answer is 4.")
    assert eval_tasks.grade_completion(make_numeric_item("7"), completion_text, "base")
    assert not eval_tasks.grade_completion(make_numeric_item("4"), completion_text, "base")


def test_chat_style_ignores_a_boxed_draft_inside_the_reasoning():
    completion_text = r"<think>Maybe it is \boxed{4}? No.</think>The total is \boxed{7}."
    assert eval_tasks.grade_completion(make_numeric_item("7"), completion_text, "chat")
    assert not eval_tasks.grade_completion(make_numeric_item("4"), completion_text, "chat")
    only_reasoned = r"<think>It must be \boxed{7}.</think>I am not sure."
    assert not eval_tasks.grade_completion(make_numeric_item("7"), only_reasoned, "chat")


def test_math_normalisation_keeps_what_distinguishes_answers():
    normalise = eval_tasks.normalise_math_answer_text
    assert normalise("(1,234)") == "(1,234)"                      # a pair, not the number 1234
    assert normalise("3, 5, 7") == "3,5,7"
    assert normalise(r"\rightarrow") == r"\rightarrow"            # \right is removed, \rightarrow is not
    assert normalise("2 x") == "2x"                               # a one-letter word is a variable, not a unit
    assert normalise(r"1 \text{ and } 2") == "1,2"


# ---------------------------------------------------------------------------
# multiple choice
# ---------------------------------------------------------------------------

four_choice_item = make_multiple_choice_item(["Paris", "Lyon", "Nice", "Lille"], answer_index=1)


@pytest.mark.parametrize("completion_text, prompt_style, is_correct", [
    (" B", "base", True),
    ("B. Lyon", "base", True),
    (" (B)", "base", True),
    (" B\n\nWhich one is wrong?\nA. x\nAnswer: A", "base", True),       # only the first line is the answer
    (" C", "base", False),
    (" Both of them", "base", False),                                   # "B" inside a word is not the letter B
    (" E", "base", False),                                              # there is no fifth option
    ("", "base", False),
    (r"\boxed{B}", "chat", True),
    (r"The answer is \boxed{\text{B}}.", "chat", True),
    (r"\boxed{(B)}", "chat", True),
    (r"\boxed{b}", "chat", True),
    (r"\boxed{Lyon}", "chat", True),                                    # the option text instead of its letter
    (r"\boxed{B. Lyon}", "chat", True),
    ("The answer is B.", "chat", True),
    ("**Answer: B**", "chat", True),
    ("B", "chat", True),
    (r"I first thought \boxed{A}, but it is \boxed{B}.", "chat", True),  # the last box counts
    (r"\boxed{A}", "chat", False),
    (r"\boxed{Marseille}", "chat", False),
    ("The answer is a city in France.", "chat", False),                 # the article "a" is not the letter A
    ("I cannot tell.", "chat", False),
])
def test_multiple_choice_grader_reads_the_letter(completion_text, prompt_style, is_correct):
    assert eval_tasks.grade_completion(four_choice_item, completion_text, prompt_style) == is_correct


def test_multiple_choice_grader_reads_a_chinese_answer():
    chinese_item = make_multiple_choice_item(["茎", "叶", "根", "果实"], answer_index=0, language="zh")
    assert eval_tasks.grade_completion(chinese_item, "A", "base")
    assert eval_tasks.grade_completion(chinese_item, "答案是 A。", "chat")
    assert eval_tasks.grade_completion(chinese_item, r"正确选项是 \boxed{A}", "chat")
    assert not eval_tasks.grade_completion(chinese_item, "答案是：B", "chat")


def test_continuations_match_the_way_shots_show_their_answers():
    """The scored letter must look like the answers in the shots, or its log-likelihood measures the format."""
    english_item = make_multiple_choice_item(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], answer_index=9)
    assert eval_tasks.multiple_choice_continuations(english_item) == [" A", " B", " C", " D", " E", " F", " G", " H",
                                                                      " I", " J"]
    assert eval_tasks.render_base_block(english_item, with_answer=True).endswith("Answer: J")
    chinese_item = make_multiple_choice_item(["茎", "叶", "根"], answer_index=2, language="zh")
    assert eval_tasks.multiple_choice_continuations(chinese_item) == ["A", "B", "C"]
    assert eval_tasks.render_base_block(chinese_item, with_answer=True).endswith("答案是：C")
    with pytest.raises(AssertionError):
        eval_tasks.multiple_choice_continuations(make_numeric_item("1"))


# ---------------------------------------------------------------------------
# short answers
# ---------------------------------------------------------------------------

trivia_item = make_short_answer_item(["Sinclair Lewis", "Harry Sinclair Lewis", "The Beatles"])


@pytest.mark.parametrize("completion_text, prompt_style, is_correct", [
    (" Sinclair Lewis", "base", True),
    (" sinclair lewis.", "base", True),
    (" Harry Sinclair Lewis\n\nQuestion: Who was next?\nAnswer: Someone Else", "base", True),
    (" the beatles!", "base", True),                                    # articles and punctuation are dropped
    (" Beatles", "base", True),
    (" Lewis", "base", False),                                          # a part of an alias is not an alias
    (" Sinclair Lewis and Upton Sinclair", "base", False),
    (" The", "base", False),                                            # normalises to nothing
    ("", "base", False),
    (r"It was \boxed{Sinclair Lewis}.", "chat", True),
    (r"\boxed{\text{The Beatles}}", "chat", True),
    ("Sinclair Lewis", "chat", True),
    (r"\boxed{Upton Sinclair}", "chat", False),
])
def test_short_answer_grader_matches_normalised_aliases(completion_text, prompt_style, is_correct):
    assert eval_tasks.grade_completion(trivia_item, completion_text, prompt_style) == is_correct


def test_short_answer_normalisation_follows_the_trivia_script():
    normalise = eval_tasks.normalise_short_answer_text
    assert normalise("The  Go-Between!") == "go between"
    assert normalise("An_apple a day") == "apple day"
    assert normalise("EXXON VALDEZ") == "exxon valdez"
    assert normalise("The") == ""


# ---------------------------------------------------------------------------
# code: how the program is put together (no sandbox needed)
# ---------------------------------------------------------------------------

def test_function_stub_program_is_prompt_plus_completion_cut_at_the_first_top_level_stop():
    item = make_function_stub_item()
    completion_text = ("    return sum(numbers)\n\n\ndef unrelated_function():\n    raise RuntimeError\n\n"
                       "print(add_all([1]))\n")
    program_text = eval_tasks.build_code_program(item, completion_text, "base")
    assert program_text.startswith(item["question"] + "    return sum(numbers)")
    assert "unrelated_function" not in program_text and "print(add_all" not in program_text
    assert program_text.endswith(item["test_code"])


@pytest.mark.parametrize("stop_sequence", ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\nassert ", "\n#"])
def test_each_top_level_stop_sequence_ends_a_function_body(stop_sequence):
    item = make_function_stub_item()
    completion_text = "    return sum(numbers)\n" + stop_sequence + "junk that must not be run"
    assert "junk" not in eval_tasks.build_code_program(item, completion_text, "base")


def test_indented_code_is_not_cut():
    """A nested def or an indented print is part of the body; only a line that starts at the margin ends it."""
    item = make_function_stub_item()
    completion_text = ("    def helper(value):\n        return value\n    print('working')\n"
                       "    return sum(helper(number) for number in numbers)\n")
    assert completion_text.rstrip("\n") in eval_tasks.build_code_program(item, completion_text, "base")


def test_task_description_completion_keeps_helper_functions_and_ends_at_the_next_docstring():
    item = make_task_description_item()
    completion_text = ("def double_one(number):\n    return 2 * number\n\ndef double_all(numbers):\n"
                       "    return [double_one(number) for number in numbers]\n\n\n"
                       '"""\nWrite a function to do something else.\n"""\ndef something_else():\n    pass\n')
    program_text = eval_tasks.build_code_program(item, completion_text, "base")
    assert "def double_one" in program_text and "def double_all" in program_text
    assert "something_else" not in program_text
    assert program_text.endswith(item["test_code"])


def test_chat_program_uses_the_first_fenced_python_block():
    extract = eval_tasks.extract_first_fenced_python_block
    answer_text = ("Install it first:\n```bash\npip install nothing\n```\nThen:\n```python\ndef f():\n    return 1\n```\n"
                   "A slower version:\n```python\ndef f():\n    return 2\n```\n")
    assert extract(answer_text) == "def f():\n    return 1\n"
    assert extract("```\ndef g():\n    return 3\n```") == "def g():\n    return 3\n"
    assert extract("```python\ndef h():\n    return 4\n") == "def h():\n    return 4\n"       # cut off before the fence closed
    assert extract("def bare():\n    return 5\n") == "def bare():\n    return 5\n"           # no fence at all
    item = make_task_description_item()
    reasoning_then_answer = ("<think>```python\ndef double_all(numbers):\n    return []\n```</think>\n"
                             "```python\ndef double_all(numbers):\n    return [2 * n for n in numbers]\n```")
    program_text = eval_tasks.build_code_program(item, reasoning_then_answer, "chat")
    assert "return [2 * n for n in numbers]" in program_text and "return []" not in program_text


def test_cut_text_at_first_stop_sequence_takes_the_earliest_one():
    cut = eval_tasks.cut_text_at_first_stop_sequence
    assert cut("abc\nprint(1)\ndef f", ["\ndef ", "\nprint("]) == "abc"
    assert cut("no stop here", ["\ndef "]) == "no stop here"


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------

def test_pass_at_k_equals_brute_force_enumeration():
    """Enumerate every way to draw k of the n samples and count the draws that contain a correct one."""
    for sample_count in range(1, 9):
        for correct_sample_count in range(sample_count + 1):
            sample_is_correct = [True] * correct_sample_count + [False] * (sample_count - correct_sample_count)
            for k in range(1, sample_count + 1):
                draws = list(itertools.combinations(sample_is_correct, k))
                fraction_of_draws_with_a_correct_sample = sum(any(draw) for draw in draws) / len(draws)
                estimate = eval_tasks.estimate_pass_at_k(correct_sample_count, sample_count, k)
                assert estimate == pytest.approx(fraction_of_draws_with_a_correct_sample, abs=1e-12)


def test_pass_at_k_edge_cases_and_bad_input():
    assert eval_tasks.estimate_pass_at_k(0, 10, 5) == 0.0
    assert eval_tasks.estimate_pass_at_k(10, 10, 1) == 1.0
    assert eval_tasks.estimate_pass_at_k(6, 10, 5) == 1.0          # only 4 wrong samples: every draw of 5 has a correct one
    assert eval_tasks.estimate_pass_at_k(1, 200, 1) == pytest.approx(1 / 200)
    with pytest.raises(AssertionError):
        eval_tasks.estimate_pass_at_k(3, 10, 11)                     # pass@11 from 10 samples does not exist
    with pytest.raises(AssertionError):
        eval_tasks.estimate_pass_at_k(11, 10, 1)


# ---------------------------------------------------------------------------
# the sandbox
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_sandbox_probe_state():
    """Tests that fake the probe must not leave their result behind for the tests that use the real sandbox."""
    saved_state = dict(eval_tasks.sandbox_probe_state)
    yield
    eval_tasks.sandbox_probe_state.update(saved_state)


@pytest.fixture
def sandbox_backend_name():
    """The working backend of this machine. No backend at all skips the test; a configured but broken one fails it."""
    if not eval_tasks.list_candidate_sandbox_backends():
        pytest.skip("no sandbox on this machine: bwrap is not on PATH and $MIDTRAIN_SANDBOX_IMAGE is not set")
    backend_name = eval_tasks.find_working_sandbox_backend()
    assert backend_name is not None, "a sandbox is configured on this machine but its probe program does not pass"
    return backend_name


@pytest.fixture
def isolating_sandbox(sandbox_backend_name):
    if sandbox_backend_name == "unsandboxed":
        pytest.skip("MIDTRAIN_ALLOW_UNSANDBOXED=1 runs programs directly: there is no isolation to test")
    return sandbox_backend_name


def test_sandbox_passes_a_correct_program(sandbox_backend_name):
    program_text = "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n"
    assert eval_tasks.run_python_program_in_sandbox(program_text) == {"passed": True, "reason": "passed"}


def test_sandbox_fails_an_assertion(sandbox_backend_name):
    verdict = eval_tasks.run_python_program_in_sandbox("assert 1 + 1 == 3\n")
    assert verdict == {"passed": False, "reason": "failed:AssertionError"}


def test_sandbox_fails_a_program_that_does_not_compile(sandbox_backend_name):
    verdict = eval_tasks.run_python_program_in_sandbox("def broken(:\n    pass\n")
    assert verdict == {"passed": False, "reason": "failed:SyntaxError"}


def test_sandbox_times_out_an_infinite_loop_and_leaves_nothing_behind(sandbox_backend_name, tmp_path, monkeypatch):
    sandbox_parent_directory = tmp_path / "sandboxes"
    sandbox_parent_directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sandbox_parent_directory))
    start_time = time.monotonic()
    verdict = eval_tasks.run_python_program_in_sandbox("while True:\n    pass\n", timeout_seconds=2)
    elapsed_seconds = time.monotonic() - start_time
    assert verdict == {"passed": False, "reason": "timeout"}
    assert 2 <= elapsed_seconds < 10
    assert list(sandbox_parent_directory.iterdir()) == [], "the program's folder was not removed"
    # the looping process must be gone too: its command line names the folder it ran in
    deadline = time.monotonic() + 5
    while True:
        command_lines = []
        for process_directory in Path("/proc").iterdir():
            if process_directory.name.isdigit():
                try:
                    command_lines.append((process_directory / "cmdline").read_bytes().decode("utf-8", "replace"))
                except OSError:
                    continue
        survivors = [command_line for command_line in command_lines if str(sandbox_parent_directory) in command_line]
        if not survivors or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    assert survivors == [], "a process of the timed-out program is still running"


def test_sandbox_blocks_the_network(isolating_sandbox):
    """A listener on this machine is reachable from the test and must be unreachable from inside the sandbox."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        socket.create_connection(("127.0.0.1", port), timeout=5).close()       # the listener is real ...
        listener.accept()[0].close()                                           # ... and saw this connection
        connect_program = ("import socket\n"
                           "try:\n"
                           f"    socket.create_connection(('127.0.0.1', {port}), timeout=5)\n"
                           "except OSError:\n"
                           "    pass\n"
                           "else:\n"
                           "    raise AssertionError('reached the listener outside the sandbox')\n")
        assert eval_tasks.run_python_program_in_sandbox(connect_program) == {"passed": True, "reason": "passed"}
        listener.settimeout(0.5)
        with pytest.raises(socket.timeout):                                    # nothing arrived from the sandbox
            listener.accept()
    outside_world_program = (
        "import socket\n"
        "attempts = [lambda: socket.create_connection(('1.1.1.1', 53), timeout=3),\n"
        "            lambda: socket.getaddrinfo('huggingface.co', 443)]\n"
        "for attempt in attempts:\n"
        "    try:\n"
        "        attempt()\n"
        "    except OSError:\n"
        "        continue\n"
        "    raise AssertionError('the network is reachable')\n")
    assert eval_tasks.run_python_program_in_sandbox(outside_world_program)["passed"]


def test_sandbox_blocks_writing_outside_its_own_folder(isolating_sandbox, tmp_path):
    home_directory = Path.home()
    repository_directory = Path(__file__).resolve().parents[1]
    forbidden_paths = [tmp_path / "written_by_sandbox.txt", home_directory / "written_by_sandbox.txt",
                       repository_directory / "written_by_sandbox.txt", Path("/usr/written_by_sandbox.txt"),
                       Path("/written_by_sandbox.txt")]
    for forbidden_path in forbidden_paths:
        write_program = f"open({str(forbidden_path)!r}, 'w').write('escaped')\n"
        verdict = eval_tasks.run_python_program_in_sandbox(write_program)
        assert not verdict["passed"] and verdict["reason"].startswith("failed:"), (forbidden_path, verdict)
        assert not forbidden_path.exists(), f"the sandboxed program created {forbidden_path}"
    # its own folder and /tmp are writable, and what it writes to /tmp does not reach this machine's /tmp
    allowed_program = ("import os, tempfile\n"
                       "open('note.txt', 'w').write('kept')\n"
                       "assert open('note.txt').read() == 'kept'\n"
                       "open('/tmp/midtrain_sandbox_test_note.txt', 'w').write('private')\n"
                       "assert os.listdir(tempfile.gettempdir()) == ['midtrain_sandbox_test_note.txt'] or "
                       "tempfile.gettempdir() != '/tmp'\n")
    assert eval_tasks.run_python_program_in_sandbox(allowed_program) == {"passed": True, "reason": "passed"}
    assert not Path("/tmp/midtrain_sandbox_test_note.txt").exists()


def test_sandbox_blocks_reading_outside_its_own_folder(isolating_sandbox, tmp_path):
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("a checkpoint path or an access token")
    conventions_path = Path(__file__).resolve().parents[1] / "docs" / "conventions.md"
    assert conventions_path.exists()
    for hidden_path in (secret_path, conventions_path):
        read_program = f"open({str(hidden_path)!r}).read()\n"
        verdict = eval_tasks.run_python_program_in_sandbox(read_program)
        assert verdict == {"passed": False, "reason": "failed:FileNotFoundError"}, (hidden_path, verdict)
    # the home folder shows at most the folders that lead to the Python installation, never the user's files
    home_program = (f"import os\nhome = {str(Path.home())!r}\n"
                    "visible = os.listdir(home) if os.path.isdir(home) else []\n"
                    "assert not [name for name in visible if name.startswith('.')], visible\n")
    assert eval_tasks.run_python_program_in_sandbox(home_program)["passed"]


def test_an_early_exit_is_not_a_pass(sandbox_backend_name):
    """A solution that calls exit() before the tests run must not look like "no test failed"."""
    assert eval_tasks.run_python_program_in_sandbox("import sys\nsys.exit(0)\nassert False\n") == {
        "passed": False, "reason": "failed:SystemExit"}
    verdict = eval_tasks.run_python_program_in_sandbox("import os\nos._exit(0)\nassert False\n")
    assert not verdict["passed"] and verdict["reason"].startswith("no_verdict")


def test_printed_text_cannot_pass_for_the_verdict(sandbox_backend_name):
    program_text = "print('passed')\nprint('0' * 32 + ' passed')\nraise ValueError('still wrong')\n"
    assert eval_tasks.run_python_program_in_sandbox(program_text) == {"passed": False, "reason": "failed:ValueError"}


def test_a_demo_under_the_main_guard_is_not_run(sandbox_backend_name):
    """Chat answers often end with a demo that reads input(); the function is tested, the demo is not."""
    program_text = ("def double(value):\n    return 2 * value\n\n"
                    "if __name__ == '__main__':\n    print(double(int(input())))\n\nassert double(4) == 8\n")
    assert eval_tasks.run_python_program_in_sandbox(program_text)["passed"]


def test_sandbox_gives_the_same_hash_seed_on_every_run(sandbox_backend_name):
    """`python -I` ignores PYTHONHASHSEED, which made the order of a set of strings differ between two runs."""
    program_text = "import sys\nassert sys.flags.hash_randomization == 0\n"
    assert eval_tasks.run_python_program_in_sandbox(program_text)["passed"]


def test_sandbox_limits_memory_and_file_size(sandbox_backend_name):
    assert eval_tasks.run_python_program_in_sandbox("block = bytearray(8 * 1024 ** 3)\n") == {
        "passed": False, "reason": "failed:MemoryError"}
    big_file_program = "with open('big.bin', 'wb') as big_file:\n    big_file.write(b'0' * (64 * 1024 ** 2))\n"
    assert eval_tasks.run_python_program_in_sandbox(big_file_program) == {"passed": False, "reason": "failed:OSError"}


def test_bubblewrap_mounts_tmp_before_the_binds_and_isolates():
    """A later mount hides an earlier one: a tmpfs on /tmp after the binds would hide the program's own folder."""
    command = eval_tasks.build_bubblewrap_command("/usr/bin/bwrap", "/tmp/midtrain_sandbox_x", ["/opt/python"],
                                                  {"PATH": "/usr/bin:/bin"})
    tmpfs_position = command.index("--tmpfs")
    assert command[tmpfs_position + 1] == "/tmp"
    assert tmpfs_position < command.index("/opt/python") and tmpfs_position < command.index("/tmp/midtrain_sandbox_x")
    assert command[command.index("/opt/python") - 1] == "--ro-bind"
    assert command[command.index("/tmp/midtrain_sandbox_x") - 1] == "--bind"
    for required_flag in ("--unshare-all", "--die-with-parent", "--clearenv", "--remount-ro"):
        assert required_flag in command


def test_without_a_sandbox_every_program_fails_closed(monkeypatch):
    monkeypatch.setattr(eval_tasks, "list_candidate_sandbox_backends", lambda: [])
    eval_tasks.sandbox_probe_state.update({"backend_name": None, "last_failed_probe_time": None})
    assert eval_tasks.run_python_program_in_sandbox("assert True\n") == {"passed": False, "reason": "no_sandbox"}
    item = make_function_stub_item()
    assert not eval_tasks.grade_completion(item, item["meta"]["reference_solution"], "base")


def test_unsandboxed_runs_only_when_it_is_explicitly_allowed(monkeypatch):
    monkeypatch.setattr(eval_tasks.shutil, "which", lambda program_name: None)
    monkeypatch.delenv("MIDTRAIN_SANDBOX_IMAGE", raising=False)
    monkeypatch.delenv("MIDTRAIN_ALLOW_UNSANDBOXED", raising=False)
    assert eval_tasks.list_candidate_sandbox_backends() == []
    monkeypatch.setenv("MIDTRAIN_ALLOW_UNSANDBOXED", "1")
    assert eval_tasks.list_candidate_sandbox_backends() == ["unsandboxed"]
    eval_tasks.sandbox_probe_state.update({"backend_name": None, "last_failed_probe_time": None})
    assert eval_tasks.run_python_program_in_sandbox("assert 2 + 2 == 4\n") == {"passed": True, "reason": "passed"}
    assert eval_tasks.sandbox_probe_state["backend_name"] == "unsandboxed"
    assert eval_tasks.run_python_program_in_sandbox("assert 2 + 2 == 5\n")["reason"] == "failed:AssertionError"


def test_backends_are_tried_in_order_bubblewrap_then_apptainer(monkeypatch):
    monkeypatch.setattr(eval_tasks.shutil, "which", lambda program_name: f"/usr/bin/{program_name}")
    monkeypatch.delenv("MIDTRAIN_ALLOW_UNSANDBOXED", raising=False)
    monkeypatch.delenv("MIDTRAIN_SANDBOX_IMAGE", raising=False)
    assert eval_tasks.list_candidate_sandbox_backends() == ["bubblewrap"]
    monkeypatch.setenv("MIDTRAIN_SANDBOX_IMAGE", "/images/python.sif")
    assert eval_tasks.list_candidate_sandbox_backends() == ["bubblewrap", "apptainer"]
    monkeypatch.setattr(eval_tasks.shutil, "which",
                        lambda program_name: None if program_name == "bwrap" else f"/usr/bin/{program_name}")
    assert eval_tasks.list_candidate_sandbox_backends() == ["apptainer"]


def test_a_failed_sandbox_probe_is_retried_after_sixty_seconds_and_a_passed_one_is_kept(monkeypatch):
    """In a sister project a failed probe was cached for the life of the process and zeroed a run's code scores."""
    probe_verdicts = [{"passed": False, "reason": "no_verdict:return_code=1"}, {"passed": True, "reason": "passed"}]
    probe_calls = []

    def fake_run_program(program_text, timeout_seconds, backend_name):
        probe_calls.append(backend_name)
        return probe_verdicts[min(len(probe_calls), len(probe_verdicts)) - 1]

    fake_time = {"now": 1000.0}
    monkeypatch.setattr(eval_tasks, "list_candidate_sandbox_backends", lambda: ["bubblewrap"])
    monkeypatch.setattr(eval_tasks, "run_program_with_sandbox_backend", fake_run_program)
    eval_tasks.sandbox_probe_state.update({"backend_name": None, "last_failed_probe_time": None})
    clock = lambda: fake_time["now"]

    assert eval_tasks.find_working_sandbox_backend(clock) is None            # the first probe fails
    assert len(probe_calls) == 1
    fake_time["now"] += 30
    assert eval_tasks.find_working_sandbox_backend(clock) is None            # too early for another probe
    assert len(probe_calls) == 1
    fake_time["now"] += 31
    assert eval_tasks.find_working_sandbox_backend(clock) == "bubblewrap"    # 61 s later the probe is repeated
    assert len(probe_calls) == 2
    fake_time["now"] += 3600
    assert eval_tasks.find_working_sandbox_backend(clock) == "bubblewrap"    # a passed probe is never repeated
    assert len(probe_calls) == 2


def test_a_probe_that_raises_counts_as_failed_and_does_not_stop_the_grading(monkeypatch):
    def run_program_that_raises(program_text, timeout_seconds, backend_name):
        raise OSError("no space left on device")

    monkeypatch.setattr(eval_tasks, "list_candidate_sandbox_backends", lambda: ["bubblewrap"])
    monkeypatch.setattr(eval_tasks, "run_program_with_sandbox_backend", run_program_that_raises)
    eval_tasks.sandbox_probe_state.update({"backend_name": None, "last_failed_probe_time": None})
    assert eval_tasks.find_working_sandbox_backend() is None
    assert eval_tasks.sandbox_probe_state["last_failed_probe_time"] is not None


# ---------------------------------------------------------------------------
# code graded through the sandbox
# ---------------------------------------------------------------------------

def test_code_grader_in_both_prompt_styles(sandbox_backend_name):
    stub_item = make_function_stub_item()
    assert eval_tasks.grade_completion(stub_item, "    return sum(numbers)\n", "base")
    assert not eval_tasks.grade_completion(stub_item, "    return len(numbers)\n", "base")
    assert not eval_tasks.grade_completion(stub_item, "return sum(numbers)\n", "base")          # lost its indentation
    complete_function = "```python\ndef add_all(numbers: List[int]) -> int:\n    return sum(numbers)\n```"
    assert eval_tasks.grade_completion(stub_item, "Here it is.\n" + complete_function, "chat")   # List comes from the prompt
    assert eval_tasks.grade_completion(stub_item, "```python\n    return sum(numbers)\n```", "chat")
    assert not eval_tasks.grade_completion(stub_item, "```python\ndef add_all(numbers):\n    return 0\n```", "chat")

    task_item = make_task_description_item()
    reference_solution = task_item["meta"]["reference_solution"]
    assert eval_tasks.grade_completion(task_item, reference_solution + '\n\n\n"""\nAnother task.\n"""\n', "base")
    assert eval_tasks.grade_completion(task_item, f"```python\n{reference_solution}\n```", "chat")
    assert not eval_tasks.grade_completion(task_item, "def double_all(numbers):\n    return numbers\n", "base")
    assert not eval_tasks.grade_completion(task_item, "I cannot write this function.", "chat")


def test_parallel_code_grading_keeps_the_order_of_the_completions(sandbox_backend_name):
    stub_item = make_function_stub_item()
    completion_texts = ["    return sum(numbers)\n" if index % 3 else "    return -1\n" for index in range(30)]
    verdicts = eval_tasks.grade_code_completions_in_parallel([stub_item] * 30, completion_texts, "base", worker_count=8)
    assert [verdict["passed"] for verdict in verdicts] == [index % 3 != 0 for index in range(30)]
    assert {verdict["reason"] for verdict in verdicts} == {"passed", "failed:AssertionError"}
    correct_sample_count = sum(verdict["passed"] for verdict in verdicts)
    assert eval_tasks.estimate_pass_at_k(correct_sample_count, 30, 1) == pytest.approx(20 / 30)
    with pytest.raises(AssertionError):
        eval_tasks.grade_code_completions_in_parallel([stub_item], [], "base")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_base_prompt_refuses_an_item_among_its_own_shots():
    """The evaluated question with its answer among the shots would give the answer away."""
    for item in (make_numeric_item("7"), four_choice_item, trivia_item, make_task_description_item()):
        other_shot = dict(item, item_id="hand:train:other", question="A different question?")
        if item["kind"] == "multiple_choice":
            other_shot["choices"] = ["w", "x", "y", "z"]
        assert eval_tasks.render_base_prompt(item, [other_shot])                 # a different question is fine
        with pytest.raises(AssertionError, match="among its own shots"):
            eval_tasks.render_base_prompt(item, [other_shot, item])
        same_question_under_another_id = dict(item, item_id="hand:train:copy", question=item["question"].upper() + " ")
        with pytest.raises(AssertionError, match="among its own shots"):
            eval_tasks.render_base_prompt(item, [same_question_under_another_id])


def test_same_stem_with_other_options_is_a_different_question():
    """"Which of the following is true?" repeats between questions; only the same options make it the same one."""
    first_item = make_multiple_choice_item(["1", "2"], 0, question="Which of the following is true?", item_id="a")
    second_item = make_multiple_choice_item(["3", "4"], 1, question="Which of the following is true?", item_id="b")
    assert not eval_tasks.is_same_question(first_item, second_item)
    assert "Answer: B\n\nWhich of the following is true?\nA. 1\nB. 2\nAnswer:" in eval_tasks.render_base_prompt(
        first_item, [second_item])


def test_base_prompt_refuses_shots_of_another_kind():
    with pytest.raises(AssertionError):
        eval_tasks.render_base_prompt(four_choice_item, [make_numeric_item("7")])


def test_multiple_choice_base_prompt_layout():
    shot = make_multiple_choice_item(["red", "blue"], 1, question="What colour is the sky?", item_id="hand:train:1")
    prompt_text = eval_tasks.render_base_prompt(four_choice_item, [shot])
    assert prompt_text == ("The following are multiple choice questions (with answers).\n\n"
                           "What colour is the sky?\nA. red\nB. blue\nAnswer: B\n\n"
                           "Which one is right?\nA. Paris\nB. Lyon\nC. Nice\nD. Lille\nAnswer:")
    chinese_shot = make_multiple_choice_item(["茎", "叶"], 0, question="洋葱是植物的哪一部分", item_id="hand:dev:1",
                                             language="zh")
    chinese_item = make_multiple_choice_item(["一", "二", "三"], 2, question="一加二等于几", language="zh")
    assert eval_tasks.render_base_prompt(chinese_item, [chinese_shot]) == (
        "以下是单项选择题，请直接给出正确答案的选项。\n\n"
        "题目：洋葱是植物的哪一部分\nA. 茎\nB. 叶\n答案是：A\n\n"
        "题目：一加二等于几\nA. 一\nB. 二\nC. 三\n答案是：")


def test_numeric_base_prompt_ends_shots_with_the_answer_sentence():
    number_shot = make_numeric_item("72", question="How many clips?", item_id="hand:train:1",
                                    worked_solution="She sold 48 + 24 = 72 clips.")
    expression_shot = make_numeric_item(r"\frac{3}{4}", question="What fraction?", item_id="hand:train:2",
                                        worked_solution="Three of four parts.")
    prompt_text = eval_tasks.render_base_prompt(make_numeric_item("5", question="How many?"),
                                                [number_shot, expression_shot])
    assert prompt_text == ("Question: How many clips?\nAnswer: She sold 48 + 24 = 72 clips. The answer is 72.\n\n"
                           "Question: What fraction?\nAnswer: Three of four parts. The answer is $\\frac{3}{4}$.\n\n"
                           "Question: How many?\nAnswer:")
    # what the shots show must be read back as the gold answer
    for shot in (number_shot, expression_shot):
        shot_answer_text = eval_tasks.render_base_block(shot, with_answer=True).split("Answer:", 1)[1]
        assert eval_tasks.grade_completion(shot, shot_answer_text, "base")


def test_short_answer_base_prompt_layout():
    shot = make_short_answer_item(["York", "City of York"], question="Where was she born?", item_id="hand:train:1")
    assert eval_tasks.render_base_prompt(trivia_item, [shot]) == (
        "Question: Where was she born?\nAnswer: York\n\nQuestion: Who wrote it?\nAnswer:")


def test_code_base_prompts():
    stub_item = make_function_stub_item()
    assert eval_tasks.render_base_prompt(stub_item, []) == stub_item["question"]        # HumanEval's own prompt
    with pytest.raises(AssertionError):
        eval_tasks.render_base_prompt(stub_item, [make_task_description_item()])
    task_item = make_task_description_item()
    shot = make_task_description_item(item_id="hand:prompt:1")
    shot["question"] = "Write a function to triple every number of a list."
    shot["meta"] = dict(shot["meta"], test_list=["assert triple_all([1]) == [3]"],
                        reference_solution="def triple_all(numbers):  \n    return [3 * n for n in numbers] ")
    assert eval_tasks.render_base_prompt(task_item, [shot]) == (
        '"""\nWrite a function to triple every number of a list.\nassert triple_all([1]) == [3]\n"""\n'
        "def triple_all(numbers):\n    return [3 * n for n in numbers]\n\n\n"
        '"""\nWrite a function to double every number of a list.\n'
        "assert double_all([1, 2]) == [2, 4]\nassert double_all([]) == []\n" '"""\n')


def test_stop_sequences_by_kind():
    assert eval_tasks.base_completion_stop_sequences(make_numeric_item("1")) == ["\nQuestion:"]
    assert eval_tasks.base_completion_stop_sequences(four_choice_item) == ["\n"]
    assert eval_tasks.base_completion_stop_sequences(trivia_item) == ["\n"]
    assert "\ndef " in eval_tasks.base_completion_stop_sequences(make_function_stub_item())
    task_stop_sequences = eval_tasks.base_completion_stop_sequences(make_task_description_item())
    assert "\ndef " not in task_stop_sequences and '\n"""' in task_stop_sequences


def test_chat_messages_ask_for_the_documented_answer_form():
    for item in (make_numeric_item("7"), four_choice_item, trivia_item, make_function_stub_item(),
                 make_task_description_item()):
        messages = eval_tasks.render_chat_messages(item)
        assert [message["role"] for message in messages] == ["system", "user"]
        assert all(message["content"].strip() for message in messages)
        user_text = messages[1]["content"]
        if item["kind"] == "code":
            assert "one fenced ```python block" in user_text and "\\boxed" not in user_text
        else:
            assert "\\boxed{}" in user_text
    assert "step by step" in eval_tasks.render_chat_messages(make_numeric_item("7"))[1]["content"]
    assert "A. Paris\nB. Lyon\nC. Nice\nD. Lille" in eval_tasks.render_chat_messages(four_choice_item)[1]["content"]
    assert make_function_stub_item()["question"].rstrip("\n") in eval_tasks.render_chat_messages(
        make_function_stub_item())[1]["content"]
    assert "assert double_all([1, 2]) == [2, 4]" in eval_tasks.render_chat_messages(
        make_task_description_item())[1]["content"]
    chinese_item = make_multiple_choice_item(["茎", "叶"], 0, question="洋葱是植物的哪一部分", language="zh")
    assert "字母" in eval_tasks.render_chat_messages(chinese_item)[1]["content"]


# ---------------------------------------------------------------------------
# build helpers (no network)
# ---------------------------------------------------------------------------

def make_grouped_items(size_of_group: dict[str, int]) -> list[dict]:
    items = []
    for group_name, size in size_of_group.items():
        for index in range(size):
            item = make_multiple_choice_item(["a", "b", "c", "d"], answer_index=index % 4,
                                             question=f"Question {index} of {group_name}?",
                                             item_id=f"hand:test:{group_name}/{index}")
            item["meta"]["category"] = group_name
            items.append(item)
    return items


def test_proportional_quotas_sum_to_the_sample_and_follow_the_sizes():
    quota_of_group = eval_tasks.allocate_proportional_quotas({"math": 1351, "law": 1101, "history": 381}, 100)
    assert sum(quota_of_group.values()) == 100
    assert quota_of_group == {"math": 48, "law": 39, "history": 13}        # 47.7, 38.9, 13.4: the two largest remainders round up
    assert eval_tasks.allocate_proportional_quotas({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    with pytest.raises(AssertionError):
        eval_tasks.allocate_proportional_quotas({"a": 3}, 4)


def test_stratified_sample_is_proportional_seeded_and_independent_of_row_order():
    items = make_grouped_items({"math": 60, "law": 30, "history": 10})
    sample = eval_tasks.sample_evaluated_items(items, 20, "category")
    counts = {group_name: sum(item["meta"]["category"] == group_name for item in sample)
              for group_name in ("math", "law", "history")}
    assert counts == {"math": 12, "law": 6, "history": 2}
    assert [item["item_id"] for item in sample] == [item["item_id"] for item in items if item in sample]   # order kept
    sample_from_reversed_rows = eval_tasks.sample_evaluated_items(list(reversed(items)), 20, "category")
    assert {item["item_id"] for item in sample_from_reversed_rows} == {item["item_id"] for item in sample}
    plain_sample = eval_tasks.sample_evaluated_items(items, 20, None)
    assert len(plain_sample) == 20 and plain_sample == eval_tasks.sample_evaluated_items(items, 20, None)
    assert eval_tasks.sample_evaluated_items(items, len(items), None) == items


def test_shots_spread_answer_letters_and_groups():
    candidates = make_grouped_items({"math": 20, "law": 20, "history": 20, "physics": 20, "biology": 20})
    shots = eval_tasks.select_few_shot_items(candidates, 5, "category")
    assert len({shot["meta"]["category"] for shot in shots}) == 5
    assert {shot["answer_index"] for shot in shots} == {0, 1, 2, 3}          # four letters cover five shots
    assert shots == eval_tasks.select_few_shot_items(list(reversed(candidates)), 5, "category")
    numeric_candidates = ([make_numeric_item(str(index), item_id=f"hand:train:n{index}") for index in range(30)]
                          + [make_numeric_item(rf"\frac{{1}}{{{index}}}", item_id=f"hand:train:e{index}")
                             for index in range(2, 5)])
    numeric_shots = eval_tasks.select_few_shot_items(numeric_candidates, 5, None)
    expression_shot_count = sum(not eval_tasks.is_plain_number_text(shot["answer_text"]) for shot in numeric_shots)
    assert expression_shot_count == 2, "math shots should show both a number and an expression as the answer"
    with pytest.raises(AssertionError):
        eval_tasks.select_few_shot_items(candidates[:3], 5, None)


def test_normalisers_on_rows_as_published():
    gsm8k_item = eval_tasks.normalise_gsm8k_row(
        {"question": "How many clips? ", "answer": "She sold 48/2 = <<48/2=24>>24 clips in May\nIn total 1,024.\n#### 1,024"},
        "test", 3)
    assert gsm8k_item["item_id"] == "gsm8k:test:3" and gsm8k_item["answer_text"] == "1024"
    assert gsm8k_item["meta"]["worked_solution"] == "She sold 48/2 = 24 clips in May. In total 1,024."

    mmlu_pro_item = eval_tasks.normalise_mmlu_pro_row(
        {"question_id": 2, "question": "Pick.", "options": ["-5", "0", "-3", "N/A", "N/A"], "answer": "C",
         "answer_index": 2, "category": "math", "src": "x"}, "validation", 0)
    assert mmlu_pro_item["choices"] == ["-5", "0", "-3"] and mmlu_pro_item["answer_index"] == 2

    arc_item = eval_tasks.normalise_arc_row(
        {"id": "X_1", "question": "Which?", "choices": {"text": ["w", "x", "y", "z"], "label": ["1", "2", "3", "4"]},
         "answerKey": "3"}, "test", 0)
    assert arc_item["answer_index"] == 2 and arc_item["choices"] == ["w", "x", "y", "z"]

    hellaswag_item = eval_tasks.normalise_hellaswag_row(
        {"ind": 1, "activity_label": "Personal Care", "ctx_a": "[header] How to gel hair [title] Choose a gel. [step] Gels vary.",
         "ctx_b": "", "endings": ["[substeps] Wet hair breaks. [title] Look for more.", "it", "they", "we"],
         "source_id": "wikihow~1", "split": "val", "split_type": "indomain", "label": "0"}, "validation", 7)
    assert hellaswag_item["meta"]["passage"] == "Personal Care: How to gel hair. Choose a gel. Gels vary."
    assert hellaswag_item["choices"][0] == "Wet hair breaks. Look for more."        # no doubled full stop
    assert hellaswag_item["question"].startswith("Which ending continues the passage most plausibly?\n")
    activity_item = eval_tasks.normalise_hellaswag_row(
        {"ind": 2, "activity_label": "Roofing", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he",
         "endings": ["a", "b", "c", "d"], "source_id": "activitynet~v", "split": "val", "split_type": "zeroshot",
         "label": "3"}, "validation", 8)
    assert activity_item["meta"]["passage"] == "Roofing: A man is sitting on a roof. He"
    assert activity_item["answer_index"] == 3

    mbpp_item = eval_tasks.normalise_mbpp_row(
        {"task_id": 6, "prompt": "Write a function to compare bits.",
         "code": "def is_power(x):\n    return x and not (x & (x - 1))\ndef differ(a, b):\n    return is_power(a ^ b)",
         "test_imports": ["import math"], "test_list": ["assert differ(13, 9) == True"]}, "prompt", 0)
    assert mbpp_item["entry_point"] == "differ"                                    # the helper is not the entry point
    assert mbpp_item["test_code"] == "import math\nassert differ(13, 9) == True\n"

    humaneval_item = eval_tasks.normalise_humaneval_row(
        {"task_id": "HumanEval/0", "prompt": "def f(x):\n    \"\"\"Return x.\"\"\"\n", "entry_point": "f",
         "canonical_solution": "    return x\n", "test": "\n\ndef check(candidate):\n    assert candidate(1) == 1\n"},
        "test", 0)
    assert humaneval_item["test_code"].endswith("\ncheck(f)\n")                    # the tests are actually called

    triviaqa_item = eval_tasks.normalise_triviaqa_row(
        {"question": "Which band?", "question_id": "tc_1", "question_source": "x",
         "answer": {"value": "The Beatles", "aliases": ["Beatles", "The", "Fab Four"],
                    "normalized_aliases": ["beatles", "fab four"], "type": "WikipediaEntity"}}, "validation", 0)
    assert triviaqa_item["answer_aliases"] == ["The Beatles", "Fab Four"]         # one per normalised form, none empty

    cmmlu_item = eval_tasks.normalise_cmmlu_row(
        {"": "0", "Question": "牛的发情周期平均为", "A": "16～17天", "B": "21天", "C": "28天", "D": "19～20天", "Answer": "B"},
        "test", "agronomy", 0)
    assert cmmlu_item["item_id"] == "cmmlu:test:agronomy/0" and cmmlu_item["answer_index"] == 1
    assert cmmlu_item["meta"]["language"] == "zh"

    for item in (gsm8k_item, mmlu_pro_item, arc_item, hellaswag_item, mbpp_item, humaneval_item, triviaqa_item,
                 cmmlu_item):
        eval_tasks.assert_item_follows_schema(item)


def test_schema_check_rejects_broken_items():
    broken_items = [
        dict(four_choice_item, answer_index=4),                                  # outside the four options
        dict(four_choice_item, choices=["only one"]),
        dict(make_numeric_item("7"), answer_text=" "),
        dict(trivia_item, answer_aliases=["The"]),                               # would match an empty answer
        dict(make_function_stub_item(), test_code=""),
        dict(make_numeric_item("7"), kind="essay"),
        {"item_id": "x", "task": "hand", "kind": "numeric", "answer_text": "1", "meta": {}},        # no question
    ]
    for broken_item in broken_items:
        with pytest.raises((AssertionError, KeyError)):
            eval_tasks.assert_item_follows_schema(broken_item)


def test_json_lines_round_trip_and_hash(tmp_path):
    items = [make_multiple_choice_item(["茎", "叶"], 0, question="洋葱 是什么", language="zh"),
             make_numeric_item(r"\frac{3}{4}")]
    file_path = tmp_path / "items.jsonl"
    file_sha256 = eval_tasks.write_json_lines(file_path, items)
    assert file_sha256 == hashlib.sha256(file_path.read_bytes()).hexdigest()
    assert eval_tasks.read_json_lines(file_path) == items
    file_text = file_path.read_text(encoding="utf-8")
    assert len(file_text.splitlines()) == 2, "a line separator inside a question must not split its line"
    assert "洋葱" in file_text                                                    # readable, not \u escapes


def test_task_table_is_complete():
    assert list(eval_tasks.evaluation_tasks) == ["gsm8k", "math500", "humaneval", "mbpp", "mmlu_pro", "arc_challenge",
                                                 "hellaswag", "commonsenseqa", "triviaqa", "cmmlu", "svamp", "asdiv_a"]
    for task_name, task in eval_tasks.evaluation_tasks.items():
        assert len(task["hub_revision"]) == 40, f"{task_name}: the revision must be a full commit sha"
        assert task["kind"] in ("multiple_choice", "numeric", "code", "short_answer")
        assert task["few_shot_item_count"] == 0 or 5 <= task["few_shot_item_count"] <= 8
        assert task["few_shot_split"] != task["evaluated_split"], f"{task_name}: shots from the evaluated split"
        assert task["evaluated_item_count"] <= task["evaluated_split_item_count"]


# ---------------------------------------------------------------------------
# the files that `build` wrote (skipped until it has run)
# ---------------------------------------------------------------------------

manifest_path = eval_tasks.evaluation_directory() / "manifest.json"
needs_built_files = pytest.mark.skipif(not manifest_path.exists(),
                                       reason="run `python -m midtrain.eval_tasks build` to write the evaluation files")


@needs_built_files
def test_manifest_describes_every_task_and_matches_the_files():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tasks_not_built"] == {}
    assert list(manifest["tasks"]) == list(eval_tasks.evaluation_tasks)
    for task_name, entry in manifest["tasks"].items():
        task = eval_tasks.evaluation_tasks[task_name]
        assert entry["hub_id"] == task["hub_id"] and entry["hub_revision"] == task["hub_revision"]
        assert entry["evaluated_split"] == task["evaluated_split"] and entry["licence"]
        assert entry["item_count"] == task["evaluated_item_count"]
        assert entry["few_shot_item_count"] == task["few_shot_item_count"]
        assert set(entry["files"]) == {f"{task_name}.jsonl", f"{task_name}_fewshot.jsonl"}
        for file_name, file_record in entry["files"].items():
            file_path = eval_tasks.evaluation_directory() / file_name
            assert hashlib.sha256(file_path.read_bytes()).hexdigest() == file_record["sha256"], file_name
            assert len(eval_tasks.read_json_lines(file_path)) == file_record["item_count"], file_name


@needs_built_files
@pytest.mark.parametrize("task_name", list(eval_tasks.evaluation_tasks))
def test_built_shots_are_never_evaluated_items_and_every_prompt_renders(task_name):
    task = eval_tasks.evaluation_tasks[task_name]
    items = eval_tasks.load_evaluation_items(task_name)
    few_shot_items = eval_tasks.load_few_shot_items(task_name)
    assert len({item["item_id"] for item in items}) == len(items)
    evaluated_item_ids = {item["item_id"] for item in items}
    evaluated_question_texts = {eval_tasks.normalise_question_for_comparison(item["question"]) for item in items}
    for few_shot_item in few_shot_items:
        assert few_shot_item["meta"]["split"] == task["few_shot_split"] != task["evaluated_split"]
        assert few_shot_item["item_id"] not in evaluated_item_ids
        assert eval_tasks.normalise_question_for_comparison(few_shot_item["question"]) not in evaluated_question_texts
    for item in items:
        eval_tasks.assert_item_follows_schema(item)
        assert item["meta"]["split"] == task["evaluated_split"] and item["kind"] == task["kind"]
        prompt_text = eval_tasks.render_base_prompt(item, few_shot_items)       # raises if the item is among its shots
        assert prompt_text.count(item["question"]) == 1 or task_name == "humaneval" and prompt_text == item["question"]
        assert len(eval_tasks.render_chat_messages(item)) == 2
    if task["kind"] == "multiple_choice" and few_shot_items:
        assert len({few_shot_item["answer_index"] for few_shot_item in few_shot_items}) >= min(
            4, len(few_shot_items)), "the shots should not all have the same answer letter"


@needs_built_files
@pytest.mark.parametrize("task_name", ["svamp", "asdiv_a"])
def test_word_problem_sets_are_answered_from_the_gsm8k_prompt(task_name):
    """SVAMP and ASDiv-A test whether a GSM8K gain carries to other problems, so only the problems may differ."""
    gsm8k_shot_ids = [shot["item_id"] for shot in eval_tasks.load_few_shot_items("gsm8k")]
    assert [shot["item_id"] for shot in eval_tasks.load_few_shot_items(task_name)] == gsm8k_shot_ids


@needs_built_files
def test_built_stratified_samples_cover_every_group():
    category_counts = {}
    for item in eval_tasks.load_evaluation_items("mmlu_pro"):
        category_counts[item["meta"]["category"]] = category_counts.get(item["meta"]["category"], 0) + 1
    assert len(category_counts) == 14 and sum(category_counts.values()) == 1200
    assert category_counts["math"] == 135 and category_counts["history"] == 38     # 1,351 and 381 of 12,032 questions
    subject_counts = {}
    for item in eval_tasks.load_evaluation_items("cmmlu"):
        subject_counts[item["meta"]["subject"]] = subject_counts.get(item["meta"]["subject"], 0) + 1
    assert len(subject_counts) == 67 and sum(subject_counts.values()) == 1000 and min(subject_counts.values()) >= 9
    assert len({item["question"] for item in eval_tasks.load_evaluation_items("triviaqa")}) == 1000


@needs_built_files
@pytest.mark.parametrize("task_name", ["gsm8k", "math500", "svamp", "asdiv_a"])
def test_built_numeric_gold_answers_are_graded_correct_and_changed_ones_wrong(task_name):
    r"""The data set's own solution (its last \boxed{} or `#### x`) must be read back as its own answer."""
    for item in eval_tasks.load_evaluation_items(task_name):
        answer_text = item["answer_text"]
        assert eval_tasks.grade_completion(item, item["meta"]["reference_solution"], "chat"), item["item_id"]
        assert eval_tasks.grade_completion(item, " " + eval_tasks.render_final_answer_sentence(answer_text), "base")
        assert not eval_tasks.grade_completion(item, "\\boxed{" + answer_text + "1}", "chat"), item["item_id"]


@needs_built_files
@pytest.mark.parametrize("task_name", ["humaneval", "mbpp"])
def test_built_reference_solutions_pass_in_the_sandbox(task_name, sandbox_backend_name):
    """Every tenth item: `python -m midtrain.eval_tasks check` runs all 421 reference solutions in both styles."""
    items = eval_tasks.load_evaluation_items(task_name)[::10]
    reference_solutions = [item["meta"]["reference_solution"] for item in items]
    verdicts = eval_tasks.grade_code_completions_in_parallel(items, reference_solutions, "base")
    assert [verdict["reason"] for verdict in verdicts] == ["passed"] * len(items)
    if task_name == "humaneval":
        broken_solutions = ["    raise RuntimeError('deliberately broken')\n"] * len(items)
    else:
        broken_solutions = [f"def {item['entry_point']}(*arguments):\n    raise RuntimeError('deliberately broken')\n"
                            for item in items]
    broken_verdicts = eval_tasks.grade_code_completions_in_parallel(items, broken_solutions, "base")
    assert [verdict["reason"] for verdict in broken_verdicts] == ["failed:RuntimeError"] * len(items)
