"""Generate "reasoning primitives": worked solutions whose every number is computed by code, plus a held-out test.

Why this file exists. The reinforcement-learning stage that follows continued pretraining needs a high pass@k: among
k sampled attempts at least one must be right, so the model has to know many ways to work a problem, has to notice
its own slips, and has to use a tool result when one is offered. Web text shows finished answers far more often than
the working, and working written by a language model is only as right as that model. Here each document comes from a
small problem generator and a step-by-step solver written in code, so the text is correct by construction and no
language model is involved. Every training family comes in three variants:
    plain            a correct worked solution; about one in five solves the problem by two methods that agree
    error_recovery   a plausible slip, a check that exposes it and the corrected step, or a dead end: an approach
                     that is tried and dropped for a stated reason; the final answer is always right
    tool_use         the trace calls a calculator, a Python interpreter, or a lookup / search over a table given in
                     the document, and goes on with the returned value; the result is what the code computed
The same solver output is laid out in eight ways (prose, question and answer, tutor dialogue, numbered steps,
notebook entry, forum answer, textbook example, scratchpad), so a family is a skill and not a template to memorise.

The held-out test (`build-eval`) uses five other families, each a chain of 6 to 12 dependent steps (function
composition, object swaps, variable substitution, modular updates, a walk on a grid). The training generator
refuses these names, and the program-tracing family leaves out the constructs they are made of, so a gain on the
test is a gain in carrying state over many steps and not recall of a seen format.

Measured facts that shaped the design:
  * `<tool_call>`, `<tool_response>` and `<think>` are single added tokens of the Qwen3.5 tokenizer (checked
    2026-11-20). The tool formats are meant to be generic text, so the tag format uses `<call>`, `<function>` and
    `<request>`, and the builder asserts that no document contains any added token of the tokenizer.
  * `84.5 * 0.85` prints `71.82499999999999` in Python. Tool expressions therefore use integers, with at most one
    final division, and every tool result is compared with the value the step-by-step solver reached.
  * A document averages about 300 tokens, so a 30,000,000-token pool holds about 100,000 documents. Work is cut
    into shards of 250,000 tokens, each with its own seed, so the output does not depend on the worker count.

Commands (from the repository root, after `source scripts/env.sh`):
    $MIDTRAIN_PYTHON -m midtrain.reasoning_primitives build --token-budget 30000000 --seed 20261120
    $MIDTRAIN_PYTHON -m midtrain.reasoning_primitives build-eval --item-count 400 --seed 20261121
    $MIDTRAIN_PYTHON -m midtrain.reasoning_primitives show --family sorting_trace --variant error_recovery --count 3
    $MIDTRAIN_PYTHON -m pytest tests/test_reasoning_primitives.py -q
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime
import fractions
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

pool_source_name = "primitives"
evaluation_task_name = "long_chain"
variant_names = ("plain", "error_recovery", "tool_use")
tool_format_names = ("json_function_call", "xml_tags", "react", "python_repl")
difficulty_levels = (1, 2, 3)
layout_names = ("prose_solution", "question_answer", "tutor_dialogue", "numbered_steps", "notebook_entry",
                "forum_answer", "textbook_example", "scratchpad")
# Share of the pool's tokens given to each variant. Plain solutions carry the method; the other two carry the
# habits the later stage needs (checking, recovering, using a returned value), so together they get the larger part.
default_variant_token_shares = {"plain": 0.40, "error_recovery": 0.30, "tool_use": 0.30}
# One plain document in five shows two methods, and one error-recovery document in three is a dead end, in the
# families that have them: enough to be seen often, few enough that the single clean solution stays the norm.
two_methods_share_of_plain = 0.20
dead_end_share_of_error_recovery = 0.35
pool_rows_per_parquet_file = 50_000
shard_token_budget = 250_000
documents_per_tokenizer_batch = 64
# Strings that are control tokens in some chat template. A document that contained one would teach the model to
# emit a control token in the middle of ordinary text, so every generated text is checked against this list and,
# in the builder, against the added tokens of the tokenizer itself.
forbidden_chat_template_strings = (
    "<|", "|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>",
    "[INST]", "[/INST]", "<<SYS>>", "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<function_calls>",
    "<invoke")

person_names = ("Ana", "Ben", "Cora", "Dev", "Elif", "Farid", "Gia", "Hugo", "Ines", "Jamal", "Kira", "Leo", "Mei",
                "Nico", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Uma", "Viktor", "Wen", "Ximena", "Yusuf",
                "Zoe", "Amara", "Bruno", "Chidi", "Dana", "Emeka", "Freya", "Goran", "Hana", "Ivo", "Jun")
countable_object_names = ("apples", "pencils", "marbles", "stickers", "bolts", "tiles", "candles", "notebooks",
                          "buttons", "bricks", "coins", "seedlings", "postcards", "batteries", "ribbons", "screws")
forum_handle_first_parts = ("quiet", "rusty", "amber", "north", "paper", "lunar", "maple", "pixel", "sandy", "brisk")
forum_handle_second_parts = ("fox", "kettle", "otter", "lantern", "sparrow", "gear", "harbor", "comet", "willow", "badger")


# ---------------------------------------------------------------------------------------------------------------
# small helpers: exact decimals, trace items
# ---------------------------------------------------------------------------------------------------------------
def format_exact_decimal(value: fractions.Fraction | int) -> str:
    """The value as a decimal string with no rounding, e.g. Fraction(1437, 20) -> '71.85', Fraction(6) -> '6'.

    Problems are built so that every stated value terminates; a repeating decimal here means a generator sampled
    numbers it should not have, and a rounded value in a "correct by construction" document would be a wrong one.
    """
    value = fractions.Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    remaining_denominator = value.denominator
    for prime_factor in (2, 5):
        while remaining_denominator % prime_factor == 0:
            remaining_denominator //= prime_factor
    assert remaining_denominator == 1, f"{value} has no terminating decimal form"
    decimal_places = 1
    while (value * 10 ** decimal_places).denominator != 1:
        decimal_places += 1
    scaled_value = abs(int(value * 10 ** decimal_places))
    digits = str(scaled_value).rjust(decimal_places + 1, "0")
    return ("-" if value < 0 else "") + digits[:-decimal_places] + "." + digits[-decimal_places:]


def join_with_and(words: list[str]) -> str:
    """['a'] -> 'a', ['a', 'b'] -> 'a and b', ['a', 'b', 'c'] -> 'a, b and c'."""
    words = [str(word) for word in words]
    if len(words) <= 2:
        return " and ".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def work_line(text: str) -> dict:
    """A correct step of the solution."""
    return {"kind": "work", "text": text}


def mistake_line(text: str) -> dict:
    """A step that is wrong on purpose; a check_line and a fix_line always follow it."""
    return {"kind": "mistake", "text": text}


def check_line(text: str) -> dict:
    """The test that shows the preceding mistake, stated with its numbers."""
    return {"kind": "check", "text": text}


def fix_line(text: str) -> dict:
    """The corrected step; the trace continues from its value."""
    return {"kind": "fix", "text": text}


def abandon_line(text: str) -> dict:
    """Why an approach is dropped: the stated reason of a dead end."""
    return {"kind": "abandon", "text": text}


def method_header(text: str) -> dict:
    """Opens one of several approaches, e.g. 'Method 1: add column by column.'"""
    return {"kind": "method", "text": text}


def tool_step(thought: str, tool_call: dict) -> dict:
    """A tool call with the sentence that says why it is made; the following work_line uses its result."""
    return {"kind": "tool", "text": thought, "tool_call": tool_call}


def choose_trace_kind(random_generator: random.Random, variant: str, has_two_methods: bool,
                      has_dead_end: bool) -> str:
    """Which shape of trace a document gets: single_method / two_methods, slip / dead_end, or tool."""
    if variant == "plain":
        wants_two_methods = random_generator.random() < two_methods_share_of_plain
        return "two_methods" if (has_two_methods and wants_two_methods) else "single_method"
    if variant == "error_recovery":
        wants_dead_end = random_generator.random() < dead_end_share_of_error_recovery
        return "dead_end" if (has_dead_end and wants_dead_end) else "slip"
    assert variant == "tool_use", f"unknown variant {variant!r}"
    return "tool"


# ---------------------------------------------------------------------------------------------------------------
# tools: the result of every call is computed here, by evaluating the call
# ---------------------------------------------------------------------------------------------------------------
modules_available_to_tools = {"math": math, "fractions": fractions, "datetime": datetime, "itertools": itertools}
builtins_available_to_tools = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "format": format, "hex": hex, "int": int, "len": len, "list": list, "max": max,
    "min": min, "oct": oct, "pow": pow, "print": print, "range": range, "reversed": reversed, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip}


def evaluate_tool_expression(python_expression: str, import_statements: list[str], tables: dict):
    """What the tool returns: the value of the expression, with the named modules and the document's tables in scope."""
    namespace = {"__builtins__": builtins_available_to_tools}
    for import_statement in import_statements:
        module_name = import_statement.removeprefix("import ")
        assert module_name in modules_available_to_tools, f"tool may not import {module_name!r}"
        namespace[module_name] = modules_available_to_tools[module_name]
    namespace.update(tables)
    return eval(python_expression, namespace)


def make_tool_call(tool_name: str, python_expression: str, expected_value, arguments: dict | None = None,
                   import_statements: tuple = (), tables: dict | None = None) -> dict:
    """Evaluate one tool call and return it with its result.

    `expected_value` is what the step-by-step solver computed for the same quantity. The two must agree: a tool
    that returned something else would put a wrong number into a document that claims to be verified. A float
    result is compared through its printed form, because that printed form is what the document shows.
    `arguments` is how the call looks in the JSON, tag and ReAct formats (default: the expression itself).
    """
    assert tool_name in ("calculator", "python", "lookup", "search"), f"unknown tool {tool_name!r}"
    tables = tables or {}
    result_value = evaluate_tool_expression(python_expression, list(import_statements), tables)
    if isinstance(result_value, float):
        assert fractions.Fraction(repr(result_value)) == fractions.Fraction(expected_value), (
            f"tool expression {python_expression!r} printed {result_value!r}, the solver computed {expected_value}")
    else:
        assert result_value == expected_value, (
            f"tool expression {python_expression!r} returned {result_value!r}, the solver computed {expected_value!r}")
    if arguments is None:
        arguments = {"expression": python_expression} if tool_name == "calculator" else {"code": python_expression}
    table_names_used = [table_name for table_name in tables if table_name in python_expression]
    return {"tool_name": tool_name, "python_expression": python_expression, "arguments": arguments,
            "import_statements": list(import_statements), "table_names": table_names_used, "tables": tables,
            "result_value": result_value}


def sample_tool_style(random_generator: random.Random) -> dict:
    """The spelling of calls and results, sampled once per document so that one document stays consistent."""
    return {
        "json_key_names": random_generator.choice([("function", "arguments", "result"), ("name", "parameters", "output"),
                                                   ("tool", "input", "output"), ("call", "args", "returned")]),
        "tag_style": random_generator.choice(["call_with_children", "function_with_children", "request_attributes"]),
        "react_action_style": random_generator.choice(["bracket", "input_line"]),
        "react_final_label": random_generator.choice(["Final Answer:", "Final answer:", "Answer:"]),
    }


def serialize_tool_call(tool_format: str, tool_style: dict, thought: str, tool_call: dict,
                        names_already_in_repl: set) -> dict:
    """Write one tool call and its result in the document's tool format.

    Returns `lines_before` (the thought, as prose), `block_lines` (the call and its result, kept together) and
    `result_text` (the result exactly as printed). In the ReAct format the thought is part of the block.
    """
    tool_name, arguments, result_value = tool_call["tool_name"], tool_call["arguments"], tool_call["result_value"]
    python_code_lines = tool_call["import_statements"] + [f"print({tool_call['python_expression']})"]
    if tool_name == "python" and tool_format != "python_repl":
        # Outside the interpreter transcript the python tool is sent a script and returns what the script printed.
        arguments = {"code": "\n".join(python_code_lines)}
        result_text = str(result_value)
        result_as_json_value = result_text
    else:
        result_as_json_value = json.loads(json.dumps(result_value))        # tuples become lists, as JSON would show
        result_text = json.dumps(result_value, ensure_ascii=False) if not isinstance(result_value, str) else result_value
    lines_before, block_lines = [thought], []
    if tool_format == "json_function_call":
        name_key, arguments_key, result_key = tool_style["json_key_names"]
        block_lines.append(json.dumps({name_key: tool_name, arguments_key: arguments}, ensure_ascii=False))
        block_lines.append(json.dumps({result_key: result_as_json_value}, ensure_ascii=False))
        result_text = block_lines[-1]
    elif tool_format == "xml_tags":
        if tool_style["tag_style"] == "call_with_children":
            block_lines.append(f'<call tool="{tool_name}">')
            block_lines.extend(f"  <{key}>{value}</{key}>" for key, value in arguments.items())
            block_lines.extend(["</call>", f"<result>{result_text}</result>"])
        elif tool_style["tag_style"] == "function_with_children":
            argument_tags = "".join(f"<{key}>{value}</{key}>" for key, value in arguments.items())
            block_lines.append(f'<function name="{tool_name}">{argument_tags}</function>')
            block_lines.append(f"<output>{result_text}</output>")
        else:
            is_single_line = all("\n" not in str(value) and '"' not in str(value) for value in arguments.values())
            if is_single_line:
                attributes = " ".join(f'{key}="{value}"' for key, value in arguments.items())
                block_lines.append(f'<request tool="{tool_name}" {attributes}/>')
            else:
                # Code with quotes or line breaks cannot sit in an attribute, so it becomes the body of the tag.
                body = "\n".join(str(value) for value in arguments.values())
                block_lines.append(f'<request tool="{tool_name}">\n{body}\n</request>')
            block_lines.append(f"<response>{result_text}</response>")
    elif tool_format == "react":
        lines_before = []
        block_lines.append(f"Thought: {thought}")
        argument_values = [str(value) for value in arguments.values()]
        if tool_style["react_action_style"] == "bracket" and all("\n" not in value for value in argument_values):
            block_lines.append(f"Action: {tool_name}[{', '.join(argument_values)}]")
        else:
            block_lines.append(f"Action: {tool_name}")
            action_input = argument_values[0] if len(argument_values) == 1 else json.dumps(arguments, ensure_ascii=False)
            block_lines.append(f"Action Input: {action_input}")
        block_lines.append(f"Observation: {result_text}")
    else:
        assert tool_format == "python_repl", f"unknown tool format {tool_format!r}"
        # An interpreter session has to define what it reads: each import and each table is typed once.
        for import_statement in tool_call["import_statements"]:
            if import_statement not in names_already_in_repl:
                names_already_in_repl.add(import_statement)
                block_lines.append(f">>> {import_statement}")
        for table_name in tool_call["table_names"]:
            if table_name not in names_already_in_repl:
                names_already_in_repl.add(table_name)
                block_lines.append(f">>> {table_name} = {tool_call['tables'][table_name]!r}")
        block_lines.append(f">>> {tool_call['python_expression']}")
        result_text = repr(result_value)
        block_lines.append(result_text)
    return {"lines_before": lines_before, "block_lines": block_lines, "result_text": result_text}


# ---------------------------------------------------------------------------------------------------------------
# layouts: the same trace written as prose, a dialogue, numbered steps, a notebook entry, a forum answer ...
# ---------------------------------------------------------------------------------------------------------------
check_openers = ("Check:", "Quick check:", "Sanity check:", "Wait, check that:", "Let me verify:", "Test it:",
                 "Hold on:", "Verify:")
fix_openers = ("Correction:", "Fix:", "Corrected step:", "Redo it:", "So, corrected:", "Put right:")
abandon_openers = ("Dead end:", "Drop this approach:", "This route is abandoned:", "Stop here:", "Not the way to go:")
student_attempt_openers = ("Let me try.", "My attempt:", "I think it goes like this:", "Here is what I get:",
                           "Can I do this step?", "I'll have a go.")
student_interjections = ("Okay.", "Got it, go on.", "I follow.", "What comes next?", "And then?",
                         "Makes sense so far.", "Right.", "Keep going.")


def build_trace_units(random_generator: random.Random, tool_format: str, tool_style: dict,
                      trace_items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Turn trace items into text units and collect the tool calls as they are printed.

    A unit is {"kind", "text", "is_block"}; a block (a tool call with its result, a table, a column layout) keeps
    its line breaks in every layout. Checks, fixes and abandoned approaches get a short opener here so that every
    layout marks them the same way and a reader can find the correction.
    """
    units, tool_call_records, names_already_in_repl = [], [], set()
    for trace_item in trace_items:
        kind, text = trace_item["kind"], trace_item["text"]
        if kind == "tool":
            serialized = serialize_tool_call(tool_format, tool_style, text, trace_item["tool_call"],
                                             names_already_in_repl)
            unit_text = "\n".join(serialized["lines_before"] + serialized["block_lines"])
            units.append({"kind": "tool", "text": unit_text, "is_block": True})
            tool_call_records.append({
                "tool_name": trace_item["tool_call"]["tool_name"],
                "python_expression": trace_item["tool_call"]["python_expression"],
                "import_statements": trace_item["tool_call"]["import_statements"],
                "tables": trace_item["tool_call"]["tables"],
                "block_text": "\n".join(serialized["block_lines"]),
                "result_text": serialized["result_text"]})
            continue
        if kind == "check":
            text = f"{random_generator.choice(check_openers)} {text}"
        elif kind == "fix":
            text = f"{random_generator.choice(fix_openers)} {text}"
        elif kind == "abandon":
            text = f"{random_generator.choice(abandon_openers)} {text}"
        units.append({"kind": kind, "text": text, "is_block": "\n" in text})
    return units, tool_call_records


def group_units_into_paragraphs(units: list[dict]) -> list[str]:
    """Join consecutive one-line units into paragraphs; a block or a method header starts a new paragraph."""
    paragraphs, sentences_of_current_paragraph = [], []
    for unit in units:
        if unit["is_block"] or unit["kind"] == "method":
            if sentences_of_current_paragraph:
                paragraphs.append(" ".join(sentences_of_current_paragraph))
            sentences_of_current_paragraph = []
            if unit["is_block"]:
                paragraphs.append(unit["text"])
                continue
        sentences_of_current_paragraph.append(unit["text"])
    if sentences_of_current_paragraph:
        paragraphs.append(" ".join(sentences_of_current_paragraph))
    return paragraphs


def lay_out_react_transcript(random_generator: random.Random, tool_style: dict, question: str, units: list[dict],
                             answer_sentence: str) -> str:
    """Thought / Action / Observation lines; reasoning between two calls becomes one Thought."""
    question_label = random_generator.choice(["Question:", "Task:", "Problem:", "Input:", "Query:", "User question:"])
    lines, pending_thought_sentences = [f"{question_label} {question}"], []
    for unit in units:
        if unit["kind"] == "tool":
            if pending_thought_sentences:
                lines.append("Thought: " + " ".join(pending_thought_sentences))
                pending_thought_sentences = []
            lines.append(unit["text"])
        else:
            pending_thought_sentences.append(unit["text"])
    if pending_thought_sentences:
        lines.append("Thought: " + " ".join(pending_thought_sentences))
    lines.append(f"{tool_style['react_final_label']} {answer_sentence}")
    return "\n".join(lines)


def lay_out_document(random_generator: random.Random, layout_name: str, question: str, units: list[dict],
                     answer_sentence: str) -> str:
    """Write the question, the trace units and the answer in one of the eight layouts.

    Every layout ends with the answer sentence, so the last line of a document always states the result.
    """
    pick = random_generator.choice
    question_separator = "\n" if "\n" in question else " "
    unit_texts = [unit["text"] for unit in units]
    if layout_name == "prose_solution":
        problem_label = pick(["Problem.", "Exercise.", "Task.", "Question.", "Problem:", "The problem:"])
        solution_label = pick(["Solution.", "Worked solution.", "Reasoning.", "Solution:", "Working it out.",
                               "How to get there."])
        paragraphs = group_units_into_paragraphs(units)
        return (f"{problem_label}{question_separator}{question}\n\n{solution_label} " + "\n\n".join(paragraphs)
                + f"\n\n{answer_sentence}")
    if layout_name == "question_answer":
        question_label, answer_label = pick([("Q:", "A:"), ("Question:", "Answer:"), ("QUESTION:", "ANSWER:"),
                                             ("Q.", "A."), ("Prompt:", "Response:"), ("Problem:", "Solution:")])
        return "\n".join([f"{question_label} {question}", f"{answer_label} {unit_texts[0]}"] + unit_texts[1:]
                         + [answer_sentence])
    if layout_name == "tutor_dialogue":
        student_name, tutor_name = pick([("Student", "Tutor"), ("Learner", "Teacher"), ("Pupil", "Coach"),
                                         tuple(random_generator.sample(person_names, 2))])
        question_opening = pick(["", "Could you help me with this? ", "I'm stuck on this one. ",
                                 "How would you work this out? ", "Can we go through this together? "])
        lines = [f"{student_name}: {question_opening}{question}"]
        sentences_of_tutor_turn, sentences_per_tutor_turn = [], random_generator.randint(2, 4)
        for unit in units:
            if unit["kind"] == "mistake":
                # The slip is the student's and the check is the tutor's: that is how a slip shows up in a lesson.
                if sentences_of_tutor_turn:
                    lines.append(f"{tutor_name}: " + " ".join(sentences_of_tutor_turn))
                    sentences_of_tutor_turn = []
                lines.append(f"{student_name}: {pick(student_attempt_openers)} {unit['text']}")
            elif unit["is_block"]:
                # A block (a tool call, a table) closes the tutor's turn, so the call stays next to its reason.
                lines.append(f"{tutor_name}: " + " ".join(sentences_of_tutor_turn + [unit["text"]]))
                sentences_of_tutor_turn = []
            else:
                if len(sentences_of_tutor_turn) >= sentences_per_tutor_turn and unit["kind"] == "work":
                    lines.append(f"{tutor_name}: " + " ".join(sentences_of_tutor_turn))
                    lines.append(f"{student_name}: {pick(student_interjections)}")
                    sentences_of_tutor_turn = []
                sentences_of_tutor_turn.append(unit["text"])
        if sentences_of_tutor_turn:
            lines.append(f"{tutor_name}: " + " ".join(sentences_of_tutor_turn))
        lines.append(f"{student_name}: {pick(['So what is the result?', 'And the final answer?', 'So in the end?'])}")
        lines.append(f"{tutor_name}: {answer_sentence}")
        return "\n".join(lines)
    if layout_name == "numbered_steps":
        number_style = pick(["{}.", "Step {}:", "({})", "{})", "Step {}.", "S{}:"])
        problem_label = pick(["Problem:", "Task:", "Given:", "Exercise:"])
        answer_label = pick(["Answer:", "Final answer:", "Result:", "Conclusion:", "So:", "Therefore:"])
        lines, step_number = [f"{problem_label}{question_separator}{question}", ""], 0
        for unit in units:
            if unit["kind"] == "method":
                lines.append(unit["text"])
                continue
            step_number += 1
            lines.append(f"{number_style.format(step_number)} {unit['text']}")
        return "\n".join(lines + [f"{answer_label} {answer_sentence}"])
    if layout_name == "notebook_entry":
        header_template = pick(["## {}", "### {}", "**{}**", "[{}]", "{}:", "== {} =="])
        bullet = pick(["- ", "* ", "", "> "])
        problem_title, work_title, result_title = pick([("Problem", "Work", "Result"), ("Task", "Steps", "Answer"),
                                                        ("Question", "Notes", "Conclusion"),
                                                        ("Given", "Working", "Outcome")])
        lines = [header_template.format(problem_title), question, "", header_template.format(work_title)]
        lines.extend(unit["text"] if (unit["is_block"] or unit["kind"] == "method") else bullet + unit["text"]
                     for unit in units)
        return "\n".join(lines + ["", header_template.format(result_title), answer_sentence])
    if layout_name == "forum_answer":
        asker_handle, answerer_handle = [
            f"{pick(forum_handle_first_parts)}_{pick(forum_handle_second_parts)}{random_generator.randint(2, 99)}"
            for _ in range(2)]
        title = pick(["Help with this one?", "Stuck on a homework problem", "Can someone check my method?",
                      "How do I work this out?", "Quick question", "Need a worked answer"])
        answer_label = pick(["Top answer, by", "Accepted answer from", "Reply from", "Best answer, by"])
        opening = pick(["Here is one way to do it.", "Let's go through it carefully.", "Working first, answer at the end.",
                        "Happy to help. Step by step:", "This one is mechanical once it is set up.",
                        "Take it one step at a time."])
        return "\n".join([title, f"{asker_handle} wrote:", question, "", f"{answer_label} {answerer_handle}:", opening]
                         + unit_texts + [answer_sentence])
    if layout_name == "textbook_example":
        example_label = pick(["Example", "Worked example", "Exercise", "Sample problem", "Practice problem",
                              "Illustration"])
        example_number = f"{random_generator.randint(1, 12)}.{random_generator.randint(1, 30)}"
        solution_label = pick(["Solution.", "Solution:", "Worked solution."])
        return "\n".join([f"{example_label} {example_number}.{question_separator}{question}", "", solution_label]
                         + unit_texts + ["", answer_sentence])
    assert layout_name == "scratchpad", f"unknown layout {layout_name!r}"
    scratch_header = pick(["Scratchpad:", "Working:", "Notes:", "scratch work:", "Rough work:", "Margin notes:"])
    indent = pick(["", "  ", "    "])
    answer_arrow = pick(["=>", "Result:", "Answer:", "So:", "Final:", "Done:"])
    indented_unit_texts = [indent + unit_text.replace("\n", "\n" + indent) for unit_text in unit_texts]
    return "\n".join([question, "", scratch_header] + indented_unit_texts + [f"{answer_arrow} {answer_sentence}"])


# ---------------------------------------------------------------------------------------------------------------
# training families. Each function samples one problem, solves it step by step and returns
#   question, trace_items, answer_sentence, final_answer (a substring of answer_sentence), trace_kind,
#   problem (the raw numbers, enough for the tests to recompute the answer by another route).
# ---------------------------------------------------------------------------------------------------------------
column_names = ("ones", "tens", "hundreds", "thousands", "ten-thousands", "hundred-thousands", "millions",
                "ten-millions")


def sum_of_digits(number: int) -> int:
    return sum(int(digit) for digit in str(abs(number)))


def describe_remainder_by_nine(number: int) -> str:
    """'457 -> 16 -> 7': the digit sums that lead to the remainder of the number on division by 9."""
    chain, current = [str(number)], number
    while current >= 10:
        current = sum_of_digits(current)
        chain.append(str(current))
    if current == 9:
        chain.append("0")
    return " -> ".join(chain)


def build_arithmetic_expression(random_generator: random.Random, operation_count: int):
    """A random expression tree with `operation_count` operations; a leaf is an int, a node (operator, left, right).

    A division is only kept when it is exact, so every intermediate value is a whole number.
    """
    if operation_count == 0:
        return random_generator.randint(2, 25)
    operator = random_generator.choice("++--***/")
    left_operation_count = random_generator.randint(0, operation_count - 1)
    left = build_arithmetic_expression(random_generator, left_operation_count)
    if operator == "/":
        left_value = evaluate_arithmetic_expression(left)
        exact_divisors = [divisor for divisor in range(2, 13) if left_value % divisor == 0 and left_value > divisor]
        if operation_count - 1 - left_operation_count == 0 and exact_divisors:
            return ("/", left, random_generator.choice(exact_divisors))
        operator = "+"
    right = build_arithmetic_expression(random_generator, operation_count - 1 - left_operation_count)
    return (operator, left, right)


def evaluate_arithmetic_expression(node) -> int:
    if isinstance(node, int):
        return node
    operator, left, right = node
    left_value, right_value = evaluate_arithmetic_expression(left), evaluate_arithmetic_expression(right)
    if operator == "/":
        assert left_value % right_value == 0, "only exact divisions are generated"
        return left_value // right_value
    return {"+": left_value + right_value, "-": left_value - right_value, "*": left_value * right_value}[operator]


def render_arithmetic_expression(node, operator_symbols: dict, parent_operator: str | None = None,
                                 is_right_operand: bool = False) -> str:
    """The expression as text with only the parentheses its tree needs."""
    if isinstance(node, int):
        return str(node)
    operator, left, right = node
    text = (f"{render_arithmetic_expression(left, operator_symbols, operator, False)} {operator_symbols[operator]} "
            f"{render_arithmetic_expression(right, operator_symbols, operator, True)}")
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    needs_parentheses = parent_operator is not None and (
        precedence[operator] < precedence[parent_operator]
        or (precedence[operator] == precedence[parent_operator] and is_right_operand))
    return f"({text})" if needs_parentheses else text


def reduce_leftmost_innermost_operation(node):
    """Evaluate the first operation whose two operands are numbers; return (new tree, (operator, left, right, value))."""
    operator, left, right = node
    if isinstance(left, int) and isinstance(right, int):
        return evaluate_arithmetic_expression(node), (operator, left, right, evaluate_arithmetic_expression(node))
    if not isinstance(left, int):
        new_left, reduction = reduce_leftmost_innermost_operation(left)
        return (operator, new_left, right), reduction
    new_right, reduction = reduce_leftmost_innermost_operation(right)
    return (operator, left, new_right), reduction


def find_tempting_left_to_right_step(node):
    """A place like 12 + 7 * 3 where reading left to right would wrongly start with 12 + 7; None when there is none."""
    if isinstance(node, int):
        return None
    operator, left, right = node
    if (operator in "+-" and isinstance(left, int) and not isinstance(right, int) and right[0] in "*/"
            and isinstance(right[1], int) and isinstance(right[2], int)):
        return (left, operator, right[1], right[0], right[2])
    return find_tempting_left_to_right_step(left) or find_tempting_left_to_right_step(right)


def make_wrong_arithmetic_result(random_generator: random.Random, operator: str, left: int, right: int,
                                 value: int) -> tuple[int, str]:
    """A plausible wrong value for `left operator right`, and the check that exposes it."""
    if operator == "+":
        wrong_value = value + random_generator.choice([-10, 10])
        return wrong_value, f"{wrong_value} - {right} = {wrong_value - right}, not {left}."
    if operator == "-":
        wrong_value = value + random_generator.choice([-10, 10, 2 * right]) if value >= 10 else value + 10
        return wrong_value, f"{wrong_value} + {right} = {wrong_value + right}, not {left}."
    if operator == "*":
        wrong_value = value + random_generator.choice([-left, left, -right, right, -10, 10])
        if wrong_value % 10 != value % 10:
            return wrong_value, (f"{left} times {right} must end in {(left % 10) * (right % 10) % 10}, "
                                 f"because {left % 10} times {right % 10} is {(left % 10) * (right % 10)}; "
                                 f"{wrong_value} ends in {wrong_value % 10}.")
        return wrong_value, (f"dividing back, {wrong_value} / {right} should be {left}, "
                             f"but {right} times {left} is {value}, not {wrong_value}.")
    wrong_value = value + random_generator.choice([-1, 1])
    return wrong_value, f"{wrong_value} times {right} is {wrong_value * right}, not {left}."


def generate_multi_step_arithmetic(random_generator: random.Random, variant: str, difficulty: int) -> dict:
    """Evaluate an expression with parentheses and mixed operations, one operation per step."""
    pick = random_generator.choice
    trace_kind = choose_trace_kind(random_generator, variant, has_two_methods=False, has_dead_end=True)
    operation_count = random_generator.randint(*{1: (2, 3), 2: (4, 5), 3: (6, 8)}[difficulty])
    for _ in range(2000):
        expression = build_arithmetic_expression(random_generator, operation_count)
        reductions, remaining = [], expression
        while not isinstance(remaining, int):
            remaining, reduction = reduce_leftmost_innermost_operation(remaining)
            reductions.append((reduction, remaining))
        # values of 0 and 1 make steps like "0 * 500" whose checks read oddly, and large ones are only long
        values_stay_readable = all(2 <= reduction[3] <= 2000 for reduction, _ in reductions)
        has_multiplication_or_division = any(reduction[0] in "*/" for reduction, _ in reductions)
        tempting_step = find_tempting_left_to_right_step(expression)
        if values_stay_readable and has_multiplication_or_division and (trace_kind != "dead_end" or tempting_step):
            break
    else:
        raise AssertionError("no readable expression found in 2000 tries")
    operator_symbols = pick([{"+": "+", "-": "-", "*": "×", "/": "÷"}, {"+": "+", "-": "-", "*": "*", "/": "/"}])
    expression_text = render_arithmetic_expression(expression, operator_symbols)
    final_value = evaluate_arithmetic_expression(expression)
    question = pick(["Evaluate {}.", "Compute the value of {}.", "What is {}?", "Work out {} step by step.",
                     "Simplify {} to a single number.", "Find the value of the expression {}."]).format(expression_text)
    step_template = pick(["{left} {symbol} {right} = {value}, which leaves {rest}.",
                          "Compute {left} {symbol} {right} = {value}. The expression is now {rest}.",
                          "{left} {symbol} {right} = {value}, so we have {rest}."])
    last_step_template = pick(["{left} {symbol} {right} = {value}.", "Finally {left} {symbol} {right} = {value}."])

    def describe_reduction(reduction, remaining_tree, value_shown=None):
        operator, left, right, value = reduction
        template = last_step_template if isinstance(remaining_tree, int) else step_template
        return template.format(left=left, symbol=operator_symbols[operator], right=right,
                               value=value if value_shown is None else value_shown,
                               rest=render_arithmetic_expression(remaining_tree, operator_symbols))

    trace_items = []
    if random_generator.random() < 0.4:
        trace_items.append(work_line(f"Order of operations: parentheses first, then {operator_symbols['*']} and "
                                     f"{operator_symbols['/']}, then + and -, left to right."))
    if trace_kind == "dead_end":
        first, low_operator, second, high_operator, third = tempting_step
        tempting_value = first + second if low_operator == "+" else first - second
        trace_items.append(work_line(
            f"First idea: read left to right and start with {first} {low_operator} {second} = {tempting_value}."))
        trace_items.append(abandon_line(
            f"{operator_symbols[high_operator]} binds tighter than {low_operator}, so {second} belongs to "
            f"{second} {operator_symbols[high_operator]} {third} and cannot be used up first. "
            f"Start again and follow the order of operations."))
    slip_index = random_generator.randrange(len(reductions)) if trace_kind == "slip" else None
    tool_operators = "*/" if trace_kind == "tool" else ""
    for reduction_index, (reduction, remaining_tree) in enumerate(reductions):
        operator, left, right, value = reduction
        if reduction_index == slip_index:
            wrong_value, check_text = make_wrong_arithmetic_result(random_generator, operator, left, right, value)
            trace_items.append(mistake_line(f"{left} {operator_symbols[operator]} {right} = {wrong_value}."))
            trace_items.append(check_line(check_text))
            trace_items.append(fix_line(describe_reduction(reduction, remaining_tree)))
        elif operator in tool_operators:
            tool_call = make_tool_call("calculator", f"{left} {operator} {right}", value)
            trace_items.append(tool_step(f"I need {left} {operator_symbols[operator]} {right}.", tool_call))
            trace_items.append(work_line(describe_reduction(reduction, remaining_tree)))
        else:
            trace_items.append(work_line(describe_reduction(reduction, remaining_tree)))
    answer_sentence = pick(["The value is {}.", "The expression equals {}.", "The result is {}."]).format(final_value)
    python_expression = render_arithmetic_expression(expression, {"+": "+", "-": "-", "*": "*", "/": "//"})
    return {"question": question, "trace_items": trace_items, "answer_sentence": answer_sentence,
            "final_answer": str(final_value), "trace_kind": trace_kind,
            "problem": {"python_expression": python_expression}}


def add_in_columns(addends: list[int], position_of_dropped_carry: int | None = None) -> tuple[list[str], int, list[int]]:
    """Column addition from the ones column up: (one line per column, the total, the carry out of each column).

    With `position_of_dropped_carry` set, the carry coming out of that column is forgotten, which is the slip the
    error-recovery variant shows; the returned total is then the wrong one that such a slip produces.
    """
    column_count = max(len(str(addend)) for addend in addends)
    column_lines, written_digits, carries_out, carry = [], [], [], 0
    for position in range(column_count):
        digits = [(addend // 10 ** position) % 10 for addend in addends if addend >= 10 ** position]
        terms = [str(digit) for digit in digits] + ([f"carry {carry}"] if carry else [])
        column_sum = sum(digits) + carry
        is_last_column = position == column_count - 1
        carry_out = column_sum // 10
        if is_last_column:
            written_digits.append(str(column_sum))
            ending = f"write {column_sum}."
        else:
            written_digits.append(str(column_sum % 10))
            ending = f"write {column_sum % 10}" + (f", carry {carry_out}." if carry_out else ".")
        equation = " + ".join(terms) + f" = {column_sum}" if len(terms) > 1 else f"{terms[0]}"
        column_lines.append(f"{column_names[position].capitalize()}: {equation}, {ending}")
        carries_out.append(carry_out)
        carry = 0 if position == position_of_dropped_carry else carry_out
    return column_lines, int("".join(reversed(written_digits))), carries_out


def generate_column_addition(random_generator: random.Random, variant: str, difficulty: int) -> dict:
    """Add two to four whole numbers column by column, with the carries written out."""
    pick = random_generator.choice
    trace_kind = choose_trace_kind(random_generator, variant, has_two_methods=True, has_dead_end=False)
    addend_count = {1: 2, 2: random_generator.randint(2, 3), 3: random_generator.randint(3, 4)}[difficulty]
    digit_count_range = {1: (3, 4), 2: (4, 5), 3: (5, 7)}[difficulty]
    while True:
        addends = [random_generator.randint(10 ** (random_generator.randint(*digit_count_range) - 1),
                                            10 ** digit_count_range[1] - 1) for _ in range(addend_count)]
        column_lines, total, carries_out = add_in_columns(addends)
        positions_with_carry = [position for position, carry_out in enumerate(carries_out[:-1]) if carry_out]
        if positions_with_carry:                      # an addition without a carry has nothing worth tracing
            break
    assert total == sum(addends), "column addition disagrees with the direct sum"
    sum_text = " + ".join(str(addend) for addend in addends)
    listed_text = join_with_and(addends)
    object_name, person_name = pick(countable_object_names), pick(person_names)
    question = pick([
        f"Add {listed_text}." if trace_kind == "tool" else f"Add {listed_text} using column addition.",
        f"Compute {sum_text}." if trace_kind == "tool" else f"Compute {sum_text} by hand.",
        f"Find the sum {sum_text}.", f"What is the total of {listed_text}?",
        f"A warehouse received shipments of {listed_text} {object_name}. How many {object_name} arrived in total?",
        f"{person_name} counted {listed_text} {object_name} in separate boxes. How many {object_name} is that altogether?"])
    intro = work_line(pick(["Line the numbers up on the right and add each column, starting with the ones.",
                            "Add column by column from the right, carrying when a column reaches 10.",
                            "Work from the ones column leftward."]))
    read_off_template = pick(["Reading the written digits from the last column back to the ones gives {}.",
                              "Put the digits together: {}.", "So the columns give {}."])
    trace_items = [intro]
    if trace_kind == "slip":
        dropped_position = pick(positions_with_carry)
        wrong_lines, wrong_total, _ = add_in_columns(addends, position_of_dropped_carry=dropped_position)
        for position, wrong_line in enumerate(wrong_lines):
            is_the_slip = position == dropped_position + 1
            trace_items.append(mistake_line(wrong_line) if is_the_slip else work_line(wrong_line))
        trace_items.append(work_line(read_off_template.format(wrong_total)))
        remainder_of_addends = sum(addend % 9 for addend in addends) % 9
        trace_items.append(check_line(
            "a number and its digit sum leave the same remainder on division by 9. "
            + "; ".join(describe_remainder_by_nine(addend) for addend in addends)
            + f"; together the addends leave {remainder_of_addends}. But {describe_remainder_by_nine(wrong_total)}, "
            + f"so {wrong_total} leaves {wrong_total % 9}. The two differ, so a carry was lost."))
        assert remainder_of_addends != wrong_total % 9, "the nines check must expose a dropped carry"
        redone_lines = " ".join(column_lines[dropped_position + 1:])
        trace_items.append(fix_line(
            f"the carry out of the {column_names[dropped_position]} column was dropped. From the "
            f"{column_names[dropped_position + 1]} column on: {redone_lines}"))
        trace_items.append(work_line(read_off_template.format(total)))
    elif trace_kind == "tool":
        running_total = addends[0]
        for addend in addends[1:]:
            tool_call = make_tool_call("calculator", f"{running_total} + {addend}", running_total + addend)
            trace_items.append(tool_step(f"Add {addend} to the running total {running_total}.", tool_call))
            running_total += addend
            trace_items.append(work_line(f"The running total is now {running_total}."))
        trace_items[0] = work_line("Add the numbers one at a time and keep a running total.")
    else:
        if trace_kind == "two_methods":
            trace_items.insert(0, method_header("Method 1: column addition."))
        trace_items.extend(work_line(column_line) for column_line in column_lines)
        trace_items.append(work_line(read_off_template.format(total)))
        if trace_kind == "two_methods":
            trace_items.append(method_header("Method 2: add each place value separately, then combine."))
            place_value_sums = []
            for position in reversed(range(len(carries_out))):
                parts = [(addend // 10 ** position) % 10 * 10 ** position for addend in addends]
                parts = [part for part in parts if part]
                if parts:
                    place_value_sums.append(sum(parts))
                    trace_items.append(work_line(f"{column_names[position].capitalize()}: "
                                                 + " + ".join(str(part) for part in parts) + f" = {sum(parts)}."))
            trace_items.append(work_line(" + ".join(str(part) for part in place_value_sums) + f" = {total}."))
            trace_items.append(work_line(f"Both methods give {total}."))
    answer_sentence = pick(["The sum is {}.", "The total is {}.", "Altogether that makes {}."]).format(total)
    return {"question": question, "trace_items": trace_items, "answer_sentence": answer_sentence,
            "final_answer": str(total), "trace_kind": trace_kind, "problem": {"addends": addends}}


def multiply_by_one_digit(multiplicand: int, digit: int, position_of_dropped_carry: int | None = None,
                          units_fact_offset: int = 0) -> tuple[str, int, list[int]]:
    """One row of long multiplication: (the digit-by-digit description, the row's product, the carry out of each digit).

    `position_of_dropped_carry` forgets the carry out of that digit, and `units_fact_offset` misremembers the first
    times-table fact by that amount: the two slips the error-recovery variant shows.
    """
    multiplicand_digits = [int(character) for character in reversed(str(multiplicand))]
    descriptions, written_digits, carries_out, carry = [], [], [], 0
    for position, multiplicand_digit in enumerate(multiplicand_digits):
        digit_product = multiplicand_digit * digit + (units_fact_offset if position == 0 else 0)
        digit_total = digit_product + carry
        description = f"{multiplicand_digit} × {digit} = {digit_product}"
        if carry:
            description += f", plus the carried {carry} is {digit_total}"
        carry_out = digit_total // 10
        if position == len(multiplicand_digits) - 1:
            written_digits.append(str(digit_total))
            description += f", write {digit_total}"
        else:
            written_digits.append(str(digit_total % 10))
            description += f", write {digit_total % 10}" + (f" and carry {carry_out}" if carry_out else "")
        descriptions.append(description)
        carries_out.append(carry_out)
        carry = 0 if position == position_of_dropped_carry else carry_out
    return "; ".join(descriptions), int("".join(reversed(written_digits))), carries_out


def generate_long_multiplication(random_generator: random.Random, variant: str, difficulty: int) -> dict:
    """Multiply a 2-4 digit number by a 1-3 digit number: one row per multiplier digit, then the rows are added."""
    pick = random_generator.choice
    multiplier_digit_count = difficulty
    while True:
        multiplicand = random_generator.randint(23, {1: 999, 2: 999, 3: 9999}[difficulty])
        multiplier = random_generator.randint(10 ** (multiplier_digit_count - 1) + 1, 10 ** multiplier_digit_count - 1)
        multiplier_digits = [int(character) for character in reversed(str(multiplier))]
        rows = []                              # (position, digit, description, row product, carries of the row)
        for position, digit in enumerate(multiplier_digits):
            if digit:
                rows.append((position, digit) + multiply_by_one_digit(multiplicand, digit))
        rows_with_carry = [row for row in rows if any(row[4][:-1])]
        if multiplier_digits[0] >= 2 and multiplicand % 10 >= 2 and rows_with_carry:
            break
    product = multiplicand * multiplier
    assert sum(row[3] * 10 ** row[0] for row in rows) == product, "the rows do not add up to the product"
    trace_kind = choose_trace_kind(random_generator, variant, has_two_methods=True, has_dead_end=multiplier >= 12)
    object_name = pick(countable_object_names)
    question = pick([
        f"Multiply {multiplicand} by {multiplier}.", f"What is {multiplicand} times {multiplier}?",
        f"Find the product {multiplicand} × {multiplier}.",
        f"Compute {multiplicand} × {multiplier}." if trace_kind == "tool"
        else f"Compute {multiplicand} × {multiplier} by long multiplication.",
        f"Each crate holds {multiplicand} {object_name}. How many {object_name} are in {multiplier} crates?",
        f"A press stamps {multiplicand} {object_name} an hour. How many does it stamp in {multiplier} hours?"])

    def describe_row(row, description=None, row_product=None):
        position, digit = row[0], row[1]
        description = row[2] if description is None else description
        row_product = row[3] if row_product is None else row_product
        text = f"{multiplicand} × {digit}: {description}. So {multiplicand} × {digit} = {row_product}"
        if position:
            place = "one place" if position == 1 else f"{position} places"
            text += (f"; this digit stands in the {column_names[position]} place, so shift {place}: "
                     f"{row_product * 10 ** position}")
        return text + "."

    def describe_sum_of_rows(row_values):
        if len(row_values) == 1:
            return f"There is only one row, so the product is {row_values[0]}."
        return "Add the rows: " + " + ".join(str(row_value) for row_value in row_values) + f" = {sum(row_values)}."

    correct_row_values = [row[3] * 10 ** row[0] for row in rows]
    trace_items = []
    if trace_kind == "dead_end":
        first_multiples = ", ".join(str(multiplicand * count) for count in (1, 2, 3))
        trace_items.append(work_line(f"First idea: repeated addition, adding {multiplicand} again and again: "
                                     f"{first_multiples}, ..."))
        trace_items.append(abandon_line(f"that needs {multiplier - 1} additions, and each one is a chance to slip. "
                                        f"Multiply digit by digit instead."))
    if trace_kind == "slip":
        slip_type = pick(["wrong_units_fact", "dropped_carry"])
        if slip_type == "wrong_units_fact":
            slipped_row = rows[0]
            true_fact = (multiplicand % 10) * slipped_row[1]
            offset = pick([-2, -1, 1, 2])
            wrong_description, wrong_row_product, _ = multiply_by_one_digit(multiplicand, slipped_row[1],
                                                                            units_fact_offset=offset)
            check_text = (f"the last digit of the product comes from {multiplicand % 10} × {multiplier % 10} = "
                          f"{true_fact}, so it must be {true_fact % 10}; the result ends in "
                          f"{(true_fact + offset) % 10}.")
            fix_reason = f"{multiplicand % 10} × {slipped_row[1]} is {true_fact}, not {true_fact + offset}"
        else:
            slipped_row = pick(rows_with_carry)
            dropped_position = pick([position for position, carry in enumerate(slipped_row[4][:-1]) if carry])
            wrong_description, wrong_row_product, _ = multiply_by_one_digit(
                multiplicand, slipped_row[1], position_of_dropped_carry=dropped_position)
            fix_reason = (f"in the row for {slipped_row[1]} the carried {slipped_row[4][dropped_position]} "
                          f"was not added")
        wrong_row_values = [wrong_row_product * 10 ** row[0] if row is slipped_row else row[3] * 10 ** row[0]
                            for row in rows]
        wrong_product = sum(wrong_row_values)
        if slip_type == "dropped_carry":
            expected_remainder = (multiplicand % 9) * (multiplier % 9) % 9
            check_text = (f"a number and its digit sum leave the same remainder on division by 9. "
                          f"{describe_remainder_by_nine(multiplicand)} and {describe_remainder_by_nine(multiplier)}, "
                          f"so the product must leave {multiplicand % 9} × {multiplier % 9} = "
                          f"{(multiplicand % 9) * (multiplier % 9)}, that is {expected_remainder}. "
                          f"But {describe_remainder_by_nine(wrong_product)}, so {wrong_product} leaves "
                          f"{wrong_product % 9}.")
            assert expected_remainder != wrong_product % 9, "the nines check must expose a dropped carry"
        assert wrong_product != product
        for row in rows:
            if row is slipped_row:
                trace_items.append(mistake_line(describe_row(row, wrong_description, wrong_row_product)))
            else:
                trace_items.append(work_line(describe_row(row)))
        trace_items.append(work_line(describe_sum_of_rows(wrong_row_values)))
        trace_items.append(check_line(check_text))
        trace_items.append(fix_line(f"{fix_reason}. {describe_row(slipped_row)}"))
        trace_items.append(work_line(describe_sum_of_rows(correct_row_values)))
    elif trace_kind == "tool":
        trace_items.append(work_line(f"Split {multiplier} by its digits and multiply {multiplicand} by each part."))
        for row, row_value in zip(rows, correct_row_values):
            part_of_multiplier = row[1] * 10 ** row[0]
            tool_call = make_tool_call("calculator", f"{multiplicand} * {part_of_multiplier}", row_value)
            trace_items.append(tool_step(f"Row for {part_of_multiplier}.", tool_call))
        if len(rows) > 1:
            sum_expression = " + ".join(str(row_value) for row_value in correct_row_values)
            trace_items.append(tool_step("Add the rows the calculator returned.",
                                         make_tool_call("calculator", sum_expression, product)))
        trace_items.append(work_line(f"So {multiplicand} × {multiplier} = {product}."))
    else:
        if trace_kind == "two_methods":
            trace_items.append(method_header("Method 1: long multiplication, one row per digit of the multiplier."))
        trace_items.extend(work_line(describe_row(row)) for row in rows)
        trace_items.append(work_line(describe_sum_of_rows(correct_row_values)))
        if trace_kind == "two_methods":
            trace_items.append(method_header(f"Method 2: split {multiplicand} by place value and multiply each part "
                                             f"by {multiplier}."))
            place_value_parts = [int(character) * 10 ** position
                                 for position, character in enumerate(reversed(str(multiplicand))) if character != "0"]
            place_value_parts.reverse()
            partial_products = [part * multiplier for part in place_value_parts]
            trace_items.append(work_line(f"{multiplicand} = " + " + ".join(str(part) for part in place_value_parts) + "."))
            trace_items.extend(work_line(f"{part} × {multiplier} = {partial_product}.")
                               for part, partial_product in zip(place_value_parts, partial_products))
            if len(partial_products) > 1:
                trace_items.append(work_line(" + ".join(str(value) for value in partial_products) + f" = {product}."))
            trace_items.append(work_line(f"Both methods give {product}."))
    answer_sentence = pick(["The product is {}.", "{} × {} = {{}}.".format(multiplicand, multiplier),
                            "The answer is {}."]).format(product)
    return {"question": question, "trace_items": trace_items, "answer_sentence": answer_sentence,
            "final_answer": str(product), "trace_kind": trace_kind,
            "problem": {"multiplicand": multiplicand, "multiplier": multiplier}}


def format_linear_expression(coefficient: int, constant: int, variable: str) -> str:
    """'3x + 5', '-x - 7', 'x', '12': a linear expression the way it is written by hand."""
    if coefficient == 0:
        return str(constant)
    text = {1: "", -1: "-"}.get(coefficient, str(coefficient)) + variable
    if constant > 0:
        text += f" + {constant}"
    elif constant < 0:
        text += f" - {-constant}"
    return text


def format_bracket_side(multiplier: int, shift: int, constant: int, inner: str) -> str:
    """'3(x + 2) - 4': a multiplier times a shifted quantity, plus a constant; `inner` is the variable or a number."""
    shifted = f"{inner} + {shift}" if shift > 0 else f"{inner} - {-shift}"
    text = {1: "", -1: "-"}.get(multiplier, str(multiplier)) + f"({shifted})"
    if constant > 0:
        text += f" + {constant}"
    elif constant < 0:
        text += f" - {-constant}"
    return text


def generate_linear_equation(random_generator: random.Random, variant: str, difficulty: int) -> dict:
    """Solve a linear equation in one unknown with a whole-number solution, and check it by substitution."""
    pick = random_generator.choice
    trace_kind = choose_trace_kind(random_generator, variant, has_two_methods=True, has_dead_end=True)
    variable = pick("xytnkm")
    nonzero = [value for value in range(-9, 10) if value != 0]
    for _ in range(5000):
        solution = random_generator.randint(-15, 15)
        left_multiplier, right_multiplier = pick(nonzero), (pick(nonzero) if difficulty >= 2 else 0)
        left_shift, right_shift = (pick(nonzero), pick(nonzero)) if difficulty == 3 else (0, 0)
        left_constant = pick(nonzero + [0] * (difficulty == 3))
        net_coefficient = left_multiplier - right_multiplier
        # after expanding, the equation reads  left_multiplier*v + expanded_left = right_multiplier*v + expanded_right
        expanded_left = left_multiplier * left_shift + left_constant
        expanded_right = net_coefficient * solution + expanded_left
        right_constant = expanded_right - right_multiplier * right_shift
        is_usable = (net_coefficient not in (0, 1) and solution != 0 and expanded_left != 0
                     and abs(right_constant) <= 99 and (difficulty < 3 or abs(left_multiplier) != 1)
                     and (difficulty < 3 or abs(right_multiplier) != 1))
        if trace_kind == "slip":
            # the slip keeps the sign of the constant when it crosses the equals sign; the wrong solution has to
            # be a whole number so that the substitution check is as easy to read as the right one
            is_usable = is_usable and (expanded_right + expanded_left) % net_coefficient == 0
        if trace_kind == "dead_end":
            is_usable = is_usable and not 1 <= solution <= 3      # the guesses 1, 2, 3 must all miss
        if is_usable:
            break
    else:
        raise AssertionError("no usable linear equation found in 5000 tries")

    def left_side(inner: str) -> str:
        if difficulty == 3:
            return format_bracket_side(left_multiplier, left_shift, left_constant, inner)
        return format_linear_expression(left_multiplier, left_constant, inner)

    def right_side(inner: str) -> str:
        if difficulty == 3:
            return format_bracket_side(right_multiplier, right_shift, right_constant, inner)
        return format_linear_expression(right_multiplier, right_constant, inner)

    def side_values(value: int) -> tuple[int, int]:
        return (left_multiplier * (value + left_shift) + left_constant,
                right_multiplier * (value + right_shift) + right_constant)

    def python_side(multiplier: int, shift: int, constant: int, value_text: str) -> str:
        return f"{multiplier} * ({value_text} + ({shift})) + ({constant})"

    assert side_values(solution)[0] == side_values(solution)[1], "the constructed solution does not satisfy the equation"
    equation_text = f"{left_side(variable)} = {right_side(variable)}"
    question = pick([f"Solve for {variable}: {equation_text}.", f"Find {variable} if {equation_text}.",
                     f"Solve the equation {equation_text}.", f"What value of {variable} makes {equation_text} true?",
                     f"Determine {variable} given that {equation_text}.",
                     f"For which number {variable} does {equation_text} hold?"])
    collected_left = format_linear_expression(net_coefficient, expanded_left, variable)
    algebra_lines = []
    if difficulty == 3:
        algebra_lines.append(
            f"Expand the brackets: the left side is {format_linear_expression(left_multiplier, expanded_left, variable)}"
            f" and the right side is {format_linear_expression(right_multiplier, expanded_right, variable)}.")
    if right_multiplier != 0:
        move_verb = (f"Subtract {format_linear_expression(right_multiplier, 0, variable)} from"
                     if right_multiplier > 0 else f"Add {format_linear_expression(-right_multiplier, 0, variable)} to")
        algebra_lines.append(f"{move_verb} both sides: {collected_left} = {expanded_right}.")
    constant_verb = f"Subtract {expanded_left} from" if expanded_left > 0 else f"Add {-expanded_left} to"
    net_term = format_linear_expression(net_coefficient, 0, variable)
    right_after_move = expanded_right - expanded_left
    left_value, right_value = side_values(solution)
    check_of_solution = (f"Substitute {variable} = {solution}: the left side is {left_side(str(solution)) if difficulty == 3 else left_value}"
                         f"{' = ' + str(left_value) if difficulty == 3 else ''} and the right side is {right_value}. "
                         f"They agree.")
    trace_items = []
    if trace_kind == "dead_end":
        guesses = []
        for guess in (1, 2, 3):
            guess_left, guess_right = side_values(guess)
            guesses.append(f"{variable} = {guess} gives {guess_left} on the left and {guess_right} on the right")
        trace_items.append(work_line("First idea: guess and check. " + "; ".join(guesses) + "."))
        trace_items.append(abandon_line("none of the guesses balances the equation, and nothing says how far away "
                                        "the solution is or whether it is negative. Solve it with algebra instead."))
    if trace_kind == "two_methods":
        trace_items.append(method_header("Method 1: isolate the unknown."))
    trace_items.extend(work_line(algebra_line) for algebra_line in algebra_lines)
    if trace_kind == "slip":
        wrong_right = expanded_right + expanded_left
        wrong_solution = wrong_right // net_coefficient
        wrong_left_value, wrong_right_value = side_values(wrong_solution)
        trace_items.append(mistake_line(f"Move the constant {expanded_left} across: {net_term} = {expanded_right} "
                                        f"{'+' if expanded_left > 0 else '-'} {abs(expanded_left)} = {wrong_right}, "
                                        f"so {variable} = {wrong_solution}."))
        trace_items.append(check_line(f"substitute {variable} = {wrong_solution} into the original equation: the left "
                                      f"side is {wrong_left_value}, the right side is {wrong_right_value}. "
                                      f"They differ."))
        assert wrong_left_value != wrong_right_value
        trace_items.append(fix_line(f"a term changes sign when it crosses the equals sign. {constant_verb} both "
                                    f"sides: {net_term} = {right_after_move}."))
    else:
        trace_items.append(work_line(f"{constant_verb} both sides: {net_term} = {right_after_move}."))
    if trace_kind == "tool":
        tool_call = make_tool_call("calculator", f"{right_after_move} / {net_coefficient}", solution)
        trace_items.append(tool_step(f"Divide both sides by {net_coefficient}.", tool_call))
        trace_items.append(work_line(f"So {variable} = {solution}."))
        for side_name, multiplier, shift, constant, value in (
                ("left", left_multiplier, left_shift, left_constant, left_value),
                ("right", right_multiplier, right_shift, right_constant, right_value)):
            tool_call = make_tool_call("calculator", python_side(multiplier, shift, constant, str(solution)), value)
            trace_items.append(tool_step(f"Check the {side_name} side with {variable} = {solution}.", tool_call))
        trace_items.append(work_line(f"Both sides equal {left_value}, so the solution is confirmed."))
    else:
        trace_items.append(work_line(f"Divide both sides by {net_coefficient}: {variable} = {solution}."))
        if trace_kind != "single_method" or random_generator.random() < 0.6:
            trace_items.append(work_line(check_of_solution))
    if trace_kind == "two_methods":
        difference_at_zero = side_values(0)[0] - side_values(0)[1]
        difference_at_one = side_values(1)[0] - side_values(1)[1]
        trace_items.append(method_header("Method 2: follow the difference between the two sides."))
        trace_items.append(work_line(
            f"Let d be the left side minus the right side. At {variable} = 0, d = {difference_at_zero}; "
            f"at {variable} = 1, d = {difference_at_one}."))
        trace_items.append(work_line(
            f"Both sides are linear, so d changes by {difference_at_one - difference_at_zero} for each step of 1 "
            f"in {variable}. It reaches 0 at {variable} = {-difference_at_zero} / {net_coefficient} = {solution}."))
        trace_items.append(work_line(f"Both methods give {variable} = {solution}."))
    answer_sentence = pick(["The solution is {} = {}.", "So {} = {}.", "Therefore {} = {}."]).format(variable, solution)
    return {"question": question, "trace_items": trace_items, "answer_sentence": answer_sentence,
            "final_answer": f"{variable} = {solution}", "trace_kind": trace_kind,
            "problem": {"left_python": python_side(left_multiplier, left_shift, left_constant, "unknown"),
                        "right_python": python_side(right_multiplier, right_shift, right_constant, "unknown"),
                        "variable": variable}}
