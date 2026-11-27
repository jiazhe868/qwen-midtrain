"""The regression suite: a checkpoint judged as a base model, before any fine-tune.

Why this file exists. Continued pretraining on a new mixture can make a model better at the new text and worse at
what it knew. The fine-tuning probe (sft_probe.py) does not see that: it tests math and code after a fine-tune on
math and code. This suite asks the checkpoint itself, with the prompts a base model understands, and reports three
groups of numbers:
    knowledge      five multiple-choice sets scored two ways, and trivia questions answered greedily. Lettered: the
                   log-likelihood of the answer letter, with five examples and without. Cloze: the log-likelihood of
                   each answer's own text after the question, per character, with no letters at all
    few-shot work  math and code solved from few-shot prompts, 16 samples per question: pass@1 and pass@16 of the
                   checkpoint with no fine-tune in between (README Appendix H, P11, compares their spread with the probe's)
    use of context how much the few-shot examples help (accuracy with shots minus accuracy without), and how much
                   the loss on the last 2,048 tokens of a long document falls when 14,336 tokens of context are
                   given compared with 2,048. These are where long-context training should show (README Appendix E).

Lettered multiple choice is scored from one forward pass per question: the prompt ends in `Answer:` and the logits
of the candidate letters at that position are compared. No text is generated, so there is no answer extraction to
get wrong, and 5,600 questions take minutes. Small models have letter preferences, though: the 0.8B base picks "B"
more often after five examples (33 of 80 ARC questions against 27 without), and after the first design run its
zero-shot lettered HellaSwag fell 6.4 points while the five-example score did not move (11-20). Cloze scoring has no
letters to prefer, which is why it is the knowledge guard of the mixture search (README Appendix C).

Commands (after `source scripts/env.sh`):
    CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m midtrain.base_regression run \
        --checkpoint $MIDTRAIN_DATA/runs/<run>/checkpoint_final --run-name <run>     # writes results/regression/<run>.json
    CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m midtrain.base_regression run --checkpoint ... --run-name ... --quick-test
    CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m midtrain.base_regression add-cloze --checkpoint ... --run-name ...
        # adds the cloze scores to a result written before they existed (about 2 minutes for the 0.8B model)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time

import numpy as np
import torch

from midtrain import eval_tasks, sft_probe
from midtrain.data_sources import data_root, open_token_source, repository_root, tokenizer_path
from midtrain.mixture import end_of_text_token_id
from midtrain.train_continue import chunked_language_model_loss, load_working_model, place_model_on_devices

multiple_choice_task_names = ["mmlu_pro", "arc_challenge", "hellaswag", "commonsenseqa", "cmmlu"]
# Few-shot generation by the base checkpoint. 16 samples at temperature 1 give pass@1 and pass@16 at a quarter of
# the probe's sampling cost; the items are the probe's (same seed, same counts), so the two can be compared item by
# item. Trivia is one greedy answer per question: its answers are names, and sampling adds only noise there.
few_shot_generation_settings = {
    "sample_count": 16, "temperature": 1.0, "top_p": 1.0, "item_seed": 20261120, "pass_at_k": [1, 16],
    "tasks": {
        "gsm8k": {"item_count": 200, "maximum_new_tokens": 320},
        "math500": {"item_count": 250, "maximum_new_tokens": 384},
        "humaneval": {"item_count": 164, "maximum_new_tokens": 384},
        "mbpp": {"item_count": 200, "maximum_new_tokens": 384},
        "triviaqa": {"item_count": 1000, "maximum_new_tokens": 24, "sample_count": 1, "temperature": 0.0},
    },
}
# The final evaluation of the dose experiment (README §2.2): every test item once, greedy, with the few-shot prompts
# of the regression suite. Sampled pass@1 on 200-250 items ranks runs well enough; a claim about the size of a gain
# wants the whole test set and the decoding that published base-model results use. The answer limits are longer than
# in the sampled suite, because a model trained on worked solutions writes longer ones; the cut share is recorded.
final_generation_settings = {
    "sample_count": 1, "temperature": 0.0, "top_p": 1.0, "item_seed": 20261120, "pass_at_k": [1],
    "tasks": {
        "gsm8k": {"item_count": 1319, "maximum_new_tokens": 512},
        "math500": {"item_count": 500, "maximum_new_tokens": 768},
        "humaneval": {"item_count": 164, "maximum_new_tokens": 512},
        "mbpp": {"item_count": 257, "maximum_new_tokens": 512},
        # word problems from other sources, answered from the GSM8K shots (eval_tasks.py; added 11-23)
        "svamp": {"item_count": 1000, "maximum_new_tokens": 512},
        "asdiv_a": {"item_count": 1218, "maximum_new_tokens": 512},
    },
}
long_document_token_count = 16384          # the long training sequence length (configs/designs/screen18.yaml)
target_token_count = 2048                  # the tokens whose loss is compared, at the end of the long window
near_context_token_count = 2048            # context given in the "near" reading, directly before the targets
long_document_count = 48                   # 48 x 2,048 = 98,304 scored tokens: the mean loss is stable to about 0.3%


# ---------------------------------------------------------------------------------------------------------------
# knowledge: multiple choice from the logits of the answer letters
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def logits_after_each_prompt(model, prompt_token_ids: list[list[int]], candidate_token_ids: list[list[int]],
                             tokens_per_forward: int, padding_id: int) -> list[np.ndarray]:
    """For every prompt, the logits of its candidate tokens at the position after the prompt's last token.

    Prompts are padded on the right. The model reads left to right, so the state at a prompt's last token does
    not depend on the padding after it, and no attention mask is needed."""
    input_device = model.get_input_embeddings().weight.device
    output_layer = model.get_output_embeddings()
    order = sorted(range(len(prompt_token_ids)), key=lambda index: -len(prompt_token_ids[index]))
    candidate_logits: list[np.ndarray | None] = [None] * len(prompt_token_ids)
    batch_start = 0
    while batch_start < len(order):
        longest = len(prompt_token_ids[order[batch_start]])
        batch = order[batch_start:batch_start + max(1, tokens_per_forward // longest)]
        token_ids = np.full((len(batch), longest), padding_id, dtype=np.int64)
        for row, index in enumerate(batch):
            token_ids[row, :len(prompt_token_ids[index])] = prompt_token_ids[index]
        hidden_states = model.model(input_ids=torch.from_numpy(token_ids).to(input_device), use_cache=False).last_hidden_state
        last_positions = torch.tensor([len(prompt_token_ids[index]) - 1 for index in batch], device=hidden_states.device)
        last_states = hidden_states[torch.arange(len(batch), device=hidden_states.device), last_positions]
        logits = output_layer(last_states.to(output_layer.weight.device)).float().cpu().numpy()
        for row, index in enumerate(batch):
            candidate_logits[index] = logits[row, candidate_token_ids[index]]
        batch_start += len(batch)
    return candidate_logits


def score_multiple_choice_task(model, tokenizer, task_name: str, with_shots: bool, tokens_per_forward: int,
                               item_count_limit: int | None) -> list[int]:
    """1 or 0 per item of the task, in file order: whether the most likely answer letter is the right one."""
    items = eval_tasks.load_evaluation_items(task_name)[:item_count_limit]
    few_shot_items = eval_tasks.load_few_shot_items(task_name) if with_shots else []
    prompts = [eval_tasks.render_base_prompt(item, few_shot_items) for item in items]
    prompt_token_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    token_id_of_continuation: dict[str, int] = {}
    candidate_token_ids = []
    for item in items:
        candidates = []
        for continuation in eval_tasks.multiple_choice_continuations(item):
            if continuation not in token_id_of_continuation:
                continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
                assert len(continuation_ids) == 1, (
                    f"the answer {continuation!r} is {len(continuation_ids)} tokens; one forward pass scores one token")
                token_id_of_continuation[continuation] = continuation_ids[0]
            candidates.append(token_id_of_continuation[continuation])
        candidate_token_ids.append(candidates)
    candidate_logits = logits_after_each_prompt(model, prompt_token_ids, candidate_token_ids, tokens_per_forward,
                                                end_of_text_token_id())
    return [int(int(np.argmax(logits)) == item["answer_index"]) for logits, item in zip(candidate_logits, items)]


def cloze_context_and_choices(item: dict) -> tuple[str, list[str]]:
    """The text before the answer and each answer's own text, for cloze scoring.

    HellaSwag is a passage and its possible endings, so the passage is the context. Every other set is a question:
    `Question: ... Answer:` and the answer text after a space, or the Chinese labels without one."""
    if item["task"] == "hellaswag":
        return item["meta"]["passage"], [" " + choice for choice in item["choices"]]
    if item["meta"].get("language") == "zh":
        return f"题目：{item['question']}\n答案：", list(item["choices"])
    return f"Question: {item['question']}\nAnswer:", [" " + choice for choice in item["choices"]]


@torch.no_grad()
def continuation_log_likelihoods(model, sequences: list[tuple[list[int], list[int]]], tokens_per_forward: int,
                                 padding_id: int) -> list[float]:
    """The summed log-probability of each continuation's tokens after its context: one value per (context, continuation).

    Only the positions that predict a continuation token go through the output layer, because its 248,320 outputs
    per position are the expensive part."""
    input_device = model.get_input_embeddings().weight.device
    output_layer = model.get_output_embeddings()
    order = sorted(range(len(sequences)), key=lambda index: -(len(sequences[index][0]) + len(sequences[index][1])))
    log_likelihoods: list[float] = [0.0] * len(sequences)
    batch_start = 0
    while batch_start < len(order):
        longest = len(sequences[order[batch_start]][0]) + len(sequences[order[batch_start]][1])
        batch = order[batch_start:batch_start + max(1, tokens_per_forward // longest)]
        token_ids = np.full((len(batch), longest), padding_id, dtype=np.int64)
        rows, positions, targets = [], [], []
        for row, index in enumerate(batch):
            context_ids, continuation_ids = sequences[index]
            token_ids[row, :len(context_ids) + len(continuation_ids)] = context_ids + continuation_ids
            for offset, target in enumerate(continuation_ids):
                rows.append(row)
                positions.append(len(context_ids) + offset - 1)       # the position that predicts this token
                targets.append(target)
        hidden_states = model.model(input_ids=torch.from_numpy(token_ids).to(input_device), use_cache=False).last_hidden_state
        states = hidden_states[torch.tensor(rows, device=hidden_states.device), torch.tensor(positions, device=hidden_states.device)]
        target_ids = torch.tensor(targets, device=output_layer.weight.device)
        picked_chunks = []
        # 1,024 positions at a time: the log-softmax over 248,320 words for a whole batch took 5.7 GB (11-20)
        for chunk_start in range(0, len(targets), 1024):
            chunk = slice(chunk_start, chunk_start + 1024)
            log_probabilities = torch.log_softmax(output_layer(states[chunk].to(output_layer.weight.device)).float(), dim=-1)
            picked_chunks.append(log_probabilities.gather(1, target_ids[chunk].unsqueeze(1)).squeeze(1))
        picked = torch.cat(picked_chunks).cpu().numpy()
        cursor = 0
        for index in batch:
            continuation_length = len(sequences[index][1])
            log_likelihoods[index] = float(picked[cursor:cursor + continuation_length].sum())
            cursor += continuation_length
        batch_start += len(batch)
    return log_likelihoods


def score_cloze_task(model, tokenizer, task_name: str, tokens_per_forward: int, item_count_limit: int | None) -> list[int]:
    """1 or 0 per item: whether the answer with the highest log-likelihood per character is the right one."""
    items = eval_tasks.load_evaluation_items(task_name)[:item_count_limit]
    sequences, owners, character_counts = [], [], []
    for item_index, item in enumerate(items):
        context, choices = cloze_context_and_choices(item)
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        for choice_index, choice in enumerate(choices):
            sequences.append((context_ids, tokenizer(choice, add_special_tokens=False)["input_ids"]))
            owners.append((item_index, choice_index))
            character_counts.append(max(1, len(choice.strip())))
    log_likelihoods = continuation_log_likelihoods(model, sequences, tokens_per_forward, end_of_text_token_id())
    best: dict[int, tuple[float, int]] = {}
    for (item_index, choice_index), log_likelihood, characters in zip(owners, log_likelihoods, character_counts):
        per_character = log_likelihood / characters
        if item_index not in best or per_character > best[item_index][0]:
            best[item_index] = (per_character, choice_index)
    return [int(best[item_index][1] == item["answer_index"]) for item_index, item in enumerate(items)]


# ---------------------------------------------------------------------------------------------------------------
# use of context: does distant text lower the loss
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def distant_context_gain(model, tokens_per_forward: int, document_count: int) -> dict:
    """Loss on the last `target_token_count` tokens of long held-out papers, read twice: with everything before
    them in the 16,384-token window, and with only the `near_context_token_count` tokens before them. The
    difference is what the distant 12,288 tokens are worth to this checkpoint, in bits per token."""
    token_source = open_token_source("papers", "validation")
    offsets = token_source["document_offsets"]
    long_documents = [position for position in range(len(offsets) - 1)
                      if offsets[position + 1] - offsets[position] >= long_document_token_count][:document_count]
    assert len(long_documents) >= 8, f"only {len(long_documents)} held-out papers are {long_document_token_count:,} tokens long"
    input_device = model.get_input_embeddings().weight.device
    output_layer = model.get_output_embeddings()

    def loss_on_targets(windows: np.ndarray) -> float:
        label_mask = np.zeros_like(windows)
        label_mask[:, -target_token_count:] = 1
        loss_sum, token_count = 0.0, 0
        rows_per_forward = max(1, tokens_per_forward // windows.shape[1])
        for row_start in range(0, len(windows), rows_per_forward):
            token_ids = torch.from_numpy(windows[row_start:row_start + rows_per_forward]).to(input_device)
            hidden_states = model.model(input_ids=token_ids, use_cache=False).last_hidden_state
            chunk_loss, chunk_tokens = chunked_language_model_loss(
                hidden_states, token_ids, output_layer, 1.0, with_gradient=False,
                label_mask=torch.from_numpy(label_mask[row_start:row_start + rows_per_forward]).to(input_device))
            loss_sum += chunk_loss
            token_count += chunk_tokens
        return loss_sum / token_count / math.log(2)

    full_windows = np.stack([np.asarray(token_source["token_ids"][offsets[position]:offsets[position] + long_document_token_count],
                                        dtype=np.int64) for position in long_documents])
    near_windows = full_windows[:, -(near_context_token_count + target_token_count):]
    full_loss, near_loss = loss_on_targets(full_windows), loss_on_targets(near_windows)
    return {"context/loss_with_full_window_bits_per_token": round(full_loss, 4),
            "context/loss_with_near_context_bits_per_token": round(near_loss, 4),
            "context/distant_gain_bits_per_token": round(near_loss - full_loss, 4)}


# ---------------------------------------------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------------------------------------------
def run_regression_suite(checkpoint_path: str, run_name: str, tokens_per_forward: int, is_quick_test: bool) -> dict:
    from transformers import AutoTokenizer

    item_count_limit = 24 if is_quick_test else None
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    device_names = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    assert device_names, "no GPU visible: set CUDA_VISIBLE_DEVICES"
    model = load_working_model(checkpoint_path)
    place_model_on_devices(model, device_names)
    model.eval()

    scores: dict[str, float] = {}
    item_correct: dict[str, list[int]] = {}
    for task_name in multiple_choice_task_names:
        for with_shots in (True, False):
            correct = score_multiple_choice_task(model, tokenizer, task_name, with_shots, tokens_per_forward, item_count_limit)
            group = "multiple_choice" if with_shots else "multiple_choice_zero_shot"
            scores[f"{group}/{task_name}"] = round(100 * float(np.mean(correct)), 2)
            item_correct[f"{group}/{task_name}"] = correct
        scores[f"shot_gain/{task_name}"] = round(scores[f"multiple_choice/{task_name}"] - scores[f"multiple_choice_zero_shot/{task_name}"], 2)
        print(f"{run_name} {task_name}: {scores[f'multiple_choice/{task_name}']:.1f} with shots, "
              f"{scores[f'multiple_choice_zero_shot/{task_name}']:.1f} without ({(time.time() - started) / 60:.1f} min)", flush=True)
    scores["shot_gain_mean"] = round(float(np.mean([scores[f"shot_gain/{name}"] for name in multiple_choice_task_names])), 2)
    for task_name in multiple_choice_task_names:
        correct = score_cloze_task(model, tokenizer, task_name, tokens_per_forward, item_count_limit)
        scores[f"cloze/{task_name}"] = round(100 * float(np.mean(correct)), 2)
        item_correct[f"cloze/{task_name}"] = correct
    print(f"{run_name} cloze: " + ", ".join(f"{name} {scores[f'cloze/{name}']:.1f}" for name in multiple_choice_task_names)
          + f" ({(time.time() - started) / 60:.1f} min)", flush=True)
    scores.update(distant_context_gain(model, tokens_per_forward, 8 if is_quick_test else long_document_count))
    print(f"{run_name} distant context: {scores['context/distant_gain_bits_per_token']:.4f} bits per token gained", flush=True)

    # the scoring model has to leave the GPU before the server starts: the server takes 80% of its memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    generation = sft_probe.evaluate_pass_at_k(checkpoint_path, run_name, sft_probe.port_for_this_process(), prompt_style="base",
                                              item_count_limit=12 if is_quick_test else None,
                                              evaluation_settings=json.loads(json.dumps(few_shot_generation_settings)))
    for task_name, task_result in generation["tasks"].items():
        if task_name == "triviaqa":
            scores["trivia_exact_match"] = task_result["pass@1"]["value"]
        else:
            for k in (1, 16):
                if f"pass@{k}" in task_result:
                    scores[f"few_shot/{task_name}/pass@{k}"] = task_result[f"pass@{k}"]["value"]
    few_shot_pass_at_1 = [value for name, value in scores.items() if name.startswith("few_shot/") and name.endswith("pass@1")]
    scores["few_shot_mean/pass@1"] = round(float(np.mean(few_shot_pass_at_1)), 2)
    scores["knowledge_mean"] = round(float(np.mean([scores[f"multiple_choice/{name}"] for name in multiple_choice_task_names]
                                                   + [scores["trivia_exact_match"]])), 2)
    # the guard of the mixture search (mixing_law.py): the five sets scored by cloze, and trivia
    scores["knowledge_cloze_mean"] = round(float(np.mean([scores[f"cloze/{name}"] for name in multiple_choice_task_names]
                                                         + [scores["trivia_exact_match"]])), 2)
    return {"run_name": run_name, "minutes": round((time.time() - started) / 60, 1), "scores": scores,
            "item_correct": item_correct,
            "few_shot_generation": {task_name: {key: task_result[key] for key in ("item_ids", "correct_counts", "sample_count")}
                                    for task_name, task_result in generation["tasks"].items()}}


def add_cloze_scores(checkpoint_path: str, run_name: str, tokens_per_forward: int) -> dict:
    """Score the five sets by cloze and merge the scores into results/regression/<run>.json, which a version of this
    file without cloze scoring wrote; the guard `knowledge_cloze_mean` is computed with the stored trivia score."""
    from transformers import AutoTokenizer

    result_path = repository_root / "results" / "regression" / f"{run_name}.json"
    results = json.loads(result_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    model = load_working_model(checkpoint_path)
    place_model_on_devices(model, [f"cuda:{index}" for index in range(torch.cuda.device_count())])
    model.eval()
    for task_name in multiple_choice_task_names:
        correct = score_cloze_task(model, tokenizer, task_name, tokens_per_forward, None)
        results["scores"][f"cloze/{task_name}"] = round(100 * float(np.mean(correct)), 2)
        results["item_correct"][f"cloze/{task_name}"] = correct
    scores = results["scores"]
    scores["knowledge_cloze_mean"] = round(float(np.mean([scores[f"cloze/{name}"] for name in multiple_choice_task_names]
                                                         + [scores["trivia_exact_match"]])), 2)
    result_path.write_text(json.dumps(results) + "\n")
    return {name: value for name, value in scores.items() if name.startswith(("cloze/", "knowledge_cloze"))}


def run_final_evaluation(checkpoint_path: str, run_name: str) -> dict:
    """Greedy few-shot accuracy on the full test sets; writes results/final_eval/<run>.json.

    Tasks that an existing result file already holds are kept and not run again. A set added to
    final_generation_settings later (SVAMP and ASDiv-A on 11-23) is then filled in for the earlier runs by running
    this command once more, and the scores already measured stay exactly as they were.
    """
    results_directory = repository_root / "results" / "final_eval"
    results_directory.mkdir(parents=True, exist_ok=True)
    result_path = results_directory / f"{run_name}.json"
    results = (json.loads(result_path.read_text()) if result_path.exists()
               else {"run_name": run_name, "decoding": "greedy, one answer per item", "tasks": {}})
    settings = json.loads(json.dumps(final_generation_settings))
    settings["tasks"] = {task_name: task_settings for task_name, task_settings in settings["tasks"].items()
                         if task_name not in results["tasks"]}
    if settings["tasks"]:
        generation = sft_probe.evaluate_pass_at_k(checkpoint_path, f"{run_name}_final", sft_probe.port_for_this_process(),
                                                  prompt_style="base", evaluation_settings=settings)
        results["tasks"].update(generation["tasks"])
        result_path.write_text(json.dumps(results) + "\n")
    return {task_name: {"accuracy": task["pass@1"]["value"], "items": len(task["item_ids"]),
                        "cut": task.get("share_cut_at_token_limit")} for task_name, task in results["tasks"].items()}


def measure_teacher_text_loss(checkpoint_path: str, run_name: str) -> dict:
    """Held-out loss of a checkpoint on the teacher-written solutions; writes results/teacher_text_loss/<run>.json.

    Training leaves these sources out of its validation (train_continue.available_validation_sources). Here they are
    the point: the loss a base model has on the teacher's held-out solutions says how much of that text it can
    already predict, and so how much there is left for it to learn (README §5.5: at 9B the solutions taught mostly
    what the model already knew). For a trained run the number is optimistic, because a held-out solution can be
    a second solution of a problem whose other solutions were trained on."""
    from midtrain.train_continue import validation_loss_by_source

    teacher_sources = [name for name in ("synthetic_solutions", "synthetic_solutions_numina", "synthetic_solutions_hard")
                       if (data_root() / "tokens" / name / "validation.document_offsets.npy").exists()]
    device_names = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    assert device_names, "no GPU visible: set CUDA_VISIBLE_DEVICES"
    model = load_working_model(checkpoint_path)
    place_model_on_devices(model, device_names)
    with torch.no_grad():
        losses = validation_loss_by_source(model, teacher_sources, sequences_per_forward=2)
    results = {"run_name": run_name, "loss": losses}
    results_directory = repository_root / "results" / "teacher_text_loss"
    results_directory.mkdir(parents=True, exist_ok=True)
    (results_directory / f"{run_name}.json").write_text(json.dumps(results, indent=1) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run", "add-cloze", "final", "teacher-text-loss"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tokens-per-forward", type=int, default=16384)
    parser.add_argument("--quick-test", action="store_true", help="24 questions per set, 8 long documents: minutes")
    arguments = parser.parse_args()
    if arguments.command == "add-cloze":
        print(json.dumps(add_cloze_scores(arguments.checkpoint, arguments.run_name, arguments.tokens_per_forward), indent=1))
        return
    if arguments.command == "final":
        (data_root() / "runs" / arguments.run_name).mkdir(parents=True, exist_ok=True)
        print(json.dumps(run_final_evaluation(arguments.checkpoint, arguments.run_name), indent=1))
        return
    if arguments.command == "teacher-text-loss":
        print(json.dumps(measure_teacher_text_loss(arguments.checkpoint, arguments.run_name), indent=1))
        return
    (data_root() / "runs" / arguments.run_name).mkdir(parents=True, exist_ok=True)
    results = run_regression_suite(arguments.checkpoint, arguments.run_name, arguments.tokens_per_forward, arguments.quick_test)
    if arguments.quick_test:
        print(json.dumps(results["scores"], indent=1))
        return
    results_directory = repository_root / "results" / "regression"
    results_directory.mkdir(parents=True, exist_ok=True)
    (results_directory / f"{arguments.run_name}.json").write_text(json.dumps(results) + "\n")
    print(json.dumps(results["scores"], indent=1))


if __name__ == "__main__":
    main()
