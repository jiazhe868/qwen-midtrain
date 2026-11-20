"""The regression suite: a checkpoint judged as a base model, before any fine-tune.

Why this file exists. Continued pretraining on a new mixture can make a model better at the new text and worse at
what it knew. The fine-tuning probe (sft_probe.py) does not see that: it tests math and code after a fine-tune on
math and code. This suite asks the checkpoint itself, with the prompts a base model understands, and reports three
groups of numbers:
    knowledge      five multiple-choice sets, scored by the log-likelihood of the answer letter after a few-shot
                   prompt, and trivia questions answered greedily
    few-shot work  math and code solved from few-shot prompts, 16 samples per question: pass@1 and pass@16 of the
                   checkpoint with no fine-tune in between (README §13 P11 compares their spread with the probe's)
    use of context how much the few-shot examples help (accuracy with shots minus accuracy without), and how much
                   the loss on the last 2,048 tokens of a long document falls when 14,336 tokens of context are
                   given compared with 2,048. These are where long-context training should show (README §9).

Multiple choice is scored from one forward pass per question: the prompt ends in `Answer:` and the logits of the
candidate letters at that position are compared. No text is generated, so there is no answer extraction to get
wrong, and 5,600 questions take minutes.

Commands (after `source scripts/env.sh`):
    CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m midtrain.base_regression run \
        --checkpoint $MIDTRAIN_DATA/runs/<run>/checkpoint_final --run-name <run>     # writes results/regression/<run>.json
    CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m midtrain.base_regression run --checkpoint ... --run-name ... --quick-test
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

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
        hidden_states = model.model(input_ids=torch.from_numpy(token_ids).to(input_device)).last_hidden_state
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
            hidden_states = model.model(input_ids=token_ids).last_hidden_state
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
    # the guard of the mixture search (mixing_law.py): the mean of the six knowledge scores
    scores["knowledge_mean"] = round(float(np.mean([scores[f"multiple_choice/{name}"] for name in multiple_choice_task_names]
                                                   + [scores["trivia_exact_match"]])), 2)
    return {"run_name": run_name, "minutes": round((time.time() - started) / 60, 1), "scores": scores,
            "item_correct": item_correct,
            "few_shot_generation": {task_name: {key: task_result[key] for key in ("item_ids", "correct_counts", "sample_count")}
                                    for task_name, task_result in generation["tasks"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tokens-per-forward", type=int, default=16384)
    parser.add_argument("--quick-test", action="store_true", help="24 questions per set, 8 long documents: minutes")
    arguments = parser.parse_args()
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
