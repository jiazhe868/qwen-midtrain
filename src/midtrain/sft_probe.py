"""The fixed short fine-tune ("SFT probe") and the pass@k test that follows it.

Why this file exists. A base checkpoint is usually judged by its loss. Loss is a weak guide to what matters after
post-training: in a study of 50 one-billion-parameter models, perplexity ordered pairs of models correctly after
fine-tuning 33-38% of the time (arXiv 2504.12491). So every candidate checkpoint here gets the same small
fine-tune, and is then asked the same questions 64 times each. pass@1 says how often one answer is right; pass@64
says how many questions the model can solve at all, which is the room a later RL stage has to work with.

What is fixed (configs/probe.yaml): the conversations and their order, the learning rate and steps, the test items,
the sampling. What varies: the checkpoint.

The fine-tuning set is built from human-written solutions where they exist (GSM8K and MATH train splits, MBPP
reference programs) and from instruction data that open models wrote (subsets of smoltalk: StarCoder2 wrote the code
instructions, Llama 3.1 the everyday conversations; the dataset card names the pipeline of the others). Its math and
code questions are worded by the same function that words the test questions (eval_tasks.render_chat_messages), so
the probe teaches the format the test uses and the test measures what the checkpoint brought along.

Conversations are rendered in plain ChatML and sent to the server as raw text, so no chat-template file is needed:
the base checkpoints ship without one.

Commands (after `source scripts/env.sh`):
    env -u HF_HUB_OFFLINE $MIDTRAIN_PYTHON -m midtrain.sft_probe build-data
    CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m midtrain.sft_probe run --checkpoint $MIDTRAIN_DATA/runs/<run>/checkpoint_final \
        --run-name <run>            # fine-tune, serve, sample, grade; writes results/probe/<run>.json
    CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m midtrain.sft_probe evaluate --checkpoint <folder> --run-name <name>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from midtrain import eval_tasks
from midtrain.data_sources import data_root, repository_root, tokenizer_path

turn_start, turn_end = "<|im_start|>", "<|im_end|>"
# Two variants of the probe (README §3.3). "in_domain", the first one, fine-tunes on the training solutions of the
# tested tasks (GSM8K, MATH, MBPP) and so teaches the test: across the 18 runs of the 0.8B design it made the mixtures
# indistinguishable (11-21, README §1.2). "generic" fine-tunes on instruction data alone, so what the model shows
# afterwards on math and code comes from the checkpoint, not from the fine-tune. Each variant has its own config,
# data folder and results folder; the variant is chosen once per process with --variant.
probe_variants = {
    "in_domain": {"configuration": "probe.yaml", "data_folder": "probe", "results_folder": "probe"},
    "generic": {"configuration": "probe_generic.yaml", "data_folder": "probe_generic", "results_folder": "probe_generic"},
    # the generic probe's fine-tune with longer answer limits, for models trained on long thinking traces (README §6.4)
    "generic_long": {"configuration": "probe_generic_long.yaml", "data_folder": "probe_generic",
                     "results_folder": "probe_generic_long"},
}
active_probe_variant = {"name": "in_domain"}


def probe_variant_settings() -> dict:
    return probe_variants[active_probe_variant["name"]]


def load_probe_configuration() -> dict:
    return yaml.safe_load((repository_root / "configs" / probe_variant_settings()["configuration"]).read_text())


def probe_directory() -> Path:
    return data_root() / probe_variant_settings()["data_folder"]


# ---------------------------------------------------------------------------------------------------------------
# ChatML: the one conversation format of the probe, at training and at test time
# ---------------------------------------------------------------------------------------------------------------
def render_chatml_prompt(messages: list[dict]) -> str:
    """The conversation so far as text, ending where the assistant's next turn begins."""
    rendered_turns = "".join(f"{turn_start}{message['role']}\n{message['content']}{turn_end}\n" for message in messages)
    return rendered_turns + f"{turn_start}assistant\n"


def tokenize_conversation_with_loss_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids of a whole conversation, and a mask that is 1 on the tokens the assistant wrote (its text and the
    end-of-turn token after it). Only those tokens carry loss: the probe teaches answering, not asking."""
    token_ids: list[int] = []
    loss_mask: list[int] = []
    for message in messages:
        header_ids = tokenizer(f"{turn_start}{message['role']}\n", add_special_tokens=False)["input_ids"]
        content_ids = tokenizer(message["content"] + turn_end, add_special_tokens=False)["input_ids"]
        newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
        is_assistant = int(message["role"] == "assistant")
        token_ids += header_ids + content_ids + newline_ids
        loss_mask += [0] * len(header_ids) + [is_assistant] * len(content_ids) + [0] * len(newline_ids)
    return token_ids, loss_mask


# ---------------------------------------------------------------------------------------------------------------
# build-data: the fixed fine-tuning set
# ---------------------------------------------------------------------------------------------------------------
def gsm8k_solution_as_answer(solution_text: str) -> str:
    """GSM8K writes `<<3*4=12>>` calculator notes and ends with `#### 12`: drop the notes, box the answer."""
    worked_steps, _, final_answer = solution_text.partition("####")
    worked_steps = re.sub(r"<<[^>]*>>", "", worked_steps).strip()
    return f"{worked_steps}\nThe final answer is \\boxed{{{final_answer.strip()}}}."


def collect_probe_conversations(configuration: dict) -> list[dict]:
    """Every conversation of the probe as {"source", "messages"}; the caller shuffles and trims."""
    from datasets import load_dataset

    wanted = configuration["data"]["sources"]
    seed = configuration["data"]["seed"]
    conversations: list[dict] = []

    def take(rows, source_name: str, build_messages) -> None:
        kept = 0
        for row in rows:
            messages = build_messages(row)
            if messages:
                conversations.append({"source": source_name, "messages": messages})
                kept += 1
            if kept >= wanted[source_name]:
                break
        print(f"  {source_name}: {kept} conversations", flush=True)

    def question_then_answer(item: dict, answer_text: str) -> list[dict]:
        return eval_tasks.render_chat_messages(item) + [{"role": "assistant", "content": answer_text}]

    if wanted.get("gsm8k_train"):
        gsm8k_train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
        take(gsm8k_train, "gsm8k_train", lambda row: question_then_answer(
            {"task": "gsm8k", "kind": "numeric", "question": row["question"]}, gsm8k_solution_as_answer(row["answer"])))
    if wanted.get("math_train"):
        take_math_training_solutions(take, question_then_answer, seed)
    if wanted.get("mbpp_train"):
        take_mbpp_training_programs(take, question_then_answer)

    smoltalk_subsets = {"code_instructions": "self-oss-instruct", "instruction_following": "smol-constraints",
                        "summarization": "smol-summarize", "rewriting": "smol-rewrite",
                        "everyday_conversation": "everyday-conversations"}
    for source_name, subset_name in smoltalk_subsets.items():
        if not wanted.get(source_name):
            continue
        rows = load_dataset("HuggingFaceTB/smoltalk", subset_name, split="train").shuffle(seed=seed)
        take(rows, source_name, lambda row: [{"role": message["role"], "content": message["content"]}
                                             for message in row["messages"]]
             if row["messages"] and row["messages"][-1]["role"] == "assistant" else None)
    return conversations


def take_math_training_solutions(take, question_then_answer, seed: int) -> None:
    """MATH training problems with their written solutions (only those that end in a boxed answer)."""
    from datasets import load_dataset

    math_rows = []
    for subject in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                    "prealgebra", "precalculus"):
        math_rows.extend(load_dataset("EleutherAI/hendrycks_math", subject, split="train").to_list())
    np.random.default_rng(seed).shuffle(math_rows)
    take(math_rows, "math_train", lambda row: question_then_answer(
        {"task": "math500", "kind": "numeric", "question": row["problem"]}, row["solution"]) if "\\boxed" in row["solution"] else None)


def take_mbpp_training_programs(take, question_then_answer) -> None:
    """MBPP training and validation tasks with their reference programs."""
    from datasets import load_dataset

    # MBPP task ids: 11-510 are the test split, 511-600 validation, 601-974 train. The probe trains on the last two,
    # so none of the tested tasks is seen; the assert keeps that true if the hub files ever change.
    mbpp_rows = (load_dataset("google-research-datasets/mbpp", "full", split="train").to_list()
                 + load_dataset("google-research-datasets/mbpp", "full", split="validation").to_list())
    assert all(row["task_id"] > 510 for row in mbpp_rows), "an MBPP training row has a task id of the test split"

    def mbpp_question_then_answer(row: dict) -> list[dict]:
        # the fields render_chat_messages reads for a task-description code item, from the unsanitized rows
        setup_lines = [row["test_setup_code"]] if row["test_setup_code"].strip() else []
        item = {"task": "mbpp", "kind": "code", "question": row["text"].strip(),
                "meta": {"question_form": "task_description", "test_imports": setup_lines, "test_list": list(row["test_list"])}}
        return question_then_answer(item, f"```python\n{row['code'].strip()}\n```")

    take(mbpp_rows, "mbpp_train", mbpp_question_then_answer)


def build_probe_data() -> None:
    from transformers import AutoTokenizer

    configuration = load_probe_configuration()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    conversations = collect_probe_conversations(configuration)
    np.random.default_rng(configuration["data"]["seed"]).shuffle(conversations)

    maximum_tokens = configuration["data"]["maximum_tokens_per_conversation"]
    kept_conversations, token_id_rows, loss_mask_rows = [], [], []
    dropped_as_too_long = 0
    for conversation in conversations:
        token_ids, loss_mask = tokenize_conversation_with_loss_mask(tokenizer, conversation["messages"])
        if len(token_ids) > maximum_tokens:
            dropped_as_too_long += 1
            continue
        kept_conversations.append(conversation)
        token_id_rows.append(np.asarray(token_ids, dtype=np.int32))
        loss_mask_rows.append(np.asarray(loss_mask, dtype=np.int8))

    probe_directory().mkdir(parents=True, exist_ok=True)
    with (probe_directory() / "probe_conversations.jsonl").open("w") as conversation_file:
        for conversation in kept_conversations:
            conversation_file.write(json.dumps(conversation, ensure_ascii=False) + "\n")
    offsets = np.concatenate([[0], np.cumsum([len(row) for row in token_id_rows])]).astype(np.int64)
    np.save(probe_directory() / "probe_token_ids.npy", np.concatenate(token_id_rows))
    np.save(probe_directory() / "probe_loss_mask.npy", np.concatenate(loss_mask_rows))
    np.save(probe_directory() / "probe_offsets.npy", offsets)
    by_source: dict[str, int] = {}
    for conversation in kept_conversations:
        by_source[conversation["source"]] = by_source.get(conversation["source"], 0) + 1
    summary = {"conversations": len(kept_conversations), "dropped_as_too_long": dropped_as_too_long,
               "tokens": int(offsets[-1]), "tokens_with_loss": int(sum(int(row.sum()) for row in loss_mask_rows)),
               "conversations_by_source": by_source}
    summary_name = "probe_data_summary.json" if active_probe_variant["name"] == "in_domain" else f"probe_data_summary_{active_probe_variant['name']}.json"
    (repository_root / "results" / summary_name).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def load_probe_rows() -> tuple[list[np.ndarray], list[np.ndarray]]:
    offsets = np.load(probe_directory() / "probe_offsets.npy")
    token_ids = np.load(probe_directory() / "probe_token_ids.npy")
    loss_mask = np.load(probe_directory() / "probe_loss_mask.npy")
    rows = [(token_ids[start:stop], loss_mask[start:stop]) for start, stop in zip(offsets[:-1], offsets[1:])]
    return [row[0] for row in rows], [row[1] for row in rows]


# ---------------------------------------------------------------------------------------------------------------
# train: the fixed fine-tune
# ---------------------------------------------------------------------------------------------------------------
def plan_probe_batches(token_id_rows: list[np.ndarray], tokens_per_forward: int, epoch_count: int, seed: int) -> list[list[int]]:
    """Batches of row positions, the same for every checkpoint. Rows of similar length share a batch (so little
    padding is wasted) and the batches of each epoch are shuffled with the probe's seed."""
    random_generator = np.random.default_rng(seed)
    batches_of_all_epochs: list[list[int]] = []
    for _ in range(epoch_count):
        # sort within shuffled windows of 512 rows: similar lengths meet, yet every epoch pairs rows differently
        positions = random_generator.permutation(len(token_id_rows))
        epoch_batches: list[list[int]] = []
        for window_start in range(0, len(positions), 512):
            window = sorted(positions[window_start:window_start + 512], key=lambda position: len(token_id_rows[position]))
            current_batch: list[int] = []
            for position in window:
                longest = max(len(token_id_rows[position]), max((len(token_id_rows[p]) for p in current_batch), default=0))
                if current_batch and longest * (len(current_batch) + 1) > tokens_per_forward:
                    epoch_batches.append(current_batch)
                    current_batch = []
                current_batch.append(int(position))
            if current_batch:
                epoch_batches.append(current_batch)
        random_generator.shuffle(epoch_batches)
        batches_of_all_epochs.extend(epoch_batches)
    return batches_of_all_epochs


def train_probe(checkpoint_path: str, output_directory: Path, tokens_per_forward: int, master_weights_on: str,
                step_count_limit: int | None = None, training_seed: int | None = None) -> dict:
    import torch

    from midtrain.mixture import end_of_text_token_id
    from midtrain.train_continue import (MasterWeightAdamW, chunked_language_model_loss, learning_rate_at_step,
                                         load_working_model, place_model_on_devices, save_checkpoint)

    configuration = load_probe_configuration()["training"]
    if training_seed is not None:      # only for measuring the probe's own noise: the same checkpoint, another batch order
        configuration["seed"] = training_seed
    torch.manual_seed(configuration["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    token_id_rows, loss_mask_rows = load_probe_rows()
    batches = plan_probe_batches(token_id_rows, tokens_per_forward, configuration["epoch_count"], configuration["seed"])
    # group the batches into optimizer steps of about `tokens_per_step` padded tokens, before any training, so that
    # each step's loss can be divided by the number of tokens that carry loss in exactly that step
    steps: list[list[list[int]]] = [[]]
    padded_tokens_in_step = 0
    for batch in batches:
        steps[-1].append(batch)
        padded_tokens_in_step += len(batch) * max(len(token_id_rows[position]) for position in batch)
        if padded_tokens_in_step >= configuration["tokens_per_step"]:
            steps.append([])
            padded_tokens_in_step = 0
    steps = [step for step in steps if step][:step_count_limit]
    step_count = len(steps)

    device_names = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    model = load_working_model(checkpoint_path)
    place_model_on_devices(model, device_names)
    optimizer = MasterWeightAdamW(model.parameters(), device_names[0] if master_weights_on == "gpu" else "cpu",
                                  betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0)
    input_device = model.get_input_embeddings().weight.device
    output_layer = model.get_output_embeddings()
    padding_id = end_of_text_token_id()

    started = time.time()
    loss_history: list[float] = []
    for step_index, batches_of_step in enumerate(steps):
        loss_tokens_of_step = sum(int(loss_mask_rows[position][1:].sum()) for batch in batches_of_step for position in batch)
        loss_sum_in_step = 0.0
        for batch in batches_of_step:
            longest = max(len(token_id_rows[position]) for position in batch)
            token_ids = np.full((len(batch), longest), padding_id, dtype=np.int64)
            loss_mask = np.zeros((len(batch), longest), dtype=np.int64)
            for row_index, position in enumerate(batch):
                token_ids[row_index, :len(token_id_rows[position])] = token_id_rows[position]
                loss_mask[row_index, :len(loss_mask_rows[position])] = loss_mask_rows[position]
            token_ids_on_device = torch.from_numpy(token_ids).to(input_device)
            hidden_states = model.model(input_ids=token_ids_on_device).last_hidden_state
            loss_sum, _ = chunked_language_model_loss(
                hidden_states, token_ids_on_device, output_layer, float(loss_tokens_of_step), with_gradient=True,
                label_mask=torch.from_numpy(loss_mask).to(input_device))
            loss_sum_in_step += loss_sum
        learning_rate = learning_rate_at_step(step_index, step_count, configuration["peak_learning_rate"],
                                              configuration["warmup_step_share"])
        optimizer.step(learning_rate, maximum_gradient_norm=1.0)
        loss_history.append(loss_sum_in_step / max(1, loss_tokens_of_step))
        assert math.isfinite(loss_history[-1]), f"probe loss is {loss_history[-1]} at step {step_index}"
        if (step_index + 1) % 50 == 0:
            print(f"probe step {step_index + 1}/{step_count} loss {np.mean(loss_history[-50:]):.3f} "
                  f"({(time.time() - started) / 60:.0f} min)", flush=True)
    save_checkpoint(model, checkpoint_path, output_directory)
    summary = {"steps": step_count, "seed": configuration["seed"], "minutes": round((time.time() - started) / 60, 1),
               "loss_first_20_steps": round(float(np.mean(loss_history[:20])), 4),
               "loss_last_20_steps": round(float(np.mean(loss_history[-20:])), 4)}
    # Drop every reference to the model before the server starts. The server checks at start-up that its share of
    # the GPU is free, and on 11-20 the 2B probe still held about 5 GiB here (the output layer, the last batch and the
    # allocator's cache) next to another user's 3 GiB, so the server refused to start.
    del model, optimizer, output_layer, hidden_states, token_ids_on_device
    gc.collect()
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------------------------------------------
# serve and sample
# ---------------------------------------------------------------------------------------------------------------
def start_inference_server(checkpoint_path: str, port: int, log_path: Path, maximum_context_tokens: int = 4096):
    """Start the server in its own session (so that stopping it takes its workers too) and wait until it answers.

    Every answer being generated needs one block of the state cache of the linear-attention layers, and how many
    blocks fit depends on the model and on the memory share. When the requested number of answers at once is too
    large the server names the number of blocks and exits ("max_num_seqs (1024) exceeds available Mamba cache blocks
    (709)", the 0.8B model at a 25% share, 11-20); it is then started again with that number.

    The server also exits when less memory is free than its share asks for ("Free memory on device cuda:0
    (30.86/39.49 GiB) on startup is less than desired GPU memory utilization (0.8, 31.59 GiB)", the 2B probe on a GPU
    that another user's job also uses, 11-20). It is then started again with the share that is free, less 1 GiB."""
    # The server may use this share of the GPU's memory; on a GPU that another job also uses, set it lower.
    memory_share = float(os.environ.get("MIDTRAIN_SERVER_MEMORY_SHARE", "0.80"))
    sequences_at_once = 1024
    for attempt in range(4):
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(checkpoint_path),
                   "--served-model-name", "candidate", "--port", str(port), "--max-model-len", str(maximum_context_tokens),
                   "--gpu-memory-utilization", str(memory_share), "--max-num-seqs", str(sequences_at_once),
                   "--mamba-ssm-cache-dtype", "float32", "--gdn-prefill-backend", "triton",
                   "--language-model-only", "--seed", "0"]
        server_process = subprocess.Popen(command, stdout=log_path.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 900          # loading takes 1-2 minutes for the 0.8B model and about 6 for the 9B
        while time.time() < deadline:
            if server_process.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3)
                return server_process
            except OSError:
                time.sleep(5)
        else:
            stop_inference_server(server_process)
            raise RuntimeError(f"the inference server did not answer within 15 minutes; see {log_path}")
        server_log_text = log_path.read_text(errors="replace")
        block_limit = re.search(r"exceeds available Mamba cache blocks \((\d+)\)", server_log_text)
        free_memory = re.search(r"Free memory on device \S+ \(([\d.]+)/([\d.]+) GiB\) on startup is less than desired",
                                server_log_text)
        if block_limit is not None:
            sequences_at_once = int(block_limit.group(1)) // 8 * 8
            print(f"the server holds state for {block_limit.group(1)} answers at once; starting it again with "
                  f"{sequences_at_once}", flush=True)
        elif free_memory is not None:
            free_gib, total_gib = float(free_memory.group(1)), float(free_memory.group(2))
            memory_share = math.floor((free_gib - 1.0) / total_gib * 100) / 100
            # below 30% of a 40 GiB A100 the 2B model's weights (4.5 GiB) would leave little room for its caches
            assert memory_share >= 0.3, f"only {free_gib:.1f} of {total_gib:.1f} GiB are free for the server; see {log_path}"
            print(f"{free_gib:.1f} of {total_gib:.1f} GiB are free; starting the server again with a share of "
                  f"{memory_share:.2f}", flush=True)
        else:
            raise RuntimeError(f"the inference server exited with code {server_process.returncode}; see {log_path}")
    raise RuntimeError(f"the inference server did not start after four attempts; see {log_path}")


def stop_inference_server(server_process) -> None:
    """End the server's whole session: its engine runs in child processes that outlive a plain terminate."""
    try:
        os.killpg(server_process.pid, signal.SIGTERM)
        server_process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(server_process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def end_token_ids() -> list[int]:
    """The ids of `<|im_end|>`, `<|endoftext|>` and `<|im_start|>` (a model that skips the end of its turn and opens
    the next one has finished its answer too).

    A generation must stop at these tokens by id. The server matches stop strings against the decoded text, and the
    decoded text leaves special tokens out, so a stop string "<|im_end|>" never matches: the first probe's answers
    ran past the end of the assistant's turn into turns the model made up, and the grader read the last \\boxed{}
    of those (median answer 7 times longer than the base model's, pass@1 below the base model's few-shot pass@1;
    11-20, PITFALLS E1)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path())
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in (turn_end, "<|endoftext|>", turn_start)]
    assert all(isinstance(token_id, int) and token_id >= 0 for token_id in token_ids), f"missing end tokens: {token_ids}"
    return token_ids


def sample_completions(port: int, prompt_text: str, sample_count: int, maximum_new_tokens: int, temperature: float,
                       top_p: float, stop_sequences: list[str], stop_token_ids: list[int]) -> list[str]:
    request_body = {"model": "candidate", "prompt": prompt_text, "n": sample_count, "max_tokens": maximum_new_tokens,
                    "temperature": temperature, "top_p": top_p, "stop": stop_sequences, "stop_token_ids": stop_token_ids}
    for attempt in range(4):
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                             json.dumps(request_body).encode("utf-8"), {"Content-Type": "application/json"})
            response = json.load(urllib.request.urlopen(request, timeout=3600))
            choices = sorted(response["choices"], key=lambda choice: choice["index"])
            return [choice["text"] for choice in choices], [choice.get("finish_reason") or "" for choice in choices]
        except OSError:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def load_evaluation_items(task_name: str, item_count: int, item_seed: int) -> list[dict]:
    """The first `item_count` items of a task after a seeded shuffle: the same items for every checkpoint."""
    items = eval_tasks.read_json_lines(data_root() / "eval" / f"{task_name}.jsonl")
    order = np.random.default_rng([item_seed, len(items)]).permutation(len(items))
    return [items[position] for position in order[:item_count]]


def pass_at_k_with_interval(correct_counts: list[int], sample_count: int, k: int, resample_count: int = 1000) -> dict:
    """Mean pass@k over the items, with a 95% interval from resampling the items."""
    per_item = np.array([eval_tasks.estimate_pass_at_k(correct, sample_count, k) for correct in correct_counts])
    random_generator = np.random.default_rng(0)
    resampled_means = per_item[random_generator.integers(0, len(per_item), size=(resample_count, len(per_item)))].mean(axis=1)
    low, high = np.percentile(resampled_means, [2.5, 97.5])
    return {"value": round(100 * float(per_item.mean()), 2), "low": round(100 * float(low), 2), "high": round(100 * float(high), 2)}


def grade_samples_of_items(items: list[dict], completions_per_item: list[list[str]], prompt_style: str) -> list[int]:
    """How many of each item's samples are correct. Identical samples of one item are graded once: at temperature 1
    a small model repeats itself often on easy items, and every code sample costs a sandbox start."""
    distinct_pairs: dict[tuple[int, str], int] = {}
    for item_index, completions in enumerate(completions_per_item):
        for completion in completions:
            distinct_pairs.setdefault((item_index, completion), len(distinct_pairs))
    pair_items = [items[item_index] for item_index, _ in distinct_pairs]
    pair_texts = [completion for _, completion in distinct_pairs]
    if items[0]["kind"] == "code":
        sandbox_verdicts = eval_tasks.grade_code_completions_in_parallel(pair_items, pair_texts, prompt_style)
        # without a sandbox every program counts as failed; a code score of zero would then look like a model
        # that cannot program, so the run stops here instead
        assert not any(verdict["reason"] == "no_sandbox" for verdict in sandbox_verdicts), (
            "no working sandbox on this machine (bubblewrap, or apptainer with MIDTRAIN_SANDBOX_IMAGE): code cannot be graded")
        verdicts = [verdict["passed"] for verdict in sandbox_verdicts]
    else:
        verdicts = [eval_tasks.grade_completion(item, text, prompt_style) for item, text in zip(pair_items, pair_texts)]
    return [sum(bool(verdicts[distinct_pairs[(item_index, completion)]]) for completion in completions)
            for item_index, completions in enumerate(completions_per_item)]


def evaluate_pass_at_k(checkpoint_path: str, run_name: str, port: int, prompt_style: str = "chat",
                       item_count_limit: int | None = None, evaluation_settings: dict | None = None) -> dict:
    """Serve the checkpoint, sample every test item `sample_count` times, grade, and return pass@k per task.

    `evaluation_settings` replaces the `evaluation` block of configs/probe.yaml; the regression suite uses it to
    ask a base checkpoint few-shot questions with its own tasks and sample counts (base_regression.py). A task may
    set its own `sample_count` and `temperature`."""
    configuration = evaluation_settings or load_probe_configuration()["evaluation"]
    if item_count_limit:        # the quick end-to-end test: few items, 8 samples each
        configuration["sample_count"] = 8
        configuration["pass_at_k"] = [k for k in configuration["pass_at_k"] if k <= 8]
        for task_settings in configuration["tasks"].values():
            if "sample_count" in task_settings:      # a greedy task keeps its single sample: the server refuses more
                task_settings["sample_count"] = min(8, task_settings["sample_count"])
    run_directory = data_root() / "runs" / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    # the context must hold the prompt and the longest answer allowed (4,096 tokens for MATH-500 in generic_long)
    longest_answer = max(task_settings["maximum_new_tokens"] for task_settings in configuration["tasks"].values())
    server_process = start_inference_server(checkpoint_path, port, run_directory / f"server_{prompt_style}.log",
                                            maximum_context_tokens=max(4096, longest_answer + 2048))
    results: dict = {"run_name": run_name, "prompt_style": prompt_style, "sample_count": configuration["sample_count"], "tasks": {}}
    stop_token_ids = end_token_ids()
    try:
        for task_name, task_settings in configuration["tasks"].items():
            items = load_evaluation_items(task_name, item_count_limit or task_settings["item_count"], configuration["item_seed"])
            if prompt_style == "chat":
                prompts = [render_chatml_prompt(eval_tasks.render_chat_messages(item)) for item in items]
                stop_sequences = [turn_end]
            else:
                few_shot_items = eval_tasks.load_few_shot_items(task_name)
                prompts = [eval_tasks.render_base_prompt(item, few_shot_items) for item in items]
                stop_sequences = eval_tasks.base_completion_stop_sequences(items[0])[:4]      # the server takes 4 at most
            started = time.time()
            sample_count = task_settings.get("sample_count", configuration["sample_count"])
            temperature = task_settings.get("temperature", configuration["temperature"])
            # as many requests in flight as fill the server's 1,024 sequence slots
            with concurrent.futures.ThreadPoolExecutor(max(16, 1024 // sample_count)) as request_pool:
                answers_per_item = list(request_pool.map(
                    lambda prompt: sample_completions(port, prompt, sample_count, task_settings["maximum_new_tokens"],
                                                      temperature, configuration["top_p"], stop_sequences, stop_token_ids),
                    prompts))
            completions_per_item = [completions for completions, _ in answers_per_item]
            finish_reasons = [reason for _, reasons in answers_per_item for reason in reasons]
            sampling_minutes = (time.time() - started) / 60
            correct_counts = grade_samples_of_items(items, completions_per_item, prompt_style)
            # the raw samples stay next to the run (never under results/: they quote benchmark questions), so that
            # a surprising score can be read sample by sample afterwards
            with (run_directory / f"samples_{prompt_style}_{task_name}.jsonl").open("w") as samples_file:
                for item, completions, correct_count in zip(items, completions_per_item, correct_counts):
                    samples_file.write(json.dumps({"item_id": item["item_id"], "correct_count": correct_count,
                                                   "completions": completions}, ensure_ascii=False) + "\n")
            task_result = {f"pass@{k}": pass_at_k_with_interval(correct_counts, sample_count, k)
                           for k in configuration["pass_at_k"] if k <= sample_count}
            task_result["sample_count"] = sample_count
            task_result["item_ids"] = [item["item_id"] for item in items]
            task_result["correct_counts"] = correct_counts
            task_result["mean_completion_characters"] = round(float(np.mean([len(c) for cs in completions_per_item for c in cs])))
            # answers cut at the token limit count as wrong; their share says whether the limit is too tight
            task_result["share_cut_at_token_limit"] = round(sum(reason == "length" for reason in finish_reasons) / len(finish_reasons), 4)
            task_result["minutes"] = round(sampling_minutes + 0.0, 1)
            results["tasks"][task_name] = task_result
            shown = "  ".join(f"pass@{k} {task_result[f'pass@{k}']['value']:.1f}" for k in configuration["pass_at_k"]
                              if k in (1, 8, 16, 64) and k <= sample_count)
            print(f"{run_name} [{prompt_style}] {task_name}: {shown}  ({100 * task_result['share_cut_at_token_limit']:.1f}% cut at "
                  f"the token limit; {sampling_minutes:.1f} min sampling)", flush=True)
    finally:
        stop_inference_server(server_process)
    return results


# ---------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------
def port_for_this_process() -> int:
    """One port per first visible GPU, so that probes on different GPUs never meet."""
    first_gpu = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
    return 8100 + int(first_gpu)


def write_probe_results(run_name: str, results: dict) -> None:
    results["probe_variant"] = active_probe_variant["name"]
    results_directory = repository_root / "results" / probe_variant_settings()["results_folder"]
    results_directory.mkdir(parents=True, exist_ok=True)
    (results_directory / f"{run_name}.json").write_text(json.dumps(results, indent=1) + "\n")
    run_file_name = "probe_results.json" if active_probe_variant["name"] == "in_domain" else f"probe_results_{active_probe_variant['name']}.json"
    (data_root() / "runs" / run_name / run_file_name).write_text(json.dumps(results, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build-data", "run", "evaluate"])
    parser.add_argument("--checkpoint", help="folder of the checkpoint to probe")
    parser.add_argument("--run-name", help="names the results file and the run folder")
    parser.add_argument("--tokens-per-forward", type=int, default=16384)
    parser.add_argument("--master-weights-on", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--keep-probe-checkpoint", action="store_true",
                        help="by default the fine-tuned weights are deleted once graded: 1.5 GB to 19 GB per candidate")
    parser.add_argument("--quick-test", action="store_true",
                        help="20 fine-tuning steps, 12 items per task, 8 samples: checks that every stage runs, in minutes")
    parser.add_argument("--training-seed", type=int, default=None,
                        help="another batch order for the fine-tune; only for measuring the probe's own noise (README §3.4)")
    parser.add_argument("--variant", choices=list(probe_variants), default="in_domain",
                        help="in_domain: fine-tunes on the tested tasks' training solutions; generic: instruction data only")
    arguments = parser.parse_args()
    active_probe_variant["name"] = arguments.variant
    step_count_limit, item_count_limit = (20, 12) if arguments.quick_test else (None, None)
    if arguments.command == "build-data":
        build_probe_data()
        return
    assert arguments.checkpoint and arguments.run_name, "give --checkpoint and --run-name"
    port = port_for_this_process()
    if arguments.command == "evaluate":
        results = evaluate_pass_at_k(arguments.checkpoint, arguments.run_name, port, item_count_limit=item_count_limit)
        write_probe_results(arguments.run_name, results)
        return
    import shutil

    probe_checkpoint = data_root() / "runs" / arguments.run_name / f"probe_checkpoint_{arguments.variant}"
    training_summary = train_probe(arguments.checkpoint, probe_checkpoint, arguments.tokens_per_forward,
                                   arguments.master_weights_on, step_count_limit, arguments.training_seed)
    print(f"{arguments.run_name}: probe fine-tune {json.dumps(training_summary)}", flush=True)
    results = evaluate_pass_at_k(str(probe_checkpoint), arguments.run_name, port, item_count_limit=item_count_limit)
    results["probe_training"] = training_summary
    write_probe_results(arguments.run_name, results)
    if not arguments.keep_probe_checkpoint:
        shutil.rmtree(probe_checkpoint)


if __name__ == "__main__":
    main()
