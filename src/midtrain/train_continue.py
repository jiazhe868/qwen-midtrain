"""Continue pretraining a Qwen3.5 base checkpoint on a recipe's tokens, with a learning rate that anneals to zero.

Why this file exists. The usual way, the model in FP32 on one GPU under a library trainer with AdamW, fails here in
three places, each measured on an A100 with 40 GB (11-20):
1. The vocabulary has 248,320 entries. Logits for a step of 16,384 tokens take 16 GB in FP32, and the 0.8B model ran
   out of memory on them. The loss is therefore computed over chunks of 1,024 tokens, and the trunk receives the
   summed gradient once (`chunked_language_model_loss`).
2. Annealing uses learning rates near 1e-5. A BF16 weight has 8 bits of mantissa, so an update of 1e-5 of its size
   rounds to nothing. The weights that the GPUs compute with are BF16; the optimizer keeps an FP32 master copy and
   the Adam moments, applies the update there, and writes the rounded result back (`MasterWeightAdamW`). For the
   0.8B model the master copy sits on the same GPU. From 2B up it sits in host memory: weights, gradients, master
   and moments of the 9B model come to 154 GB.
3. The 9B model in BF16 with its gradients is 36 GB before any activation. Its layers are spread over the GPUs of
   one machine (`place_model_on_devices`), which costs speed (one GPU works at a time) and nothing else.
Gradient checkpointing is always on: without it the 0.8B model ran out of memory at 16,384 tokens per forward pass.

Measured speed on one shared A100, 4,096-token sequences: 0.8B 9.0K tokens/s, 2B 5.7K, 4B 2.8K.

The learning rate rises linearly for `warmup_step_share` of the steps and then falls linearly to zero.

Commands (after `source scripts/env.sh`):
    CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m midtrain.train_continue \
        --base-model $MIDTRAIN_MODELS/qwen35_0.8B_base --recipe configs/recipes/pilot_replay_only.yaml \
        --run-name pilot_replay_only_0.8B --peak-learning-rate 1e-5
    CUDA_VISIBLE_DEVICES=1,2,3 $MIDTRAIN_PYTHON -m midtrain.train_continue --base-model $MIDTRAIN_MODELS/qwen35_9B_base \
        --recipe configs/recipes/best_mix.yaml --run-name best_mix_9B --master-weights-on cpu --tokens-per-forward 8192
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional

from midtrain.data_sources import data_root, open_token_source, repository_root
from midtrain.mixture import iterate_training_steps, iterate_validation_sequences, load_recipe

loss_chunk_token_count = 1024           # 1,024 tokens of logits over 248,320 words are 1 GB in FP32
validation_sequence_length = 4096
# 32 sequences = 131,072 held-out tokens per source, the same ones for every run. The pilot runs used 6 (24,576
# tokens): enough to see a 2% shift, too few to trust a 0.3% one, and the design's effects are of that size.
validation_sequences_per_source = 32


# ---------------------------------------------------------------------------------------------------------------
# model: BF16 working weights, on one GPU or spread over several
# ---------------------------------------------------------------------------------------------------------------
def load_working_model(base_model_path: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.train()
    return model


def place_model_on_devices(model, device_names: list[str]) -> None:
    """One device: move everything there. Several: embeddings and the tied output layer on the first, the decoder
    layers in equal runs across all, the final norm on the last; activations follow through accelerate's hooks."""
    if len(device_names) == 1:
        model.to(device_names[0])
        return
    from accelerate import dispatch_model

    text_model = model.model
    layer_count = len(text_model.layers)
    device_map = {"lm_head": device_names[0]}
    for child_name, _ in text_model.named_children():
        if child_name == "layers":
            for layer_index in range(layer_count):
                device_map[f"model.layers.{layer_index}"] = device_names[layer_index * len(device_names) // layer_count]
        elif child_name == "norm":
            device_map["model.norm"] = device_names[-1]
        else:
            device_map[f"model.{child_name}"] = device_names[0]
    dispatch_model(model, device_map=device_map)


def chunked_language_model_loss(hidden_states: torch.Tensor, token_ids: torch.Tensor, output_layer,
                                loss_denominator: float, with_gradient: bool, label_mask: torch.Tensor | None = None):
    """Next-token loss summed over the batch and divided by `loss_denominator`, with logits for one chunk at a time.

    With `with_gradient`, each chunk's loss is back-propagated into the output layer and into a detached copy of
    the hidden states, and the trunk then receives the summed gradient in one backward pass; the caller must not
    call backward again. `label_mask` (same shape as `token_ids`, 1 where the token counts) restricts the loss, as
    the fine-tuning probe needs. Returns (loss sum in nats, number of predicted tokens).
    """
    predicting_states = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    next_token_ids = token_ids[:, 1:].reshape(-1)
    counts_toward_loss = None if label_mask is None else label_mask[:, 1:].reshape(-1).bool()
    output_device = output_layer.weight.device
    detached_states = predicting_states.detach().requires_grad_(with_gradient)
    loss_sum_in_nats = 0.0
    predicted_token_count = 0
    for chunk_start in range(0, next_token_ids.numel(), loss_chunk_token_count):
        chunk = slice(chunk_start, chunk_start + loss_chunk_token_count)
        targets = next_token_ids[chunk].to(output_device)
        if counts_toward_loss is not None:
            targets = torch.where(counts_toward_loss[chunk].to(output_device), targets, torch.full_like(targets, -100))
            if (targets == -100).all():
                continue
        logits = output_layer(detached_states[chunk].to(output_device))
        chunk_loss = torch_functional.cross_entropy(logits.float(), targets, reduction="sum", ignore_index=-100)
        if with_gradient:
            (chunk_loss / loss_denominator).backward()
        loss_sum_in_nats += chunk_loss.item()
        predicted_token_count += int((targets != -100).sum())
    if with_gradient and detached_states.grad is not None:
        predicting_states.backward(detached_states.grad)
    return loss_sum_in_nats, predicted_token_count


# ---------------------------------------------------------------------------------------------------------------
# optimizer: AdamW on an FP32 master copy
# ---------------------------------------------------------------------------------------------------------------
class MasterWeightAdamW:
    """AdamW whose weights of record are FP32 and live on `master_device`; the model computes with BF16 copies.

    State lives across steps (master weights, Adam moments, transfer buffers), so this is a class. Each step moves the
    BF16 gradients to the master device as FP32, lets `torch.optim.AdamW` update the master weights, and copies them
    back, rounded to BF16. The global gradient norm is clipped before the move.

    Master weights on the CPU (the 2B model and larger, whose FP32 weights and moments do not fit next to the model on
    a 40 GB A100). The first version moved each tensor with a plain `.to()`, FP32 through pageable memory in both
    directions, and ran a for-each AdamW: about 9 s per step for the 2B model next to about 4 s of forward and
    backward (2,162-2,751 tokens/s in the 2B smoke test, 11-20). The CPU side now keeps one page-locked BF16 buffer
    per tensor, so each direction moves half the bytes at full bus speed with a single wait for all tensors; the
    BF16/FP32 conversions run on the CPU; and AdamW runs as one fused CPU kernel (0.7 s for 2B parameters against
    1.9 s for the for-each version at 16 threads). The numbers it produces are the same: the conversions are exact or
    the same round-to-nearest-even, and the fused AdamW matched the for-each one on every element after BF16 rounding
    over 20 test steps (largest FP32 difference 7e-9). Master weights on the GPU (the 0.8B model) keep the first
    version, so the 0.8B design runs that started before this change stay comparable with the later ones.
    """

    def __init__(self, working_parameters, master_device: str, betas: tuple[float, float], epsilon: float,
                 weight_decay: float):
        self.working_parameters = [parameter for parameter in working_parameters if parameter.requires_grad]
        self.master_parameters = [torch.nn.Parameter(parameter.detach().to(master_device, dtype=torch.float32))
                                  for parameter in self.working_parameters]
        self.masters_on_host = master_device == "cpu"
        if self.masters_on_host:
            # One page-locked BF16 buffer per tensor carries the gradient out and the rounded weight back, and one FP32
            # buffer per tensor holds the gradient for AdamW, so no step allocates gigabytes. For the 2B model these
            # take 4.5 GB and 9 GB of host memory.
            self.host_staging_buffers = [torch.empty(parameter.shape, dtype=torch.bfloat16, pin_memory=True)
                                         for parameter in self.working_parameters]
            self.host_gradient_buffers = [torch.empty_like(master) for master in self.master_parameters]
            self.working_devices = sorted({parameter.device for parameter in self.working_parameters}, key=str)
            self.optimizer = torch.optim.AdamW(self.master_parameters, lr=0.0, betas=betas, eps=epsilon,
                                               weight_decay=weight_decay, fused=True)
        else:
            self.optimizer = torch.optim.AdamW(self.master_parameters, lr=0.0, betas=betas, eps=epsilon,
                                               weight_decay=weight_decay)

    def gradient_norm(self) -> float:
        squared_sum = 0.0
        for parameter in self.working_parameters:
            if parameter.grad is not None:
                squared_sum += float(parameter.grad.float().pow(2).sum())
        return math.sqrt(squared_sum)

    def step(self, learning_rate: float, maximum_gradient_norm: float) -> float:
        gradient_norm = self.gradient_norm()
        clip_factor = min(1.0, maximum_gradient_norm / (gradient_norm + 1e-12))
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        if self.masters_on_host:
            self.step_with_masters_on_host(clip_factor)
            return gradient_norm
        for master, working in zip(self.master_parameters, self.working_parameters):
            if working.grad is None:
                master.grad = None
                continue
            master.grad = working.grad.to(master.device, dtype=torch.float32).mul_(clip_factor)
            working.grad = None
        self.optimizer.step()
        with torch.no_grad():
            for master, working in zip(self.master_parameters, self.working_parameters):
                working.copy_(master)
        return gradient_norm

    def wait_for_working_devices(self) -> None:
        """Wait until every queued copy on every GPU that holds part of the model has finished."""
        for device in self.working_devices:
            torch.cuda.synchronize(device)

    def step_with_masters_on_host(self, clip_factor: float) -> None:
        """One update with the master weights on the CPU: gradients out, AdamW on the CPU, rounded weights back.

        A tensor without a gradient this step (not used by the forward pass) is left out by AdamW, as in the GPU
        version. Multiplying by a clip factor of exactly 1 changes nothing, so it is skipped.
        """
        has_gradient = [working.grad is not None for working in self.working_parameters]
        # gradients out: every copy is queued at once into page-locked memory, then one wait
        for staging, working, gradient_exists in zip(self.host_staging_buffers, self.working_parameters, has_gradient):
            if gradient_exists:
                staging.copy_(working.grad, non_blocking=True)
        self.wait_for_working_devices()
        for master, staging, gradient_buffer, working, gradient_exists in zip(
                self.master_parameters, self.host_staging_buffers, self.host_gradient_buffers, self.working_parameters,
                has_gradient):
            if not gradient_exists:
                master.grad = None
                continue
            gradient_buffer.copy_(staging)
            if clip_factor < 1.0:
                gradient_buffer.mul_(clip_factor)
            master.grad = gradient_buffer
            working.grad = None
        self.optimizer.step()
        # weights back: rounded to BF16 on the CPU into the same buffers, every copy queued at once, then one wait
        with torch.no_grad():
            for master, staging, working in zip(self.master_parameters, self.host_staging_buffers, self.working_parameters):
                staging.copy_(master)
                working.copy_(staging, non_blocking=True)
        self.wait_for_working_devices()


def learning_rate_at_step(step_index: int, step_count: int, peak_learning_rate: float, warmup_step_share: float) -> float:
    """Linear warm-up, then linear decay that reaches zero after the last step."""
    warmup_steps = int(round(warmup_step_share * step_count))
    if step_index < warmup_steps:
        return peak_learning_rate * (step_index + 1) / warmup_steps
    return peak_learning_rate * (step_count - step_index) / max(1, step_count - warmup_steps)


# ---------------------------------------------------------------------------------------------------------------
# validation: held-out loss per source, before, halfway and after
# ---------------------------------------------------------------------------------------------------------------
def available_validation_sources() -> list[str]:
    """Every tokenized source with enough held-out tokens, except the teacher-written ones.

    The rewritten sources are left out: their loss measures how well a run fits the teacher's text, which no
    analysis uses, and the rewrites were made again after the first batch failed the diversity gate (11-20), so a
    run that read their files at its halfway point could have read them while they were being replaced."""
    token_root = data_root() / "tokens"
    return sorted(directory.name for directory in token_root.iterdir()
                  if not directory.name.startswith("synthetic_")
                  and (directory / "validation.document_offsets.npy").exists()
                  and np.load(directory / "validation.document_offsets.npy")[-1] >= validation_sequence_length)


@torch.no_grad()
def validation_loss_by_source(model, source_names: list[str], sequences_per_forward: int) -> dict:
    """Mean loss in nats per token and bits per byte on fixed held-out sequences of every source."""
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    losses = {}
    for source_name in source_names:
        sequences = list(iterate_validation_sequences(source_name, validation_sequence_length,
                                                      validation_sequences_per_source))
        loss_sum = 0.0
        token_count = 0
        for start in range(0, len(sequences), sequences_per_forward):
            token_ids = torch.from_numpy(np.stack(sequences[start:start + sequences_per_forward])).to(input_device)
            hidden_states = model.model(input_ids=token_ids).last_hidden_state
            chunk_loss, chunk_tokens = chunked_language_model_loss(hidden_states, token_ids, model.get_output_embeddings(),
                                                                   1.0, with_gradient=False)
            loss_sum += chunk_loss
            token_count += chunk_tokens
        token_source = open_token_source(source_name, "validation")
        bytes_per_token = float(token_source["utf8_byte_counts"].sum()) / max(1, int(token_source["document_offsets"][-1]))
        nats_per_token = loss_sum / max(1, token_count)
        losses[source_name] = {"nats_per_token": round(nats_per_token, 4),
                               "bits_per_byte": round(nats_per_token / math.log(2) / bytes_per_token, 4)}
    model.train()
    return losses


# ---------------------------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------------------------
def run_continued_pretraining(arguments: argparse.Namespace) -> None:
    recipe = load_recipe(arguments.recipe)
    if arguments.token_budget:
        recipe["token_budget"] = arguments.token_budget
    run_directory = data_root() / "runs" / arguments.run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(recipe["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    device_names = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    assert device_names, "no GPU visible: set CUDA_VISIBLE_DEVICES"
    model = load_working_model(arguments.base_model)
    place_model_on_devices(model, device_names)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    master_device = device_names[0] if arguments.master_weights_on == "gpu" else "cpu"
    optimizer = MasterWeightAdamW(model.parameters(), master_device, betas=(0.9, 0.95), epsilon=1e-8,
                                  weight_decay=arguments.weight_decay)
    input_device = model.get_input_embeddings().weight.device
    output_layer = model.get_output_embeddings()
    validation_sources = available_validation_sources()
    sequences_per_validation_forward = max(1, arguments.tokens_per_forward // validation_sequence_length)

    settings = {"run_name": arguments.run_name, "base_model": Path(arguments.base_model).name, "recipe": recipe,
                "parameter_count": parameter_count, "peak_learning_rate": arguments.peak_learning_rate,
                "warmup_step_share": arguments.warmup_step_share, "tokens_per_step": arguments.tokens_per_step,
                "tokens_per_forward": arguments.tokens_per_forward, "weight_decay": arguments.weight_decay,
                "maximum_gradient_norm": arguments.maximum_gradient_norm, "master_weights_on": arguments.master_weights_on,
                "device_count": len(device_names)}
    (run_directory / "settings.json").write_text(json.dumps(settings, indent=1) + "\n")
    log_file = (run_directory / "training_log.jsonl").open("w")

    def write_log(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    validation_record = {"event": "validation", "step": 0, "tokens_seen": 0,
                         "loss": validation_loss_by_source(model, validation_sources, sequences_per_validation_forward)}
    write_log(validation_record)
    print(f"{arguments.run_name}: {parameter_count / 1e9:.2f}B parameters on {len(device_names)} GPU(s); "
          f"validation before: " + ", ".join(f"{name} {entry['nats_per_token']:.3f}"
                                             for name, entry in validation_record["loss"].items()), flush=True)

    tokens_seen = 0
    tokens_by_stream_total: dict[str, int] = {}
    started = time.time()
    halfway_validation_done = False
    for step in iterate_training_steps(recipe, arguments.tokens_per_step):
        step_started = time.time()
        step_loss_sum = 0.0
        step_token_count = 0
        for micro_batch in step["micro_batches"]:
            sequences_per_forward = max(1, arguments.tokens_per_forward // micro_batch.shape[1])
            for row_start in range(0, micro_batch.shape[0], sequences_per_forward):
                token_ids = torch.from_numpy(micro_batch[row_start:row_start + sequences_per_forward]).to(input_device)
                hidden_states = model.model(input_ids=token_ids).last_hidden_state
                loss_sum, token_count = chunked_language_model_loss(
                    hidden_states, token_ids, output_layer, float(arguments.tokens_per_step), with_gradient=True)
                step_loss_sum += loss_sum
                step_token_count += token_count
        learning_rate = learning_rate_at_step(step["step_index"], step["step_count"], arguments.peak_learning_rate,
                                              arguments.warmup_step_share)
        gradient_norm = optimizer.step(learning_rate, arguments.maximum_gradient_norm)
        tokens_seen += arguments.tokens_per_step
        for stream_name, tokens in step["tokens_by_stream"].items():
            tokens_by_stream_total[stream_name] = tokens_by_stream_total.get(stream_name, 0) + tokens
        step_loss = step_loss_sum / max(1, step_token_count)
        assert math.isfinite(step_loss), f"loss is {step_loss} at step {step['step_index']}: stopping before it spreads"
        write_log({"event": "step", "step": step["step_index"] + 1, "tokens_seen": tokens_seen,
                   "loss": round(step_loss, 4), "learning_rate": learning_rate, "gradient_norm": round(gradient_norm, 3),
                   "tokens_per_second": round(arguments.tokens_per_step / (time.time() - step_started))})
        if (step["step_index"] + 1) % arguments.print_every_steps == 0 or step["step_index"] + 1 == step["step_count"]:
            elapsed = time.time() - started
            print(f"{arguments.run_name}: step {step['step_index'] + 1}/{step['step_count']} loss {step_loss:.3f} "
                  f"lr {learning_rate:.2e} grad {gradient_norm:.2f} {tokens_seen / elapsed:,.0f} tokens/s "
                  f"({elapsed / 60:.0f} min)", flush=True)
        if not halfway_validation_done and step["progress"] >= 0.5:
            halfway_validation_done = True
            write_log({"event": "validation", "step": step["step_index"] + 1, "tokens_seen": tokens_seen,
                       "loss": validation_loss_by_source(model, validation_sources, sequences_per_validation_forward)})
        if arguments.save_every_steps and (step["step_index"] + 1) % arguments.save_every_steps == 0:
            save_checkpoint(model, arguments.base_model, run_directory / "checkpoint_latest")

    final_validation = validation_loss_by_source(model, validation_sources, sequences_per_validation_forward)
    write_log({"event": "validation", "step": step["step_index"] + 1, "tokens_seen": tokens_seen, "loss": final_validation})
    save_checkpoint(model, arguments.base_model, run_directory / "checkpoint_final")
    if (run_directory / "checkpoint_latest").exists():
        shutil.rmtree(run_directory / "checkpoint_latest")
    summary = {"run_name": arguments.run_name, "tokens_seen": tokens_seen, "hours": round((time.time() - started) / 3600, 2),
               "tokens_by_stream": tokens_by_stream_total, "passes_over_each_stream": step["epochs_started"],
               "validation_before": validation_record["loss"], "validation_after": final_validation}
    (run_directory / "training_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    # the committed copy: every number in the README has to come from a file under results/
    committed_copy = repository_root / "results" / "training" / f"{arguments.run_name}.json"
    committed_copy.parent.mkdir(parents=True, exist_ok=True)
    committed_copy.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{arguments.run_name}: done in {summary['hours']} h; validation after: "
          + ", ".join(f"{name} {entry['nats_per_token']:.3f}" for name, entry in final_validation.items()), flush=True)


def save_checkpoint(model, base_model_path: str, checkpoint_directory: Path) -> None:
    """Write the trained weights in the base checkpoint's own layout, so that any server can load the result.

    The base checkpoint is a vision-language model: its file also holds the vision tower and the multi-token
    prediction head, and its config describes all three. We train the language model only. A folder with just the
    language model and a text-only config loads in transformers, but the inference server refuses it ("Invalid type
    of HuggingFace config", 11-20). So the trained language-model weights go out under their original names, the
    other weights are copied from the base unchanged, and the base's config and tokenizer files come along.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    # In memory the language model's weights are named `model.<rest>`; in the base file they are
    # `model.language_model.<rest>`. The output layer is tied to the embedding and has no entry in the file.
    tensors = {"model.language_model." + name[len("model."):]: tensor.detach().to("cpu")
               for name, tensor in model.state_dict().items() if name != "lm_head.weight"}
    trained_names = set(tensors)
    base_names = set()
    for base_weight_file in sorted(Path(base_model_path).glob("*.safetensors")):
        with safe_open(base_weight_file, "pt") as base_weights:
            for name in base_weights.keys():
                base_names.add(name)
                if name not in trained_names:
                    tensors[name] = base_weights.get_tensor(name)
    assert trained_names <= base_names, (
        f"{len(trained_names - base_names)} trained weights have no counterpart in the base checkpoint, e.g. "
        f"{sorted(trained_names - base_names)[:2]}: the saved file would not load")
    save_file(tensors, str(checkpoint_directory / "model.safetensors"), metadata={"format": "pt"})
    for file_name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                      "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        source_file = Path(base_model_path) / file_name
        if source_file.exists():
            shutil.copy(source_file, checkpoint_directory / file_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", required=True, help="folder of the checkpoint to continue from")
    parser.add_argument("--recipe", required=True, help="a recipe YAML (midtrain.mixture)")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--token-budget", type=int, default=None, help="overrides the recipe's token budget")
    parser.add_argument("--peak-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-step-share", type=float, default=0.02,
                        help="2%% of the steps: the Adam moments start empty, so the first steps are damped")
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    parser.add_argument("--tokens-per-forward", type=int, default=16384,
                        help="tokens in one forward and backward pass; lower it when a larger model runs out of memory")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="0: the runs are short and the study is about data, so nothing else pulls on the weights")
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--master-weights-on", choices=["gpu", "cpu"], default="gpu",
                        help="gpu fits the 0.8B model (12 GB with its moments); from 2B up use cpu")
    parser.add_argument("--save-every-steps", type=int, default=0, help="0: only at the end")
    parser.add_argument("--print-every-steps", type=int, default=25)
    arguments = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("MIDTRAIN_CPU_THREADS", "16")))
    run_continued_pretraining(arguments)


if __name__ == "__main__":
    main()
