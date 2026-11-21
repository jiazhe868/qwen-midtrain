"""Tests of the continued-pretraining trainer (train_continue.py) that need one GPU; seconds. Skipped without a GPU.

Why these tests exist. The 2B model and larger keep their FP32 master weights and Adam moments on the CPU. The step
that moves the gradients out and the rounded weights back was rewritten for speed on 11-20 (page-locked buffers, one
wait per direction, fused AdamW: 7.6 s to 2.1 s per step for 1.8B parameters). The rewrite must give the weights that
the first version gave, including for a tensor that gets no gradient and for a step whose gradient is clipped.

    source scripts/env.sh && CUDA_VISIBLE_DEVICES=0 $MIDTRAIN_PYTHON -m pytest tests/test_train_continue.py -q
"""
import pytest
import torch

from midtrain.train_continue import MasterWeightAdamW

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the host-master step moves tensors to and from a GPU")


def run_steps_with_masters_on_host(use_rewritten_step: bool, step_count: int = 6) -> list[torch.Tensor]:
    """Return the BF16 working weights after `step_count` steps with the same seeded weights and gradients."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    shapes = [(512, 256), (256,), (1000, 64)]
    working_parameters = [torch.nn.Parameter((torch.randn(shape, generator=generator, device="cuda") * 0.02).bfloat16())
                          for shape in shapes]
    parameter_without_gradient = torch.nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
    optimizer = MasterWeightAdamW(working_parameters + [parameter_without_gradient], "cpu", betas=(0.9, 0.95),
                                  epsilon=1e-8, weight_decay=0.0)
    if not use_rewritten_step:
        # the first version: plain .to() moves and the default (for-loop) CPU AdamW
        optimizer.masters_on_host = False
        optimizer.optimizer = torch.optim.AdamW(optimizer.master_parameters, lr=0.0, betas=(0.9, 0.95), eps=1e-8,
                                                weight_decay=0.0)
    for step_index in range(step_count):
        for parameter in working_parameters:
            parameter.grad = (torch.randn(parameter.shape, generator=generator, device="cuda") * 1e-2).bfloat16()
        # a maximum norm below the gradient norm on even steps, so the clipping branch runs too
        optimizer.step(learning_rate=1e-3, maximum_gradient_norm=0.1 if step_index % 2 == 0 else 1e9)
    return [parameter.detach().cpu() for parameter in working_parameters + [parameter_without_gradient]]


@needs_gpu
def test_rewritten_host_master_step_gives_the_weights_of_the_first_version():
    first_version_weights = run_steps_with_masters_on_host(use_rewritten_step=False)
    rewritten_weights = run_steps_with_masters_on_host(use_rewritten_step=True)
    element_count = sum(weights.numel() for weights in first_version_weights)
    differing_count = sum(int((first != rewritten).sum()) for first, rewritten in zip(first_version_weights, rewritten_weights))
    # fused and for-loop AdamW differ in the last FP32 bit on a few elements, which rarely changes the BF16 rounding
    # (96 of 1.8B elements after 4 steps in the 2B-sized benchmark)
    assert differing_count <= element_count * 1e-4, f"{differing_count} of {element_count} weights differ"
    assert torch.equal(rewritten_weights[-1], torch.ones(8, dtype=torch.bfloat16)), "a tensor without gradient moved"
    initial_weights = run_steps_with_masters_on_host(use_rewritten_step=True, step_count=0)
    assert not torch.equal(initial_weights[0], rewritten_weights[0]), "the weights did not move"
