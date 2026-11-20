"""Tests of the fixed fine-tuning probe (sft_probe.py) that need no GPU; seconds.

Why these tests exist. The probe teaches a model to end its turn with one token and then asks a server to stop
there. The first probe asked with a stop string; the server matches stop strings against text from which special
tokens are removed, so the answers ran on into turns the model invented and the grader read the wrong answer
(11-20, PITFALLS E1). The two sides of that contract are tested here: what the fine-tune teaches and what the
request asks for.

    source scripts/env.sh && $MIDTRAIN_PYTHON -m pytest tests/test_sft_probe.py -q
"""
import io
import json

import numpy as np
import pytest

from midtrain import sft_probe


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from midtrain.data_sources import tokenizer_path
    return AutoTokenizer.from_pretrained(tokenizer_path())


def test_the_fine_tune_teaches_the_end_of_turn_as_one_special_token_with_loss(tokenizer):
    messages = [{"role": "user", "content": "What is 2 + 3?"}, {"role": "assistant", "content": "The answer is \\boxed{5}."}]
    token_ids, loss_mask = sft_probe.tokenize_conversation_with_loss_mask(tokenizer, messages)
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    positions = [index for index, token_id in enumerate(token_ids) if token_id == end_of_turn_id]
    assert len(positions) == 2, "each of the two turns must end with the single end-of-turn token"
    assert loss_mask[positions[0]] == 0 and loss_mask[positions[1]] == 1, "only the assistant's end of turn carries loss"
    assert tokenizer.decode([token_id for token_id, mask in zip(token_ids, loss_mask) if mask]) == "The answer is \\boxed{5}.<|im_end|>"


def test_sampling_stops_at_the_end_tokens_by_id(monkeypatch, tokenizer):
    sent_bodies = []

    def fake_urlopen(request, timeout):
        sent_bodies.append(json.loads(request.data))
        return io.StringIO(json.dumps({"choices": [{"index": 0, "text": "x", "finish_reason": "stop"}]}))

    monkeypatch.setattr(sft_probe.urllib.request, "urlopen", fake_urlopen)
    stop_token_ids = sft_probe.end_token_ids()
    texts, finish_reasons = sft_probe.sample_completions(8100, "prompt", 1, 16, 1.0, 1.0, ["<|im_end|>"], stop_token_ids)
    assert texts == ["x"] and finish_reasons == ["stop"]
    assert set(sent_bodies[0]["stop_token_ids"]) == {tokenizer.convert_tokens_to_ids(token)
                                                     for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>")}


def test_pass_at_k_interval_contains_the_value_and_is_zero_width_for_identical_items():
    result = sft_probe.pass_at_k_with_interval([3] * 50, 64, 16)
    assert result["low"] == result["value"] == result["high"]
    mixed = sft_probe.pass_at_k_with_interval(list(np.arange(50) % 65), 64, 1)
    assert mixed["low"] <= mixed["value"] <= mixed["high"]
