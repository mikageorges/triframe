"""tests/test_adapter.py — ModelAdapter contract, applied via
MinimalAdapter."""

import pytest
import torch

from triframe.adapter import ModelAdapter
from triframe.examples.beta_vae_subgroup_minimal.model import to_block_inputs


def test_abc_enforces_all_methods():
    class Incomplete(ModelAdapter):
        def get_token_module(self):
            return None
    with pytest.raises(TypeError):
        Incomplete()


def test_run_forward_and_get_output_value(untrained_adapter, data, registry):
    batch = to_block_inputs(data, registry)
    out = untrained_adapter.run_forward(batch)
    logits = untrained_adapter.get_output_value(out)
    assert logits.shape == (len(data["labels"]), 2)


def test_token_module_hookable(untrained_adapter, data, registry):
    batch = to_block_inputs(data, registry)
    captured = {}

    def hook(module, inp, output):
        captured["tokens"] = output.detach()

    handle = untrained_adapter.get_token_module().register_forward_hook(hook)
    try:
        untrained_adapter.run_forward(batch)
    finally:
        handle.remove()

    assert captured["tokens"].shape == (len(data["labels"]), registry.total_tokens, 8)


def test_token_module_patchable(untrained_adapter, data, registry):
    """Patching all token positions makes the output fully independent of
    the batch's own input — the strongest correctness property an
    adapter's hookable module must support."""
    batch_a = to_block_inputs(data, registry)
    batch_b = {k: torch.randn_like(v) for k, v in batch_a.items()}

    captured = {}

    def capture_hook(module, inp, output):
        captured["tokens"] = output.detach()

    handle = untrained_adapter.get_token_module().register_forward_hook(capture_hook)
    try:
        untrained_adapter.run_forward(batch_b)
    finally:
        handle.remove()
    source_tokens = captured["tokens"]

    n_tokens = registry.total_tokens

    def patch_hook(module, inp, output):
        return source_tokens

    handle = untrained_adapter.get_token_module().register_forward_hook(patch_hook)
    try:
        out_patched_a = untrained_adapter.run_forward(batch_a)
        out_patched_other = untrained_adapter.run_forward(
            {k: torch.randn_like(v) for k, v in batch_a.items()})
    finally:
        handle.remove()

    logits_a = untrained_adapter.get_output_value(out_patched_a)
    logits_other = untrained_adapter.get_output_value(out_patched_other)
    assert torch.allclose(logits_a, logits_other, atol=1e-5)


def test_optional_get_attention_weights_raises_by_default(untrained_adapter):
    with pytest.raises(NotImplementedError):
        untrained_adapter.get_attention_weights(None)