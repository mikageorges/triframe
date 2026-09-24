"""
tests/conftest.py

Shared fixtures, built on triframe.examples.beta_vae_subgroup_minimal.
The trained-model fixture is session-scoped (expensive: ~300 epochs) and
shared read-only across every test that needs a model with real learned
structure; the untrained fixture is cheap and function-scoped.
"""

import numpy as np
import pytest
import torch

from triframe.examples.beta_vae_subgroup_minimal.synthetic_data import build_registry, generate
from triframe.examples.beta_vae_subgroup_minimal.model import (
    build_model, MinimalAdapter, to_block_inputs, train,
)


@pytest.fixture(scope="session")
def registry():
    return build_registry()


@pytest.fixture(scope="session")
def data():
    """n=2000, seed=0 — matches the values validated by hand during
    development. Changing n or seed invalidates the hardcoded ground-truth
    thresholds in test_layer2.py/test_layer1.py; keep in sync if you do."""
    return generate(n=2000, seed=0)


@pytest.fixture(scope="session")
def block_data(data, registry):
    return {b: data[b] for b in registry.block_names}


@pytest.fixture()
def untrained_adapter(registry):
    """Fresh untrained model per test — fast, deterministic (seed=0), no
    training required. Use for mechanics tests (shapes, hook correctness,
    error handling) that don't need the model to have learned anything."""
    model = build_model(registry, d_proj=8, seed=0)
    return MinimalAdapter(model)


@pytest.fixture(scope="session")
def trained_synergy_adapter(registry, data):
    """
    Trained on synergy_target (the XOR of synergy_a/synergy_b's hidden
    bits) — the ONLY fixture where Layer 3 synergy detection has anything
    real to find, since an untrained model has no reason to encode XOR
    structure. Session-scoped: training is the expensive part of the
    suite (~300 epochs), and the model is read-only after training, so
    every test sharing this fixture must not call .train() on it or
    mutate its weights.

    Recipe (n=4000, d_proj=8, seed=1, lr=0.01, epochs=300, no weight decay)
    is exactly the one validated by hand to produce clean single-token IIA
    separation (synergy_a/synergy_b ~0.5 each, every other token exactly
    0.0) and correct greedy-search synergy detection (joint IIA -> 1.0 at
    step 2). Changing this recipe requires re-validating those thresholds
    in test_layer3.py.
    """
    big_data = generate(n=4000, seed=0)
    block_inputs = to_block_inputs(big_data, registry)
    labels = torch.tensor(big_data["synergy_target"], dtype=torch.long)

    model = build_model(registry, d_proj=8, seed=1)
    model = train(model, block_inputs, labels, epochs=300, lr=0.01)

    adapter = MinimalAdapter(model)
    adapter._synergy_data = big_data  # stash for test_layer3.py's pair construction
    return adapter