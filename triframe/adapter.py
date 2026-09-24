"""
triframe/adapter.py

The ModelAdapter interface: how triframe talks to your model.

Every layer of triframe (attention/ablation, Frechet/pattern alignment,
activation patching/IIA) needs three things from your model, and nothing
else:

  1. a hookable module that produces the token tensor (B, N_tokens, d_proj)
     somewhere in the forward pass — this is the point every layer reads
     from and, for causal patching, writes to.
  2. a way to run a forward pass on one batch of inputs.
  3. a way to pull the outcome value (e.g. logits, a probability, a score)
     out of whatever your model's forward pass returns.

Subclass ModelAdapter and implement its three abstract methods to make
your model usable with triframe. Your model's forward() signature is
untouched — call it however your model already expects to be called from
inside run_forward().

Quickstart
----------
>>> import torch.nn as nn
>>> from triframe.adapter import ModelAdapter
>>>
>>> class MyAdapter(ModelAdapter):
...     def __init__(self, model):
...         self.model = model
...
...     def get_token_module(self) -> nn.Module:
...         return self.model.token_norm  # whatever layer exposes your
...                                        # (B, N_tokens, d_proj) tensor
...
...     def run_forward(self, batch):
...         return self.model(**batch)    # call your model however it
...                                        # actually expects to be called
...
...     def get_output_value(self, model_output):
...         return model_output["logits"]  # or however your model
...                                         # exposes the outcome

Everything else in triframe (Layer 1/2/3 functions) takes a ModelAdapter
instance, not your raw model, so this is the only integration point you
need to write.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class ModelAdapter(ABC):
    """
    Minimal contract triframe needs from your model. Implement all three
    methods; the rest of triframe only ever talks to your model through
    this interface.
    """

    @abstractmethod
    def get_token_module(self) -> nn.Module:
        """
        Return the nn.Module whose forward output IS the token tensor,
        shape (B, N_tokens, d_proj), that triframe reads from and (for
        Layer 3 patching) hooks to substitute token values.

        This is usually a normalization or projection layer sitting right
        after your model builds its token representations — whichever
        module's output tensor has one row per token, in the same order
        as your TokenRegistry.all_subgroups.
        """
        raise NotImplementedError

    @abstractmethod
    def run_forward(self, batch: Any) -> Any:
        """
        Run one forward pass on `batch` and return whatever your model's
        forward() returns, unmodified. `batch` is passed through exactly
        as given by the caller (e.g. a triframe Layer 1/2/3 function) — it
        is not inspected or reshaped by triframe, so its structure is
        entirely up to you (a single tensor, a dict of named tensors,
        whatever your model already expects).
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_value(self, model_output: Any) -> Any:
        """
        Extract the outcome value triframe should measure changes in
        (e.g. logits, a probability, a scalar score) from whatever
        run_forward() returned. Return a tensor or array triframe can
        take differences of and threshold — the exact shape/meaning is up
        to you and should stay consistent across calls.
        """
        raise NotImplementedError

    def get_attention_weights(self, model_output: Any) -> Any:
        """
        Optional. Extract per-token attention weights from whatever
        run_forward() returned, shape (B, ..., N_tokens) — any leading
        dims (heads, sequence positions) are fine, as long as the last
        dim is N_tokens matching TokenRegistry.all_subgroups order.

        Only needed if you use triframe.layer1.aggregate_attention().
        Not abstract: models without an attention mechanism can skip this.
        """
        raise NotImplementedError(
            "This model adapter does not implement get_attention_weights(). "
            "Implement it to use triframe.layer1.aggregate_attention()."
        )