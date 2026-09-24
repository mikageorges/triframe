"""
triframe/layer3/__init__.py

Layer 3 (causal): intervene on the model's internal token representation
and measure the resulting change in behavior.

Core primitive:
  patch_and_forward() — one interchange intervention, one direction:
    substitute a source sample's token vector(s) into a target sample's
    forward pass, return (baseline_output, patched_output).

Aggregate metrics (built from many patch_and_forward calls over pairs):
  symmetry_score() — natural indirect effect magnitude: how much the
    output moves, regardless of whether it crosses any decision boundary.
  interchange_intervention_accuracy() — stricter: fraction of pairs where
    patching succeeded per your own success_fn.

Search:
  greedy_token_search() — greedily grow a token SET, one token at a time,
    picking whichever addition most increases joint IIA. Reveals synergy
    a fixed single/block/all partition can't detect (tokens from
    different "natural" groupings can combine super-additively).
  additive_null_iia() — the independence-null IIA a token set would show
    if each token's causal contribution were independent; compare against
    the real joint IIA from greedy_token_search to test for synergy.

Robustness:
  shuffled_null_control() — repeats a patching run with SOURCE identity
    scrambled (same target/direction, source drawn at random rather than
    via your pairing strategy), to test whether IIA reflects genuine
    pair-specific structure or just class/group-conditional signal any
    same-group source would produce.

A "direction" here is just (source_group -> target_group); for more than
two groups, call these functions once per ordered pair of groups you care
about — nothing here assumes exactly two.

Quickstart
----------
>>> from triframe.layer3 import (patch_and_forward, symmetry_score,
...     interchange_intervention_accuracy, greedy_token_search)
>>> from triframe.presets import classification_success_fn
>>>
>>> baseline_out, patched_out = patch_and_forward(
...     adapter, target_batch, source_batch, token_positions=[0, 1])
>>>
>>> sym = symmetry_score(adapter, target_batches, source_batches, token_positions=[0])
>>> iia = interchange_intervention_accuracy(
...     adapter, target_batches, source_batches, token_positions=[0],
...     expected=target_expected_classes, success_fn=classification_success_fn)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..adapter import ModelAdapter
from ..data.tokens import TokenRegistry


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


SuccessFn = Callable[[Any, Any, Any], bool]
"""success_fn(baseline_output, patched_output, expected) -> bool. Yours to
define: what counts as the intervention succeeding. See
triframe.presets.classification_success_fn for a ready classification
version (did the patched prediction match the expected class)."""


# ---------------------------------------------------------------------------
# Core primitive: one intervention, one direction
# ---------------------------------------------------------------------------

def patch_and_forward(
    adapter: ModelAdapter,
    target_batch: Any,
    source_batch: Any,
    token_positions: List[int],
):
    """
    One interchange intervention: run target_batch through adapter twice —
    once unpatched (baseline), once with the token(s) at token_positions
    replaced by source_batch's token values at the same positions.

    target_batch, source_batch : whatever adapter.run_forward() expects.
    Must be alignable — same batch size, one source sample per target
    sample (build these batches yourself from a pairing strategy's output,
    e.g. triframe.pairing.nearest_neighbor).
    token_positions : indices into the token dimension (registry.
                     indices_for_subgroup / a full token set) to patch.

    Returns (baseline_output, patched_output) — both whatever
    adapter.run_forward() returns, unmodified; call
    adapter.get_output_value() on each yourself.
    """
    baseline_output = adapter.run_forward(target_batch)

    source_output_holder: Dict[str, Any] = {}

    def _capture_source_tokens(module, inp, output):
        source_output_holder["tokens"] = output

    capture_handle = adapter.get_token_module().register_forward_hook(_capture_source_tokens)
    try:
        adapter.run_forward(source_batch)
    finally:
        capture_handle.remove()
    source_tokens = source_output_holder["tokens"]

    def _patch_hook(module, inp, output):
        patched = output.clone()
        for pos in token_positions:
            patched[:, pos, :] = source_tokens[:, pos, :]
        return patched

    patch_handle = adapter.get_token_module().register_forward_hook(_patch_hook)
    try:
        patched_output = adapter.run_forward(target_batch)
    finally:
        patch_handle.remove()

    return baseline_output, patched_output


# ---------------------------------------------------------------------------
# Symmetry score (natural indirect effect magnitude)
# ---------------------------------------------------------------------------

def symmetry_score(
    adapter: ModelAdapter,
    target_batches: Sequence[Any],
    source_batches: Sequence[Any],
    token_positions: List[int],
    value_fn: Optional[Callable[[Any], Any]] = None,
) -> float:
    """
    Mean |patched_output - baseline_output| over every (target, source)
    batch pair — the natural indirect effect magnitude: how much patching
    moves the output, regardless of whether any decision boundary is
    crossed.

    target_batches, source_batches : paired sequences, same length — item
    i's target is patched with item i's source.
    value_fn : extracts the scalar/array to difference from
    adapter.get_output_value()'s return; defaults to
    adapter.get_output_value itself.
    """
    if value_fn is None:
        value_fn = adapter.get_output_value

    deltas = []
    for target_batch, source_batch in zip(target_batches, source_batches):
        baseline_out, patched_out = patch_and_forward(
            adapter, target_batch, source_batch, token_positions)
        baseline_val = _to_numpy(value_fn(baseline_out))
        patched_val = _to_numpy(value_fn(patched_out))
        deltas.append(np.abs(patched_val - baseline_val))

    return float(np.mean(np.concatenate([np.atleast_1d(d) for d in deltas])))


# ---------------------------------------------------------------------------
# Interchange Intervention Accuracy
# ---------------------------------------------------------------------------

def interchange_intervention_accuracy(
    adapter: ModelAdapter,
    target_batches: Sequence[Any],
    source_batches: Sequence[Any],
    token_positions: List[int],
    expected: Sequence[Any],
    success_fn: SuccessFn,
) -> float:
    """
    Stricter than symmetry_score: fraction of (target, source) pairs where
    the intervention succeeded per success_fn(baseline_out, patched_out,
    expected_i). success_fn is entirely yours to define — see
    triframe.presets.classification_success_fn for the ready
    "did the patched prediction match the expected class" version.

    target_batches, source_batches, expected : paired sequences, same
    length.
    """
    successes = []
    for target_batch, source_batch, exp in zip(target_batches, source_batches, expected):
        baseline_out, patched_out = patch_and_forward(
            adapter, target_batch, source_batch, token_positions)
        successes.append(success_fn(baseline_out, patched_out, exp))

    return float(np.mean(successes)) if successes else float("nan")


# ---------------------------------------------------------------------------
# Shuffled-null control
# ---------------------------------------------------------------------------

def shuffled_null_control(
    adapter: ModelAdapter,
    target_batches: Sequence[Any],
    shuffled_source_batches: Sequence[Any],
    token_positions: List[int],
    expected: Sequence[Any],
    success_fn: SuccessFn,
) -> float:
    """
    Same as interchange_intervention_accuracy(), but intended to be called
    with shuffled_source_batches drawn independently of your real pairing
    strategy (e.g. random same-group sources, not nearest-neighbor
    matches) — build shuffled_source_batches yourself (same shape
    contract as source_batches elsewhere in this module) and pass here.

    Compare the result against interchange_intervention_accuracy() on the
    real matched pairs: IIA_matched >> IIA_shuffled supports pair-specific
    structure; IIA_matched ~= IIA_shuffled indicates IIA mainly reflects
    group-conditional signal any same-group source would produce.
    """
    return interchange_intervention_accuracy(
        adapter, target_batches, shuffled_source_batches,
        token_positions, expected, success_fn,
    )


# ---------------------------------------------------------------------------
# Additive-independence null (for synergy testing)
# ---------------------------------------------------------------------------

def additive_null_iia(single_token_iias: Sequence[float]) -> float:
    """
    IIA_additive(S) = 1 - prod_{i in S} (1 - IIA({i})) — the independence-
    null IIA for a token set S, treating each token's single-token IIA as
    the probability it alone would have flipped the intervention. Compare
    against the real joint IIA for the same set: joint >> additive is
    evidence of synergy; joint ~= additive is consistent with independent,
    non-synergistic contributions.
    """
    p_none = 1.0
    for iia in single_token_iias:
        p_none *= (1.0 - iia)
    return 1.0 - p_none


# ---------------------------------------------------------------------------
# Greedy multi-token search
# ---------------------------------------------------------------------------

def greedy_token_search(
    all_tokens: List[str],
    registry: TokenRegistry,
    single_token_iia: Dict[str, float],
    iia_fn: Callable[[List[int]], float],
    max_set_size: int,
) -> List[Dict[str, Any]]:
    """
    Greedily grow a token set: start empty, at each step add whichever
    remaining token most increases joint IIA (per iia_fn, called with the
    token POSITIONS for the trial set). Stops at max_set_size or when no
    addition improves IIA.

    iia_fn : takes a list of GLOBAL TOKEN positions (0-indexed into the
    (B, N_tokens, d_proj) token dimension — i.e. registry.all_subgroups'
    order, one integer per token, NOT registry.indices_for_subgroup's raw
    per-block feature indices) and returns a joint IIA float — you supply
    this, typically a closure wrapping interchange_intervention_accuracy()
    with your target/source batches and success_fn already bound.

    Returns one dict per step: {step, added_token, current_set, set_size,
    iia_joint, iia_additive_expected, delta_vs_additive} — a search-
    derived sufficiency curve, directly comparable to a fixed single/
    block/all curve, but not constrained to any fixed token grouping.
    """
    current_set: List[str] = []
    remaining = list(all_tokens)
    rows: List[Dict[str, Any]] = []
    token_order = registry.all_subgroups

    def positions_for(token_set: List[str]) -> List[int]:
        return [token_order.index(tok) for tok in token_set]

    for step in range(1, max_set_size + 1):
        best_token, best_iia = None, None
        for candidate in remaining:
            trial_positions = positions_for(current_set + [candidate])
            trial_iia = iia_fn(trial_positions)
            if best_iia is None or trial_iia > best_iia:
                best_token, best_iia = candidate, trial_iia

        if best_token is None:
            break

        current_set.append(best_token)
        remaining.remove(best_token)

        expected = additive_null_iia([single_token_iia[t] for t in current_set])
        rows.append({
            "step": step, "added_token": best_token,
            "current_set": "+".join(current_set), "set_size": len(current_set),
            "iia_joint": best_iia, "iia_additive_expected": expected,
            "delta_vs_additive": best_iia - expected,
        })

        prev_iia = rows[-2]["iia_joint"] if len(rows) > 1 else 0.0
        if best_iia <= prev_iia:
            break  # plateaued

    return rows