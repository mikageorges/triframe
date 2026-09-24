"""
triframe/presets.py

Ready-made scoring/success functions for common task types. The core
framework (layer1.py, layer3.py) is task-agnostic — every scoring
function is something you supply. This module holds convenience
implementations for common cases so you don't have to write the obvious
ones yourself; import what fits, or ignore this module and write your own.

Classification
---------------
classification_accuracy_fn : Layer 1 score_fn — exact-match accuracy on
    argmax(output_value) vs. integer class labels.
classification_success_fn : Layer 3 success_fn — did patching flip the
    predicted class to match the expected (source) class.

Nothing here is imported by layer1.py/layer3.py — these are opt-in.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def classification_accuracy_fn(output_value: Any, labels: Any) -> float:
    """
    Layer 1 score_fn preset. Assumes output_value is (B, n_classes)
    per-class scores/logits and labels is (B,) integer class indices.
    Returns exact-match accuracy over the batch, mean over the batch —
    HIGHER = BETTER, as run_ablation requires.
    """
    output_arr = _to_numpy(output_value)
    labels_arr = _to_numpy(labels)
    preds = output_arr.argmax(axis=-1)
    return float((preds == labels_arr).mean())


def classification_success_fn(
    baseline_output: Any,
    patched_output: Any,
    expected_class: Any,
) -> bool:
    """
    Layer 3 success_fn preset. Assumes both outputs are (n_classes,)
    per-class scores/logits for a SINGLE sample (Layer 3 patching is
    typically evaluated per-pair, not batched) and expected_class is a
    scalar integer class index. Returns True if the patched prediction
    matches expected_class — i.e. the intervention successfully flipped
    (or preserved) the prediction toward the expected class.

    baseline_output is accepted but unused here — included so this
    matches the general success_fn(baseline, patched, expected) -> bool
    signature layer3.py expects; some custom success_fn implementations
    will want to compare against the baseline (e.g. "did the prediction
    CHANGE" rather than "does it MATCH X"), but a simple classification
    flip check only needs the patched output and the target.
    """
    patched_arr = _to_numpy(patched_output)
    expected_idx = int(_to_numpy(expected_class))
    return int(patched_arr.argmax(axis=-1)) == expected_idx