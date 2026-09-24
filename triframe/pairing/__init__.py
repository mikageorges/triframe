"""
triframe/pairing/__init__.py

Strategies for selecting matched (source, target) sample pairs for a
Layer 3 intervention. An intervention needs, for each pair, a source
sample (to draw a token vector from) and a target sample (to patch that
vector into) — nearest_neighbor is one way to choose which samples get
paired; write your own for anything else (e.g. pre-computed pairs, random
pairing, stratified sampling).

Quickstart
----------
>>> from pairing import nearest_neighbor
>>>
>>> embeddings = np.column_stack([lengths, gc_content])  # any (N, k) embedding
>>> pairs = nearest_neighbor(embeddings, group_a_mask, group_b_mask, max_dist=0.2)
>>> pairs  # list of (idx_in_group_a, idx_in_group_b) index pairs
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def nearest_neighbor(
    embeddings: Any,
    group_a_mask: Any,
    group_b_mask: Any,
    max_dist: Optional[float] = None,
    n_pairs: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Greedily pair each group_a sample with its nearest unused group_b
    sample in `embeddings` space (Euclidean distance, per-dimension —
    scale/normalize embeddings yourself first if dimensions aren't
    comparable).

    embeddings   : (N, k) array — any similarity space (e.g. [length,
                   GC content], or a learned representation). Same
                   indexing as group_a_mask/group_b_mask.
    group_a_mask, group_b_mask : (N,) boolean masks selecting the two
                   pools to pair between.
    max_dist     : reject a candidate pair if its distance exceeds this
                   (None = no limit).
    n_pairs      : stop after this many pairs (None = pair every group_a
                   sample that finds a match).
    seed         : group_a iteration order is shuffled with this seed, so
                   which samples get paired first (and therefore first
                   claim on close matches) is reproducible but not biased
                   by array order.

    Returns a list of (idx_a, idx_b) GLOBAL indices into embeddings (not
    local positions within each group) — one pair per matched sample.
    """
    emb = _to_numpy(embeddings)
    mask_a = _to_numpy(group_a_mask).astype(bool)
    mask_b = _to_numpy(group_b_mask).astype(bool)

    pool_a = np.where(mask_a)[0]
    pool_b = np.where(mask_b)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(pool_a)

    used_b = set()
    pairs: List[Tuple[int, int]] = []

    for idx_a in pool_a:
        if n_pairs is not None and len(pairs) >= n_pairs:
            break

        available_b = np.array([i for i in pool_b if i not in used_b])
        if len(available_b) == 0:
            break

        dists = np.linalg.norm(emb[available_b] - emb[idx_a], axis=1)
        if max_dist is not None:
            valid = dists <= max_dist
            if not valid.any():
                continue
            available_b, dists = available_b[valid], dists[valid]

        best_b = available_b[np.argmin(dists)]
        pairs.append((int(idx_a), int(best_b)))
        used_b.add(best_b)

    return pairs