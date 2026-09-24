"""tests/test_pairing.py — pairing strategies."""

import numpy as np

from triframe.pairing import nearest_neighbor


def test_nearest_neighbor_pairs_from_correct_groups():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    assert all(group_a[a] and group_b[b] for a, b in pairs)


def test_nearest_neighbor_no_reuse():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    b_indices = [b for _, b in pairs]
    assert len(b_indices) == len(set(b_indices))


def test_nearest_neighbor_respects_max_dist():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    for a, b in pairs:
        assert np.linalg.norm(embeddings[a] - embeddings[b]) <= 500


def test_nearest_neighbor_n_pairs_limit():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs = nearest_neighbor(embeddings, group_a, group_b, n_pairs=5)
    assert len(pairs) == 5


def test_nearest_neighbor_reproducible():
    rng = np.random.default_rng(0)
    n = 100
    embeddings = rng.uniform(0, 1000, (n, 2))
    group_a = np.zeros(n, dtype=bool); group_a[:50] = True
    group_b = ~group_a

    pairs_1 = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    pairs_2 = nearest_neighbor(embeddings, group_a, group_b, max_dist=500)
    assert pairs_1 == pairs_2