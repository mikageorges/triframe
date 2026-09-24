"""
triframe/examples/beta_vae_subgroup_minimal/synthetic_data.py

Generates synthetic per-token feature data with known structure,
so every triframe layer has something real to find (not pure noise) and
tests can assert against ground truth rather than just "runs without error".

Ground truth baked in (see generate() docstring for exact values):
  - one token ("strong_signal") with a large, easy class-separating mean
    shift — Layer 2's Frechet distance and Layer 1's ablation/attention
    should all rank it highest.
  - one token ("covariance_signal") with NO mean shift but a real
    covariance-structure difference between classes — visible to Frechet's
    cov_term and pattern_alignment's residual variant, invisible to a
    naive mean-only comparison.
  - one token ("redundant_with_z") whose content is a linear function of
    the "sequence" representation z — high representation_redundancy(tok, z).
  - two tokens ("synergy_a", "synergy_b") individually weak but, when
    patched together, jointly sufficient to flip a downstream decision —
    exercises Layer 3's greedy_token_search synergy detection.
  - several "noise" tokens with no relationship to anything — the null
    case every metric should score near zero.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ...data.tokens import TokenRegistry, BlockMeta, FeatureMeta


def build_registry() -> TokenRegistry:
    """
    3 blocks, 9 tokens total, so block-boundary logic in plotting/registry
    gets real multi-block exercise.

    block_a (4 tokens): strong_signal, covariance_signal, noise_a1, noise_a2
    block_b (3 tokens): redundant_with_z, synergy_a, noise_b1
    block_c (2 tokens): synergy_b, noise_c1
    """
    def feats(block, subgroups_dims):
        out, i = [], 0
        for sg, dim in subgroups_dims:
            for _ in range(dim):
                out.append(FeatureMeta(name=f"{sg}_f{i}", block=block, subgroup=sg,
                                       category="continuous", scale="standard"))
                i += 1
        return out

    block_a = feats("block_a", [
        ("strong_signal", 3), ("covariance_signal", 3),
        ("noise_a1", 2), ("noise_a2", 2),
    ])
    block_b = feats("block_b", [
        ("redundant_with_z", 4), ("synergy_a", 2), ("noise_b1", 2),
    ])
    block_c = feats("block_c", [
        ("synergy_b", 2), ("noise_c1", 2),
    ])

    return TokenRegistry(blocks={
        "block_a": (BlockMeta(name="block_a", source="single",
                              description="Strong mean-shift and covariance-shift signal"), block_a),
        "block_b": (BlockMeta(name="block_b", source="single",
                              description="z-redundant token and one synergy partner"), block_b),
        "block_c": (BlockMeta(name="block_c", source="single",
                              description="Second synergy partner"), block_c),
    })


def generate(
    n: int = 2000,
    latent_dim: int = 6,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data matching build_registry()'s token layout.

    Returns dict with:
      labels       : (n,) int, 0/1, balanced
      z            : (n, latent_dim) — a synthetic "sequence encoder" latent
      block_a, block_b, block_c : (n, block_dim) raw feature matrices,
                     column order matching build_registry()'s FeatureMeta order
      synergy_target : (n,) int — a SEPARATE binary target only flippable by
                     jointly patching synergy_a + synergy_b (see below)

    Ground truth (exact, so tests can assert real numbers, not just signs):
      strong_signal      : mean shift of ~4 SD between classes, low noise
                            -> Frechet dist and Cohen's d should be large
      covariance_signal   : zero mean shift, 3x variance in class 1 vs class 0
                            -> Frechet mean_term ~0, cov_term large
      redundant_with_z    : = z @ random_matrix + small noise
                            -> representation_redundancy(token, z) ~high R^2
      synergy_a, synergy_b: each individually near-orthogonal to
                            synergy_target (weak univariate signal), but
                            synergy_target = XOR-like function of both
                            jointly -> single-token patching should show
                            low IIA for each; patching BOTH together should
                            show high IIA, and greedy_token_search should
                            discover the pair with a positive delta_vs_additive
      noise_*              : pure Gaussian noise, unrelated to labels/z/target
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)

    z = rng.normal(0, 1, size=(n, latent_dim))

    # strong_signal: 3 dims, mean shift ~4 SD, low noise
    strong_signal = rng.normal(0, 0.3, size=(n, 3))
    strong_signal += (labels[:, None] - 0.5) * 8.0  # class 0 -> -4, class 1 -> +4

    # covariance_signal: 3 dims, zero mean shift, variance differs by class
    covariance_signal = rng.normal(0, 1, size=(n, 3))
    class1_mask = labels == 1
    covariance_signal[class1_mask] *= 3.0  # same mean (0), 3x std in class 1

    # redundant_with_z: linear function of z + small noise, unrelated to label
    z_to_token = rng.normal(0, 1, size=(latent_dim, 4))
    redundant_with_z = z @ z_to_token + rng.normal(0, 0.2, size=(n, 4))

    # synergy_a / synergy_b: each is a single noisy bit; synergy_target is
    # their XOR — individually near-uninformative about synergy_target,
    # jointly fully determines it.
    bit_a = rng.integers(0, 2, size=n)
    bit_b = rng.integers(0, 2, size=n)
    synergy_target = np.bitwise_xor(bit_a, bit_b)

    synergy_a = rng.normal(0, 0.3, size=(n, 2))
    synergy_a += (bit_a[:, None] - 0.5) * 2.0
    synergy_b = rng.normal(0, 0.3, size=(n, 2))
    synergy_b += (bit_b[:, None] - 0.5) * 2.0

    # noise tokens: pure Gaussian, unrelated to anything
    noise_a1 = rng.normal(0, 1, size=(n, 2))
    noise_a2 = rng.normal(0, 1, size=(n, 2))
    noise_b1 = rng.normal(0, 1, size=(n, 2))
    noise_c1 = rng.normal(0, 1, size=(n, 2))

    block_a = np.hstack([strong_signal, covariance_signal, noise_a1, noise_a2])
    block_b = np.hstack([redundant_with_z, synergy_a, noise_b1])
    block_c = np.hstack([synergy_b, noise_c1])

    return dict(
        labels=labels, z=z,
        block_a=block_a, block_b=block_b, block_c=block_c,
        synergy_target=synergy_target,
    )