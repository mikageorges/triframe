"""tests/test_plotting.py — generic plotting primitives, applied to real
Layer 2/3 output on the minimal BVAE example. Rendering-only checks (figure
produced, no exception) — these are not pixel/visual regression tests."""

import matplotlib
matplotlib.use("Agg")

from triframe.plotting import (
    ranked_bar, stacked_bar, grouped_bar, sufficiency_curve, concentration_panels,
)
from triframe.layer2 import frechet_by_token, concentration_by_token


def test_ranked_bar_on_real_frechet_output(block_data, data, registry):
    fd = frechet_by_token(registry, block_data, data["labels"])
    fig, ax = ranked_bar(
        labels=fd["token"].tolist(), values=fd["frechet_dist"].tolist(),
        blocks=fd["block"].tolist(), ylabel="Frechet distance",
        log_scale=True, annotate_top_n=2)
    assert fig is not None


def test_stacked_bar_on_real_frechet_output(block_data, data, registry):
    fd = frechet_by_token(registry, block_data, data["labels"])
    fig, ax = stacked_bar(
        labels=fd["token"].tolist(),
        series={"mean": fd["mean_term"].tolist(), "cov": fd["cov_term"].tolist()},
        blocks=fd["block"].tolist(), ylabel="Frechet distance")
    assert fig is not None


def test_grouped_bar_runs():
    fig, ax = grouped_bar(
        labels=["a", "b", "c"],
        series={"mode1": [0.1, 0.2, 0.05], "mode2": [0.15, 0.1, 0.02]},
        blocks=["x", "x", "y"], ylabel="drop")
    assert fig is not None


def test_sufficiency_curve_runs():
    fig, ax = sufficiency_curve(
        set_sizes=[1, 2, 3], joint_values=[0.1, 0.6, 0.62],
        additive_values=[0.1, 0.19, 0.23], joint_std=[0.01, 0.05, 0.04])
    assert fig is not None


def test_concentration_panels_on_real_output(block_data, data, registry):
    conc = concentration_by_token(registry, block_data, data["labels"])
    fig, axes = concentration_panels(
        labels=conc["token"].tolist(),
        elbow_pct=conc["pct_features_for_threshold"].tolist(),
        gini=conc["gini"].tolist(),
        participation_pct=(conc["participation_ratio"] / conc["n_features"] * 100).tolist(),
        blocks=conc["block"].tolist())
    assert fig is not None
    assert len(axes) == 3


def test_ranked_bar_palette_cycles_beyond_default_size():
    labels = [f"tok{i}" for i in range(12)]
    values = list(range(12))
    blocks = [f"block{i}" for i in range(12)]  # more blocks than default palette size
    fig, ax = ranked_bar(labels, values, blocks=blocks)
    assert fig is not None