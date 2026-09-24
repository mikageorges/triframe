"""
triframe/plotting/__init__.py

Generic chart primitives for Layer 1/2/3 output. Block-colored (any
number of blocks — colors cycle through a palette), labels/titles as
parameters, no domain-specific defaults.

ranked_bar()        — single-series bar, tokens ranked by value, optional
                       log scale and top-N value annotations. Fits
                       symmetry_score, IIA, redundancy, mean_abs_d, etc.
stacked_bar()        — two-component stacked bar (e.g. Frechet distance's
                       mean_term + cov_term).
grouped_bar()        — N-series grouped bar per token (e.g. ablation's
                       token_zero vs feature_zero, or raw vs residual
                       pattern alignment).
sufficiency_curve()  — line + error band over increasing set size vs. an
                       additive-null baseline (greedy_token_search output).
concentration_panels() — 3-panel elbow / Gini / participation-ratio bars,
                       side by side.

Quickstart
----------
>>> from triframe.plotting import ranked_bar, stacked_bar
>>>
>>> fig, ax = ranked_bar(
...     labels=fd_df["token"], values=fd_df["frechet_dist"],
...     blocks=fd_df["block"], ylabel="Frechet distance", log_scale=True)
>>>
>>> fig, ax = stacked_bar(
...     labels=fd_df["token"], series={"mean": fd_df["mean_term"],
...     "cov": fd_df["cov_term"]}, blocks=fd_df["block"],
...     ylabel="Frechet distance")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


_DEFAULT_PALETTE = [
    "#4A90D9", "#F4B942", "#9B7FD4", "#E8743B", "#5CB85C",
    "#D9534F", "#5BC0DE", "#8E5EA2",
]


def _block_colors(blocks: Sequence[str]) -> Dict[str, str]:
    """Assign one color per unique block, in first-appearance order,
    cycling through _DEFAULT_PALETTE if there are more blocks than colors."""
    seen = list(dict.fromkeys(blocks))
    return {b: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)] for i, b in enumerate(seen)}


def _legend_handles(blocks: Sequence[str], color_map: Dict[str, str]):
    import matplotlib.patches as mpatches
    seen = list(dict.fromkeys(blocks))
    return [mpatches.Patch(color=color_map[b], label=b) for b in seen]


# ---------------------------------------------------------------------------
# Ranked bar (single series)
# ---------------------------------------------------------------------------

def ranked_bar(
    labels: Sequence[str],
    values: Sequence[float],
    blocks: Optional[Sequence[str]] = None,
    ylabel: str = "",
    title: str = "",
    log_scale: bool = False,
    annotate_top_n: int = 0,
    ascending: bool = False,
    figsize: Optional[tuple] = None,
):
    """
    Single-series bar chart, tokens ranked by value. blocks (optional)
    colors each bar and its x-tick label by block, with a legend.
    annotate_top_n prints the exact value above the top N bars.
    """
    import matplotlib.pyplot as plt

    order = np.argsort(values)
    if not ascending:
        order = order[::-1]
    labels_ord = [labels[i] for i in order]
    values_ord = [values[i] for i in order]
    blocks_ord = [blocks[i] for i in order] if blocks is not None else None

    fig, ax = plt.subplots(figsize=figsize or (0.55 * len(labels) + 2, 6))
    x = np.arange(len(labels_ord))

    if blocks_ord is not None:
        color_map = _block_colors(blocks)
        colors = [color_map[b] for b in blocks_ord]
        ax.bar(x, values_ord, color=colors, edgecolor="black", linewidth=0.5)
        ax.legend(handles=_legend_handles(blocks, color_map), fontsize=9, frameon=True)
        for tick, b in zip(ax.get_xticklabels(), blocks_ord):
            pass  # tick color set after set_xticklabels below
    else:
        ax.bar(x, values_ord, color="#4A90D9", edgecolor="black", linewidth=0.5)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ord, rotation=45, ha="right", fontsize=9)
    if blocks_ord is not None:
        color_map = _block_colors(blocks)
        for tick, b in zip(ax.get_xticklabels(), blocks_ord):
            tick.set_color(color_map[b])

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate_top_n > 0:
        for i in range(min(annotate_top_n, len(values_ord))):
            ax.text(i, values_ord[i], f"{values_ord[i]:.3f}",
                   ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Stacked bar (decomposed value, e.g. Frechet mean+cov terms)
# ---------------------------------------------------------------------------

def stacked_bar(
    labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    blocks: Optional[Sequence[str]] = None,
    ylabel: str = "",
    title: str = "",
    order_by: Optional[str] = None,
    figsize: Optional[tuple] = None,
):
    """
    Stacked bar chart, one bar per token, components stacked from `series`
    (a dict of {component_name: values}, drawn bottom-to-top in dict
    order). order_by (optional) sorts tokens by the sum of that
    component's values descending; defaults to sorting by total stack.
    """
    import matplotlib.pyplot as plt

    total = np.sum([np.asarray(v) for v in series.values()], axis=0)
    sort_key = np.asarray(series[order_by]) if order_by else total
    order = np.argsort(sort_key)[::-1]

    labels_ord = [labels[i] for i in order]
    blocks_ord = [blocks[i] for i in order] if blocks is not None else None

    fig, ax = plt.subplots(figsize=figsize or (0.55 * len(labels) + 2, 5))
    x = np.arange(len(labels_ord))

    bottom = np.zeros(len(labels_ord))
    for name, values in series.items():
        values_ord = np.asarray(values)[order]
        ax.bar(x, values_ord, bottom=bottom, label=name, edgecolor="none")
        bottom += values_ord

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ord, rotation=45, ha="right", fontsize=9)
    if blocks_ord is not None:
        color_map = _block_colors(blocks)
        for tick, b in zip(ax.get_xticklabels(), blocks_ord):
            tick.set_color(color_map[b])

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Grouped bar (N series side by side per token)
# ---------------------------------------------------------------------------

def grouped_bar(
    labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    blocks: Optional[Sequence[str]] = None,
    ylabel: str = "",
    title: str = "",
    order_by: Optional[str] = None,
    figsize: Optional[tuple] = None,
):
    """
    Grouped bar chart, N series drawn side by side per token (e.g. two
    ablation modes, or raw vs residual alignment). order_by (optional)
    sorts tokens by that series' values descending; defaults to the mean
    across series.
    """
    import matplotlib.pyplot as plt

    n_series = len(series)
    width = 0.8 / n_series

    mean_vals = np.mean([np.asarray(v) for v in series.values()], axis=0)
    sort_key = np.asarray(series[order_by]) if order_by else mean_vals
    order = np.argsort(sort_key)[::-1]

    labels_ord = [labels[i] for i in order]
    blocks_ord = [blocks[i] for i in order] if blocks is not None else None

    fig, ax = plt.subplots(figsize=figsize or (0.6 * len(labels) + 2, 6))
    x = np.arange(len(labels_ord))

    for i, (name, values) in enumerate(series.items()):
        values_ord = np.asarray(values)[order]
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, values_ord, width, label=name,
              edgecolor="black", linewidth=0.5, alpha=0.9)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_ord, rotation=45, ha="right", fontsize=9)
    if blocks_ord is not None:
        color_map = _block_colors(blocks)
        for tick, b in zip(ax.get_xticklabels(), blocks_ord):
            tick.set_color(color_map[b])

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Sufficiency curve (greedy_token_search output vs additive null)
# ---------------------------------------------------------------------------

def sufficiency_curve(
    set_sizes: Sequence[int],
    joint_values: Sequence[float],
    additive_values: Sequence[float],
    joint_std: Optional[Sequence[float]] = None,
    additive_std: Optional[Sequence[float]] = None,
    ylabel: str = "IIA",
    title: str = "",
    figsize: Optional[tuple] = None,
):
    """
    Line plot of joint (searched) value vs. the additive-independence
    null, over increasing token-set size — the standard output shape of
    triframe.layer3.greedy_token_search(), optionally with error bars
    (e.g. cross-fold std).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize or (6, 4.5))

    if joint_std is not None:
        ax.errorbar(set_sizes, joint_values, yerr=joint_std, marker="o",
                   markersize=5, linewidth=1.6, capsize=3, label="Joint (searched)")
    else:
        ax.plot(set_sizes, joint_values, marker="o", markersize=5,
               linewidth=1.6, label="Joint (searched)")

    if additive_std is not None:
        ax.errorbar(set_sizes, additive_values, yerr=additive_std, marker="s",
                   markersize=5, linewidth=1.2, linestyle="--", capsize=3,
                   label="Additive null")
    else:
        ax.plot(set_sizes, additive_values, marker="s", markersize=5,
               linewidth=1.2, linestyle="--", label="Additive null")

    joint_arr = np.asarray(joint_values)
    additive_arr = np.asarray(additive_values)
    ax.fill_between(set_sizes, joint_arr, additive_arr,
                    where=(joint_arr >= additive_arr), alpha=0.12)

    ax.set_xlabel("Token set size", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(list(set_sizes))
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Concentration panels (elbow / Gini / participation ratio)
# ---------------------------------------------------------------------------

def concentration_panels(
    labels: Sequence[str],
    elbow_pct: Sequence[float],
    gini: Sequence[float],
    participation_pct: Sequence[float],
    blocks: Optional[Sequence[str]] = None,
    figsize: Optional[tuple] = None,
):
    """
    3-panel horizontal bar chart: elbow concentration %, Gini coefficient,
    participation ratio %, all ordered by elbow_pct ascending (most
    concentrated first) — the standard triframe.layer2.concentration_by_token()
    output shape.
    """
    import matplotlib.pyplot as plt

    order = np.argsort(elbow_pct)
    labels_ord = [labels[i] for i in order]
    blocks_ord = [blocks[i] for i in order] if blocks is not None else None
    color_map = _block_colors(blocks) if blocks is not None else None
    colors = [color_map[b] for b in blocks_ord] if blocks_ord is not None else "#4A90D9"

    fig, axes = plt.subplots(1, 3, figsize=figsize or (18, 6))
    panels = [
        (axes[0], np.asarray(elbow_pct)[order], "% features for threshold"),
        (axes[1], np.asarray(gini)[order], "Gini coefficient"),
        (axes[2], np.asarray(participation_pct)[order], "Participation ratio (%)"),
    ]
    for ax, values, xlabel in panels:
        ax.barh(labels_ord, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if blocks is not None:
        fig.legend(handles=_legend_handles(blocks, color_map), fontsize=9,
                  loc="lower center", ncol=len(set(blocks)), bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    return fig, axes