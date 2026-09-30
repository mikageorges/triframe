"""
triframe/tokens.py

Domain-agnostic token registry for the triframe three-layer interpretability
framework (diagnostics -> associational -> interventional).

A TokenRegistry organizes a model's input features into named TOKENS
(called "subgroups" in some domains) grouped into BLOCKS (a block is a
logically related set of tokens sharing an input source, e.g. one feature
modality). This is the data structure every layer of triframe (attention/
ablation, Frechet/pattern alignment, activation patching/IIA) operates
against: it answers "which raw feature indices belong to token X" and
"which block does token X live in", nothing more.

Quickstart
----------
>>> from triframe.tokens import TokenRegistry, BlockMeta, FeatureMeta
>>>
>>> block_a_features = [
...     FeatureMeta(name="x1", block="a", subgroup="foo", category="continuous", scale="standard"),
...     FeatureMeta(name="x2", block="a", subgroup="foo", category="continuous", scale="standard"),
...     FeatureMeta(name="x3", block="a", subgroup="bar", category="count",      scale="none"),
... ]
>>> registry = TokenRegistry(blocks={
...     "a": (BlockMeta(name="a", source="single", description="Block A"), block_a_features),
... })
>>> registry.all_subgroups          # ["foo", "bar"]
>>> registry.total_tokens           # 2
>>> registry.token_block("foo")     # "a"
>>> registry.indices_for_subgroup("foo")  # [0, 1]

Category/scale conventions
---------------------------
FeatureMeta.category and FeatureMeta.scale are free-form strings — the
registry does not interpret them, only groups and exposes them via
scale_strategy(). If you don't have your own vocabulary, DEFAULT_CATEGORIES
below offers a minimal generic starting point (categorical / continuous /
count / binary / other) you can import and reuse; it is a convenience
preset, not a requirement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureMeta:
    """One raw input feature's metadata.

    name     : exact identifier for this feature (e.g. a CSV column name,
               a tensor index label) — must be unique within its block.
    block    : the block this feature belongs to (a key into the
               TokenRegistry's `blocks` mapping).
    subgroup : the token this feature is grouped into. Multiple features
               commonly share a subgroup (a token is usually built from
               several raw features); a subgroup belongs to exactly one
               block.
    category : free-form label for what KIND of value this is (e.g.
               "continuous", "count", "binary"). Not interpreted by the
               registry — used only for scale_strategy() grouping and your
               own downstream scaling code. See DEFAULT_CATEGORIES for an
               optional preset.
    scale    : free-form label for how this feature should be
               normalized/scaled before use (e.g. "standard", "minmax",
               "none"). Not interpreted by the registry.
    """
    name:     str
    block:    str
    subgroup: str
    category: str = "other"
    scale:    str = "none"


@dataclass(frozen=True)
class BlockMeta:
    """Metadata for one block (a logically related group of tokens sharing
    an input source).

    name        : block identifier — must match the key used in the
                  TokenRegistry's `blocks` mapping.
    source      : free-form label describing how this block's raw inputs
                  are supplied to your model (e.g. "single", "paired",
                  "genomic+processed" — whatever distinction matters for
                  your architecture's input-loading code). The registry
                  stores this but does not interpret it; it exists so your
                  own model/dataset code has one place to look up "how do
                  I feed block X's inputs."
    description : free-form human-readable description, used in summary().
    """
    name:        str
    source:      str = "single"
    description: str = ""


# Optional preset — import and reuse if you don't have your own category
# vocabulary. Purely a convenience default; FeatureMeta.category accepts
# any string.
DEFAULT_CATEGORIES = ("categorical", "continuous", "count", "binary", "other")


# ---------------------------------------------------------------------------
# TokenRegistry — the public API
# ---------------------------------------------------------------------------

class TokenRegistry:
    """
    Indexed registry of tokens (subgroups) organized into blocks.

    The registry is the single source of truth, for any model built on
    triframe's three-layer framework, for:
      - which raw feature indices make up each token
      - which block each token belongs to
      - the concatenation order tokens appear in when fed to a model
        (block insertion order, then subgroup first-appearance order
        within each block) — this order must match how your model
        actually concatenates/stacks its token tensor, since every
        triframe layer (attention aggregation, ablation, patching)
        indexes tokens positionally against this order.

    Construction is fully data-driven: you supply a mapping of
    block_name -> (BlockMeta, [FeatureMeta, ...]); the registry does not
    define or hardcode any blocks, tokens, or feature lists itself. See
    beta_vae_lnclassifier's data/feature_registry.py for a full worked
    example of building real feature lists and wiring them into this
    constructor.

    Parameters
    ----------
    blocks : dict mapping block_name -> (BlockMeta, list of FeatureMeta).
             Insertion order of this dict is preserved and defines the
             block order used throughout (all_subgroups, total_tokens,
             token concatenation order).
    check_scale_conflicts : if True, scale_strategy() raises ValueError
             when a category maps to more than one scale within the same
             block (a common LNC-repo invariant: every feature of a given
             category should be scaled the same way). Off by default,
             since this is a useful convention for some domains but an
             unnecessary constraint for others.
    """

    def __init__(
        self,
        blocks: Dict[str, Tuple[BlockMeta, List[FeatureMeta]]],
        check_scale_conflicts: bool = False,
    ) -> None:
        self._block_meta: Dict[str, BlockMeta] = {}
        self._blocks: Dict[str, List[FeatureMeta]] = {}
        self._check_scale_conflicts = check_scale_conflicts

        for block_name, (meta, features) in blocks.items():
            if meta.name != block_name:
                raise ValueError(
                    f"BlockMeta.name ('{meta.name}') must match the dict key "
                    f"('{block_name}') it's registered under"
                )
            for f in features:
                if f.block != block_name:
                    raise ValueError(
                        f"FeatureMeta '{f.name}' has block='{f.block}' but was "
                        f"placed under registry block '{block_name}'"
                    )
            self._block_meta[block_name] = meta
            self._blocks[block_name] = list(features)

        # Name -> index maps, per block (index within that block's feature list)
        self._name_to_idx: Dict[str, Dict[str, int]] = {
            block_name: {m.name: i for i, m in enumerate(features)}
            for block_name, features in self._blocks.items()
        }

        # Subgroup -> block lookup (built once; a subgroup must live in
        # exactly one block)
        self._sg_to_block: Dict[str, str] = {}
        for block_name, features in self._blocks.items():
            for m in features:
                if m.subgroup in self._sg_to_block:
                    if self._sg_to_block[m.subgroup] != block_name:
                        raise ValueError(
                            f"Subgroup '{m.subgroup}' appears in multiple "
                            f"blocks: {self._sg_to_block[m.subgroup]} and "
                            f"{block_name}. A subgroup (token) must belong "
                            f"to exactly one block."
                        )
                self._sg_to_block[m.subgroup] = block_name

    # ------------------------------------------------------------------
    # Block-level API
    # ------------------------------------------------------------------

    @property
    def block_names(self) -> List[str]:
        """Ordered list of block names — matches token concatenation order."""
        return list(self._blocks.keys())

    def block_source(self, block_name: str) -> str:
        """Return the free-form 'source' label for a block."""
        return self._block_meta[block_name].source

    def block_features(self, block_name: str) -> List[FeatureMeta]:
        """Return the ordered FeatureMeta list for a block."""
        return self._blocks[block_name]

    def block_dim(self, block_name: str) -> int:
        """Return the raw feature dimension (number of FeatureMeta entries)
        for a block."""
        return len(self._blocks[block_name])

    def block_subgroups(self, block_name: str) -> List[str]:
        """Return the ordered list of subgroup (token) names for a block,
        in first-appearance order."""
        seen: Dict[str, None] = {}
        for m in self._blocks[block_name]:
            seen[m.subgroup] = None
        return list(seen)

    def token_block(self, subgroup: str) -> str:
        """Return the block name that contains a given subgroup (token)."""
        if subgroup not in self._sg_to_block:
            raise KeyError(
                f"Subgroup '{subgroup}' not found in any block. "
                f"Known subgroups: {list(self._sg_to_block)}"
            )
        return self._sg_to_block[subgroup]

    @property
    def all_subgroups(self) -> List[str]:
        """All subgroup (token) names, in the order tokens are concatenated:
        block insertion order, then first-appearance order within each
        block. This is the order every triframe layer indexes against."""
        result: Dict[str, None] = {}
        for block_name in self.block_names:
            for sg in self.block_subgroups(block_name):
                result[sg] = None
        return list(result)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens (subgroups) across all blocks."""
        return len(self.all_subgroups)

    def indices_for_subgroup(self, subgroup: str) -> List[int]:
        """Return the raw feature indices for a subgroup, indexed within
        its own block's feature vector (not the global token index)."""
        block = self.token_block(subgroup)
        features = self._blocks[block]
        return [i for i, m in enumerate(features) if m.subgroup == subgroup]

    def indices_for_subgroups(self, subgroups: List[str]) -> Dict[str, List[int]]:
        """Convenience batch form of indices_for_subgroup, for callers
        (e.g. a joint multi-token patching search) that need indices for
        several subgroups at once. Returns {subgroup: indices}, grouped
        per-block since indices are only meaningful within their own
        block's feature vector."""
        return {sg: self.indices_for_subgroup(sg) for sg in subgroups}

    def scale_strategy(self, block_name: str) -> Dict[str, str]:
        """Return {category: scale} for a block, i.e. which scale each
        category of feature should use. If check_scale_conflicts was set
        at construction time, raises ValueError when a category maps to
        more than one distinct scale within the block; otherwise the last
        feature's scale wins silently for a given category (matches plain
        dict-overwrite semantics)."""
        result: Dict[str, str] = {}
        for m in self._blocks[block_name]:
            if (self._check_scale_conflicts
                    and m.category in result
                    and result[m.category] != m.scale):
                raise ValueError(
                    f"Block '{block_name}' category '{m.category}' has "
                    f"conflicting scale strategies: {result[m.category]} vs "
                    f"{m.scale} (feature: {m.name}). Pass "
                    f"check_scale_conflicts=False to allow this."
                )
            result[m.category] = m.scale
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        columns: Optional[Dict[str, List[str]]] = None,
        strict: bool = True,
    ) -> None:
        """
        Validate the registry against actual data column lists, e.g. the
        real column order of a CSV or tensor your data loader produces.

        Parameters
        ----------
        columns : {block_name: [column_name, ...]} for the blocks you want
                  checked. Omit a block to skip its validation (e.g. if
                  you only have the real columns for one block on hand).
        strict  : if True, raises ValueError on any mismatch; if False,
                  emits a warning and continues.
        """
        columns = columns or {}
        errors: List[str] = []

        for block_name, expected_columns in columns.items():
            if block_name not in self._blocks:
                errors.append(f"'{block_name}': not a known block in this "
                              f"registry (known: {self.block_names})")
                continue

            registry_names = [m.name for m in self._blocks[block_name]]
            if registry_names == expected_columns:
                continue

            missing = set(registry_names) - set(expected_columns)
            extra = set(expected_columns) - set(registry_names)
            order_only = not missing and not extra

            if missing:
                errors.append(
                    f"{block_name}: in registry but not in data: {sorted(missing)}")
            if extra:
                errors.append(
                    f"{block_name}: in data but not in registry: {sorted(extra)}")
            if order_only:
                errors.append(
                    f"{block_name}: feature ORDER differs — indices will be wrong")

        if errors:
            msg = "\n".join(errors)
            if strict:
                raise ValueError(f"TokenRegistry validation failed:\n{msg}")
            else:
                warnings.warn(f"TokenRegistry validation warnings:\n{msg}")
        else:
            validated = list(columns.keys())
            if validated:
                dims = {b: self.block_dim(b) for b in validated}
                print(f"TokenRegistry validation passed: {dims}")

    # ------------------------------------------------------------------
    # Display names (optional)
    # ------------------------------------------------------------------

    def set_display_names(
        self,
        subgroup_names: Optional[Dict[str, str]] = None,
        block_names: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Register optional cosmetic display names for subgroups and/or
        blocks (e.g. an internal key that doesn't match the label you want
        in plots/reports — see beta_vae_lnclassifier's TE_* -> REP_*
        relabeling for a real example). Entries not covered by these maps
        display as their internal name unchanged. Safe to call multiple
        times; later calls merge into (don't replace) prior display names.
        """
        if not hasattr(self, "_display_subgroups"):
            self._display_subgroups: Dict[str, str] = {}
            self._display_blocks: Dict[str, str] = {}
        if subgroup_names:
            self._display_subgroups.update(subgroup_names)
        if block_names:
            self._display_blocks.update(block_names)

    def display_subgroup(self, subgroup: str) -> str:
        """Return the display label for a subgroup, or the subgroup's own
        name if no display name was registered."""
        return getattr(self, "_display_subgroups", {}).get(subgroup, subgroup)

    def display_block(self, block_name: str) -> str:
        """Return the display label for a block, or the block's own name
        if no display name was registered."""
        return getattr(self, "_display_blocks", {}).get(block_name, block_name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["TokenRegistry summary", "=" * 65]
        for block_name in self.block_names:
            meta = self._block_meta[block_name]
            dim = self.block_dim(block_name)
            sgs = self.block_subgroups(block_name)
            lines.append(f"\n{block_name} block — {dim} features  "
                         f"[source={meta.source}]")
            if meta.description:
                lines.append(f"  {meta.description}")
            lines.append(f"  Subgroups: {sgs}")
            for sg in sgs:
                idx = self.indices_for_subgroup(sg)
                span = f"{idx[0]}-{idx[-1]}" if idx else "none"
                lines.append(f"    {sg:20s}: {len(idx):3d} features  "
                             f"(indices {span})")
            cat_counts: Dict[str, int] = {}
            for m in self.block_features(block_name):
                cat_counts[m.category] = cat_counts.get(m.category, 0) + 1
            lines.append(f"  Categories: {cat_counts}")

        block_counts = " + ".join(
            str(len(self.block_subgroups(b))) for b in self.block_names)
        lines.append(f"\nTotal tokens : {self.total_tokens}  "
                     f"({block_counts} = {' + '.join(self.block_names)})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"TokenRegistry(blocks={self.block_names}, "
                f"total_tokens={self.total_tokens})")