"""tests/test_tokens.py — TokenRegistry, applied to the minimal example's
3-block/9-token registry."""

import pytest

from triframe.data.tokens import TokenRegistry, BlockMeta, FeatureMeta


def test_block_structure(registry):
    assert registry.block_names == ["block_a", "block_b", "block_c"]
    assert registry.total_tokens == 9
    assert registry.block_dim("block_a") == 10
    assert registry.block_dim("block_b") == 8
    assert registry.block_dim("block_c") == 4


def test_subgroup_membership(registry):
    assert registry.block_subgroups("block_a") == [
        "strong_signal", "covariance_signal", "noise_a1", "noise_a2"]
    assert registry.token_block("synergy_a") == "block_b"
    assert registry.token_block("synergy_b") == "block_c"


def test_all_subgroups_order_matches_concatenation(registry):
    """all_subgroups' order is block-insertion-order then first-appearance
    within block — this is the order every layer indexes tokens against,
    so it must be stable and exactly this."""
    assert registry.all_subgroups == [
        "strong_signal", "covariance_signal", "noise_a1", "noise_a2",
        "redundant_with_z", "synergy_a", "noise_b1",
        "synergy_b", "noise_c1",
    ]


def test_indices_for_subgroup(registry):
    assert registry.indices_for_subgroup("strong_signal") == [0, 1, 2]
    assert registry.indices_for_subgroup("covariance_signal") == [3, 4, 5]
    assert registry.indices_for_subgroup("synergy_a") == [4, 5]  # within block_b


def test_indices_for_subgroups_batch(registry):
    result = registry.indices_for_subgroups(["strong_signal", "synergy_a"])
    assert result == {"strong_signal": [0, 1, 2], "synergy_a": [4, 5]}


def test_validate_matching_columns(registry):
    columns = {"block_a": [f.name for f in registry.block_features("block_a")]}
    registry.validate(columns=columns)  # should not raise


def test_validate_mismatch_strict_raises(registry):
    with pytest.raises(ValueError):
        registry.validate(columns={"block_a": ["wrong_col"]}, strict=True)


def test_validate_mismatch_nonstrict_warns(registry):
    with pytest.warns(UserWarning):
        registry.validate(columns={"block_a": ["wrong_col"]}, strict=False)


def test_duplicate_subgroup_across_blocks_raises():
    with pytest.raises(ValueError):
        TokenRegistry(blocks={
            "x": (BlockMeta(name="x"), [FeatureMeta(name="a", block="x", subgroup="shared")]),
            "y": (BlockMeta(name="y"), [FeatureMeta(name="b", block="y", subgroup="shared")]),
        })


def test_scale_conflict_off_by_default():
    features = [
        FeatureMeta(name="a", block="x", subgroup="tok", category="c", scale="standard"),
        FeatureMeta(name="b", block="x", subgroup="tok", category="c", scale="minmax"),
    ]
    reg = TokenRegistry(blocks={"x": (BlockMeta(name="x"), features)})
    reg.scale_strategy("x")  # should not raise


def test_scale_conflict_opted_in_raises():
    features = [
        FeatureMeta(name="a", block="x", subgroup="tok", category="c", scale="standard"),
        FeatureMeta(name="b", block="x", subgroup="tok", category="c", scale="minmax"),
    ]
    reg = TokenRegistry(blocks={"x": (BlockMeta(name="x"), features)}, check_scale_conflicts=True)
    with pytest.raises(ValueError):
        reg.scale_strategy("x")


def test_display_names(registry):
    registry.set_display_names(subgroup_names={"strong_signal": "SS"})
    assert registry.display_subgroup("strong_signal") == "SS"
    assert registry.display_subgroup("noise_a1") == "noise_a1"  # unchanged
    # reset for other tests sharing the session-scoped fixture
    registry._display_subgroups.pop("strong_signal", None)


def test_summary_runs(registry):
    s = registry.summary()
    assert "block_a" in s
    assert "Total tokens : 9" in s