"""tests/test_presets.py — task-specific convenience presets."""

import pytest
import torch

from triframe.presets import classification_success_fn, classification_accuracy_fn


def test_classification_success_fn_correct_usage():
    array = torch.tensor([0.1, 9.5])
    assert classification_success_fn(None, array, 1) is True
    assert classification_success_fn(None, array, 0) is False


def test_classification_success_fn_REGRESSION_dict_input_raises():
    # dict input used to silently read as class 0 every time; must raise now
    patched_dict = {"logits": torch.tensor([0.1, 9.5])}  # actually predicts class 1
    with pytest.raises(TypeError):
        classification_success_fn(None, patched_dict, 1)


def test_classification_accuracy_fn_basic():
    import numpy as np
    output = np.array([[0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0])
    assert classification_accuracy_fn(output, labels) == 1.0