import pytest
import torch
import numpy as np
from mech_interp_toolkit.activation_dict import ActivationDict
from mech_interp_toolkit.linear_probes import LinearProbe
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


@pytest.fixture
def mock_config():
    """Fixture for a mock model configuration."""
    config = Qwen3Config(num_attention_heads=4, num_hidden_layers=2, hidden_size=128)
    config._attn_implementation = "eager"
    return config


def test_linear_probe_classification(mock_config):
    """Tests the LinearProbe class for classification."""
    input_dict = ActivationDict(mock_config, positions=None)
    input_dict[(0, "z")] = torch.randn(100, 1, 128)
    target = np.random.randint(0, 2, 100)

    probe = LinearProbe(target_type="classification")
    probe.fit(input_dict, target)

    assert probe.linear_model is not None
    assert probe.weight is not None
    assert probe.bias is not None

    # Test predict
    new_input_dict = ActivationDict(mock_config, positions=None)
    new_input_dict[(0, "z")] = torch.randn(10, 1, 128)
    predictions = probe.predict(new_input_dict)
    assert predictions.shape == (10,)


def test_linear_probe_regression(mock_config):
    """Tests the LinearProbe class for regression."""
    input_dict = ActivationDict(mock_config, positions=None)
    input_dict[(0, "z")] = torch.randn(100, 1, 128)
    target = np.random.rand(100)

    probe = LinearProbe(target_type="regression")
    probe.fit(input_dict, target)

    assert probe.linear_model is not None
    assert probe.weight is not None
    assert probe.bias is not None

    # Test predict
    new_input_dict = ActivationDict(mock_config, positions=None)
    new_input_dict[(0, "z")] = torch.randn(10, 1, 128)
    predictions = probe.predict(new_input_dict)
    assert predictions.shape == (10,)


def test_linear_probe_value_error(mock_config):
    """Tests that LinearProbe raises ValueError for more than one component."""
    input_dict = ActivationDict(mock_config, positions=None)
    input_dict[(0, "z")] = torch.randn(100, 128)
    input_dict[(1, "z")] = torch.randn(100, 128)
    target = np.random.rand(100)

    probe = LinearProbe(target_type="regression")
    with pytest.raises(ValueError):
        probe.fit(input_dict, target)