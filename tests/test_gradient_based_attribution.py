import pytest
import torch
from unittest.mock import MagicMock, patch
from mech_interp_toolkit.gradient_based_attribution import (
    eap_integrated_gradients,
    simple_integrated_gradients,
)
from mech_interp_toolkit.activation_dict import ActivationDict
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


@pytest.fixture
def mock_config():
    """Fixture for a mock model configuration."""
    config = Qwen3Config(num_attention_heads=4, num_hidden_layers=2, hidden_size=128)
    config._attn_implementation = "eager"
    return config


def test_eap_integrated_gradients(mock_config):
    """Tests the eap_integrated_gradients function."""
    model = MagicMock()
    model.model.config = mock_config
    inputs = {"input_ids": torch.randint(0, 100, (1, 5))}
    baseline_embeddings = ActivationDict(mock_config, positions=slice(None))
    baseline_embeddings[(0, "layer_in")] = torch.zeros(1, 5, 128)

    with patch(
        "mech_interp_toolkit.gradient_based_attribution.get_activations"
    ) as mock_get_activations, patch(
        "mech_interp_toolkit.gradient_based_attribution.patch_activations"
    ) as mock_patch_activations:
        input_embeddings = ActivationDict(mock_config, positions=slice(None))
        input_embeddings[(0, "layer_in")] = torch.ones(1, 5, 128)
        mock_get_activations.return_value = (input_embeddings.cpu(), None, None)

        grads = ActivationDict(mock_config, positions=slice(None))
        grads[(0, "layer_in")] = torch.ones(1, 5, 128)
        mock_patch_activations.return_value = (None, grads.cpu(), None)

        result = eap_integrated_gradients(
            model, inputs, baseline_embeddings.cpu(), steps=10
        )

        assert isinstance(result, ActivationDict)
        assert (0, "layer_in") in result
        assert result[(0, "layer_in")].shape == (1, 5)
        assert torch.allclose(result[(0, "layer_in")], torch.ones(1, 5) * 128)


def test_simple_integrated_gradients(mock_config):
    """Tests the simple_integrated_gradients function."""
    model = MagicMock()
    model.model.config = mock_config
    inputs = {"input_ids": torch.randint(0, 100, (1, 5))}
    baseline_embeddings = ActivationDict(mock_config, positions=slice(None))
    baseline_embeddings[(0, "layer_in")] = torch.zeros(1, 5, 128)

    with patch(
        "mech_interp_toolkit.gradient_based_attribution.get_activations"
    ) as mock_get_activations, patch(
        "mech_interp_toolkit.gradient_based_attribution.patch_activations"
    ) as mock_patch_activations:
        input_embeddings = ActivationDict(mock_config, positions=slice(None))
        input_embeddings[(0, "layer_in")] = torch.ones(1, 5, 128)
        mock_get_activations.return_value = (input_embeddings.cpu(), None, None)

        grads = ActivationDict(mock_config, positions=slice(None))
        grads[(0, "layer_in")] = torch.ones(1, 5, 128)
        mock_patch_activations.return_value = (None, grads.cpu(), None)

        result = simple_integrated_gradients(
            model, inputs, baseline_embeddings.cpu(), steps=10
        )

        assert isinstance(result, ActivationDict)
        assert (0, "layer_in") in result
        assert result[(0, "layer_in")].shape == (1, 5)
        assert torch.allclose(result[(0, "layer_in")], torch.ones(1, 5) * 128)
