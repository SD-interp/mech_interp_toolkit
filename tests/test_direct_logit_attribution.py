import pytest
import torch
from unittest.mock import MagicMock, patch
from mech_interp_toolkit.activation_dict import ActivationDict
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from mech_interp_toolkit.direct_logit_attribution import (
    get_pre_rms_logit_diff_direction,
    run_componentwise_dla,
    run_headwise_dla_for_layer,
)


@pytest.fixture
def mock_config():
    """Fixture for a mock model configuration."""
    config = Qwen3Config(num_attention_heads=4, num_hidden_layers=2, hidden_size=128)
    config._attn_implementation = "eager"
    return config


def test_get_pre_rms_logit_diff_direction():
    """Tests the get_pre_rms_logit_diff_direction function."""
    mock_model = MagicMock()
    mock_model.get_output_embeddings.return_value.weight = torch.randn(100, 128)
    mock_model.model.norm.weight = torch.randn(128)

    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenizer.encode.side_effect = [[1], [2]]

    direction = get_pre_rms_logit_diff_direction(["a", "b"], mock_tokenizer, mock_model)
    assert direction.shape == (128,)


def test_run_componentwise_dla():
    """Tests the run_componentwise_dla function."""
    with patch("mech_interp_toolkit.utils.input_dict_to_tuple") as mock_input_dict:
        mock_input_dict.return_value = (
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )
        mock_model = MagicMock()
        mock_model.model.config.num_hidden_layers = 2
        with patch(
            "mech_interp_toolkit.direct_logit_attribution.get_activations"
        ) as mock_get_activations:
            activations = ActivationDict(mock_model.model.config, slice(None))
            activations[(1, "layer_out")] = torch.ones(1, 1, 128)
            activations[(0, "attn")] = torch.ones(1, 1, 128)
            activations[(0, "mlp")] = torch.ones(1, 1, 128)
            mock_get_activations.return_value = (activations, None, None)
            dla_results = run_componentwise_dla(
                mock_model,
                {"input_ids": torch.randint(0, 100, (1, 10))},
                torch.zeros(128),
            )
            assert (0, "attn") in dla_results
            assert (0, "mlp") in dla_results


def test_run_headwise_dla_for_layer():
    """Tests the run_headwise_dla_for_layer function."""
    with patch("mech_interp_toolkit.utils.input_dict_to_tuple") as mock_input_dict:
        mock_input_dict.return_value = (
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )
        mock_model = MagicMock()
        mock_model.model.config.num_attention_heads = 4
        mock_model.model.layers[0].self_attn.o_proj.weight = torch.randn(128, 128)
        with patch.object(mock_model, "trace") as mock_trace:
            mock_trace.return_value.__enter__.return_value = mock_trace
            mock_model.model.layers[
                0
            ].self_attn.o_proj.input.__getitem__.return_value.save.return_value = torch.randn(1, 128)
            mock_model.model.layers[
                -1
            ].output.__getitem__.return_value.norm.return_value.save.return_value = torch.randn(1)
            dla_results = run_headwise_dla_for_layer(
                mock_model,
                {"input_ids": torch.randint(0, 100, (1, 10))},
                torch.zeros(128),
                0,
            )
            assert dla_results is not None