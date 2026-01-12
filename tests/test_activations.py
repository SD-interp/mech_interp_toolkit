import pytest
import torch
from unittest.mock import MagicMock, patch
from mech_interp_toolkit.activation_dict import ActivationDict, FrozenError
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from mech_interp_toolkit.activations import (
    create_z_patch_dict,
    get_activations,
    get_activations_and_grads,
    patch_activations,
)


@pytest.fixture
def mock_config():
    """Fixture for a mock model configuration."""
    config = Qwen3Config(num_attention_heads=4, num_hidden_layers=2, hidden_size=128)
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def activation_dict(mock_config):
    """Fixture for an ActivationDict."""
    return ActivationDict(mock_config, positions=slice(None))


def test_activation_dict_init(activation_dict, mock_config):
    """Tests the initialization of the ActivationDict."""
    assert activation_dict.config == mock_config
    assert activation_dict.num_heads == 4
    assert activation_dict.num_layers == 2
    assert activation_dict.head_dim == 32
    assert activation_dict.model_dim == 128
    assert not activation_dict._frozen


def test_activation_dict_freeze_unfreeze(activation_dict):
    """Tests the freeze and unfreeze methods."""
    activation_dict.freeze()
    assert activation_dict._frozen
    with pytest.raises(FrozenError):
        activation_dict["test"] = torch.randn(1)

    activation_dict.unfreeze()
    assert not activation_dict._frozen
    activation_dict["test"] = torch.randn(1)
    assert "test" in activation_dict


def test_split_merge_heads(activation_dict):
    """Tests the split_heads and merge_heads methods."""
    activation_dict[(0, "z")] = torch.randn(1, 10, 128)

    activation_dict.split_heads()
    assert not activation_dict.fused_heads
    assert activation_dict[(0, "z")].shape == (1, 10, 4, 32)

    activation_dict.merge_heads()
    assert activation_dict.fused_heads
    assert activation_dict[(0, "z")].shape == (1, 10, 128)


def test_get_mean_activations(activation_dict):
    """Tests the get_mean_activations method."""
    activation_dict[(0, "z")] = torch.ones(2, 10, 128)
    mean_acts = activation_dict.apply(torch.mean, dim=0)
    assert torch.allclose(mean_acts[(0, "z")], torch.ones(10, 128))


def test_cuda(activation_dict):
    """Tests the cuda method."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    activation_dict[(0, "z")] = torch.randn(1, 10, 128)
    activation_dict.cuda()
    assert activation_dict[(0, "z")].is_cuda


def test_create_z_patch_dict(activation_dict, mock_config):
    """Tests the create_z_patch_dict method."""
    new_acts = ActivationDict(mock_config, positions=slice(None))
    activation_dict[(0, "z")] = torch.randn(1, 10, 128)
    new_acts[(0, "z")] = torch.ones(1, 10, 128)

    activation_dict.split_heads()
    new_acts.split_heads()

    patch_dict = create_z_patch_dict(activation_dict, new_acts, [(0, 1)], position=[5])

    assert patch_dict.fused_heads
    assert (0, "z") in patch_dict
    assert torch.allclose(
        patch_dict[(0, "z")].view(1, 10, 4, 32)[:, 5, 1, :], torch.ones(1, 32)
    )


@patch("mech_interp_toolkit.activations.input_dict_to_tuple")
@patch("mech_interp_toolkit.activations._extract_or_patch")
def test_get_activations(mock_extract, mock_input_dict_to_tuple, mock_config):
    """Tests the get_activations function."""
    mock_input_dict_to_tuple.return_value = (
        torch.randint(0, 100, (1, 10)),
        torch.ones(1, 10),
        torch.arange(10),
    )
    mock_extract.return_value = (torch.randn(1, 10, 128), None)

    mock_model = MagicMock()
    mock_model.model.config = mock_config
    mock_model.trace.return_value.__enter__.return_value = MagicMock()
    mock_model.lm_head.output.__getitem__.return_value.save.return_value = (
        torch.randn(1, 100)
    )

    acts, grads, logits = get_activations(
        mock_model, {"input_ids": torch.randint(0, 100, (1, 10))}, [(0, "attn")]
    )
    assert (0, "attn") in acts
    assert logits is not None


@patch("mech_interp_toolkit.activations.input_dict_to_tuple")
@patch("mech_interp_toolkit.activations._extract_or_patch")
def test_get_activations_and_grads(mock_extract, mock_input_dict_to_tuple, mock_config):
    """Tests the get_activations_and_grads function."""
    mock_input_dict_to_tuple.return_value = (
        torch.randint(0, 100, (1, 10)),
        torch.ones(1, 10),
        torch.arange(10),
    )
    mock_extract.return_value = (torch.randn(1, 10, 128), torch.randn(1, 10, 128))

    mock_model = MagicMock()
    mock_model.model.config = mock_config
    mock_model.trace.return_value.__enter__.return_value = MagicMock()
    mock_model.lm_head.output.__getitem__.return_value.save.return_value = (
        torch.randn(1, 100)
    )

    metric_fn = MagicMock()
    metric_fn.return_value = MagicMock()

    acts, grads, logits = get_activations_and_grads(
        mock_model,
        {"input_ids": torch.randint(0, 100, (1, 10))},
        [(0, "attn")],
        metric_fn=metric_fn,
    )
    assert (0, "attn") in acts
    assert (0, "attn") in grads
    metric_fn.assert_called()


@patch("mech_interp_toolkit.activations.input_dict_to_tuple")
@patch("mech_interp_toolkit.activations._extract_or_patch")
def test_patch_activations(
    mock_extract, mock_input_dict_to_tuple, activation_dict, mock_config
):
    """Tests the patch_activations function."""
    mock_input_dict_to_tuple.return_value = (
        torch.randint(0, 100, (1, 10)),
        torch.ones(1, 10),
        torch.arange(10),
    )
    mock_extract.return_value = (torch.randn(1, 10, 128), None)

    mock_model = MagicMock()
    mock_model.model.config = mock_config
    mock_model.trace.return_value.__enter__.return_value = MagicMock()
    mock_model.lm_head.output.__getitem__.return_value.save.return_value = (
        torch.randn(1, 100)
    )

    inputs = {"input_ids": torch.randint(0, 100, (1, 10))}
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    activation_dict[(0, "attn")] = torch.randn(1, 10, 128)

    _, _, logits = patch_activations(mock_model, inputs, activation_dict, position=0)

    assert logits is not None
