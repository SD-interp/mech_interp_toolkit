import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, PropertyMock, patch
from mech_interp_toolkit.interpret import ActivationDict, FrozenError, LinearProbe
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

@pytest.fixture
def mock_config():
    """Fixture for a mock model configuration."""
    config = Qwen2Config(
        num_attention_heads=4,
        num_hidden_layers=2,
        hidden_size=128
    )
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
    mean_acts = activation_dict.get_mean_activations()
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
    
    patch_dict = activation_dict.create_z_patch_dict(new_acts, [(0, 1)], position=[5])
    
    assert patch_dict.fused_heads
    assert (0, "z") in patch_dict
    assert torch.allclose(patch_dict[(0,"z")].view(1,10,4,32)[:,5,1,:], torch.ones(1, 32))

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

def test_get_pre_rms_logit_diff_direction():
    """Tests the get_pre_rms_logit_diff_direction function."""
    from mech_interp_toolkit.interpret import get_pre_rms_logit_diff_direction
    
    mock_model = MagicMock()
    mock_model.get_output_embeddings.return_value.weight = torch.randn(100, 128)
    mock_model.model.norm.weight = torch.randn(128)

    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenizer.encode.side_effect = [[1], [2]]

    direction = get_pre_rms_logit_diff_direction(["a", "b"], mock_tokenizer, mock_model)
    assert direction.shape == (128,)

@patch('mech_interp_toolkit.interpret.input_dict_to_tuple')
def test_get_activations(mock_input_dict_to_tuple, mock_config):
    """Tests the get_activations function."""
    from mech_interp_toolkit.interpret import get_activations
    
    mock_input_dict_to_tuple.return_value = (torch.randint(0, 100, (1, 10)), torch.ones(1, 10), torch.arange(10))

    mock_model = MagicMock()
    mock_model.model.config = mock_config
    
    with patch.object(mock_model, 'trace') as mock_trace:
        mock_trace.return_value.__enter__.return_value = None # No trace object yielded
        get_activations(mock_model, {}, [])


@patch('mech_interp_toolkit.interpret.input_dict_to_tuple')
def test_patch_activations(mock_input_dict_to_tuple, activation_dict, mock_config):
    """Tests the patch_activations function."""
    from mech_interp_toolkit.interpret import patch_activations

    mock_input_dict_to_tuple.return_value = (torch.randint(0, 100, (1, 10)), torch.ones(1, 10), torch.arange(10))

    mock_model = MagicMock()
    mock_model.model.config = mock_config
    
    with patch.object(mock_model, 'trace') as mock_trace:
        mock_trace.return_value.__enter__.return_value = None
        mock_model.lm_head.output.__getitem__.return_value.save.return_value = torch.randn(1, 100)

        inputs = {"input_ids": torch.randint(0, 100, (1, 10))}
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        activation_dict[(0, "attn")] = torch.randn(1, 1, 128)
        
        logits = patch_activations(mock_model, inputs, activation_dict, position=0)
        
        assert logits.shape == (1, 100)

def test_run_layerwise_dla():
    """Tests the run_layerwise_dla function."""
    from mech_interp_toolkit.interpret import run_layerwise_dla
    with patch('mech_interp_toolkit.interpret.input_dict_to_tuple') as mock_input_dict:
        mock_input_dict.return_value = (torch.zeros(1,1), torch.zeros(1,1), torch.zeros(1,1))
        mock_model = MagicMock()
        mock_model.model.config.num_hidden_layers = 2
        with patch.object(mock_model, 'trace') as mock_trace:
            mock_trace.return_value.__enter__.return_value = None
            mock_model.model.layers[-1].output.__getitem__.return_value.norm.return_value.save.return_value = torch.randn(1)
            dla_results = run_layerwise_dla(mock_model, {}, torch.zeros(1))
            assert 'attn' in dla_results
            assert 'mlp' in dla_results

def test_run_headwise_dla_for_layer():
    """Tests the run_headwise_dla_for_layer function."""
    from mech_interp_toolkit.interpret import run_headwise_dla_for_layer
    with patch('mech_interp_toolkit.interpret.input_dict_to_tuple') as mock_input_dict:
        mock_input_dict.return_value = (torch.zeros(1,1), torch.zeros(1,1), torch.zeros(1,1))
        mock_model = MagicMock()
        mock_model.model.config.num_attention_heads = 4
        mock_model.model.layers[0].self_attn.o_proj.weight = torch.randn(128,128)
        with patch.object(mock_model, 'trace') as mock_trace:
            mock_trace.return_value.__enter__.return_value = mock_trace
            mock_model.model.layers[0].self_attn.o_proj.input.__getitem__.return_value.save.return_value = torch.randn(1, 128)
            mock_model.model.layers[-1].output.__getitem__.return_value.norm.return_value.save.return_value = torch.randn(1)
            dla_results = run_headwise_dla_for_layer(mock_model, {}, torch.zeros(1), 0)
            assert dla_results is not None

def test_get_attention_pattern(mock_config):
    """Tests the get_attention_pattern function."""
    from mech_interp_toolkit.interpret import get_attention_pattern
    with patch('mech_interp_toolkit.interpret.input_dict_to_tuple') as mock_input_dict:
        mock_input_dict.return_value = (torch.zeros(1,1), torch.zeros(1,1), torch.zeros(1,1))
        mock_model = MagicMock()
        mock_model.model.config = mock_config
        with patch.object(mock_model, 'trace') as mock_trace:
            mock_trace.return_value.__enter__.return_value = mock_trace
            patterns = get_attention_pattern(mock_model, {}, [0], [(0,1)])
            assert patterns is not None
