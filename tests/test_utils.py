import pytest
import torch
import numpy as np
import random
from unittest.mock import patch, MagicMock
from pathlib import Path
from mech_interp_toolkit.utils import (
    set_global_seed,
    load_model_tokenizer_config,
    get_prompts_from_url,
    build_dataloader,
    get_position_ids,
    input_dict_to_tuple,
    get_logit_difference,
)
from mech_interp_toolkit.tokenizer import ChatTemplateTokenizer

def test_set_global_seed():
    """Tests the set_global_seed function."""
    set_global_seed(42)

@patch("transformers.AutoConfig.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("mech_interp_toolkit.utils.NNsight")
def test_load_model_tokenizer_config(mock_nnsight, mock_model, mock_tokenizer, mock_config):
    """Tests the load_model_tokenizer_config function."""
    mock_config.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()
    model_mock = MagicMock()
    model_mock.eval.return_value = model_mock
    model_mock.to.return_value = model_mock
    mock_model.return_value = model_mock
    mock_nnsight.return_value = MagicMock()
    
    model, tokenizer, config = load_model_tokenizer_config("test-model")
    
    assert model is not None
    assert tokenizer is not None
    assert config is not None
    mock_config.assert_called_with("test-model")
    mock_tokenizer.assert_called_with("test-model", use_fast=True, padding_side="left", suffix="")
    mock_model.assert_called_with("test-model", config=mock_config.return_value)
    mock_nnsight.assert_called_with(model_mock)

@patch("requests.get")
def test_get_prompts_from_url(mock_get, tmp_path):
    """Tests the get_prompts_from_url function."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"metadata": {}, "questions": []}
    mock_get.return_value = mock_response
    
    save_path = tmp_path / "prompts.json"
    get_prompts_from_url("http://fakeurl.com", save_path=save_path)
    
    assert save_path.exists()
    mock_get.assert_called_with("http://fakeurl.com")

def test_build_dataloader():
    """Tests the build_dataloader function."""
    dataset = torch.randn(10, 5)
    dataloader = build_dataloader(dataset, batch_size=2)
    
    assert len(dataloader) == 5
    for batch in dataloader:
        assert batch.shape == (2, 5)

def test_get_position_ids():
    """Tests the get_position_ids function."""
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    position_ids = get_position_ids(attention_mask)
    expected = torch.tensor([[0, 1, 2, 1, 1]])
    assert torch.equal(position_ids, expected)

def test_input_dict_to_tuple():
    """Tests the input_dict_to_tuple function."""
    input_dict = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 0]])
    }
    input_ids, attention_mask, position_ids = input_dict_to_tuple(input_dict)
    
    assert input_ids.shape == (1, 3)
    assert attention_mask.shape == (1, 3)
    assert position_ids.shape == (1, 3)

def test_get_logit_difference():
    """Tests the get_logit_difference function."""
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenizer.encode.side_effect = [[1], [2]] # Corresponds to tokens 'A' and 'B'
    
    diff = get_logit_difference(logits, mock_tokenizer, tokens=["A", "B"])
    
    assert torch.isclose(diff, torch.tensor([-0.1]))

def test_input_dict_to_tuple_type_error():
    """Tests that input_dict_to_tuple raises TypeError for wrong input type."""
    with pytest.raises(TypeError):
        input_dict_to_tuple("not a dict")
