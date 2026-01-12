import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from mech_interp_toolkit.misc import get_attention_pattern


@pytest.fixture
def mock_config():
    """Fixture for a mock model configuration."""
    config = Qwen3Config(num_attention_heads=4, num_hidden_layers=2, hidden_size=128)
    config._attn_implementation = "eager"
    return config


def test_get_attention_pattern(mock_config):
    """Tests the get_attention_pattern function."""
    with patch("mech_interp_toolkit.utils.input_dict_to_tuple") as mock_input_dict:
        mock_input_dict.return_value = (
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )
        mock_model = MagicMock()
        mock_model.model.config = mock_config
        with patch.object(mock_model, "trace") as mock_trace:
            mock_trace.return_value.__enter__.return_value = mock_trace
            patterns = get_attention_pattern(
                mock_model, {"input_ids": torch.randint(0, 100, (1, 10))}, [0], [(0, 1)]
            )
            assert patterns is not None