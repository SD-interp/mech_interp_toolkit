import random
from collections.abc import Sequence
from typing import Optional, Tuple, Union, cast

import numpy as np
import torch
from nnsight import Envoy, NNsight
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)

from .tokenizer import ChatTemplateTokenizer


def set_global_seed(seed: int = 0) -> None:
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed: The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def load_model_tokenizer_config(
    model_name: str,
    device: Optional[str] = None,
    padding_side: str = "left",
    attn_type: str = "sdpa",
    suffix: str = "",
    system_prompt: str = "",
) -> Tuple[Envoy, ChatTemplateTokenizer, PretrainedConfig]:
    """
    Load a Hugging Face model, tokenizer, and config by name.

    Args:
        model_name: The model identifier from the Hugging Face Hub or a local path.
        device: The device to load the model on. If None, defaults to 'cuda' if available, otherwise 'cpu'.
        padding_side: The side to pad the tokenizer on.
        attn_type: The attention implementation to use.
        suffix: A suffix to append to the model name.

    Returns:
        A tuple containing the NNsight-wrapped model, the chat tokenizer, and the model config.
    """
    if device is None:
        device = get_default_device()

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side=padding_side)
    tokenizer = ChatTemplateTokenizer(tokenizer, suffix=suffix, system_prompt=system_prompt)
    config._attn_implementation = attn_type
    if attn_type == "eager":
        config.return_dict_in_generate = True
        config.output_attentions = True

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.eval().to(device)  # type: ignore
    model = NNsight(model)

    return model, tokenizer, config


def get_logit_difference(
    logits: torch.Tensor,
    tokenizer: ChatTemplateTokenizer,
    tokens: list[str] = ["A", "B"],
) -> torch.Tensor:
    """
    Calculates the difference in logits between two tokens.

    Args:
        logits: The logits tensor.
        tokenizer: The chat tokenizer.
        tokens: A list of two tokens to compare.

    Returns:
        The difference in logits.
    """
    tok_a_id = tokenizer.tokenizer.encode(tokens[0], add_special_tokens=False)[0]
    tok_b_id = tokenizer.tokenizer.encode(tokens[1], add_special_tokens=False)[0]
    return logits[:, tok_a_id] - logits[:, tok_b_id]


def regularize_position(
    position: Union[int, slice, Sequence, None],
) -> Union[list[int], slice, Sequence]:
    if isinstance(position, int):
        position = [position]
    elif position is None:
        position = slice(None)
    elif isinstance(position, (slice, Sequence)):
        pass
    else:
        raise ValueError("position must be int, slice, None or Sequence")
    return position


def get_num_layers(model: NNsight) -> int:
    """
    Get the number of hidden layers in the model.

    Args:
        model: The NNsight model wrapper.

    Returns:
        The number of hidden layers.
    """
    return cast(int, model.model.config.num_hidden_layers)  # type: ignore


def get_layer_components(model: NNsight, stop_at: Optional[int] = None) -> list[tuple[int, str]]:
    """
    Get a list of all (layer, component) tuples for attention and MLP components.

    Args:
        model: The NNsight model wrapper.

    Returns:
        A list of tuples containing (layer_index, component_name) for all layers.
    """
    if stop_at is not None:
        n_layers = stop_at + 1
    else:
        n_layers = get_num_layers(model)
    return [(i, c) for i in range(n_layers) for c in ["attn", "mlp"]]


def get_default_device() -> str:
    """
    Get the default device (cuda if available, otherwise cpu).

    Returns:
        The device string ('cuda' or 'cpu').
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
