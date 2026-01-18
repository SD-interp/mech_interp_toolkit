import json
import random
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Optional, Tuple, Union, cast

import numpy as np
import requests
import torch
from nnsight import Envoy, NNsight
from torch.utils.data import DataLoader
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side=padding_side, suffix=suffix
    )
    tokenizer = ChatTemplateTokenizer(tokenizer)
    config._attn_implementation = attn_type
    if attn_type == "eager":
        config.return_dict_in_generate = True
        config.output_attentions = True

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.eval().to(device)
    model = NNsight(model)

    return model, tokenizer, config


def get_prompts_from_url(url: str, save_path: Union[str, Path] = "data/prompts.json") -> None:
    """
    Downloads prompts from a URL and saves them to a local file.

    Args:
        url: The URL to download the prompts from.
        save_path: The local path to save the prompts to.
    """
    response = requests.get(url)
    data = response.json()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if "metadata" not in data or ("questions" not in data and "pairs" not in data):
        raise ValueError("Incorrect schema in fetched data.")

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def build_dataloader(
    dataset: Union[torch.Tensor, list],
    batch_size: Optional[int],
) -> DataLoader:
    """
    Creates a DataLoader from a given dataset.

    Args:
        dataset: The dataset to create the DataLoader from.
        batch_size: The batch size for the DataLoader.

    Returns:
        The created DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )


def get_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Generates position IDs from an attention mask.

    Args:
        attention_mask: The attention mask tensor.

    Returns:
        The position IDs tensor.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def input_dict_to_tuple(
    input_dict: dict[str, torch.Tensor], device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts an input dictionary of tensors to a tuple of tensors on a specified device.

    Args:
        input_dict: The input dictionary.
        device: The device to move the tensors to. If None, defaults to 'cuda' if available, otherwise 'cpu'.

    Returns:
        A tuple of tensors (input_ids, attention_mask, position_ids).
    """
    if device is None:
        device = get_default_device()

    if isinstance(input_dict, MutableMapping):
        input_ids = input_dict["input_ids"].to(device)
        attention_mask = input_dict["attention_mask"].to(device)
        position_ids = get_position_ids(attention_mask).to(device)
        return input_ids, attention_mask, position_ids
    else:
        raise TypeError("Input must be a dictionary-like object.")


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
    tokA_id = tokenizer.tokenizer.encode(tokens[0], add_special_tokens=False)[0]
    tokB_id = tokenizer.tokenizer.encode(tokens[1], add_special_tokens=False)[0]
    return logits[:, tokA_id] - logits[:, tokB_id]


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


def get_all_layer_components(model: NNsight) -> list[tuple[int, str]]:
    """
    Get a list of all (layer, component) tuples for attention and MLP components.

    Args:
        model: The NNsight model wrapper.

    Returns:
        A list of tuples containing (layer_index, component_name) for all layers.
    """
    n_layers = get_num_layers(model)
    return [(i, c) for i in range(n_layers) for c in ["attn", "mlp"]]


def get_default_device() -> str:
    """
    Get the default device (cuda if available, otherwise cpu).

    Returns:
        The device string ('cuda' or 'cpu').
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
