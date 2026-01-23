from collections.abc import Sequence
from typing import cast

import einops
import torch
from nnsight import NNsight

from .activation_dict import ActivationDict
from .activation_utils import locate_layer_component
from .utils import ChatTemplateTokenizer, get_layer_components, get_num_layers


def get_pre_rms_logit_diff_direction(
    token_pair: Sequence[str], tokenizer: ChatTemplateTokenizer, model: NNsight
) -> torch.Tensor:
    """
    Calculates the direction in the residual stream that corresponds to the difference
    in logits between two tokens, before the final LayerNorm.

    Args:
        token_pair: A sequence of two tokens.
        tokenizer: The tokenizer.
        model: The NNsight model wrapper.

    Returns:
        The direction vector.
    """
    unembedding_matrix = model.get_output_embeddings().weight
    gamma = model.model.norm.weight  # type: ignore (d_model,)
    token_ids = []
    if len(token_pair) != 2:
        raise ValueError("Provide exactly two target tokens.")

    for token in token_pair:
        encoding = tokenizer.tokenizer.encode(token, add_special_tokens=False)
        if len(encoding) != 1:
            raise ValueError(f"Token '{token}' is tokenized into multiple tokens.")
        token_ids.append(encoding[0])

    post_rms_logit_diff_direction = (
        unembedding_matrix[token_ids[0]] - unembedding_matrix[token_ids[1]]
    )  # (d_model,)
    pre_rms_logit_diff_direction = post_rms_logit_diff_direction * gamma  # (d_model,)
    return pre_rms_logit_diff_direction


def run_componentwise_dla(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    pre_rms_direction: torch.Tensor,
    eps: float = 1e-6,
) -> ActivationDict:
    """
    Performs component-wise Direct Logit Attribution (DLA).

    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        pre_rms_direction: The direction vector in the residual stream.
        eps: A small value to prevent division by zero.

    Returns:
        A dictionary containing the DLA results for attention and MLP layers.
    """

    n_layers = get_num_layers(model)

    output = ActivationDict(model.model.config, [-1])

    # Prepare components to fetch
    layers_components = get_layer_components(model)
    layers_components.append((n_layers - 1, "layer_out"))

    with model.trace() as tracer:
        with tracer.invoke(**inputs):
            for layer_component in layers_components:
                output[layer_component] = locate_layer_component(model, layer_component)[
                    :, [-1], :
                ].save()  # type: ignore

    # Calculate divisor (RMS normalization factor)
    final_layer_output = output[(n_layers - 1, "layer_out")].squeeze(1)
    divisor = torch.sqrt(torch.mean(final_layer_output**2, dim=-1, keepdim=True) + eps)

    output.pop((n_layers - 1, "layer_out"))

    # Calculate DLA
    for layer_component, activation in output.items():
        output[layer_component] = (activation.squeeze(1) @ pre_rms_direction) / divisor.squeeze(1)
    output.value_type = "dla_scores"
    return output


def run_headwise_dla_for_layer(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    pre_rms_direction: torch.Tensor,
    layer: int,
    eps: float = 1e-6,
) -> ActivationDict:
    """
    Performs head-wise Direct Logit Attribution (DLA) for a specific layer.
    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        pre_rms_direction: The direction vector in the residual stream.
        layer: The layer index.
        eps: A small value to prevent division by zero.
    Returns:
        An ActivationDict containing the DLA results for each head.
    """
    proj_weight = model.model.layers[layer].self_attn.o_proj.weight  # type: ignore
    num_heads = cast(int, model.model.config.num_attention_heads)  # type: ignore
    n_layers = get_num_layers(model)

    # Define components to fetch
    layers_components = [(layer, "z"), (n_layers - 1, "layer_out")]

    output = ActivationDict(model.model.config, [-1])

    with model.trace() as tracer:
        with tracer.invoke(**inputs):
            for layer_component in layers_components:
                output[layer_component] = locate_layer_component(model, layer_component)[
                    :, [-1], :
                ].save()  # type: ignore

    head_inputs = output.split_heads()[(layer, "z")].squeeze(1)

    final_layer_output = output[(n_layers - 1, "layer_out")].squeeze(1)
    divisor = torch.sqrt(torch.mean(final_layer_output**2, dim=-1, keepdim=True) + eps)

    output.pop((n_layers - 1, "layer_out"))

    W_O = proj_weight.view(proj_weight.shape[0], num_heads, -1)  # type: ignore # noqa: N806

    # Calculate the contribution of each head to the final output in the given direction.
    projections = einops.einsum(
        head_inputs,
        W_O,
        pre_rms_direction,
        "batch n_heads head_dim, d_model n_heads head_dim, d_model -> batch n_heads",
    )
    output[(layer, "z")] = projections / divisor
    output.value_type = "dla_scores"
    return output
