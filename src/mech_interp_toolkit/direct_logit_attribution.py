from collections.abc import Sequence
from typing import cast

import einops
import torch
from nnsight import NNsight

from .utils import ChatTemplateTokenizer, get_all_layer_components, get_num_layers


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
) -> dict[str, dict[int, torch.Tensor]]:
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
    device = pre_rms_direction.device

    # Prepare components to fetch
    layers_components = get_all_layer_components(model)
    layers_components.append((n_layers - 1, "layer_out"))

    # Get activations
    spec_dict: SpecDict = {"activations": {"positions": -1, "locations": layers_components}}
    with UAP(model, inputs, spec_dict) as uap:
        activations, _ = uap.unified_access_and_patching()

    if activations is None:
        raise RuntimeError("Failed to retrieve activations from the model.")

    # Calculate divisor
    final_layer_output = activations[(n_layers - 1, "layer_out")].squeeze(1).to(device)
    rms_final = final_layer_output.norm(dim=-1, keepdim=True)
    divisor = torch.sqrt(rms_final**2 + eps).squeeze()

    # Calculate DLA
    dla_scores = {}
    for layer_component, activation in activations.items():
        _, component = layer_component
        if component == "layer_out":
            continue
        dla_scores[layer_component] = (
            activation.squeeze(1).to(device) @ pre_rms_direction
        ) / divisor
    return dla_scores


def run_headwise_dla_for_layer(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    pre_rms_direction: torch.Tensor,
    layer: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Performs head-wise Direct Logit Attribution (DLA) for a specific layer.
    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        pre_rms_direction: The direction vector in the residual stream.
        layer: The layer index.
        eps: A small value to prevent division by zero.
    Returns:
        A tensor containing the DLA results for each head.
    """
    proj_weight = model.model.layers[layer].self_attn.o_proj.weight  # type: ignore
    num_heads = cast(int, model.model.config.num_attention_heads)  # type: ignore
    n_layers = get_num_layers(model)

    # Define components to fetch
    layers_components = [(layer, "z"), (n_layers - 1, "layer_out")]

    # Get activations
    spec_dict: SpecDict = {"activations": {"positions": -1, "locations": layers_components}}
    with UAP(model, inputs, spec_dict) as uap:
        activations, _ = uap.unified_access_and_patching()

    if activations is None:
        raise RuntimeError("Failed to retrieve activations from the model.")

    head_inputs = activations[(layer, "z")].squeeze(1)
    final_layer_output = activations[(n_layers - 1, "layer_out")].squeeze(1)

    rms_final = final_layer_output.norm(dim=-1, keepdim=True)

    divisor = torch.sqrt(rms_final**2 + eps)

    batch_size = head_inputs.shape[0]
    head_inputs = head_inputs.view(batch_size, num_heads, -1)

    W_O = proj_weight.view(proj_weight.shape[0], num_heads, -1)  # type: ignore # noqa: N806

    # Calculate the contribution of each head to the final output in the given direction.
    projections = einops.einsum(
        head_inputs,
        W_O,
        pre_rms_direction,
        "batch n_heads head_dim, d_model n_heads head_dim, d_model -> batch n_heads",
    )

    return projections / divisor
