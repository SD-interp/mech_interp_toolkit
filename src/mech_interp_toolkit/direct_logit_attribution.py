import torch
from nnsight import NNsight
from .utils import ChatTemplateTokenizer, input_dict_to_tuple
import einops
from collections.abc import Sequence


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
    gamma = model.model.norm.weight  # (d_model,)
    token_ids = []
    assert len(token_pair) == 2, "Provide exactly two target tokens."

    for token in token_pair:
        encoding = tokenizer.tokenizer.encode(token, add_special_tokens=False)
        assert len(encoding) == 1, f"Token '{token}' is tokenized into multiple tokens."
        token_ids.append(encoding[0])

    post_rms_logit_diff_direction = (
        unembedding_matrix[token_ids[0]] - unembedding_matrix[token_ids[1]]
    )  # (d_model,)
    pre_rms_logit_diff_direction = post_rms_logit_diff_direction * gamma  # (d_model,)
    return pre_rms_logit_diff_direction


def run_layerwise_dla(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    pre_rms_direction: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Performs layer-wise Direct Logit Attribution (DLA).

    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        pre_rms_direction: The direction vector in the residual stream.
        eps: A small value to prevent division by zero.

    Returns:
        A dictionary containing the DLA results for attention and MLP layers.
    """
    attn_records = dict()
    mlp_records = dict()
    n_layers = model.model.config.num_hidden_layers
    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)

    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:  # noqa: F841
            for layer in range(n_layers):
                attn_records[layer] = (
                    model.model.layers[layer].self_attn.output[0][:, -1, :].save()
                )
                mlp_records[layer] = (
                    model.model.layers[layer].mlp.output[:, -1, :].save()
                )
            rms_final = (
                model.model.layers[-1]
                .output[:, -1, :]
                .norm(dim=-1, keepdim=True)
                .save()
            )

    # The divisor for DLA is the L2 norm of the final residual stream.
    divisor = torch.sqrt(rms_final**2 + eps).squeeze()

    attn_dla = {k: (v @ pre_rms_direction) / divisor for k, v in attn_records.items()}

    mlp_dla = {k: (v @ pre_rms_direction) / divisor for k, v in mlp_records.items()}

    return {"attn": attn_dla, "mlp": mlp_dla}


def run_headwise_dla_for_layer(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    pre_rms_direction: torch.Tensor,
    layer: int,
    eps: float = 1e-6,
    scale: bool = True,
) -> torch.Tensor:
    """
    Performs head-wise Direct Logit Attribution (DLA) for a specific layer.

    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        pre_rms_direction: The direction vector in the residual stream.
        layer: The layer index.
        eps: A small value to prevent division by zero.
        scale: Whether to scale the DLA results by the final LayerNorm.

    Returns:
        A tensor containing the DLA results for each head.
    """
    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)

    proj_weight = model.model.layers[layer].self_attn.o_proj.weight
    num_heads = model.model.config.num_attention_heads

    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:  # noqa: F841
            head_inputs = (
                model.model.layers[layer].self_attn.o_proj.input[:, -1, :].save()
            )
            rms_final = (
                model.model.layers[-1]
                .output[:, -1, :]
                .norm(dim=-1, keepdim=True)
                .save()
            )

    divisor = torch.sqrt(rms_final**2 + eps)

    batch_size = head_inputs.shape[0]
    head_inputs = head_inputs.view(batch_size, num_heads, -1)

    W_O = proj_weight.view(proj_weight.shape[0], num_heads, -1)

    # Calculate the contribution of each head to the final output in the given direction.
    projections = einops.einsum(
        head_inputs,
        W_O,
        pre_rms_direction,
        "batch n_heads head_dim, d_model n_heads head_dim, d_model -> batch n_heads",
    )

    return projections / divisor if scale else projections
