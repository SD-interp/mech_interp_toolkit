import warnings
from collections.abc import Sequence

import torch
from nnsight import NNsight


def get_attention_pattern(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    layers: list[int],
    head_indices: Sequence[Sequence[int]],
    query_position: int = -1,
) -> dict[int, torch.Tensor]:
    """
    Retrieves the attention patterns for specific heads and layers.

    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        layers: A list of layer indices.
        head_indices: A list of head indices for each layer.
        query_position: The position of the query token.

    Returns:
        A dictionary mapping layer indices to attention patterns.
    """
    if model.model.config._attn_implementation != "eager":  # type: ignore
        warnings.warn("Attention patterns may not be accurate for non-eager implementations.")
    output = dict()

    if len(layers) != len(head_indices):
        raise ValueError("each layer# provided must have corresponding head indices")

    with torch.no_grad():
        with model.trace() as tracer:  # noqa: F841
            with tracer.invoke(**inputs):
                for i, layer in enumerate(layers):
                    heads = list(head_indices[i])
                    output[layer] = (
                        model.model.layers[layer]  # type: ignore
                        .self_attn.output[1][:, heads, query_position, :]  # type: ignore
                        .save()  # type: ignore
                    )

    return output
