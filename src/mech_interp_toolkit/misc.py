import torch
from nnsight import NNsight
from .utils import input_dict_to_tuple


def get_attention_pattern(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    layers: list[int],
    head_indices: list[tuple[int]],
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
    assert model.model.config._attn_implementation == "eager", (
        "Model must use eager attention to get attention patterns."
    )
    output = dict()

    assert len(layers) == len(head_indices), (
        "each layer# provided must have corresponding head indices"
    )

    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)

    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:
            for i, layer in enumerate(layers):
                output[layer] = (
                    model.model.layers[layer]
                    .self_attn.output[1][:, head_indices[i], query_position, :]
                    .save()
                )
            trace.stop()

    return output
