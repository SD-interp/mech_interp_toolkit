import warnings
from collections.abc import Sequence
from typing import Any, cast

import torch
from nnsight import NNsight
from torch.nn import functional as F  # noqa: N812

from .activation_dict import ActivationDict, LayerComponent
from .utils import empty_dict_like, regularize_position

type Position = slice | int | Sequence | None


def get_activations(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    layer_components: list[LayerComponent],
    retain_grads: bool = True,
    positions: Position = None,
) -> ActivationDict:
    positions = regularize_position(positions)
    output = ActivationDict(model.model.config, positions=positions)
    with model.trace() as tracer:
        with tracer.invoke(**inputs):
            for layer_component in layer_components:
                output[layer_component] = locate_layer_component(model, layer_component)[
                    :, positions, :
                ].save()
                if retain_grads:
                    if positions is not None:
                        warnings.warn("retain_grads is set to True, setting positions to None")
                    output[layer_component].requires_grad_()
                    output[layer_component].retain_grad()
            tracer.stop()
    output.attention_mask = inputs["attention_mask"]
    output.value_type = "activation"
    return output


def get_embeddings_dict(model: NNsight, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "inputs_embeds" in inputs:
        pass
    else:
        embeds = get_activations(
            model, inputs, [(0, "layer_in")], retain_grads=False, positions=None
        )
        inputs.pop("input_ids", None)
        inputs["inputs_embeds"] = embeds[(0, "layer_in")]
    return inputs


def interpolate_activations(
    clean_activations: torch.Tensor,
    baseline_activations: torch.Tensor,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """
    Interpolates between clean and corrupted inputs.
    """
    interpolated_activations = (1 - alpha) * clean_activations + alpha * baseline_activations
    return interpolated_activations


def locate_layer_component(model: NNsight, layer_component: LayerComponent) -> Any:
    layer, component = layer_component

    layers = cast(Any, model.model.layers)
    if component == "logits":
        comp = model.lm_head.output
    elif component == "inputs_embeds":
        comp = layers[0].input
    elif component == "attn":
        comp = layers[layer].self_attn.output[0]
    elif component == "mlp":
        comp = layers[layer].mlp.output
    elif component == "z":
        comp = layers[layer].self_attn.o_proj.input
    elif component == "layer_in":
        comp = layers[layer].input
    elif component == "layer_out":
        comp = layers[layer].output
    else:
        raise ValueError("component must be one of {'attn', 'mlp', 'z', 'layer_in', 'layer_out'}")
    return comp


def _pad_and_concat(tensors, padding_value, dim):
    dim = dim % tensors[0].ndim
    max_len = max(t.shape[dim] for t in tensors)

    padded = []
    ndim = tensors[0].ndim

    for t in tensors:
        pad_len = max_len - t.shape[dim]
        if pad_len > 0:
            pad = [0, 0] * ndim
            # left-padding: put pad_len on the "left" side of `dim`
            pad[2 * (ndim - dim - 1)] = pad_len
            t = F.pad(t, pad, value=padding_value)
        padded.append(t)

    return torch.cat(padded, dim=0)


def concat_activations(list_activations: list[ActivationDict], pad_value=None) -> ActivationDict:
    new_obj = empty_dict_like(list_activations[0])

    if new_obj.attention_mask.numel() > 0:
        new_obj.attention_mask = _pad_and_concat(
            [activation.attention_mask for activation in list_activations],
            padding_value=0,
            dim=1,
        )

    for key in new_obj.keys():
        if pad_value is None:
            new_obj[key] = torch.cat([activation[key] for activation in list_activations])
        else:
            new_obj[key] = _pad_and_concat(
                [activation[key] for activation in list_activations],
                padding_value=pad_value,
                dim=1,
            )

    return new_obj


def expand_mask(mask: torch.Tensor, expansion: int):
    batch_size, seq_len = mask.shape

    padding = torch.zeros((batch_size, expansion), dtype=mask.dtype, device=mask.device)

    merged_tensor = torch.cat([padding, mask], dim=1)

    return merged_tensor[:, :seq_len]
