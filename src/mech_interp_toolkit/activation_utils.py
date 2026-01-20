from typing import Any, cast

import torch
from nnsight import NNsight

from .activation_dict import ActivationDict, LayerComponent


def get_activations(
    model: NNsight, inputs: dict[str, torch.Tensor], layer_components: list[LayerComponent]
) -> ActivationDict:
    output = ActivationDict(model.model.config, slice(None))
    with model.trace() as tracer:
        with tracer.invoke(**inputs):
            for layer_component in layer_components:
                output[layer_component] = locate_layer_component(model, layer_component).save()  # type: ignore
                output[layer_component].requires_grad_()
                output[layer_component].retain_grad()
            tracer.stop()
    return output


def get_embeddings(model: NNsight, inputs: dict[str, torch.Tensor]) -> ActivationDict:
    return get_activations(model, inputs, [(0, "layer_in")])


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
    if component == "attn":
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
