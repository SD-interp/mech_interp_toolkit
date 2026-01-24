from typing import Any, cast

import torch
from nnsight import NNsight

from .activation_dict import ActivationDict, LayerComponent


def get_activations(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    layer_components: list[LayerComponent],
    retain_grads: bool = True,
) -> ActivationDict:
    output = ActivationDict(model.model.config, slice(None))
    with model.trace() as tracer:
        with tracer.invoke(**inputs):
            for layer_component in layer_components:
                output[layer_component] = locate_layer_component(model, layer_component).save()  # type: ignore
                if retain_grads:
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
        embeds = get_activations(model, inputs, [(0, "layer_in")])
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
