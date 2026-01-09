import torch
from nnsight import NNsight
from .utils import input_dict_to_tuple
from collections.abc import Sequence, Callable
from .interpret import ActivationDict
import warnings
from typing import Optional


def _locate_layer_component(model, trace, layer: int, component: str):
    if trace is None:
        raise ValueError("Active trace is required to locate layer components.")

    if component == "attn":
        comp = model.model.layers[layer].self_attn.output[0]
    elif component == "mlp":
        comp = model.model.layers[layer].mlp.output
    elif component == "z":
        comp = model.model.layers[layer].self_attn.o_proj.input
    elif component == "layer_in":
        comp = model.model.layers[layer].input
    elif component == "layer_out":
        comp = model.model.layers[layer].output
    else:
        raise ValueError(
            "component must be one of {'attn', 'mlp', 'z', 'layer_in', 'layer_out'}"
        )
    return comp


def _extract_or_patch(
    model,
    trace,
    layer,
    component,
    position,
    capture_grad: bool = False,
    patch_value: Optional[torch.Tensor] = None,
):
    if trace is None:
        raise ValueError("Active trace is required to locate layer components.")

    attn_implementation = model.model.config._attn_implementation

    if attn_implementation != "eager":
        warnings.warn(
            f"attn_implementation {attn_implementation} can give incorrect results"
        )

    if patch_value is not None and capture_grad:
        warnings.warn(
            "Capturing gradients post patching is ot tested and might give incorrect results."
        )

    comp = _locate_layer_component(model, trace, layer, component)
    if patch_value is not None:
        comp[:, position, :] = patch_value
        act = None
    else:
        act = comp[:, position, :].save()

    if capture_grad:
        grad = comp.grad[:, position, :].save()
    else:
        grad = None
    return act, grad


def _regularize_position(position):
    if isinstance(position, int):
        position = [position]
    elif position is None:
        position = slice(None)
    elif isinstance(position, (slice, Sequence)):
        pass
    else:
        raise ValueError("postion must be int, slice, None or Seqeuence")
    return position


def get_activations_and_grads(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    layers_components: Sequence[tuple[int, str]],
    metric_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    position: slice | int | Sequence | None = -1,
) -> tuple[ActivationDict, Optional[ActivationDict], torch.Tensor]:
    """Get activations and gradients of specific components at given layers."""

    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)
    position = _regularize_position(position)

    acts_output = ActivationDict(model.model.config, position)
    grads_output = (
        ActivationDict(model.model.config, position) if metric_fn is not None else None
    )

    context = torch.enable_grad if metric_fn is not None else torch.no_grad

    with context():
        with model.trace(input_ids, attention_mask, position_ids) as trace:
            for layer, component in layers_components:
                act, grad = _extract_or_patch(
                    model, trace, layer, component, position, capture_grad=True
                )
                acts_output[(layer, component)] = act.cpu()
                if grad is not None:
                    grads_output[(layer, component)] = grad.cpu()
            logits = model.lm_head.output[:, -1, :].save()

            if metric_fn is not None:
                metric = metric_fn(logits)
                metric.backward()

    return acts_output, grads_output, logits


def get_activations(*args, **kwargs):
    return get_activations_and_grads(*args, metric_fn=None, **kwargs)


def patch_activations(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    patching_dict: ActivationDict,
    position: slice | Sequence[int] | int | None = -1,
) -> torch.Tensor:
    if not patching_dict.fused_heads:
        raise ValueError("head activations must be fused before patching")

    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)

    position = _regularize_position(position)

    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:  # noqa: F841
            for (layer, component), patch in patching_dict.items():
                _, _ = _extract_or_patch(
                    model,
                    trace,
                    layer,
                    component,
                    position,
                    capture_grad=False,
                    patch_value=patch,
                )
            logits = model.lm_head.output[:, -1, :].save()
    return logits
