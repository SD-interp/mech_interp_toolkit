import torch
from nnsight import NNsight
from .utils import input_dict_to_tuple, regularize_position
from collections.abc import Sequence, Callable
import warnings
from typing import Optional, Any, cast
from .activation_dict import ActivationDict


def create_z_patch_dict(
    original_acts: ActivationDict,
    new_acts: ActivationDict,
    layer_head: list[tuple[int, int]],
    position: None | int | Sequence[int] | slice = None,
):
    """
    Creates a new ActivationDict for patching 'z' activations.

    Args:
        new_acts: An ActivationDict containing the new activations.
        layer_head: A list of (layer, head) tuples to patch.
        position: The sequence position(s) to patch.

    Returns:
        A new ActivationDict with the patched activations.
    """
    assert not (original_acts.fused_heads or new_acts.fused_heads), (
        "Both ActivationDicts must have unfused heads for patching."
    )

    if isinstance(position, int):
        position = [position]

    if isinstance(position, Sequence):

        def check_pos(pos_spec, pos_list):
            if isinstance(pos_spec, slice):
                if pos_spec.start is None and pos_spec.stop is None:
                    return True

                # Assume positive indices
                max_pos = 0
                if pos_list:
                    max_pos = max(pos_list)

                start, stop, step = pos_spec.indices(max_pos + 1)
                return all(p in range(start, stop, step) for p in pos_list)
            else:
                return all(p in pos_spec for p in pos_list)

        self_check = check_pos(original_acts.positions, position)
        new_check = check_pos(new_acts.positions, position)

        if not self_check or not new_check:
            raise ValueError("For cross-position patching, implement custom logic.")
    elif position is None:
        if original_acts.positions != new_acts.positions:
            warnings.warn(
                "Patching all positions but ActivationDicts have different position sets."
            )
        position = slice(None)

    patch_dict = ActivationDict(original_acts.config, position)
    patch_dict.fused_heads = False

    for layer, head in layer_head:
        patch_dict[(layer, "z")] = original_acts[(layer, "z")].clone()
        patch_dict[(layer, "z")][:, position, head, :] = new_acts[(layer, "z")][
            :, position, head, :
        ].clone()
    patch_dict.merge_heads()
    return patch_dict


def _locate_layer_component(model, trace, layer: int, component: str):
    if trace is None:
        raise ValueError("Active trace is required to locate layer components.")

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
            f"attn_implementation '{attn_implementation}' can give incorrect results for z patches or gradients."
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


def get_activations_and_grads(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    layers_components: Sequence[tuple[int, str]],
    metric_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    position: slice | int | Sequence | None = -1,
    stop_at_layer: Optional[int] = None,
) -> tuple[ActivationDict, ActivationDict, torch.Tensor]:
    """Get activations and gradients of specific components at given layers."""

    capture_grad = metric_fn is not None

    if capture_grad and stop_at_layer is not None:
        warnings.warn(
            "stop_at_layer is not compatible with gradient computation. Skipping gradient computation."
        )
        capture_grad = False

    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)
    position = regularize_position(position)

    acts_output = ActivationDict(model.model.config, position, value_type="activation")
    grads_output = ActivationDict(model.model.config, position, value_type="gradient")

    acts_output.attention_mask = attention_mask
    grads_output.attention_mask = attention_mask

    context = torch.enable_grad if capture_grad else torch.no_grad

    logits = None

    with context():
        with model.trace(input_ids, attention_mask, position_ids) as tracer:
            for layer, component in layers_components:
                if stop_at_layer is not None:
                    if layer >= stop_at_layer:
                        tracer.stop()
                act, grad = _extract_or_patch(
                    model, tracer, layer, component, position, capture_grad=capture_grad
                )
                acts_output[(layer, component)] = act
                if grad is not None:
                    grads_output[(layer, component)] = grad
            logits = model.lm_head.output[:, -1, :].save()  # type: ignore

            if capture_grad:
                metric = metric_fn(logits)  # type: ignore
                metric.backward()

    return acts_output, grads_output, logits


def get_activations(*args, **kwargs):
    return get_activations_and_grads(*args, metric_fn=None, **kwargs)


def patch_activations(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    patching_dict: ActivationDict,
    layers_components: Sequence[tuple[int, str]] = [],
    metric_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    position: slice | Sequence[int] | int | None = -1,
) -> tuple[ActivationDict, ActivationDict, torch.Tensor]:
    if not patching_dict.fused_heads:
        raise ValueError("head activations must be fused before patching")

    capture_grad = metric_fn is not None

    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)

    position = regularize_position(position)

    acts_output = ActivationDict(model.model.config, position, value_type="activation")
    grads_output = ActivationDict(model.model.config, position, value_type="gradient")

    acts_output.attention_mask = attention_mask
    grads_output.attention_mask = attention_mask

    logits = None
    grads = None

    patching_dict.unfreeze()
    patching_dict.update([(x, None) for x in layers_components])
    patching_dict.reorganize()

    context = torch.enable_grad if capture_grad else torch.no_grad

    with context():
        with model.trace(input_ids, attention_mask, position_ids) as trace:  # noqa: F841
            for (layer, component), patch in patching_dict.items():
                if (layer, component) not in layers_components:
                    _capture_grad = False
                else:
                    _capture_grad = capture_grad

                acts, grads = _extract_or_patch(
                    model,
                    trace,
                    layer,
                    component,
                    position,
                    capture_grad=_capture_grad,
                    patch_value=patch,
                )

            logits = model.lm_head.output[:, -1, :].save()  # type: ignore
            if capture_grad:
                with metric_fn(logits).backward():          # type: ignore
                    for (layer, component), patch in reversed(patching_dict.items()):
                        if (layer, component) in layers_components:
                            acts_output[(layer, component)] = acts
                            if grads is not None:
                                grads_output[(layer, component)] = grads
                    

    return acts_output, grads_output, logits
