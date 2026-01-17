import warnings
from collections.abc import Callable, Sequence
from typing import Any, NotRequired, TypedDict, cast

import torch
from nnsight import NNsight

from .activation_dict import ActivationDict
from .utils import input_dict_to_tuple, regularize_position

type Position = slice | int | Sequence | None
type LayerComponent = list[tuple[int, str]]
type LayerHead = list[tuple[int, int]]


class AccessGradients(TypedDict):
    metric_fn: Callable[[torch.Tensor], torch.Tensor]
    compute_metric_at: LayerComponent


class AccessActivations(TypedDict):
    positions: Position
    locations: LayerComponent
    gradients: AccessGradients


class SpecDict(TypedDict):
    patching: NotRequired[ActivationDict]
    activations: NotRequired[AccessActivations]
    stop_at_layer: NotRequired[int]


def create_z_patch_dict(
    original_acts: ActivationDict,
    new_acts: ActivationDict,
    layer_head: LayerHead,
    position: Position = None,
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


class UnifiedAccessAndPatching:
    def __init__(
        self,
        model: NNsight,
        inputs: dict[str, torch.Tensor],
        spec_dict: SpecDict,
    ):
        input_tuple = input_dict_to_tuple(inputs)
        self.input_ids, self.attention_mask, self.position_ids = input_tuple
        self.model = model
        self.spec_dict = spec_dict

        self.stop_at_layer = spec_dict.get("stop_at_layer")

        self.patching_dict = spec_dict.get("patching")
        self.acts_dict = spec_dict.get("activations")

        self.loop_components = ActivationDict(model.model.config, -1)  # dummy

        if self.patching_dict:
            temp = [(x, None) for x in self.patching_dict]
            self.loop_components.update(temp)

        if self.acts_dict:
            temp = [(x, None) for x in self.acts_dict["locations"]]
            self.loop_components.update(temp)
            self.output = ActivationDict(
                model.model.config, self.acts_dict["positions"]
            )
            self.capture_grads = "gradients" in self.acts_dict
        else:
            self.capture_grads = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def warning_for_attn_type(self, loop_components):
        attn_implementation = self.model.model.config._attn_implementation  # type: ignore
        for _, component in loop_components:
            if component == "z":
                if attn_implementation != "eager":
                    warnings.warn(
                        f"attn_implementation '{attn_implementation}' can give incorrect results for z"
                    )

    @staticmethod
    def patch_fn(original, new_value, position):
        original = original.clone()
        original[:, position, :] = new_value
        return original

    def unified_access_and_patching(self):
        self.warning_for_attn_type(self.loop_components)

        context = torch.enable_grad if self.capture_grads else torch.no_grad

        with (
            context(),
            self.model.trace(
                self.input_ids, self.attention_mask, self.position_ids
            ) as tracer,
        ):
            for layer, component in self.loop_components:
                # This is here to prevent autoformatter from removing line
                if self.stop_at_layer is not None:
                    if layer > self.stop_at_layer:
                        tracer.stop()

                comp = _locate_layer_component(self.model, tracer, layer, component)

                if (
                    self.patching_dict is not None
                    and (layer, component) in self.patching_dict
                ):
                    patch_pos = self.patching_dict.positions
                    patch_pos = regularize_position(patch_pos)
                    comp[:] = self.patch_fn(
                        comp, self.patching_dict[(layer, component)], patch_pos
                    )

                if (
                    self.acts_dict is not None
                    and (layer, component) in self.acts_dict["locations"]
                ):
                    if self.capture_grads:
                        self.output[(layer, component)] = comp.save()
                        self.output[(layer, component)].retain_grad()
                    else:
                        cache_pos = self.acts_dict["positions"]
                        self.output[(layer, component)] = comp[:, cache_pos, :].save()

            logits = self.model.lm_head.output[:, [-1], :].save()  # type: ignore

            if self.capture_grads:
                assert self.acts_dict is not None

                metric_fn = self.acts_dict["gradients"]["metric_fn"]
                metric_layer, metric_component = self.acts_dict["gradients"][
                    "compute_metric_at"
                ]

                if metric_component == "logits":
                    metric = metric_fn(logits)
                else:
                    metric = metric_fn(self.output[(metric_layer, metric_component)])

                metric.backward()

        return self.output if hasattr(self, "output") else None, logits

    @staticmethod
    def patch_fn_example(acts):
        return acts[:, -1, :].sum()
