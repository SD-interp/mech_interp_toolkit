import warnings
from collections.abc import Callable, Sequence
from typing import Any, NotRequired, Optional, TypedDict, cast

import torch
from nnsight import NNsight

from .activation_dict import ActivationDict, LayerComponent, LayerHead, Position
from .utils import input_dict_to_tuple, regularize_position


class AccessGradients(TypedDict):
    metric_fn: Callable[[torch.Tensor], torch.Tensor]
    compute_metric_at: LayerComponent


class AccessActivations(TypedDict):
    positions: Position
    locations: list[LayerComponent]
    gradients: NotRequired[AccessGradients]


class SpecDict(TypedDict):
    patching: NotRequired[ActivationDict]
    activations: NotRequired[AccessActivations]
    stop_at_layer: NotRequired[int]


def create_z_patch_dict(
    original_acts: ActivationDict,
    new_acts: ActivationDict,
    layer_head: list[LayerHead],
    position: Position = None,
) -> ActivationDict:
    """
    Creates a new ActivationDict for patching 'z' activations.

    Args:
        new_acts: An ActivationDict containing the new activations.
        layer_head: A list of (layer, head) tuples to patch.
        position: The sequence position(s) to patch.

    Returns:
        A new ActivationDict with the patched activations.
    """
    if original_acts.fused_heads or new_acts.fused_heads:
        raise ValueError("Both ActivationDicts must have unfused heads for patching.")

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


def _locate_layer_component(model: NNsight, trace: Any, layer: int, component: str) -> Any:
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
        raise ValueError("component must be one of {'attn', 'mlp', 'z', 'layer_in', 'layer_out'}")
    return comp


class UnifiedAccessAndPatching:
    def __init__(
        self,
        model: NNsight,
        inputs: dict[str, torch.Tensor],
        spec_dict: SpecDict,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        input_tuple = input_dict_to_tuple(inputs)
        self.input_ids, self.attention_mask, self.position_ids = input_tuple
        self.model = model
        self.spec_dict = spec_dict
        self.inputs_embeds = inputs_embeds

        self.stop_at_layer = spec_dict.get("stop_at_layer")

        self.patching_dict = spec_dict.get("patching")
        self.acts_dict = spec_dict.get("activations")

        self.loop_components = ActivationDict(model.model.config, -1)  # dummy

        if self.patching_dict:
            temp = [(x, None) for x in self.patching_dict]
            self.loop_components.update(temp)

        if self.acts_dict:
            self.acts_dict["positions"] = regularize_position(self.acts_dict["positions"])
            temp = [(x, None) for x in self.acts_dict["locations"]]
            self.loop_components.update(temp)
            self.output = ActivationDict(model.model.config, self.acts_dict["positions"])
            self._capture_grads = bool(self.acts_dict.get("gradients"))

            if self._capture_grads and self.acts_dict["positions"] != slice(None):
                warnings.warn(
                    """slicing and indexing when capturing gradients is not supported.
                    All positions will be captured by default. Use output.extract_positions() instead"""
                )
        else:
            self._capture_grads = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # clean-up references
        del self.model
        del self.input_ids
        del self.attention_mask
        del self.position_ids
        del self.spec_dict
        del self.loop_components
        del self.patching_dict
        del self.acts_dict
        del self._capture_grads
        del self.output

    def warning_for_attn_type(self, loop_components: ActivationDict) -> None:
        attn_implementation = self.model.model.config._attn_implementation  # type: ignore
        for _, component in loop_components:
            if component == "z":
                if attn_implementation != "eager":
                    warnings.warn(
                        f"attn_implementation '{attn_implementation}' can give incorrect results for z"
                    )

    def patch_fn(
        self, original: torch.Tensor, new_value: torch.Tensor, position: Position
    ) -> torch.Tensor:
        mask = torch.zeros_like(original, dtype=torch.bool)
        mask[:, position, :] = True
        return torch.where(mask, new_value, original)

    def unified_access_and_patching(self) -> tuple[Optional[ActivationDict], torch.Tensor]:
        self.warning_for_attn_type(self.loop_components)

        context = torch.enable_grad if self._capture_grads else torch.no_grad
        model_input_dict = {
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
        }

        if self.inputs_embeds is None:
            model_input_dict["input_ids"] = self.input_ids
        else:
            model_input_dict["inputs_embeds"] = self.inputs_embeds

        with (
            context(),
            self.model.trace() as tracer,
        ):
            with tracer.invoke(**model_input_dict):
                for layer, component in self.loop_components:
                    # This is here to prevent autoformatter from removing line
                    if self.stop_at_layer is not None:
                        if layer > self.stop_at_layer:
                            tracer.stop()

                    comp = _locate_layer_component(self.model, tracer, layer, component)

                    if self.patching_dict is not None and (layer, component) in self.patching_dict:
                        patch_pos = self.patching_dict.positions
                        patch_pos = regularize_position(patch_pos)
                        comp[:] = self.patch_fn(
                            comp, self.patching_dict[(layer, component)], patch_pos
                        )

                    if (
                        self.acts_dict is not None
                        and (layer, component) in self.acts_dict["locations"]
                    ):
                        if self._capture_grads:
                            self.output[(layer, component)] = comp.save()
                            self.output[(layer, component)].retain_grad()
                        else:
                            cache_pos = self.acts_dict["positions"]
                            self.output[(layer, component)] = comp[:, cache_pos, :].save()

                logits = self.model.lm_head.output[:, [-1], :].save()  # type: ignore

                if self._capture_grads:
                    metric_fn = self.acts_dict["gradients"]["metric_fn"]  # type: ignore
                    metric_layer, metric_component = self.acts_dict["gradients"][  # type: ignore
                        "compute_metric_at"
                    ]

                    if metric_component == "logits":
                        metric = metric_fn(logits)
                    else:
                        metric = metric_fn(self.output[(metric_layer, metric_component)])

                    metric.backward()

        return self.output if hasattr(self, "output") else None, logits

    @staticmethod
    def metric_fn_example(acts: torch.Tensor) -> torch.Tensor:
        return acts[:, -1, :].sum()
