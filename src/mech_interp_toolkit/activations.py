import torch
from nnsight import NNsight
from .utils import input_dict_to_tuple, regularize_position
from collections.abc import Sequence, Callable
import warnings
from typing import Optional
import einops


class FrozenError(RuntimeError):
    """Raised when attempting to modify a frozen ActivationDict."""

    pass


class ActivationDict(dict):
    """
    A dictionary-like object to store and manage model activations.

    This class extends the standard dictionary to provide features specific to handling
    activations from neural networks, such as freezing the state, managing head
    dimensions, and moving data to a GPU.

    Args:
        config: The model's configuration object.
        positions: The sequence positions of the activations.
    """

    def __init__(self, config, positions):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.model_dim = config.hidden_size
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.fused_heads = True
        self.positions = positions
        self._frozen = False

    def reorganize(self):
        execution_order = {
            "layer_in": 0,
            "z": 1,
            "attn": 2,
            "mlp": 3,
            "layer_out": 4,
        }

        new_dict = ActivationDict(self.config, self.positions)
        keys = list(self.keys())
        keys = sorted(
            keys,
            key=lambda x: (x[0], execution_order[x[1]]),
        )

        for key in keys:
            new_dict[key] = self[key]
        return new_dict

    def freeze(self):
        """Freeze the ActivationDict, making it immutable."""
        self._frozen = True
        return self

    def unfreeze(self):
        """Unfreeze the ActivationDict, making it mutable."""
        self._frozen = False
        return self

    def _check_frozen(self):
        if self._frozen:
            raise FrozenError(
                "ActivationDict is frozen and cannot be modified, heads can still be split and merged."
            )

    def __setitem__(self, key, value):
        self._check_frozen()
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        self._check_frozen()
        return super().__delitem__(key)

    def clear(self):
        self._check_frozen()
        return super().clear()

    def pop(self, *args):
        self._check_frozen()
        return super().pop(*args)

    def popitem(self):
        self._check_frozen()
        return super().popitem()

    def setdefault(self, *args):
        self._check_frozen()
        return super().setdefault(*args)

    def update(self, *args, **kwargs):
        self._check_frozen()
        return super().update(*args, **kwargs)

    def split_heads(self):
        """
        Splits the 'z' activations into individual heads.
        Assumes 'z' activations are stored with fused heads.
        """
        pre_state = self._frozen
        self.unfreeze()
        if not self.fused_heads:
            return self

        for layer, component in self.keys():
            if component != "z":
                continue
            # Rearrange from (batch, pos, n_heads * d_head) to (batch, pos, n_heads, d_head)
            self[(layer, "z")] = einops.rearrange(
                self[(layer, "z")],
                "batch pos (head d_head) -> batch pos head d_head",
                head=self.num_heads,
                d_head=self.head_dim,
            )
        self.fused_heads = False
        self._frozen = pre_state
        return self

    def merge_heads(self):
        """
        Merges the 'z' activations from individual heads back into a single tensor.
        """
        pre_state = self._frozen
        self.unfreeze()
        if self.fused_heads:
            return self
        for layer, component in self.keys():
            if component != "z":
                continue
            # Rearrange from (batch, pos, n_heads, d_head) to (batch, pos, n_heads * d_head)
            self[(layer, "z")] = einops.rearrange(
                self[(layer, "z")],
                "batch pos head d_head -> batch pos (head d_head)",
                head=self.num_heads,
                d_head=self.head_dim,
            )
        self.fused_heads = True
        self._frozen = pre_state
        return self

    def apply(
        self, function: Callable[[torch.Tensor, int], torch.Tensor], dim: int = -1
    ) -> dict:
        output = dict()
        for layer, component in self.keys():
            output[(layer, component)] = function(self[(layer, component)], dim=dim)
        return output

    def cuda(self):
        """Moves all activation tensors to the GPU."""
        for key in self.keys():
            self[key] = self[key].cuda()
        return self

    def cpu(self):
        """Moves all activation tensors to the CPU."""
        for key in self.keys():
            self[key] = self[key].cpu()
        return self


def create_z_patch_dict(
    original_acts,
    new_acts,
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
) -> tuple[ActivationDict, Optional[ActivationDict], torch.Tensor]:
    """Get activations and gradients of specific components at given layers."""
    
    capture_grad = metric_fn is not None

    if capture_grad and stop_at_layer is not None:
        warnings.warn(
            "stop_at_layer is not compatible with gradient computation. Skipping gradient computation."
        )
        capture_grad = False

    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)
    position = regularize_position(position)

    acts_output = ActivationDict(model.model.config, position)
    grads_output = (
        ActivationDict(model.model.config, position) if capture_grad else None
    )

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
            logits = model.lm_head.output[:, -1, :].save()

            if capture_grad:
                metric = metric_fn(logits)
                metric.backward()

    return acts_output, grads_output, logits


def get_activations(*args, **kwargs):
    return get_activations_and_grads(*args, metric_fn=None, **kwargs)


def patch_activations(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    patching_dict: ActivationDict,
    layers_components: Optional[Sequence[tuple[int, str]]] = None,
    metric_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    position: slice | Sequence[int] | int | None = -1,
) -> torch.Tensor:
    if not patching_dict.fused_heads:
        raise ValueError("head activations must be fused before patching")
    
    capture_grad = metric_fn is not None

    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)

    position = regularize_position(position)

    acts_output = ActivationDict(model.model.config, position)
    grads_output = (
        ActivationDict(model.model.config, position) if capture_grad else None
    )

    logits = None
    grads = None

    patching_dict.unfreeze()
    patching_dict.update(
        [(x, None) for x in layers_components]
    )  # Need to find a better way to do this
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

                if (layer, component) in layers_components:
                    acts_output[(layer, component)] = acts
                    if grads is not None:
                        grads_output[(layer, component)] = grads

            logits = model.lm_head.output[:, -1, :].save()

            if capture_grad:
                metric = metric_fn(logits)
                metric.backward()

    return acts_output, grads_output, logits
