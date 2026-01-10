import torch
from nnsight import NNsight
from .utils import input_dict_to_tuple
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

    def get_mean_activations(self) -> dict:
        """
        Computes the mean of activations across the batch dimension.

        Returns:
            A dictionary with the same keys but with mean activations.
        """
        output = dict()
        for layer, component in self.keys():
            output[(layer, component)] = self[(layer, component)].mean(dim=0)
        return output

    def cuda(self):
        """Moves all activation tensors to the GPU."""
        for key in self.keys():
            self[key] = self[key].cuda()
        return self

    def create_z_patch_dict(
        self,
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
        assert not (self.fused_heads or new_acts.fused_heads), (
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

            self_check = check_pos(self.positions, position)
            new_check = check_pos(new_acts.positions, position)

            if not self_check or not new_check:
                raise ValueError("For cross-position patching, implement custom logic.")
        elif position is None:
            if self.positions != new_acts.positions:
                warnings.warn(
                    "Patching all positions but ActivationDicts have different position sets."
                )
            position = slice(None)

        patch_dict = ActivationDict(self.config, position)
        patch_dict.fused_heads = False

        for layer, head in layer_head:
            patch_dict[(layer, "z")] = self[(layer, "z")].clone()
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
