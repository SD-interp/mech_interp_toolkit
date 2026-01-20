import warnings
from abc import ABC
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Literal, Optional, Self

import einops
import torch

type Position = slice | int | Sequence | None
type LayerComponent = tuple[int, str]
type LayerHead = tuple[int, int]


class FrozenError(RuntimeError):
    """Raised when attempting to modify a frozen ActivationDict."""

    pass


class FreezableDict(ABC, dict[LayerComponent, torch.Tensor]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._frozen = False

    def freeze(self) -> Self:
        """Freeze the dictionary, making it immutable."""
        self._frozen = True
        return self

    def unfreeze(self) -> Self:
        """Unfreeze the dictionary, making it mutable."""
        self._frozen = False
        return self

    def _check_frozen(self):
        if getattr(self, "_frozen", False):
            raise FrozenError("This object is frozen and cannot be modified.")

    def __setitem__(self, key, value):
        self._check_frozen()
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        self._check_frozen()
        return super().__delitem__(key)

    def clear(self) -> None:
        self._check_frozen()
        return super().clear()

    def pop(self, *args) -> torch.Tensor:
        self._check_frozen()
        return super().pop(*args)

    def popitem(self) -> tuple[LayerComponent, torch.Tensor]:
        self._check_frozen()
        return super().popitem()

    def setdefault(self, *args) -> torch.Tensor:
        self._check_frozen()
        return super().setdefault(*args)

    def update(self, *args, **kwargs) -> None:
        self._check_frozen()
        return super().update(*args, **kwargs)

    def clone(self) -> Self:
        return deepcopy(self)


class ArithmeticOperation(FreezableDict):
    def __init__(self, config=None, positions=None) -> None:
        super().__init__()
        self.config = config
        self.positions = positions

    def check_compatibility(self, other) -> None:
        if not isinstance(other, ActivationDict):
            raise ValueError("Operand must be an instance of ActivationDict.")

        if self.keys() != other.keys():
            warnings.warn(
                "ActivationDicts have different keys; only matching keys will be processed."
            )

    def __add__(self, other) -> Self:
        self.check_compatibility(other)
        if isinstance(other, ActivationDict):
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] + other[key]
            return result
        else:
            raise NotImplementedError("Addition only supported between ActivationDicts.")

    def __radd__(self, other) -> Self:
        return self.__add__(other)

    def __sub__(self, other) -> Self:
        self.check_compatibility(other)
        if isinstance(other, ActivationDict):
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] - other[key]
            return result
        else:
            raise NotImplementedError("Subtraction only supported between ActivationDicts.")

    def __mul__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_compatibility(other)
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] * other[key]
            return result
        elif isinstance(other, (int, float, torch.Tensor)):
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                result[key] = self[key] * other
            return result
        else:
            raise NotImplementedError("Multiplication not supported for this type.")

    def __rmul__(self, other) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_compatibility(other)
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] / other[key]
            return result
        elif isinstance(other, (int, float, torch.Tensor)):
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                result[key] = self[key] / other
            return result
        else:
            raise NotImplementedError("Division not supported for this type.")


class ActivationDict(ArithmeticOperation):
    """
    A dictionary-like object to store and manage model activations.

    This class extends the standard dictionary to provide features specific to handling
    activations from neural networks, such as freezing the state, managing head
    dimensions, and moving data to a GPU.

    Args:
        config: The model's configuration object.
        positions: The sequence positions of the activations.
    """

    def __init__(
        self,
        config,
        positions,
        value_type: Literal["gradient", "scores", "activation"] = "activation",
    ):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.model_dim = config.hidden_size
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.fused_heads = True
        self.positions = positions
        self.value_type = value_type  # e.g., 'activation' or 'gradient' or 'scores'
        self.attention_mask = torch.tensor([])  # Placeholder for attention mask

    def reorganize(self) -> Self:
        execution_order = {
            "layer_in": 0,
            "z": 1,
            "attn": 2,
            "mlp": 3,
            "layer_out": 4,
        }

        new_dict = type(self)(self.config, self.positions)
        keys = list(self.keys())
        keys = sorted(
            keys,
            key=lambda x: (x[0], execution_order[x[1]]),
        )

        for key in keys:
            new_dict[key] = self[key]
        return new_dict

    def split_heads(self) -> Self:
        """
        Splits the 'z' activations into individual heads.
        Assumes 'z' activations are stored with fused heads.
        """
        if self.value_type not in ["activation", "gradient"]:
            warnings.warn(
                f"Splitting heads is typically only relevant for activations or gradients, not '{self.value_type}'."
            )

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

    def merge_heads(self) -> Self:
        """
        Merges the 'z' activations from individual heads back into a single tensor.
        """
        if self.value_type not in ["activation", "gradient"]:
            warnings.warn(
                f"Splitting heads is typically only relevant for activations or gradients, not '{self.value_type}'."
            )

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
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        *args,
        **kwargs,
    ) -> Self:
        output = type(self)(self.config, self.positions)
        output.value_type = self.value_type
        output.attention_mask = self.attention_mask
        for layer, component in self.keys():
            output[(layer, component)] = function(self[(layer, component)], *args, **kwargs)

        return output

    def cuda(self) -> Self:
        """Moves all activation tensors to the GPU."""
        for key in self.keys():
            self[key] = self[key].cuda()
        return self

    def cpu(self) -> Self:
        """Moves all activation tensors to the CPU."""
        for key in self.keys():
            self[key] = self[key].cpu()
        return self

    def zeros_like(self, keys: Optional[list[LayerComponent]] = None) -> Self:
        """
        Creates a new ActivationDict with zero tensors for the specified keys,
        matching the shapes of the original tensors.
        """
        new_obj = type(self)(self.config, self.positions)
        new_obj.value_type = self.value_type
        new_obj.attention_mask = self.attention_mask

        if keys is None:
            keys = list(self.keys())

        for key in keys:
            if key in self:
                new_obj[key] = torch.zeros_like(self[key])
            else:
                warnings.warn(f"Key {key} not found in ActivationDict. Skipping.")

        return new_obj

    def extract_positions(self, keys: Optional[list[LayerComponent]] = None) -> Self:
        new_obj = type(self)(self.config, self.positions)
        new_obj.value_type = self.value_type
        new_obj.attention_mask = self.attention_mask

        if keys is None:
            keys = list(self.keys())

        for key in keys:
            if key in self:
                new_obj[key] = self[key][:, self.positions, :]
                if self[key].grad is not None:
                    new_obj[key].grad = self[key].grad[:, self.positions, :]  # type: ignore
            else:
                warnings.warn(f"Key {key} not found in ActivationDict. Skipping.")

        return new_obj

    def get_grads(self, keys: Optional[list[LayerComponent]] = None) -> Self:
        new_obj = type(self)(self.config, self.positions)
        new_obj.value_type = "gradient"
        new_obj.attention_mask = self.attention_mask

        if keys is None:
            keys = list(self.keys())

        for key in keys:
            if key in self:
                new_obj[key] = self[key].grad
            else:
                warnings.warn(f"Key {key} not found in ActivationDict. Skipping.")

        return new_obj
