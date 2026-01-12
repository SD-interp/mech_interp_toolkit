import torch
import einops
from typing import Callable, Literal, Self
from abc import ABC
import warnings


class FrozenError(RuntimeError):
    """Raised when attempting to modify a frozen ActivationDict."""

    pass


class ArithmeticOperation(ABC, dict[tuple[int, str], torch.Tensor]):
    def __init__(self, config=None, positions=None) -> None:
        super().__init__()
        self.config = config
        self.positions = positions

    def check_compatibility(self, other):
        if not isinstance(other, ActivationDict):
            raise ValueError("Operand must be an instance of ActivationDict.")
        if self.config != other.config:
            raise ValueError(
                "ActivationDicts must have the same config for arithmetic operations."
            )
        if self.positions != other.positions:
            raise ValueError(
                "ActivationDicts must have the same positions for arithmetic operations."
            )
        if self.keys() != other.keys():
            raise ValueError(
                "ActivationDicts must have the same keys for arithmetic operations."
            )

    def __add__(self, other) -> Self:
        self.check_compatibility(other)
        if isinstance(other, ActivationDict):
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                result[key] = self[key] + other[key]
            return result
        else:
            raise NotImplementedError(
                "Addition only supported between ActivationDicts."
            )
    
    def __radd__(self, other) -> Self:
        return self.__add__(other)
    
    def __sub__(self, other) -> Self:
        self.check_compatibility(other)
        if isinstance(other, ActivationDict):
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                result[key] = self[key] - other[key]
            return result
        else:
            raise NotImplementedError(
                "Subtraction only supported between ActivationDicts."
            )

    def __mul__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_compatibility(other)
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                result[key] = self[key] * other[key]
            return result
        elif isinstance(other, (int, float, torch.Tensor)):
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                result[key] = self[key] * other
            return result
        else:
            raise NotImplementedError(
                "Multiplication not supported for this type."
            )

    def __rmul__(self, other) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_compatibility(other)
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                result[key] = self[key] / other[key]
            return result
        elif isinstance(other, (int, float, torch.Tensor)):
            result = type(self)(self.config, self.positions)
            for key in self.keys():
                result[key] = self[key] / other
            return result
        else:
            raise NotImplementedError(
                "Division not supported for this type."
            )


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
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.model_dim = config.hidden_size
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.fused_heads = True
        self.positions = positions
        self._frozen = False
        self.value_type = value_type  # e.g., 'activation' or 'gradient' or 'scores'
        self.attention_mask = torch.tensor([])  # Placeholder for attention mask

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

    def merge_heads(self):
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
        *args, **kwargs,
    ) -> Self:
        output = type(self)(self.config, self.positions)
        output.value_type = self.value_type
        output.attention_mask = self.attention_mask
        for layer, component in self.keys():
            output[(layer, component)] = function(
                self[(layer, component)], *args, **kwargs)
    
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
