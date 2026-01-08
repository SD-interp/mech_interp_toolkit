import torch
from nnsight import NNsight
from .utils import ChatTemplateTokenizer, input_dict_to_tuple
import einops
from collections.abc import Sequence
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Optional, Literal
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

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
            raise FrozenError("ActivationDict is frozen and cannot be modified, heads can still be split and merged.")
        
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
        
        for (layer, component) in self.keys():
            if component != "z":
                continue
            # Rearrange from (batch, pos, n_heads * d_head) to (batch, pos, n_heads, d_head)
            self[(layer, "z")] = \
                einops.rearrange(self[(layer, "z")],
                                 "batch pos (head d_head) -> batch pos head d_head",
                                 head=self.num_heads, d_head=self.head_dim
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
        for (layer, component) in self.keys():
            if component != "z":
                continue
            # Rearrange from (batch, pos, n_heads, d_head) to (batch, pos, n_heads * d_head)
            self[(layer, "z")] = \
                einops.rearrange(self[(layer, "z")],
                                 "batch pos head d_head -> batch pos (head d_head)",
                                 head=self.num_heads, d_head=self.head_dim
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
        for (layer, component) in self.keys():
            output[(layer, component)] = self[(layer, component)].mean(dim=0)
        return output
   
    def cuda(self):
        """Moves all activation tensors to the GPU."""
        for key in self.keys():
            self[key] = self[key].cuda()
        return self
    
    def create_z_patch_dict(self, new_acts, layer_head: list[tuple[int, int]], position: None | int | Sequence[int] | slice =None):
        """
        Creates a new ActivationDict for patching 'z' activations.

        Args:
            new_acts: An ActivationDict containing the new activations.
            layer_head: A list of (layer, head) tuples to patch.
            position: The sequence position(s) to patch.

        Returns:
            A new ActivationDict with the patched activations.
        """
        assert not (self.fused_heads or new_acts.fused_heads), "Both ActivationDicts must have unfused heads for patching."
        
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
                warnings.warn("Patching all positions but ActivationDicts have different position sets.")
            position = slice(None)
                
        patch_dict = ActivationDict(self.config, position)
        patch_dict.fused_heads = False
        
        for layer, head in layer_head:
            patch_dict[(layer, "z")] = self[(layer, "z")].clone()
            patch_dict[(layer, "z")][:, position, head, :] = new_acts[(layer, "z")][:, position, head, :].clone()
        patch_dict.merge_heads()
        return patch_dict

class LinearProbe:
    def __init__(self, target_type: Literal["classification", "regression"],
                 broadcast_target: bool = True, test_split: float = 0.2, **kwargs):
        
        self.target_type = target_type
        self.broadcast_target = broadcast_target

        if test_split <= 0.0 or test_split >= 1.0:
             raise ValueError("test_split must be between 0.0 and 1.0")
        self.test_split = test_split
        
        self.linear_model = None
        
        # 3. FIX: Separate kwargs or handle explicitly to avoid passing invalid args
        if target_type == "classification":
            self.linear_model = LogisticRegression(**kwargs)
        elif target_type == "regression":
            self.linear_model = LinearRegression(**kwargs)
        else:
            raise ValueError("target_type must be 'classification' or 'regression'")
        
        self.weight: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
    
    def _process_batch(self, inputs: np.ndarray, target: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Helper to flatten/broadcast a specific batch (Train or Test)."""
        if inputs.ndim == 2:
            # Shape: (Batch, D_model) -> Treat as single position
            positions = 1
        elif inputs.ndim == 3:
            # Shape: (Batch, Pos, D_model)
            positions = inputs.shape[1]
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")

        d_model = inputs.shape[-1]
        
        inputs_flat = inputs.reshape(-1, d_model)
        
        target_flat = None
        if target is not None:
            if self.broadcast_target and positions > 1:
                if target.ndim != 1:
                    raise ValueError(
                        f"broadcast_target=True expects 1D targets (batch_size,), "
                        f"but got shape {target.shape}. If you have token-level targets, "
                        "set broadcast_target=False."
                    )
                target_flat = np.repeat(target, positions, axis=0)
            else:
                target_flat = target.reshape(-1) if target.ndim > 1 else target
                
        return inputs_flat, target_flat

    def prepare_data(self, activations: ActivationDict, target: torch.Tensor | np.ndarray):
        if len(activations) != 1:
            raise ValueError("Only single components are supported")
        
        # Raw inputs: (Batch, [Pos], D_model)
        inputs_full = list(activations.values())[0].cpu().numpy()
        
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        indices = np.arange(inputs_full.shape[0])
        train_idx, test_idx = train_test_split(indices, test_size=self.test_split)

        # Slice data based on indices
        X_train_raw = inputs_full[train_idx]
        y_train_raw = target[train_idx]
        X_test_raw = inputs_full[test_idx]
        y_test_raw = target[test_idx]

        # Now flatten/broadcast train and test independently
        X_train, y_train = self._process_batch(X_train_raw, y_train_raw)
        X_test, y_test = self._process_batch(X_test_raw, y_test_raw)
        
        return X_train, X_test, y_train, y_test

    def display_metrics(self, pred: np.ndarray, y: np.ndarray, label: str):
        if self.target_type == "classification":
            metric_name = "Accuracy"
            metric = accuracy_score(y, pred)
        elif self.target_type == "regression":
            metric_name = "RMSE"
            metric = root_mean_squared_error(y, pred)
        
        print(f"{label} {metric_name}: {metric:.4f}")

    def fit(self, activations: ActivationDict, target: torch.Tensor | np.ndarray):
        X_train, X_test, y_train, y_test = self.prepare_data(activations, target)
        
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        self.linear_model.fit(X_train, y_train)
        self.weight = self.linear_model.coef_
        self.bias = self.linear_model.intercept_
        
        pred_train = self.linear_model.predict(X_train)
        pred_test = self.linear_model.predict(X_test)
        
        self.display_metrics(pred_train, y_train, label="Train")
        self.display_metrics(pred_test, y_test, label="Test")
        
        return self

    def predict(self, activations: ActivationDict, target: Optional[torch.Tensor | np.ndarray] = None, label="Inference") -> np.ndarray:
        if self.weight is None: # Simple check
            raise ValueError("The linear probe has not been fitted yet.")

        inputs_full = list(activations.values())[0].cpu().numpy()
        
        if target is not None:
             if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

        # Process the entire batch for inference (no splitting needed here)
        inputs, target = self._process_batch(inputs_full, target)
        
        preds = self.linear_model.predict(inputs)
        print(f"{label} set size: {len(inputs)}")
        
        if target is not None:
            self.display_metrics(preds, target, label=label)

        return preds

def get_pre_rms_logit_diff_direction(token_pair: Sequence[str], tokenizer: ChatTemplateTokenizer, model: NNsight) -> torch.Tensor:
    """
    Calculates the direction in the residual stream that corresponds to the difference
    in logits between two tokens, before the final LayerNorm.

    Args:
        token_pair: A sequence of two tokens.
        tokenizer: The tokenizer.
        model: The NNsight model wrapper.

    Returns:
        The direction vector.
    """
    unembedding_matrix = model.get_output_embeddings().weight
    gamma = model.model.norm.weight  # (d_model,)
    token_ids = []
    assert len(token_pair) == 2, "Provide exactly two target tokens."

    for token in token_pair:
        encoding = tokenizer.tokenizer.encode(token, add_special_tokens=False)
        assert len(encoding) == 1, f"Token '{token}' is tokenized into multiple tokens."
        token_ids.append(encoding[0])

    post_rms_logit_diff_direction = unembedding_matrix[token_ids[0]] - unembedding_matrix[token_ids[1]]  # (d_model,)
    pre_rms_logit_diff_direction = post_rms_logit_diff_direction * gamma  # (d_model,)
    return pre_rms_logit_diff_direction

def get_activations(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    layers_components: Sequence[tuple[int, str]],
    position: slice | int | Sequence | None = -1,
) -> tuple[ActivationDict, torch.Tensor]:
    """
    Get activations of specific components at given layers.

    Components:
      - 'attn': attention output (post-attention, pre-residual)
      - 'mlp' : MLP output (post-MLP, pre-residual)
      - 'z'   : per-head attention output before o_proj
      - 'layer_in': Input to a transformer layer
      - 'layer_out': Output of a transformer layer

    Notes:
      - layer/component pairs must follow the forward execution order
    """
    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)
    attn_implementation = model.model.config._attn_implementation
    
    if isinstance(position, int):
        position = [position]
    elif position is None:
        position = slice(None)
    elif isinstance(position, (slice, Sequence)):
        pass
    else:
        raise ValueError("postion must be int, slice, None or Seqeuence")

    output: ActivationDict = ActivationDict(model.model.config, position)

    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:
            for layer, component in layers_components:
            
                if component == "attn":
                    act = model.model.layers[layer].self_attn.output[0][:, position, :].save()
                elif component == "mlp":
                    act = model.model.layers[layer].mlp.output[:, position, :].save()
                elif component == "z":
                    if attn_implementation != "eager":
                        warnings.warn(f"attn_implementation {attn_implementation} can give incorrect results")
                    z = model.model.layers[layer].self_attn.o_proj.input[:, position, :]
                    act = z.save()
                elif component == "layer_in":
                    act = model.model.layers[layer].input[:, position, :].save()
                elif component == "layer_out":
                    act = model.model.layers[layer].output[:, position, :].save()
                else:
                    raise ValueError(
                        "component must be one of {'attn', 'mlp', 'z', 'layer_in', 'layer_out'}"
                    )
                assert act.ndim == 3, "Malformed activation tensor"
                output[(layer, component)] = act.cpu()
            logits = model.lm_head.output[:,-1,:].save()
    return output, logits

def patch_activations(model: NNsight, 
                   inputs: dict[str, torch.Tensor],
                   patching_dict: ActivationDict,
                   position:  slice | Sequence[int] | int | None = -1,
                   ) -> torch.Tensor:
    """ 
    Patches activations of specific components at given layers and returns the new logits.
    
    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        patching_dict: An ActivationDict containing the activations to patch.
        position: The sequence position(s) to apply the patch.

    Returns:
        The logits after patching.
    """
    if not patching_dict.fused_heads:
        raise ValueError("head activations must be fused before patching")
    
    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)
    
    attn_implementation = model.model.config._attn_implementation
    
    if isinstance(position, int):
        position = [position]
    elif position is None:
        position = slice(None)
        
    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:
            for (layer, component), patch in patching_dict.items():
                if component == 'attn':
                    model.model.layers[layer].self_attn.output[0][:, position, :] = patch
                elif component == 'mlp':
                    model.model.layers[layer].mlp.output[:, position, :] = patch
                elif component == 'z':
                    if attn_implementation != "eager":
                        warnings.warn(f"attn_implementation {attn_implementation} might give incorrect results")
                    model.model.layers[layer].self_attn.o_proj.input[:, position, :] = patch
                elif component == 'layer_in':
                    model.model.layers[layer].input[:, position, :] = patch
                elif component == 'layer_out':
                    model.model.layers[layer].output[:, position, :] = patch
                else:
                    raise ValueError("Component must be 'attn', 'mlp', 'z', 'layer_in', or 'layer_out'.")
            logits = model.lm_head.output[:,-1,:].save()
    return logits

def run_layerwise_dla(model: NNsight, 
                      inputs: dict[str, torch.Tensor],
                      pre_rms_direction: torch.Tensor,
                      eps: float = 1e-6) -> dict[str, dict[int, torch.Tensor]]:
    """
    Performs layer-wise Direct Logit Attribution (DLA).

    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        pre_rms_direction: The direction vector in the residual stream.
        eps: A small value to prevent division by zero.

    Returns:
        A dictionary containing the DLA results for attention and MLP layers.
    """
    attn_records = dict()
    mlp_records = dict()
    n_layers = model.model.config.num_hidden_layers
    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)
        
    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:
            for layer in range(n_layers):
                attn_records[layer] = model.model.layers[layer].self_attn.output[0][:, -1, :].save()
                mlp_records[layer] = model.model.layers[layer].mlp.output[:, -1, :].save()
            rms_final = model.model.layers[-1].output[:, -1, :].norm(dim=-1, keepdim=True).save()

    # The divisor for DLA is the L2 norm of the final residual stream.
    divisor = torch.sqrt(rms_final**2 + eps).squeeze()

    attn_dla = {
        k: (v @ pre_rms_direction) / divisor
        for k, v in attn_records.items()
    }
    
    mlp_dla = {
        k: (v @ pre_rms_direction) / divisor
        for k, v in mlp_records.items()
    }
    
    return {'attn': attn_dla, 'mlp': mlp_dla}

def run_headwise_dla_for_layer(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    pre_rms_direction: torch.Tensor,
    layer: int,
    eps: float = 1e-6,
    scale: bool = True,
) -> torch.Tensor:
    """
    Performs head-wise Direct Logit Attribution (DLA) for a specific layer.

    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        pre_rms_direction: The direction vector in the residual stream.
        layer: The layer index.
        eps: A small value to prevent division by zero.
        scale: Whether to scale the DLA results by the final LayerNorm.

    Returns:
        A tensor containing the DLA results for each head.
    """
    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)
    
    proj_weight = model.model.layers[layer].self_attn.o_proj.weight
    num_heads = model.model.config.num_attention_heads

    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:
            head_inputs = model.model.layers[layer].self_attn.o_proj.input[:, -1, :].save()
            rms_final = model.model.layers[-1].output[:, -1, :].norm(dim=-1, keepdim=True).save()

    divisor = torch.sqrt(rms_final**2 + eps)
    
    batch_size = head_inputs.shape[0]
    head_inputs = head_inputs.view(batch_size, num_heads, -1)

    W_O = proj_weight.view(proj_weight.shape[0], num_heads, -1)

    # Calculate the contribution of each head to the final output in the given direction.
    projections = einops.einsum(head_inputs, W_O, pre_rms_direction,
                              "batch n_heads head_dim, d_model n_heads head_dim, d_model -> batch n_heads")
        
    return projections / divisor if scale else projections

def get_attention_pattern(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    layers: list[int],
    head_indices: list[tuple[int]],
    query_position: int = -1,
    ) -> dict[int, torch.Tensor]:
    """ 
    Retrieves the attention patterns for specific heads and layers.

    Args:
        model: The NNsight model wrapper.
        inputs: The input tensors for the model.
        layers: A list of layer indices.
        head_indices: A list of head indices for each layer.
        query_position: The position of the query token.

    Returns:
        A dictionary mapping layer indices to attention patterns.
    """
    assert model.model.config._attn_implementation == "eager", "Model must use eager attention to get attention patterns."
    output = dict()
    
    assert len(layers) == len(head_indices), "each layer# provided must have corresponding head indices"
    
    input_ids, attention_mask, position_ids = input_dict_to_tuple(inputs)
    
    with torch.no_grad():
        with model.trace(input_ids, attention_mask, position_ids) as trace:
            for i, layer in enumerate(layers):
                output[layer] = model.model.layers[layer].self_attn.output[1][:, head_indices[i],query_position, :].save()
            trace.stop()
    
    return output
