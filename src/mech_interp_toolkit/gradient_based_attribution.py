import torch
from nnsight import NNsight
from collections.abc import Sequence, Callable
from .activations import get_activations_and_grads, get_activations, patch_activations
from typing import Literal
from .activations import ActivationsDict
from .utils import regularize_position


def _get_acts_and_grads(
    model: NNsight,
    clean_inputs: dict[str, torch.Tensor],
    corrupted_inputs: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = torch.mean,
    position: slice | int | Sequence | None = -1,
) -> tuple[ActivationsDict, ActivationsDict, ActivationsDict]:
    """
    Helper function to get activations and gradients for clean and corrupted inputs.
    """
    n_layers = model.model.config.num_hidden_layers

    layer_components = [(i, c) for i in range(n_layers) for c in ["attn", "mlp"]]

    if compute_grad_at == "clean":
        clean_acts, grads, _ = get_activations_and_grads(
            model,
            clean_inputs,
            layers_components=layer_components,
            metric_fn=metric_fn,
            position=position,
        )
        clean_acts.cpu()
        grads.cpu()

        corrupted_acts, _, _ = get_activations(
            model,
            corrupted_inputs,
            layers_components=layer_components,
            position=position,
        )
        corrupted_acts.cpu()

    elif compute_grad_at == "corrupted":
        corrupted_acts, grads, _ = get_activations_and_grads(
            model,
            corrupted_inputs,
            layers_components=layer_components,
            metric_fn=metric_fn,
            position=position,
        )
        corrupted_acts.cpu()
        grads.cpu()

        clean_acts, _, _ = get_activations(
            model,
            clean_inputs,
            layers_components=layer_components,
            position=position,
        )
        clean_acts.cpu()
    else:
        raise ValueError("compute_grad_at must be either 'clean' or 'corrupted'")

    return clean_acts, corrupted_acts, grads


def _interpolate_activations(
    clean_activations: torch.Tensor | ActivationsDict,
    baseline_activations: torch.Tensor | ActivationsDict,
    alpha: float,
    key: tuple[int, str] = (0, "layer_in"),
) -> ActivationsDict:
    """
    Interpolates between clean and corrupted inputs.
    """
    if isinstance(clean_activations, ActivationsDict):
        clean_activations = clean_activations[key]
    
    if isinstance(baseline_activations, ActivationsDict):
        baseline_activations = baseline_activations[key]
    
    interpolated_activations = ActivationsDict()
    interpolated_activations[key] = (1 - alpha) * clean_activations + alpha * baseline_activations
    return interpolated_activations

def edge_attribution_patching(
    model: NNsight,
    clean_inputs: dict[str, torch.Tensor],
    corrupted_inputs: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = torch.mean,
    position: slice | int | Sequence | None = -1,
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Computes edge attributions for attention heads using simple gradient x activation.
    """

    clean_acts, corrupted_acts, grads = _get_acts_and_grads(
        model,
        clean_inputs,
        corrupted_inputs,
        compute_grad_at=compute_grad_at,
        metric_fn=metric_fn,
        position=position,
    )

    eap_scores = {}

    for layer_component in clean_acts.keys():
        clean_act = clean_acts[layer_component]
        corrupted_act = corrupted_acts[layer_component]
        grad = grads[layer_component]

        eap_scores[layer_component] = ((clean_act - corrupted_act) * grad).sum(dim=-1)

    return eap_scores


def vanilla_integrated_gradients(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    baseline_inputs: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = torch.mean,
    position: slice | int | Sequence | None = -1,
    steps: int = 50,
) -> dict[str, dict[int, torch.Tensor]]:
    """
    Computes vanilla integrated w.r.t. input embeddings.
    """

    position = regularize_position(position)
    embedding_key = (0, "layer_in")

    input_embeddings = get_activations(
        model,
        inputs,
        layers_components=embedding_key,
        position=position,
        stop_at_layer=1,
    )
    baseline_embeddings = get_activations(
        model,
        baseline_inputs,
        layers_components=embedding_key,
        position=position,
        stop_at_layer=1,
    )

    alphas = torch.linspace(0, 1, steps).to(input_embeddings.device)

    accumulated_grads = torch.zeros_like(input_embeddings)

    for alpha in alphas:
        interpolated_embeddings = _interpolate_activations(
            input_embeddings, baseline_embeddings, alpha
        )

        _, _, grads = patch_activations(
            model,
            inputs,
            interpolated_embeddings,
            layers_components=embedding_key,
            metric_fn=metric_fn,
            position=position,
        )
        accumulated_grads += grads[embedding_key] / steps
    integrated_grads = ((input_embeddings - baseline_embeddings) * accumulated_grads).sum(dim=-1)
    return integrated_grads.cpu()
