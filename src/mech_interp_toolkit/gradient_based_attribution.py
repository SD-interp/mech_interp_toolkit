import torch
from nnsight import NNsight
from collections.abc import Sequence, Callable
from .activations import get_activations_and_grads, get_activations, patch_activations
from typing import Literal, Optional, cast
from .activation_dict import ActivationDict
from .utils import regularize_position
from functools import partial


def _get_acts_and_grads(
    model: NNsight,
    clean_inputs: dict[str, torch.Tensor],
    corrupted_inputs: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = partial(torch.mean, dim=-1),
    position: slice | int | Sequence | None = -1,
) -> tuple[ActivationDict, ActivationDict, Optional[ActivationDict]]:
    """
    Helper function to get activations and gradients for clean and corrupted inputs.
    """
    n_layers = model.model.config.num_hidden_layers  # type: ignore
    n_layers = cast(int, n_layers)

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
        if grads is not None:
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
        if grads is not None:
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
    clean_activations: ActivationDict,
    baseline_activations: ActivationDict,
    alpha: float | torch.Tensor,
    key: tuple[int, str] = (0, "layer_in"),
) -> ActivationDict:
    """
    Interpolates between clean and corrupted inputs.
    """
    interpolated_activations = (
        1 - alpha
    ) * clean_activations + alpha * baseline_activations
    return interpolated_activations


def edge_attribution_patching(
    model: NNsight,
    clean_inputs: dict[str, torch.Tensor],
    corrupted_inputs: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = partial(torch.mean, dim=-1),
    position: slice | int | Sequence | None = -1,
) -> ActivationDict:
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

    eap_scores = ((clean_acts - corrupted_acts) * grads).apply(torch.sum, dim=-1)
    return eap_scores


def simple_integrated_gradients(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    baseline_embeddings: ActivationDict,
    metric_fn: Callable = partial(torch.mean, dim=-1),
    steps: int = 50,
) -> ActivationDict:
    """
    Computes vanilla integrated w.r.t. input embeddings.
    Implements the method from "Axiomatic Attribution for Deep Networks" by Sundararajan et al., 2017.
    https://arxiv.org/abs/1703.01365
    """

    n_layers = model.model.config.num_hidden_layers  # type: ignore
    n_layers = cast(int, n_layers)

    position = regularize_position(slice(None))
    embedding_key = (0, "layer_in")

    input_embeddings, _, _ = get_activations(
        model,
        inputs,
        layers_components=[embedding_key],
        position=position,
        stop_at_layer=1,
    )

    device = input_embeddings[embedding_key].device
    alphas = torch.linspace(0, 1, steps).to(device)
    accumulated_grads = None

    for alpha in alphas:
        interpolated_embeddings = _interpolate_activations(
            input_embeddings, baseline_embeddings, alpha
        )

        _, grads, _ = patch_activations(
            model,
            inputs,
            interpolated_embeddings,
            layers_components=[embedding_key],
            metric_fn=metric_fn,
            position=position,
        )
        if accumulated_grads is None:
            accumulated_grads = grads / steps
        else:
            accumulated_grads = accumulated_grads + (grads / steps)

    integrated_grads = (
        (input_embeddings - baseline_embeddings) * accumulated_grads
    ).apply(torch.sum, dim=-1)
    return integrated_grads


def eap_integrated_gradients(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    baseline_embeddings: ActivationDict,
    layer_components: list[tuple[int, str]] | None = None,
    metric_fn: Callable = partial(torch.mean, dim=-1),
    position: slice | int | Sequence | None = -1,
    steps: int = 5,
) -> ActivationDict:
    """
    Computes integrated gradients for edge attributions.
    Implements the method from "Have Faith in Faithfulness: Going Beyond Circuit Overlap ..."
    by Hanna et al., 2024. https://arxiv.org/pdf/2403.17806
    """

    n_layers = model.model.config.num_hidden_layers  # type: ignore
    n_layers = cast(int, n_layers)

    position = regularize_position(position)

    if layer_components is None:
        layer_components = [(i, c) for i in range(n_layers) for c in ["attn", "mlp"]]

    embeddings, _, _ = get_activations(
        model,
        inputs,
        layers_components=layer_components,
        position=position,
        stop_at_layer=1,
    )

    device = list(embeddings.values())[0].device if embeddings else torch.device("cpu")
    alphas = torch.linspace(0, 1, steps).to(device)
    accumulated_grads = None

    for alpha in alphas:
        interpolated_embeddings = _interpolate_activations(
            embeddings, baseline_embeddings, alpha
        )

        _, grads, _ = patch_activations(
            model,
            inputs,
            interpolated_embeddings,
            layers_components=layer_components,
            metric_fn=metric_fn,
            position=position,
        )
        if accumulated_grads is None:
            accumulated_grads = grads / steps
        else:
            accumulated_grads = accumulated_grads + (grads / steps)

    integrated_grads = (
        (embeddings - baseline_embeddings) * accumulated_grads
    ).apply(torch.sum, dim=-1)
    return integrated_grads
