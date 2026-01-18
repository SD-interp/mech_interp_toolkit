import copy
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch
from nnsight import NNsight

from .activation_dict import ActivationDict
from .activations import SpecDict
from .activations import UnifiedAccessAndPatching as UAP  # noqa: N814
from .utils import get_all_layer_components, regularize_position


def _get_acts_and_grads(
    model: NNsight,
    clean_inputs: dict[str, torch.Tensor],
    corrupted_inputs: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = torch.mean,
    position: slice | int | Sequence | None = -1,
) -> tuple[ActivationDict, ActivationDict, Optional[ActivationDict]]:
    """
    Helper function to get activations and gradients for clean and corrupted inputs.
    """
    layer_components = get_all_layer_components(model)

    with_grad_dict: SpecDict = {
        "activations": {
            "positions": position,
            "locations": layer_components,
            "gradients": {"metric_fn": metric_fn, "compute_metric_at": (0, "logits")},
        }
    }

    without_grad_dict: SpecDict = {
        "activations": {"positions": position, "locations": layer_components},
    }

    if compute_grad_at == "clean":
        with UAP(model, clean_inputs, with_grad_dict) as uap:
            clean_acts, _ = uap.unified_access_and_patching()
            if clean_acts is None:
                raise RuntimeError("Failed to retrieve clean activations.")
            grads = clean_acts.get_grads()

        with UAP(model, corrupted_inputs, without_grad_dict) as uap:
            corrupted_acts, _ = uap.unified_access_and_patching()
            if corrupted_acts is None:
                raise RuntimeError("Failed to retrieve corrupted activations.")

    elif compute_grad_at == "corrupted":
        with UAP(model, corrupted_inputs, with_grad_dict) as uap:
            corrupted_acts, _ = uap.unified_access_and_patching()
            if corrupted_acts is None:
                raise RuntimeError("Failed to retrieve corrupted activations.")
            grads = corrupted_acts.get_grads()

        with UAP(model, clean_inputs, without_grad_dict) as uap:
            clean_acts, _ = uap.unified_access_and_patching()
            if clean_acts is None:
                raise RuntimeError("Failed to retrieve clean activations.")
    else:
        raise ValueError("compute_grad_at must be either 'clean' or 'corrupted'")

    return clean_acts, corrupted_acts, grads


def _interpolate_activations(
    clean_activations: ActivationDict,
    baseline_activations: ActivationDict,
    alpha: float | torch.Tensor,
) -> ActivationDict:
    """
    Interpolates between clean and corrupted inputs.
    """
    interpolated_activations = (1 - alpha) * clean_activations + alpha * baseline_activations
    return interpolated_activations


def edge_attribution_patching(
    model: NNsight,
    clean_inputs: dict[str, torch.Tensor],
    corrupted_inputs: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = torch.mean,
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

    eap_scores = ((clean_acts - corrupted_acts) * grads).apply(torch.sum, dim=-1).apply(torch.mean)
    return eap_scores


def simple_integrated_gradients(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    baseline_embeddings: ActivationDict,
    metric_fn: Callable = torch.mean,
    steps: int = 50,
) -> ActivationDict:
    """
    Computes vanilla integrated w.r.t. input embeddings.
    Implements the method from "Axiomatic Attribution for Deep Networks" by Sundararajan et al., 2017.
    https://arxiv.org/abs/1703.01365
    """

    position = regularize_position(slice(None))
    embedding_key = (0, "layer_in")
    embedding_spec_dict: SpecDict = {
        "stop_at_layer": 0,
        "activations": {"positions": position, "locations": [embedding_key]},
    }

    with UAP(model, inputs, embedding_spec_dict) as uap:
        input_embeddings, _ = uap.unified_access_and_patching()
        if input_embeddings is None:
            raise RuntimeError("Failed to retrieve input embeddings.")

    device = input_embeddings[embedding_key].device
    alphas = torch.linspace(0, 1, steps + 1)[1:].to(device)
    accumulated_grads = None

    for alpha in alphas:
        interpolated_embeddings = _interpolate_activations(
            input_embeddings, baseline_embeddings, alpha
        )

        spec_dict: SpecDict = {
            "patching": interpolated_embeddings,
            "activations": {
                "positions": position,
                "locations": [embedding_key],
                "gradients": {
                    "metric_fn": metric_fn,
                    "compute_metric_at": (0, "logits"),
                },
            },
        }

        with UAP(model, inputs, spec_dict) as uap:
            acts, _ = uap.unified_access_and_patching()
            if acts is None:
                raise RuntimeError("Failed to retrieve activations.")
            grads = acts.get_grads()

        if accumulated_grads is None:
            accumulated_grads = grads / steps
        else:
            accumulated_grads = accumulated_grads + (grads / steps)

    integrated_grads = ((input_embeddings - baseline_embeddings) * accumulated_grads).apply(
        torch.sum, dim=-1
    )
    return integrated_grads


def eap_integrated_gradients(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    baseline_embeddings: ActivationDict,
    layer_components: list[tuple[int, str]] | None = None,
    metric_fn: Callable = torch.mean,
    position: slice | int | Sequence | None = -1,
    intermediate_points: int = 5,
) -> ActivationDict:
    """
    Computes integrated gradients for edge attributions.
    Implements the method from "Have Faith in Faithfulness: Going Beyond Circuit Overlap ..."
    by Hanna et al., 2024. https://arxiv.org/pdf/2403.17806
    """

    position = regularize_position(position)
    embedding_key = (0, "layer_in")

    if layer_components is None:
        layer_components = get_all_layer_components(model)

    embedding_spec_dict: SpecDict = {
        "stop_at_layer": 0,
        "activations": {"positions": position, "locations": [embedding_key]},
    }

    template_spec: SpecDict = {
        "activations": {
            "positions": position,
            "locations": layer_components,
            "gradients": {"metric_fn": metric_fn, "compute_metric_at": (0, "logits")},
        }
    }

    with UAP(model, inputs, embedding_spec_dict) as uap:
        embeddings, _ = uap.unified_access_and_patching()
        if embeddings is None:
            raise RuntimeError("Failed to retrieve embeddings.")

    device = list(embeddings.values())[0].device if embeddings else torch.device("cpu")
    alphas = torch.linspace(0, 1, intermediate_points + 1)[1:].to(device)
    accumulated_grads = None

    for alpha in alphas:
        interpolated_embeddings = _interpolate_activations(embeddings, baseline_embeddings, alpha)

        spec_dict = copy.deepcopy(template_spec)
        spec_dict["patching"] = interpolated_embeddings

        with UAP(model, inputs, spec_dict) as uap:
            acts, _ = uap.unified_access_and_patching()
            if acts is None:
                raise RuntimeError("Failed to retrieve activations.")
            grads = acts.get_grads()

        if accumulated_grads is None:
            accumulated_grads = grads / intermediate_points
        else:
            accumulated_grads = accumulated_grads + (grads / intermediate_points)

    integrated_grads = ((embeddings - baseline_embeddings) * accumulated_grads).apply(
        torch.sum, dim=-1
    )
    return integrated_grads
