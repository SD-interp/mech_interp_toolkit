import gc
from collections.abc import Callable
from typing import Literal

import torch
from nnsight import NNsight

from .activation_dict import ActivationDict
from .activation_utils import (
    get_activations,
    get_embeddings,
    interpolate_activations,
    locate_layer_component,
)
from .utils import get_all_layer_components


def simple_integrated_gradients(
    model: NNsight,
    inputs: dict[str, torch.Tensor],
    baseline_embeddings: torch.Tensor,
    metric_fn: Callable = torch.mean,
    steps: int = 50,
) -> ActivationDict:
    """
    Computes vanilla integrated w.r.t. input embeddings.
    Implements the method from "Axiomatic Attribution for Deep Networks" by Sundararajan et al., 2017.
    https://arxiv.org/abs/1703.01365
    """

    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Integrated Gradients requires gradient computation. Run with torch.enable_grad()"
        )

    input_embeddings = get_embeddings(model, inputs)[(0, "layer_in")]

    if baseline_embeddings.shape != input_embeddings.shape:
        raise ValueError(
            f"Baseline and input embeddings must have identical shape. "
            f"Got baseline: {baseline_embeddings.shape}, input: {input_embeddings.shape}"
        )
    if baseline_embeddings.device != input_embeddings.device:
        raise ValueError(
            f"Baseline and input embeddings must be on the same device. "
            f"Got baseline: {baseline_embeddings.device}, input: {input_embeddings.device}"
        )
    if baseline_embeddings.dtype != input_embeddings.dtype:
        raise ValueError(
            f"Baseline and input embeddings must have the same dtype. "
            f"Got baseline: {baseline_embeddings.dtype}, input: {input_embeddings.dtype}"
        )

    synthetic_inputs = inputs.copy()
    synthetic_inputs.pop("input_ids", None)

    alphas = torch.linspace(0, 1, steps + 1)[1:]
    accumulated_grads = torch.zeros_like(input_embeddings)

    for alpha in alphas:
        interpolated_embeddings = (
            interpolate_activations(baseline_embeddings, input_embeddings, alpha)
            .detach()
            .requires_grad_(True)
        )

        with model.trace() as tracer:
            with tracer.invoke(**synthetic_inputs, inputs_embeds=interpolated_embeddings):
                logits = model.lm_head.output.save()  # type: ignore
                metric = metric_fn(logits)
                if metric.ndim != 0:
                    metric = metric.sum()
                metric.backward()

        if interpolated_embeddings.grad is None:
            raise RuntimeError("Failed to retrieve gradients.")
        accumulated_grads = accumulated_grads + (interpolated_embeddings.grad / steps)

        gc.collect()
        torch.cuda.empty_cache()

    integrated_grads = (
        ((input_embeddings - baseline_embeddings) * accumulated_grads).sum(dim=-1).mean()
    )
    output = ActivationDict(model.model.config, slice(None))
    output[(0, "layer_in")] = integrated_grads
    output.value_type = "integrated_grads"
    return output


def edge_attribution_patching(
    model: NNsight,
    clean_inputs: dict[str, torch.Tensor],
    corrupted_inputs: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = torch.mean,
) -> ActivationDict:
    """
    Computes edge attributions for attention heads using simple gradient x activation.
    """

    if not torch.is_grad_enabled():
        raise RuntimeError("EAP requires gradient computation. Run with torch.enable_grad()")

    layer_components = get_all_layer_components(model)

    # Determine which inputs to use for gradient computation
    if compute_grad_at == "clean":
        grad_inputs = clean_inputs
    else:
        grad_inputs = corrupted_inputs

    # Get activations for ALL layer components, not just embeddings
    clean_activations = get_activations(model, clean_inputs, layer_components)
    corrupted_activations = get_activations(model, corrupted_inputs, layer_components)

    activation_cache = ActivationDict(model.model.config, slice(None))

    with model.trace() as tracer:
        with tracer.invoke(**grad_inputs):
            for layer_component in layer_components:
                comp = locate_layer_component(model, layer_component).save()
                comp.requires_grad_()
                comp.retain_grad()
                activation_cache[layer_component] = comp
            logits = model.lm_head.output.save()  # type: ignore
            metric = metric_fn(logits)
            metric.backward()

    grads = activation_cache.get_grads()

    eap_scores = (
        ((clean_activations - corrupted_activations) * grads)
        .apply(torch.sum, dim=-1)
        .apply(torch.mean)
    )
    eap_scores.value_type = "eap_scores"
    return eap_scores


def eap_integrated_gradients(
    model: NNsight,
    clean_dict: dict[str, torch.Tensor],
    corrupted_dict: dict[str, torch.Tensor],
    metric_fn: Callable = torch.mean,
    intermediate_points: int = 5,
) -> ActivationDict:
    """
    Computes integrated gradients for edge attributions.
    Implements the method from "Have Faith in Faithfulness: Going Beyond Circuit Overlap ..."
    by Hanna et al., 2024. https://arxiv.org/pdf/2403.17806
    """

    if not torch.is_grad_enabled():
        raise RuntimeError("EAP-IG requires gradient computation. Run with torch.enable_grad()")

    layer_components = get_all_layer_components(model)

    clean_activations = get_activations(model, clean_dict, [(0, "layer_in")] + layer_components)
    corrupted_activations = get_activations(
        model, corrupted_dict, [(0, "layer_in")] + layer_components
    )

    clean_embeddings = clean_activations[(0, "layer_in")]
    corrupted_embeddings = corrupted_activations[(0, "layer_in")]

    if clean_embeddings.shape != corrupted_embeddings.shape:
        raise ValueError(
            f"Clean and corrupted embeddings must have identical shape. "
            f"Got clean: {clean_embeddings.shape}, corrupted: {corrupted_embeddings.shape}"
        )
    if clean_embeddings.device != corrupted_embeddings.device:
        raise ValueError(
            f"Clean and corrupted embeddings must be on the same device. "
            f"Got clean: {clean_embeddings.device}, corrupted: {corrupted_embeddings.device}"
        )
    if clean_embeddings.dtype != corrupted_embeddings.dtype:
        raise ValueError(
            f"Clean and corrupted embeddings must have the same dtype. "
            f"Got clean: {clean_embeddings.dtype}, corrupted: {corrupted_embeddings.dtype}"
        )

    clean_embeddings.grad = None
    corrupted_embeddings.grad = None

    alphas = torch.linspace(0, 1, intermediate_points + 1)[1:]

    synthetic_input_dict = clean_dict.copy()
    synthetic_input_dict.pop("input_ids", None)
    grad_accumulator = clean_activations.zeros_like(layer_components)

    for alpha in alphas:
        interpolated_embeddings = (
            interpolate_activations(clean_embeddings, corrupted_embeddings, alpha)
            .detach()
            .requires_grad_(True)
        )
        synthetic_input_dict["inputs_embeds"] = interpolated_embeddings

        dummy_activation_cache = ActivationDict(model.model.config, slice(None))

        with model.trace() as tracer:
            with tracer.invoke(**synthetic_input_dict):
                for layer_component in layer_components:
                    comp = locate_layer_component(model, layer_component).save()
                    comp.requires_grad_()
                    comp.retain_grad()
                    dummy_activation_cache[layer_component] = comp
                logits = model.lm_head.output.save()  # type: ignore
                metric = metric_fn(logits)
                metric.backward()

        temp_grads = dummy_activation_cache.get_grads()
        grad_accumulator = grad_accumulator + (temp_grads / intermediate_points)

        del temp_grads, dummy_activation_cache
        gc.collect()
        torch.cuda.empty_cache()

    eap_ig_scores = (
        ((clean_activations - corrupted_activations) * grad_accumulator)
        .apply(torch.sum, dim=-1)
        .apply(torch.mean)
    )

    eap_ig_scores.value_type = "eap_ig_scores"
    return eap_ig_scores


def eap_ig_with_probes():
    pass
