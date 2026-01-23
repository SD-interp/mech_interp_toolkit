import gc
from collections.abc import Callable
from typing import Literal

import torch
from einops import einsum
from nnsight import NNsight

from .activation_dict import ActivationDict
from .activation_utils import (
    get_activations,
    get_embeddings_dict,
    interpolate_activations,
    locate_layer_component,
)
from .linear_probes import LinearProbe
from .utils import get_layer_components


def _validate_embeddings(
    input_embeddings: torch.Tensor,
    baseline_embeddings: torch.Tensor,
) -> None:
    """Validate that input and baseline embeddings have matching shape, device, and dtype."""
    if input_embeddings.shape != baseline_embeddings.shape:
        raise ValueError(
            f"Input and baseline embeddings must have identical shape. "
            f"Got input: {input_embeddings.shape}, baseline: {baseline_embeddings.shape}"
        )
    if input_embeddings.device != baseline_embeddings.device:
        raise ValueError(
            f"Input and baseline embeddings must be on the same device. "
            f"Got input: {input_embeddings.device}, baseline: {baseline_embeddings.device}"
        )
    if input_embeddings.dtype != baseline_embeddings.dtype:
        raise ValueError(
            f"Input and baseline embeddings must have the same dtype. "
            f"Got input: {input_embeddings.dtype}, baseline: {baseline_embeddings.dtype}"
        )


def _prepare_synthetic_inputs(input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create synthetic input dict by removing input_ids and inputs_embeds."""
    synthetic_inputs = input_dict.copy()
    synthetic_inputs.pop("input_ids", None)
    synthetic_inputs.pop("inputs_embeds", None)
    return synthetic_inputs


def _get_alpha_values(steps: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate alpha values for midpoint Riemann sum integration."""
    return (torch.arange(steps, dtype=dtype) + 0.5) / steps


def _setup_probe_components(probe: LinearProbe) -> tuple[tuple[int, str], torch.Tensor, torch.Tensor]:
    """Extract and prepare probe location, weight, and bias tensors."""
    if probe.location is None:
        raise RuntimeError("probe.location cannot be None.")
    probe_location, probe_component = probe.location

    if probe.weight is None or probe.bias is None:
        raise RuntimeError("probe.weight and probe.bias cannot be None.")

    probe_weight = torch.tensor(probe.weight)
    probe_weight.requires_grad_(True)
    probe_bias = torch.tensor(probe.bias)
    probe_bias.requires_grad_(True)

    return (probe_location, probe_component), probe_weight, probe_bias


def _cleanup_memory() -> None:
    """Clear garbage collection and CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def simple_integrated_gradients(
    model: NNsight,
    input_dict: dict[str, torch.Tensor],
    baseline_dict: dict[str, torch.Tensor],
    metric_fn: Callable = torch.mean,
    steps: int = 5,
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

    input_embeddings = get_embeddings_dict(model, input_dict)["inputs_embeds"]
    baseline_embeddings = get_embeddings_dict(model, baseline_dict)["inputs_embeds"]

    _validate_embeddings(input_embeddings, baseline_embeddings)

    synthetic_inputs = _prepare_synthetic_inputs(input_dict)
    alphas = _get_alpha_values(steps, input_embeddings.dtype)
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

        _cleanup_memory()

    integrated_grads = ((input_embeddings - baseline_embeddings) * accumulated_grads).sum(dim=-1)
    output = ActivationDict(model.model.config, slice(None))
    output[(0, "layer_in")] = integrated_grads
    output.value_type = "simple_ig_scores"
    return output


def edge_attribution_patching(
    model: NNsight,
    input_dict: dict[str, torch.Tensor],
    baseline_dict: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = torch.mean,
) -> ActivationDict:
    """
    Computes edge attributions for attention heads using simple gradient x activation.
    """

    if not torch.is_grad_enabled():
        raise RuntimeError("EAP requires gradient computation. Run with torch.enable_grad()")

    layer_components = get_layer_components(model)

    # Determine which inputs to use for gradient computation
    if compute_grad_at == "clean":
        grad_inputs = input_dict
    elif compute_grad_at == "corrupted":
        grad_inputs = baseline_dict
    else:
        raise ValueError(f"Unknown compute_grad_at value: {compute_grad_at}")

    # Get activations for ALL layer components, not just embeddings
    input_activations = get_activations(model, input_dict, layer_components)
    baseline_activations = get_activations(model, baseline_dict, layer_components)

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

    eap_scores = ((input_activations - baseline_activations) * grads).apply(torch.sum, dim=-1)
    eap_scores.value_type = "eap_scores"

    _cleanup_memory()
    return eap_scores


def eap_integrated_gradients(
    model: NNsight,
    input_dict: dict[str, torch.Tensor],
    baseline_dict: dict[str, torch.Tensor],
    metric_fn: Callable = torch.mean,
    steps: int = 5,
) -> ActivationDict:
    """
    Computes integrated gradients for edge attributions.
    Implements the method from "Have Faith in Faithfulness: Going Beyond Circuit Overlap ..."
    by Hanna et al., 2024. https://arxiv.org/pdf/2403.17806
    """

    if not torch.is_grad_enabled():
        raise RuntimeError("EAP-IG requires gradient computation. Run with torch.enable_grad()")

    layer_components = get_layer_components(model)

    input_activations = get_activations(model, input_dict, [(0, "layer_in")] + layer_components)
    baseline_activations = get_activations(
        model, baseline_dict, [(0, "layer_in")] + layer_components
    )

    input_embeddings = input_activations[(0, "layer_in")]
    baseline_embeddings = baseline_activations[(0, "layer_in")]

    _validate_embeddings(input_embeddings, baseline_embeddings)

    input_embeddings.grad = None
    baseline_embeddings.grad = None

    # Midpoint Riemann sum
    alphas = _get_alpha_values(steps, input_embeddings.dtype)
    synthetic_input_dict = _prepare_synthetic_inputs(input_dict)

    accumulated_grads = input_activations.zeros_like(layer_components)

    for alpha in alphas:
        interpolated_embeddings = (
            interpolate_activations(baseline_embeddings, input_embeddings, alpha)
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
        accumulated_grads = accumulated_grads + (temp_grads / steps)

        del temp_grads, dummy_activation_cache
        _cleanup_memory()

    eap_ig_scores = ((input_activations - baseline_activations) * accumulated_grads).apply(
        torch.sum, dim=-1
    )

    eap_ig_scores.value_type = "eap_ig_scores"
    return eap_ig_scores


def simple_ig_with_probes(
    model: NNsight,
    input_dict: dict[str, torch.Tensor],
    baseline_dict: dict[str, torch.Tensor],
    probe: LinearProbe,
    metric_fn: Callable = torch.mean,
    steps: int = 5,
):
    """
    Computes vanilla integrated w.r.t. input embeddings. Metric function is applied to the output of the linear probe.
    Implements the method from "Axiomatic Attribution for Deep Networks" by Sundararajan et al., 2017.
    https://arxiv.org/abs/1703.01365
    """

    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Integrated Gradients requires gradient computation. Run with torch.enable_grad()"
        )

    input_embeddings = get_embeddings_dict(model, input_dict)["inputs_embeds"]
    baseline_embeddings = get_embeddings_dict(model, baseline_dict)["inputs_embeds"]

    _validate_embeddings(input_embeddings, baseline_embeddings)

    synthetic_inputs = _prepare_synthetic_inputs(input_dict)
    alphas = _get_alpha_values(steps, input_embeddings.dtype)
    accumulated_grads = torch.zeros_like(input_embeddings)

    (probe_location, probe_component), probe_weight, probe_bias = _setup_probe_components(probe)

    for alpha in alphas:
        interpolated_embeddings = (
            interpolate_activations(baseline_embeddings, input_embeddings, alpha)
            .detach()
            .requires_grad_(True)
        )

        with model.trace() as tracer:
            with tracer.invoke(**synthetic_inputs, inputs_embeds=interpolated_embeddings):
                acts = locate_layer_component(model, (probe_location, probe_component)).save()

                probe_output = einsum(
                    acts,
                    probe_weight,
                    "batch pos d_model, d_model d_probe -> batch pos d_probe",
                ) + probe_bias.view(1, 1, -1)

                metric = metric_fn(probe_output)
                metric.backward()

        if interpolated_embeddings.grad is None:
            raise RuntimeError("Failed to retrieve gradients.")
        accumulated_grads = accumulated_grads + (interpolated_embeddings.grad / steps)

        _cleanup_memory()

    integrated_grads = ((input_embeddings - baseline_embeddings) * accumulated_grads).sum(dim=-1)
    output = ActivationDict(model.model.config, slice(None))
    output[(0, "layer_in")] = integrated_grads
    output.value_type = "ig_with_probe_scores"
    return output


def eap_ig_with_probes(
    model: NNsight,
    input_dict: dict[str, torch.Tensor],
    baseline_dict: dict[str, torch.Tensor],
    probe: LinearProbe,
    metric_fn: Callable = torch.mean,
    steps: int = 50,
):
    """
    Computes vanilla integrated w.r.t. input embeddings. Metric function is applied to the output of the linear probe.
    Implements the method from "Axiomatic Attribution for Deep Networks" by Sundararajan et al., 2017.
    https://arxiv.org/abs/1703.01365
    """

    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Integrated Gradients requires gradient computation. Run with torch.enable_grad()"
        )

    (probe_location, probe_component), probe_weight, probe_bias = _setup_probe_components(probe)

    layer_components = get_layer_components(model, stop_at=probe_location)

    input_activations = get_activations(model, input_dict, [(0, "layer_in")] + layer_components)
    baseline_activations = get_activations(
        model, baseline_dict, [(0, "layer_in")] + layer_components
    )

    input_embeddings = input_activations[(0, "layer_in")]
    baseline_embeddings = baseline_activations[(0, "layer_in")]

    _validate_embeddings(input_embeddings, baseline_embeddings)

    synthetic_inputs = _prepare_synthetic_inputs(input_dict)
    alphas = _get_alpha_values(steps, input_embeddings.dtype)
    accumulated_grads = torch.zeros_like(input_embeddings)

    for alpha in alphas:
        interpolated_embeddings = (
            interpolate_activations(baseline_embeddings, input_embeddings, alpha)
            .detach()
            .requires_grad_(True)
        )

        with model.trace() as tracer:
            with tracer.invoke(**synthetic_inputs, inputs_embeds=interpolated_embeddings):
                for layer_component in layer_components:
                    locate_layer_component(model, layer_component).requires_grad_()
                    locate_layer_component(model, layer_component).retain_grad()
                acts = locate_layer_component(model, (probe_location, probe_component)).save()

                probe_output = einsum(
                    acts,
                    probe_weight,
                    "batch pos d_model, d_model d_probe -> batch pos d_probe",
                ) + probe_bias.view(1, 1, -1)

                metric = metric_fn(probe_output)
                if metric.ndim != 0:
                    metric = metric.sum()
                metric.backward()

        if interpolated_embeddings.grad is None:
            raise RuntimeError("Failed to retrieve gradients.")
        accumulated_grads = accumulated_grads + (interpolated_embeddings.grad / steps)

        _cleanup_memory()

    integrated_grads = ((input_embeddings - baseline_embeddings) * accumulated_grads).sum(dim=-1)
    output = ActivationDict(model.model.config, slice(None))
    output[(0, "layer_in")] = integrated_grads
    output.value_type = "integrated_grads"
    return output
