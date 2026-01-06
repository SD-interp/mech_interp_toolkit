# MI-Toolkit

A comprehensive library for mechanistic interpretability of language models.

## Installation

```bash
pip install mech_interp_toolkit
```

## Usage

```python
from mech_interp_toolkit.utils import load_model_tokenizer_config
from mech_interp_toolkit.interpret import get_activations

# Load a model and tokenizer
model, tokenizer, config = load_model_tokenizer_config("qwen/qwen3-0.6b")

# Prepare your inputs
prompts = ["Hello, world!"]
inputs = tokenizer(prompts)

# Define the layers and components to get activations from
layers_components = [(0, "attn"), (1, "mlp")]

# Get the activations
activations, logits = get_activations(model, inputs, layers_components)

print(activations)
```

## Ablation

Here's an example of how to perform zero ablation on an attention component:

```python
import torch
from mech_interp_toolkit.utils import load_model_tokenizer_config
from mech_interp_toolkit.interpret import get_activations, patch_activations, ActivationDict

# Load model and tokenizer
model, tokenizer, config = load_model_tokenizer_config("qwen/qwen3-0.6b")

# Prepare inputs
prompts = ["The capital of France is"]
inputs = tokenizer(prompts)
position = -1

# Get clean activations and logits
layers_components = [(0, "attn")]
clean_activations, clean_logits = get_activations(model, inputs, layers_components, position=position)

# Create a patch dictionary with zeroed activations
patch_dict = ActivationDict(config, positions=position)
patch_dict[(0, "attn")] = torch.zeros_like(clean_activations[(0, "attn")], device="cuda")

# Patch the model and get patched logits
patched_logits = patch_activations(model, inputs, patch_dict, position=position)

# Compare top 5 logits
topk_clean = torch.topk(clean_logits, 5)
topk_patched = torch.topk(patched_logits, 5)

print("Clean top 5 tokens:", tokenizer.tokenizer.decode(topk_clean.indices.tolist()[0]))
print("Patched top 5 tokens:", tokenizer.tokenizer.decode(topk_patched.indices.tolist()[0]))
```

This example demonstrates how ablating (zeroing out) the attention component in layer 0 significantly changes the model's output prediction.

## Compatibility

The current version requires the underlying model's layers to follow the standard Llama layer naming convention. Hence, GPT-2 cannot be used with the package.