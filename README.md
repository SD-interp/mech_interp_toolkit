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

## Linear Probing

Here's an example of how to fit a linear probe on model activations:

```python
import torch
import numpy as np
from mech_interp_toolkit.interpret import ActivationDict, LinearProbe
from mech_interp_toolkit.utils import load_model_tokenizer_config

# Load a model and tokenizer to get the config
_, _, config = load_model_tokenizer_config("qwen/qwen3-0.6b")

# 1. Create dummy activations and targets
# Typically, you would get these activations from get_activations()
batch_size = 20
num_positions = 10
d_model = config.hidden_size

# Create a dummy ActivationDict
activations = ActivationDict(config, positions=slice(0, num_positions))
activations[(0, "mlp")] = torch.randn(batch_size, num_positions, d_model)

# Create dummy binary targets for classification
targets = np.random.randint(0, 2, size=batch_size)

# 2. Initialize and fit the probe
probe = LinearProbe(target_type="classification", random_state=42)
probe.fit(activations, targets)

# 3. Make predictions on new data
new_batch_size = 5
new_activations = ActivationDict(config, positions=slice(0, num_positions))
new_activations[(0, "mlp")] = torch.randn(new_batch_size, num_positions, d_model)
new_targets = np.random.randint(0, 2, size=new_batch_size)

predictions = probe.predict(new_activations, new_targets)
print("Predictions:", predictions.reshape(new_batch_size, num_positions))
```

This example demonstrates how to train a logistic regression model to classify activations from an MLP layer. The `LinearProbe` class handles the data preparation, training, and evaluation automatically.

## Compatibility

The current version requires the underlying model's layers to follow the standard Llama layer naming convention. Hence, GPT-2 cannot be used with the package.