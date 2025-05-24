# Ember Model Examples

This directory contains examples demonstrating how to use Ember's simplified models API to work with various LLM providers.

## Examples

- `function_style_api.py` - **Start here!** Demonstrates the simplified models() function API
- `model_api_example.py` - Basic model invocation patterns with the simplified API
- `list_models.py` - How to list available models and their capabilities
- `model_registry_example.py` - Advanced patterns including binding, batching, and custom models
- `model_registry_direct.py` - Direct model usage with environment variables
- `dependency_injection.py` - Configuration patterns and model binding for different use cases
- `manual_model_registration.py` - Registering custom models with specific configurations
- `register_models_directly.py` - Custom model registration without environment variables

## Running Examples

To run any example, use the following command format:

```bash
# Using uv (recommended)
uv run python src/ember/examples/models/example_name.py

# Or if in an activated virtual environment
python src/ember/examples/models/example_name.py
```

Replace `example_name.py` with the desired example file, such as:

```bash
# Example: Run the model registry example
uv run python src/ember/examples/models/model_registry_example.py

# Example: List available models
uv run python src/ember/examples/models/list_models.py
```

## Required Environment Variables

Most examples require API keys for LLM providers to be set in your environment:

```bash
# Set your API keys before running examples
export OPENAI_API_KEY="sk-xxxxxxxxxxxxx"
export ANTHROPIC_API_KEY="sk_ant_xxxxxxxxxxxxx"
```

Some examples (like `register_models_directly.py` and `model_registry_direct.py`) allow you to set API keys directly in the code as an alternative to environment variables.

## Best Practices with the Simplified API

### 1. Direct Model Invocation

```python
from ember.api import models

# The simplest way - direct invocation
response = models("gpt-4", "What is the capital of France?")
print(response.text)

# With parameters
response = models("claude-3", "Write a haiku", temperature=0.7, max_tokens=50)
```

### 2. Model Binding for Reuse

```python
from ember.api import models

# Create reusable model configurations
factual_model = models.bind("gpt-4", temperature=0.1)
creative_model = models.bind("gpt-4", temperature=0.9)

# Use them repeatedly
response1 = factual_model("What is gravity?")
response2 = creative_model("Write a story about gravity")
```

### 3. Listing and Discovering Models

```python
from ember.api import models

# List all available models
available = models.list()
print(f"Found {len(available)} models")

# List by provider
openai_models = models.list(provider='openai')

# Get model information
info = models.info('gpt-4')
print(f"Context window: {info['context_window']} tokens")
```

### 4. Custom Model Registration

```python
from ember.api import models
from ember.api.models import ModelInfo, ModelCost, RateLimit

# Define custom model
custom_model = ModelInfo(
    id="custom:my-model",
    name="My Custom Model",
    context_window=32000,
    cost=ModelCost(
        input_cost_per_thousand=0.001,
        output_cost_per_thousand=0.002,
    ),
    rate_limit=RateLimit(
        tokens_per_minute=100000,
        requests_per_minute=1000
    ),
    provider={
        "name": "MyProvider",
        "default_api_key": "${MY_API_KEY}",
        "base_url": "https://api.myprovider.com/v1"
    }
)

# Register the model
registry = models.get_registry()
registry.register_model(model_info=custom_model)

# Use it
response = models("custom:my-model", "Hello!")
```

## Quick Start

```python
from ember.api import models

# Just call models() with a model name and prompt
response = models("gpt-4", "What is the meaning of life?")
print(response.text)
```

## Next Steps

After understanding model usage, explore:

- `operators/` - For examples of building computation with models
- `advanced/` - For complex model usage patterns
