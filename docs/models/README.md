# Ember Model System

The Ember Model System provides a clean, intuitive interface for interacting with language models from various providers. It follows key design principles:

1. **One obvious way to do things**: A single, consistent entry point to access models
2. **Hide complexity behind simple interfaces**: Clean abstractions for the underlying components
3. **Prioritize common use cases**: Make simple things simple and complex things possible
4. **Explicit dependency management**: First-class support for dependency injection and isolation

## Quick Start

```python
from ember.api import models

# One-shot usage
response = models.model("gpt-4o")("What is the capital of France?")
print(response)  # "Paris is the capital of France."

# Reusable model with configuration
gpt4 = models.model("gpt-4o", temperature=0.7)
response1 = gpt4("Tell me a joke")
response2 = gpt4("Explain quantum computing")

# Provider namespaces for cleaner access
response = models.openai.gpt4o("What is the capital of France?")

# Temporary configuration overrides
with models.configure(temperature=0.2, max_tokens=100):
    response = models.model("gpt-4o")("Write a haiku")
```

## Key Features

### Function-Style API

The API is designed to be more intuitive and natural, using Python's function calling semantics:

```python
# Traditional API
model = ModelAPI("openai:gpt-4o")
response = model.generate(prompt="What is the capital of France?")

# New Function-Style API
response = model("gpt-4o")("What is the capital of France?")
```

### Provider Namespaces

Provider namespaces provide a cleaner way to access models from specific providers:

```python
from ember.api.models import openai, anthropic, deepmind

# OpenAI models
response = openai.gpt4o("What is the capital of France?")

# Anthropic models
response = anthropic.claude("Tell me a joke")

# Deepmind models
response = deepmind.gemini("Explain quantum mechanics")
```

### Response Object

The Response object provides a clean, consistent interface for working with model responses:

```python
# Get response text via string conversion
print(response)  # "Paris is the capital of France."

# Access usage information
print(f"Tokens: {response.usage.total_tokens}")
print(f"Cost: ${response.usage.cost:.6f}")

# Visual representation in notebooks
response.visualize()
```

### Dependency Injection

The API provides first-class support for dependency injection, allowing you to create isolated contexts:

```python
from ember.api.models import ModelContext, ContextConfig, model

# Create a test context
test_context = ModelContext(
    config=ContextConfig(
        api_keys={"openai": "test-key"},
        auto_discover=False
    )
)

# Use the context
test_model = model("gpt-4o", context=test_context)
response = test_model("This is a test")
```

### Complete Function

The `complete()` function provides a convenient way to generate completions in one line:

```python
from ember.api.models import complete

answer = complete(
    "Explain the significance of the year 1969 in space exploration.",
    model="gpt-4o",
    temperature=0.7
)
```

## Advanced Features

### Configuration Management

The API provides multiple layers of configuration management:

```python
# Global configuration
from ember.api.models import config
config.temperature = 0.7

# Instance configuration
gpt4 = model("gpt-4o", temperature=0.7)

# Temporary configuration
with configure(temperature=0.2):
    response = model("gpt-4o")("Write a haiku")

# Call-specific configuration
response = gpt4("Tell me a joke", temperature=0.9)
```

### Custom Contexts

You can create custom contexts with different configurations for testing, production, etc:

```python
# Production context
prod_context = create_context(
    config=ContextConfig(
        api_keys={"openai": os.environ.get("OPENAI_API_KEY")},
        auto_discover=True
    )
)

# Test context
test_context = create_context(
    config=ContextConfig(
        api_keys={"openai": "test-key"},
        auto_discover=False
    )
)

# Use the contexts
prod_model = model("gpt-4o", context=prod_context)
test_model = model("gpt-4o", context=test_context)
```

## Migration Guide

If you're migrating from the previous API, see the [Migration Guide](MIGRATION_GUIDE.md) for details.