# Ember Integration Examples

This directory contains examples demonstrating how to integrate Ember with other systems, libraries, and frameworks.

## Quick Start with Simplified API

```python
from ember.api.models import models
from ember.api.non import non
from ember.api.operators import Operator, Specification, EmberModel, Field

# Direct model usage
response = models("gpt-4", "Hello world")

# Using pre-built operators
ensemble = non.UniformEnsemble(num_units=3, model_name="gpt-4")
synthesizer = non.JudgeSynthesisOperator(model_name="gpt-4")

# Custom operator with simplified model binding
class MyOperator(Operator):
    def __init__(self, model_name="gpt-4"):
        self.model = models.bind(model_name)
    
    def forward(self, *, inputs):
        return self.model(inputs.query)
```

## Examples

- `api_operators_example.py` - Using Ember's simplified API with custom operators and graceful fallbacks

## Running Examples

To run any example, use the following command format:

```bash
# Using uv (recommended)
uv run python src/ember/examples/integration/example_name.py

# Or if in an activated virtual environment
python src/ember/examples/integration/example_name.py
```

Replace `example_name.py` with the desired example file.

## Integration Concepts

These examples show how to:

- Integrate Ember with external APIs and services
- Use Ember as part of a larger system
- Connect Ember to data processing pipelines
- Extend Ember with custom functionality

## Next Steps

After exploring these integration examples, you may want to check out:

- `advanced/` - For advanced Ember usage patterns
- Contributing to Ember with your own extensions
