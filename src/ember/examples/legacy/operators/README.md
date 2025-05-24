# Ember Operator Examples

This directory contains examples demonstrating how to create, compose, and use Ember operators for building AI systems using the simplified API.

## Examples

- `simplified_ensemble_example.py` - Creating ensemble operators without API access
- `composition_example.py` - Composing multiple operators together with JIT
- `container_operator_example.py` - Using container operators for complex logic
- `container_simplified.py` - Simplified container operator pattern
- `custom_prompt_example_caravan.py` - Using custom prompts with operators
- `diverse_ensemble_operator_example.py` - Building ensembles with different models

## Running Examples

To run any example, use the following command format:

```bash
# Using uv (recommended)
uv run python src/ember/examples/operators/example_name.py

# Or if in an activated virtual environment
python src/ember/examples/operators/example_name.py
```

Replace `example_name.py` with the desired example file.

## Quick Start

Here's a simple example using the Ember API:

```python
from ember.api import models, non
from ember.api.operators import Operator, EmberModel, Field

# Define input/output models
class QueryInput(EmberModel):
    query: str = Field(description="User query")

class QueryOutput(EmberModel):
    answer: str = Field(description="Model response")

# Create an operator
class SimpleOperator(Operator[QueryInput, QueryOutput]):
    def forward(self, *, inputs: QueryInput) -> QueryOutput:
        # Use the models API directly
        response = models("gpt-4", inputs.query)
        return QueryOutput(answer=response.text)

# Or use built-in operators
ensemble = non.UniformEnsemble(num_units=3, model_name="gpt-4")
result = ensemble(inputs={"query": "What is the capital of France?"})
```

## Operator Concepts

Ember operators are the fundamental building blocks for AI systems, similar to PyTorch's `nn.Module`. They provide:

- Type-safe input/output interfaces
- Specification-driven execution
- Support for composition and nesting
- Automatic optimization via XCS
- Direct integration with the models API

## Next Steps

After mastering operators, explore:

- `xcs/` - For advanced execution optimization
- `advanced/` - For complex operator patterns and architectures
