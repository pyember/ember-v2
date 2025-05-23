# Ember Data Examples

This directory contains examples demonstrating Ember's data handling capabilities, including dataset creation, manipulation, and evaluation.

## Examples

- `data_api_example.py` - A standalone example that demonstrates core dataset concepts
- `mcq_experiment_example.py` - Multiple-choice question evaluation with various answer strategies
- `transformation_example.py` - XCS transformations for efficient data processing

## Running Examples

The examples in this directory are designed to run without requiring external API keys or dependencies. To run any example, use the following command format:

```bash
# Using uv (recommended)
uv run python src/ember/examples/data/example_name.py

# Or if in an activated virtual environment
python src/ember/examples/data/example_name.py
```

Replace `example_name.py` with the desired example file.

## Example Details

### data_api_example.py

A self-contained example that demonstrates core dataset concepts without requiring the full Ember infrastructure:

- Creating and defining dataset entries with queries, choices, and metadata
- Displaying formatted dataset contents
- Sampling random entries from a dataset
- Processing and extracting information from dataset entries

This example uses a simplified `DatasetEntry` class that mimics the structure of Ember's actual data API but doesn't require any external dependencies or model registration.

### mcq_experiment_example.py

Demonstrates multiple approach strategies for answering multiple-choice questions:

- `SingleModelBaseline` - Simulates using a single model for answering
- `MultiModelEnsemble` - Simulates using an ensemble of similar models with voting
- `VariedModelEnsemble` - Simulates using specialized models for different domains

This example includes a mock dataset of multiple-choice questions across different subjects and simulates how different ensemble strategies might perform. It uses deterministic randomness (with a configurable seed) to demonstrate both correct and incorrect answers.

Command-line options:
```
--num_samples N   Number of samples to process (default: 5)
--seed N          Random seed for reproducibility (default: 42)
```

### Using the Simplified Ember Data API

The Ember Data API provides a clean interface for working with datasets:

- Access to standard benchmarks like MMLU, TruthfulQA, AIME, GPQA, and Codeforces
- Integration with the Hugging Face datasets library
- Advanced sampling and filtering with builder pattern
- Dataset registration and discovery
- Memory-efficient streaming
- Clean attribute access with DataItem wrapper

To use the Ember Data API:

```python
from ember.api import data
from ember.api.data import DataItem

# Simple loading with streaming
for item in data("mmlu", streaming=True, limit=10):
    print(f"Question: {item.question}")
    print(f"Answer: {item.answer}")

# Advanced configuration with builder pattern
dataset = (
    data.builder()
    .from_registry("mmlu")
    .subset("abstract_algebra")
    .split("test")
    .sample(100)
    .transform(lambda x: {"query": f"Q: {x['question']}"})
    .build()
)

# Clean access with DataItem
items = [DataItem(item) for item in dataset]
for item in items[:5]:
    # Direct property access - no getattr needed!
    print(f"Q: {item.question}")
    if item.options:
        for key, value in item.options.items():
            print(f"  {key}: {value}")
```

## Next Steps

After learning about data handling, explore:

- `models/` - For using models with datasets
- `operators/` - For building evaluation pipelines
- `advanced/` - For complex workflows combining these components