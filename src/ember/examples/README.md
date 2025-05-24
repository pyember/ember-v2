# Ember Examples

Welcome to the Ember examples library! This collection demonstrates how to build compound AI systems using the Ember framework, from simple model calls to complex production systems.

## 🚀 Quick Start

New to Ember? Start here:

```bash
# Check your environment is set up correctly
uv run python src/ember/examples/01_getting_started/hello_world.py

# Your first model call
uv run python src/ember/examples/01_getting_started/first_model_call.py
```

## 📚 Learning Paths

### Path 1: Quick Start (2 hours)
Perfect for getting a feel for Ember's capabilities:
1. `01_getting_started/hello_world.py` - Verify your setup
2. `01_getting_started/first_model_call.py` - Basic model interaction
3. `02_core_concepts/operators_basics.py` - Understanding operators
4. `04_compound_ai/simple_ensemble.py` - Your first ensemble

### Path 2: Building Systems (1 day)
Learn to build real AI systems:
1. Complete Quick Start path
2. `03_operators/custom_operator.py` - Create custom operators
3. `05_data_processing/loading_datasets.py` - Work with data
4. `06_performance/jit_basics.py` - Optimize performance
5. `09_practical_patterns/rag_pattern.py` - Implement RAG
6. `10_evaluation_suite/accuracy_evaluation.py` - Measure success

### Path 3: Production Systems (1 week)
Master advanced patterns for production:
1. Complete Building Systems path
2. All examples in `06_performance/`
3. `07_advanced_patterns/production_pipeline.py`
4. `08_integrations/fastapi_server.py`
5. All examples in `09_practical_patterns/`
6. `10_evaluation_suite/benchmark_harness.py`

## 📁 Directory Structure

- **[01_getting_started/](./01_getting_started/)** - Entry point for new users
- **[02_core_concepts/](./02_core_concepts/)** - Fundamental Ember concepts
- **[03_operators/](./03_operators/)** - Creating and composing operators
- **[04_compound_ai/](./04_compound_ai/)** - Networks of Networks (NON) patterns
- **[05_data_processing/](./05_data_processing/)** - Data loading and transformation
- **[06_performance/](./06_performance/)** - Optimization and scaling techniques
- **[07_advanced_patterns/](./07_advanced_patterns/)** - Complex architectural patterns
- **[08_integrations/](./08_integrations/)** - External system integrations
- **[09_practical_patterns/](./09_practical_patterns/)** - Common real-world patterns
- **[10_evaluation_suite/](./10_evaluation_suite/)** - Testing and evaluation tools
- **[notebooks/](./notebooks/)** - Interactive Jupyter notebooks
- **[legacy/](./legacy/)** - Previous examples structure (for reference)

## 🎯 Example Standards

Every example follows these principles:
- **Single Learning Objective**: Each example teaches one main concept
- **Self-Contained**: Runs without external dependencies
- **Well-Documented**: Clear explanations of what and why
- **Tested**: Included in our golden test suite

## 🧪 Running Tests

All examples are tested in CI. To run tests locally:

```bash
# Run all golden tests for examples
uv run pytest tests/golden/ -v

# Run tests for a specific example category
uv run pytest tests/golden/test_01_getting_started.py -v
```

## 🔧 Prerequisites

Before running examples, ensure you have:

1. Python 3.11 or higher
2. Ember installed: `uv pip install -e .`
3. API keys configured (see `01_getting_started/setup_guide.md`)

## 📖 Additional Resources

- [Ember Documentation](https://docs.ember.ai)
- [API Reference](https://api.ember.ai)
- [Community Discord](https://discord.gg/ember)

## 🤝 Contributing

Found an issue or have an example idea? We'd love your contribution! See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

---

**Note**: The `legacy/` directory contains our previous examples structure. While these examples still work, we recommend following the new numbered structure for a better learning experience.