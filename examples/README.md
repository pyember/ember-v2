# Ember Examples

Welcome to the Ember examples library! This collection demonstrates how to build compound AI systems using the Ember framework, from simple model calls to complex production systems.

**Important**: These examples are NOT included in the `pip install ember-ai` package. They are only available in the source repository to keep the installed package lightweight.

## 🚀 Quick Start

New to Ember? Start here:

```bash
# First, install Ember
pip install ember-ai

# Clone the repository to access examples
git clone https://github.com/anthropics/ember
cd ember

# Run your first example
python examples/01_getting_started/hello_world.py

# Your first model call
python examples/01_getting_started/first_model_call.py
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
2. `02_core_concepts/rich_specifications.py` - Data validation
3. `05_data_processing/loading_datasets.py` - Work with data
4. `06_performance_optimization/jit_basics.py` - Optimize performance
5. `09_practical_patterns/rag_pattern.py` - Implement RAG
6. `10_evaluation_suite/accuracy_evaluation.py` - Measure success

### Path 3: Production Systems (1 week)
Master advanced patterns for production:
1. Complete Building Systems path
2. All examples in `06_performance_optimization/`
3. `08_advanced_patterns/jax_xcs_integration.py`
4. All examples in `09_practical_patterns/`
5. `10_evaluation_suite/benchmark_harness.py`

## 📁 Directory Structure

- **[01_getting_started/](./01_getting_started/)** - Entry point for new users
- **[02_core_concepts/](./02_core_concepts/)** - Fundamental Ember concepts
- **[03_simplified_apis/](./03_simplified_apis/)** - High-level API patterns
- **[04_compound_ai/](./04_compound_ai/)** - Networks of Networks (NON) patterns
- **[05_data_processing/](./05_data_processing/)** - Data loading and transformation
- **[06_performance_optimization/](./06_performance_optimization/)** - JIT optimization and scaling techniques
- **[07_error_handling/](./07_error_handling/)** - Robust error handling patterns
- **[08_advanced_patterns/](./08_advanced_patterns/)** - Complex architectural patterns
- **[09_practical_patterns/](./09_practical_patterns/)** - Common real-world patterns
- **[10_evaluation_suite/](./10_evaluation_suite/)** - Testing and evaluation tools
- **[notebooks/](./notebooks/)** - Interactive Jupyter notebooks
- **[_shared/](./\\_shared/)** - Shared utilities and helper functions

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

**Note**: Examples have been consolidated to remove redundancy and provide clearer learning paths. The numbered directory structure guides you from basic concepts to production-ready patterns efficiently.