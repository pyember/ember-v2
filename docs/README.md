# Ember Documentation

Welcome to Ember - a powerful framework for building LLM-powered applications with a focus on simplicity, composability, and performance.

## Quick Links

- 🚀 [Getting Started](./getting_started.md) - Start here if you're new to Ember
- 📖 [Quick Reference](./QUICK_REFERENCE.md) - Common patterns and examples
- 🔧 [API Reference](./api_reference/) - Detailed API documentation

## Documentation Structure

### Getting Started
- [Getting Started Guide](./getting_started.md) - Your first Ember application
- [Installation](./quickstart/installation.md) - Setup instructions

### Core Concepts
- [Operators](./quickstart/operators.md) - Building blocks of Ember applications
- [Prompt Engineering](./quickstart/prompt_signatures.md) - Type-safe prompting
- [NON Patterns](./quickstart/non.md) - Networks of Networks for robust AI
- [Models](./quickstart/models.md) - Working with different LLM providers
- [Data Processing](./quickstart/data.md) - Loading and processing data

### Advanced Topics
- [XCS Execution](./xcs/) - Advanced execution strategies
- [Performance Optimization](./quickstart/performance.md) - Making your apps fast
- [Context System](./CONTEXT_SYSTEM.md) - Managing application context

### Reference
- [API Reference](./api_reference/) - Complete API documentation
- [CLI Reference](./cli/) - Command-line interface
- [Examples](./examples/) - Working code examples

## Key Features

### 🎯 Simple by Default
```python
from ember.api import ember

# Just works
response = await ember.llm("Hello, Ember!")
```

### 🏗️ Progressive Complexity
```python
# Start simple
@ember.op
async def summarize(text: str) -> str:
    return await ember.llm(f"Summarize: {text}")

# Add validation when needed
@ember.op
@validate
async def analyze(input: AnalysisInput) -> AnalysisOutput:
    return await ember.llm(input.text, output_type=AnalysisOutput)
```

### ⚡ Automatic Optimization
```python
# Ember handles parallelization and JIT compilation
results = await ember.parallel([
    analyze(doc) for doc in documents
])
```

### 🔧 Composable Operators
```python
# Build complex workflows from simple parts
pipeline = compose(
    load_document,
    extract_text,
    summarize,
    translate
)
```

## Design Principles

1. **Simple for the 90% case** - Common tasks should be trivial
2. **Progressive disclosure** - Complexity only when needed
3. **Type-safe by default** - Catch errors early with type hints
4. **Performance built-in** - Automatic optimization without configuration
5. **Composable primitives** - Build complex systems from simple parts

## Getting Help

- 📚 Read the [Getting Started Guide](./getting_started.md)
- 🔍 Search the [API Reference](./api_reference/)
- 💡 Check out [Examples](./examples/)
- 🐛 Report issues on [GitHub](https://github.com/ember-ai/ember)

## What's New

See the [SIMPLIFICATION_GUIDE.md](./SIMPLIFICATION_GUIDE.md) for details on Ember's recent architectural improvements that make it easier than ever to build AI applications.

---

Ready to build something amazing? [Get started now!](./getting_started.md)