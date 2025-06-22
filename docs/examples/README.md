# Ember Examples

This directory contains working examples demonstrating Ember's capabilities.

## Quick Start Examples

### [Hello World](./hello_world.py)
Your first Ember application - simple LLM calls.

### [Structured Output](./structured_output.py)
Using Pydantic models for type-safe outputs.

### [Simple Operators](./simple_operators.py)
Creating reusable operators with `@ember.op`.

## Core Patterns

### [Parallel Processing](./parallel_processing.py)
Efficiently process multiple items in parallel.

### [Error Handling](./error_handling.py)
Robust error handling and retry patterns.

### [Caching](./caching.py)
Cache expensive operations for better performance.

## Advanced Examples

### [NON Patterns](./non_patterns.py)
Building robust AI systems with Networks of Networks.

### [Custom Operators](./custom_operators.py)
Creating domain-specific operators.

### [Learnable Router](./learnable_router.py)
Advanced operator with JAX integration and learnable parameters.

### [Model Comparison](./model_comparison.py)
Compare outputs from multiple models.

### [Data Pipeline](./data_pipeline.py)
Complete data processing pipeline example.

## Real-World Applications

### [Document Analyzer](./document_analyzer.py)
Analyze documents with multiple operators.

### [Q&A System](./qa_system.py)
Build a robust question-answering system.

### [Code Reviewer](./code_reviewer.py)
Automated code review with LLMs.

### [Research Assistant](./research_assistant.py)
Multi-step research with web search and synthesis.

## Running the Examples

1. Install Ember:
```bash
pip install ember-ai
```

2. Set up API keys:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

3. Run an example:
```bash
python docs/examples/hello_world.py
```

## Example Structure

Each example follows this pattern:

```python
"""
Example: [Name]
Description: [What it demonstrates]
Concepts: [Key concepts shown]
"""

from ember.api import ember

async def main():
    # Example implementation
    pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Contributing Examples

To add a new example:

1. Create a descriptive file name
2. Add clear documentation at the top
3. Keep it focused on one concept
4. Include error handling
5. Add it to this README

Examples should be:
- Self-contained
- Well-documented
- Demonstrate best practices
- Actually runnable