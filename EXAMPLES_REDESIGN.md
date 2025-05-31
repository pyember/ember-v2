# Ember Examples System Design

## Core Philosophy

Examples are the primary documentation. They should be:
- **Immediately runnable** - Copy, paste, run
- **Self-contained** - No hidden dependencies or setup
- **Progressive** - Build from simple to complex
- **Real-world** - Solve actual problems, not toy examples

## Example Categories

### 1. Getting Started (5 minutes to first success)
```
examples/
├── 01_hello_world.py          # Simplest possible LLM call
├── 02_structured_output.py    # Get typed responses  
├── 03_batch_processing.py     # Process multiple inputs
└── 04_error_handling.py       # Handle failures gracefully
```

### 2. Core Patterns (Building blocks)
```
examples/
├── patterns/
│   ├── ensemble.py            # Multiple models for robustness
│   ├── chain_of_thought.py    # Step-by-step reasoning
│   ├── self_critique.py       # Generate then verify
│   ├── few_shot.py           # Examples in prompts
│   └── retry_with_feedback.py # Iterative improvement
```

### 3. Real Applications (Complete solutions)
```
examples/
├── applications/
│   ├── code_reviewer/         # AI code review system
│   ├── research_assistant/    # Multi-source research
│   ├── data_extractor/        # Structure from unstructured
│   ├── test_generator/        # Generate tests from code
│   └── doc_writer/            # Technical documentation
```

### 4. Performance & Scale
```
examples/
├── performance/
│   ├── parallel_processing.py # Using vmap/pmap
│   ├── caching_strategies.py  # Avoid redundant calls
│   ├── streaming_responses.py # Handle large outputs
│   └── cost_optimization.py   # Balance quality/cost
```

## Example Structure

Each example follows a consistent pattern:

```python
"""One-line description of what this does.

This example shows how to [specific technique] to [achieve goal].
You'll learn:
- Key concept 1
- Key concept 2
- Key concept 3

Requirements:
- ember
- Models: gpt-4 or gpt-3.5-turbo
"""

from ember.api import models
from ember.xcs import jit, vmap  # If needed

# Constants at top for easy modification
MODEL = "gpt-4"
TEMPERATURE = 0.7


def main():
    """Main example logic."""
    # Step 1: Setup (if any)
    model = models.instance(MODEL, temperature=TEMPERATURE)
    
    # Step 2: Core logic (the teaching part)
    result = model("Hello, world!")
    print(f"Response: {result.text}")
    
    # Step 3: Show results
    print(f"Tokens used: {result.usage['total_tokens']}")


if __name__ == "__main__":
    main()
```

## Progressive Learning Path

### Level 1: Basic Operations (Day 1)
1. **Hello World** → Make first API call
2. **Structured Output** → Get typed responses
3. **Batch Processing** → Handle multiple inputs
4. **Error Handling** → Graceful failures

### Level 2: Composition Patterns (Week 1)
1. **Ensemble** → Multiple models for robustness
2. **Chain** → Sequential processing
3. **Judge** → Select best output
4. **Retry** → Handle failures intelligently

### Level 3: Advanced Techniques (Week 2)
1. **Streaming** → Handle long responses
2. **Caching** → Optimize costs
3. **Parallel** → Scale with vmap/pmap
4. **State** → Build stateful operators

### Level 4: Production Systems (Month 1)
1. **Code Review Bot** → Complete GitHub integration
2. **Research Assistant** → Multi-step research
3. **Data Pipeline** → ETL with LLMs
4. **Test Generator** → Code to tests

## Implementation Requirements

### 1. Example Runner System
```python
# examples/runner.py
class ExampleRunner:
    """Run examples with proper setup/teardown."""
    
    def run(self, example_path: str):
        # Check requirements
        # Setup mock models if needed
        # Run example
        # Capture output
        # Verify assertions
```

### 2. Testing Framework
```python
# tests/test_examples.py
def test_all_examples_run():
    """Ensure all examples execute without errors."""
    for example in find_examples():
        runner.run(example)
```

### 3. Cost Simulator
```python
# examples/utils/cost.py
class CostSimulator:
    """Simulate API costs without making real calls."""
    
    def estimate(self, example_func):
        # Intercept model calls
        # Calculate tokens
        # Return cost estimate
```

### 4. Interactive Mode
```python
# examples/interactive.py
def interactive_example(example_name: str):
    """Run example step-by-step with explanations."""
    # Load example
    # Parse into steps
    # Execute with pauses
    # Explain each step
```

## Documentation Integration

Each example should:
1. Link to relevant API docs
2. Show common variations
3. Highlight gotchas
4. Provide next steps

```python
"""
See also:
- API Docs: ember.api.models
- Related: examples/patterns/ensemble.py
- Advanced: examples/applications/code_reviewer/

Common variations:
- Different models: MODEL = "gpt-3.5-turbo"
- Streaming: See examples/performance/streaming.py

Gotchas:
- Token limits: Long prompts may hit limits
- Rate limits: Use batch processing for many calls
"""
```

## Quality Criteria

Every example must:
- **Run in <5 seconds** (or explain why not)
- **Cost <$0.01** to run (or use mocks)
- **Teach one thing well**
- **Include error handling**
- **Show real output**

## Anti-patterns to Avoid

1. **No toy examples** - "Calculate fibonacci" teaches nothing about LLMs
2. **No walls of code** - If it's >100 lines, split it up
3. **No magic values** - Explain every constant
4. **No "left as exercise"** - Complete, working code only
5. **No complex setup** - Should work with just `pip install ember`

## Maintenance Strategy

1. **Automated testing** - All examples run in CI
2. **Cost tracking** - Monitor example costs
3. **Version pinning** - Examples specify Ember version
4. **Regular updates** - Quarterly review for relevance
5. **Community examples** - Process for contributions

## Success Metrics

- Time to first successful run: <5 minutes
- Examples that become production code: >50%
- Questions answered by examples: >80%
- Examples run per week: >1000

## Next Steps

1. Implement example runner infrastructure
2. Create first 10 examples covering basics
3. Build one complete application example
4. Add interactive mode for learning
5. Set up automated testing