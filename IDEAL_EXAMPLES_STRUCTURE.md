# Ideal Ember Examples Structure

## Design Principles

### 1. Progressive Complexity
Examples should follow a clear learning path from simple to complex, allowing users to build understanding incrementally.

### 2. Real-World Alignment
Each example should solve a problem users actually face, not just demonstrate syntax.

### 3. Self-Contained Execution
Every example should be runnable with minimal setup - ideally just the Ember installation.

### 4. Clear Intent
Each example should have a single, clear learning objective stated upfront.

### 5. Composition Demonstration
Show how Ember's components compose together to build larger systems.

## Proposed Structure

```
ember/examples/
├── README.md                    # Navigation guide and learning paths
├── requirements.txt             # Minimal dependencies for all examples
├── _shared/                     # Shared utilities and helpers
│   ├── __init__.py
│   ├── data_helpers.py         # Common data loading utilities
│   ├── evaluation_helpers.py   # Shared evaluation utilities
│   └── visualization.py        # Output formatting utilities
│
├── 01_getting_started/         # Entry point for new users
│   ├── README.md               # "Start here" guide
│   ├── hello_world.py          # Simplest possible example
│   ├── first_model_call.py     # Basic model invocation
│   ├── model_comparison.py     # Compare responses from different models
│   └── basic_prompt_engineering.py  # Temperature, system prompts, etc.
│
├── 02_core_concepts/           # Fundamental building blocks
│   ├── README.md               # Conceptual overview
│   ├── operators_basics.py     # What are operators and why use them
│   ├── type_safety.py          # EmberModel and specifications
│   ├── context_management.py   # Understanding EmberContext
│   └── error_handling.py       # Proper error handling patterns
│
├── 03_operators/               # Deep dive into operators
│   ├── README.md               # Operator patterns guide
│   ├── custom_operator.py      # Build your first operator
│   ├── operator_composition.py # Combining operators
│   ├── stateful_operators.py   # Operators with internal state
│   └── operator_testing.py     # How to test operators
│
├── 04_compound_ai/             # NON patterns and ensemble methods
│   ├── README.md               # Introduction to compound AI
│   ├── simple_ensemble.py      # Basic ensemble with voting
│   ├── judge_synthesis.py      # Using a judge to synthesize
│   ├── self_consistency.py     # Self-consistency checking
│   ├── debate_pattern.py       # Multi-agent debate
│   └── hierarchical_systems.py # Nested NON structures
│
├── 05_data_processing/         # Working with datasets
│   ├── README.md               # Data handling guide
│   ├── loading_datasets.py     # Using the dataset registry
│   ├── custom_datasets.py      # Creating custom datasets
│   ├── streaming_data.py       # Memory-efficient processing
│   ├── data_transformations.py # Preprocessing pipelines
│   └── batch_evaluation.py     # Evaluating on benchmarks
│
├── 06_performance/             # Optimization and scaling
│   ├── README.md               # Performance guide
│   ├── jit_basics.py          # Introduction to JIT
│   ├── parallelization.py     # Automatic parallelization
│   ├── batching_strategies.py  # Efficient batching with vmap
│   ├── caching_patterns.py     # Caching and memoization
│   └── profiling_example.py    # Finding bottlenecks
│
├── 07_advanced_patterns/       # Advanced use cases
│   ├── README.md               # Advanced patterns overview
│   ├── custom_schedulers.py    # Building custom schedulers
│   ├── distributed_execution.py # Multi-machine execution
│   ├── dynamic_graphs.py       # Runtime graph modification
│   ├── plugin_system.py        # Extending Ember
│   └── production_pipeline.py  # Production-ready example
│
├── 08_integrations/            # External integrations
│   ├── README.md               # Integration guide
│   ├── fastapi_server.py       # REST API with FastAPI
│   ├── gradio_ui.py           # Interactive UI with Gradio
│   ├── langchain_bridge.py     # Using with LangChain
│   ├── mlflow_tracking.py      # Experiment tracking
│   └── kubernetes_deployment.py # K8s deployment example
│
├── 09_practical_patterns/      # Common real-world patterns
│   ├── README.md               # Pattern catalog
│   ├── rag_pattern.py          # RAG in 100 lines
│   ├── chain_of_thought.py     # CoT reasoning pattern
│   ├── tool_use_pattern.py     # Function calling pattern
│   ├── multi_turn_conversation.py # Conversation management
│   ├── structured_output.py    # Guaranteed structured output
│   ├── retry_with_feedback.py  # Self-correcting systems
│   └── cost_optimization.py    # Optimizing API costs
│
├── 10_evaluation_suite/        # Systematic evaluation examples
│   ├── README.md               # Evaluation methodology
│   ├── accuracy_evaluation.py  # Measuring accuracy
│   ├── consistency_testing.py  # Testing consistency
│   ├── robustness_testing.py   # Adversarial testing
│   ├── performance_profiling.py # Detailed performance analysis
│   ├── a_b_testing.py         # Comparing approaches
│   ├── regression_testing.py   # Preventing regressions
│   └── benchmark_harness.py    # Reusable benchmark framework
│
└── notebooks/                 # Jupyter notebooks
    ├── README.md              # Notebook guide
    ├── interactive_tutorial.ipynb  # Interactive learning
    ├── visualization_guide.ipynb   # Result visualization
    └── debugging_guide.ipynb       # Debugging techniques
```

## Section 09: Practical Patterns - Design Rationale

This section focuses on **implementable patterns** that users frequently need. Each example is:
- **Complete but minimal**: Full working code in ~100-200 lines
- **Pattern-focused**: Demonstrates a reusable approach, not a specific application
- **Immediately useful**: Can be adapted to real projects with minimal changes

### Pattern Selection Criteria
1. **Frequency**: Patterns that come up repeatedly in real projects
2. **Ember-specific value**: Where Ember's features provide clear advantages
3. **Composability**: Patterns that combine well with each other
4. **Teaching value**: Each pattern teaches a different aspect of the framework

### Detailed Pattern Descriptions

**rag_pattern.py**: Minimal RAG showing how to combine retrieval with generation using Ember's operators. Demonstrates operator composition and data flow.

**chain_of_thought.py**: Implements CoT prompting with automatic reasoning extraction. Shows structured output handling and prompt engineering.

**tool_use_pattern.py**: Demonstrates function calling with proper error handling and retry logic. Uses Ember's type system for validation.

**multi_turn_conversation.py**: Stateful conversation management with context windowing. Shows how to maintain state across operator calls.

**structured_output.py**: Guarantees valid JSON/structured output using verification and retry operators. Critical for production systems.

**retry_with_feedback.py**: Self-correcting system that learns from errors. Demonstrates feedback loops and conditional execution.

**cost_optimization.py**: Techniques for reducing API costs - caching, model routing, and batch optimization. Shows Ember's performance features.

## Section 10: Evaluation Suite - Design Rationale

This section provides **evaluation infrastructure** that users can adapt for their own systems. Focus on:
- **Measurement methodology**: How to properly evaluate compound AI systems
- **Statistical rigor**: Proper sampling, confidence intervals, and significance testing
- **Ember-specific tools**: Leveraging Ember's features for efficient evaluation

### Evaluation Examples Design

**accuracy_evaluation.py**: Framework for measuring task-specific accuracy. Includes proper train/test splitting and metric calculation.

**consistency_testing.py**: Tests for output stability across runs. Important for production reliability.

**robustness_testing.py**: Adversarial testing framework - prompt injection, edge cases, and failure modes.

**performance_profiling.py**: Deep performance analysis using Ember's built-in metrics. Identifies bottlenecks and optimization opportunities.

**a_b_testing.py**: Statistical framework for comparing different approaches. Includes power analysis and significance testing.

**regression_testing.py**: Automated testing to prevent quality regressions. Integrates with CI/CD pipelines.

**benchmark_harness.py**: Reusable framework for creating custom benchmarks. Handles data loading, execution, and reporting.

## Example Template

Each example should follow this structure:

```python
"""
Example: [Clear, descriptive title]
Learning Objective: [What the user will learn]
Prerequisites: [What examples to complete first]
Concepts: [Key concepts demonstrated]
"""

import ember
from ember.api import models, operators, non, xcs, data

# Configuration (if needed)
# Keep this minimal and well-documented

def main():
    """Main example logic with clear sections."""
    
    # Step 1: Setup
    # Clear explanation of what we're setting up
    
    # Step 2: Core demonstration
    # The main learning objective
    
    # Step 3: Results
    # Show and explain the results
    
    # Step 4: Variations (optional)
    # Show variations or extensions

if __name__ == "__main__":
    # Make it runnable
    main()
```

## Documentation Standards

### README Structure
Each directory should have a README with:
1. **Overview**: What this section covers
2. **Learning Path**: Recommended order
3. **Key Concepts**: Important ideas introduced
4. **Common Patterns**: Reusable patterns shown
5. **Next Steps**: Where to go next

### Code Comments
- Explain the "why" not the "what"
- Link to relevant documentation
- Highlight important patterns
- Warn about common pitfalls

### Output Examples
Include example output in comments or docstrings:
```python
# Example output:
# Model: gpt-4
# Response: Machine learning is...
# Tokens used: 150
# Latency: 1.2s
```

## Learning Paths

### Path 1: Quick Start (2 hours)
1. `01_getting_started/hello_world.py`
2. `01_getting_started/first_model_call.py`
3. `02_core_concepts/operators_basics.py`
4. `04_compound_ai/simple_ensemble.py`

### Path 2: Building Systems (1 day)
1. Complete Quick Start
2. `03_operators/custom_operator.py`
3. `05_data_processing/loading_datasets.py`
4. `06_performance/jit_basics.py`
5. `09_practical_patterns/rag_pattern.py`
6. `10_evaluation_suite/accuracy_evaluation.py`

### Path 3: Production Systems (1 week)
1. Complete Building Systems
2. All of `06_performance/`
3. `07_advanced_patterns/production_pipeline.py`
4. `08_integrations/fastapi_server.py`
5. `09_practical_patterns/` (all patterns)
6. `10_evaluation_suite/benchmark_harness.py`

## Success Metrics

Good examples should:
1. Run without errors on first try
2. Complete in under 30 seconds (except benchmarks)
3. Produce clear, understandable output
4. Teach exactly one main concept
5. Build on previous examples
6. Include error cases and how to handle them

## Anti-Patterns to Avoid

1. **Kitchen Sink Examples**: Trying to show everything at once
2. **Undefined Variables**: Assuming imports or setup
3. **Hidden Dependencies**: Requiring external services without clear setup
4. **Unexplained Magic**: Using advanced features without explanation
5. **Broken Examples**: Code that doesn't run
6. **Unclear Purpose**: Examples without clear learning objectives