# Ember Framework: LLM Specifications

This document provides precise specifications for building systems with the Ember framework. It defines essential patterns, APIs, and concepts with minimal overhead.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Architecture Overview](#architecture-overview)
- [Model Registry](#model-registry)
- [Operators](#operators)
- [Execution Engine (XCS)](#execution-engine-xcs)
- [Network of Networks (NON)](#network-of-networks-non)
- [Data System](#data-system)
- [Context System](#context-system)
- [Code Style Guidelines](#code-style-guidelines)
- [Examples and Templates](#examples-and-templates)

## Core Concepts

### EmberModel
Base class for structured input/output with type validation:

```python
from ember.core.types.ember_model import EmberModel, Field

class QueryInput(EmberModel):
    query: str = Field(..., description="Query to process")
    temperature: float = Field(0.7, description="Sampling temperature")

class AnswerOutput(EmberModel):
    answer: str = Field(..., description="Generated response")
    confidence: float = Field(..., description="Confidence score 0-1")
```

### Operator
Fundamental computational unit with typed I/O:

```python
from typing import ClassVar
from ember.api.operators import Operator, Specification

class MySpec(Specification):
    input_model = QueryInput
    structured_output = AnswerOutput

class MyOperator(Operator[QueryInput, AnswerOutput]):
    specification: ClassVar[Specification] = MySpec()
    
    def forward(self, *, inputs: QueryInput) -> AnswerOutput:
        # Implementation here
        return AnswerOutput(answer="...", confidence=0.9)
```

### JIT Compilation
Execute optimized computational graphs:

```python
from ember.xcs import jit

@jit  # Auto-optimize execution path
class FastOperator(Operator[QueryInput, AnswerOutput]):
    # Implementation with parallel execution where possible
```

## Architecture Overview

Ember's layered architecture:

1. **Execution Engine (XCS)** - Base layer providing computation graph definition and execution
2. **Core Component Layer** - Building blocks including model registry, operators, specifications
3. **Application Layer** - High-level abstractions like NON patterns and auto graph building
4. **Public API Layer** - Clean interfaces exposed through the `ember.api` namespace

## Model Registry

The model registry provides unified access to LLM providers:

### Function-Style API (Recommended)

```python
from ember.api import models

# Direct invocation 
response = models.model("gpt-4o")("What is quantum computing?")

# Provider namespaces
response = models.openai.gpt4o("What is quantum computing?")

# Reusable models
gpt4 = models.model("gpt-4o", temperature=0.7)
response1 = gpt4("Question 1")
response2 = gpt4("Question 2")

# Configuration context
with models.configure(temperature=0.2, max_tokens=100):
    response = models.model("gpt-4o")("Write a haiku")
```

### Type-Safe Enum References

```python
from ember.api.models import ModelEnum
response = models.from_enum(ModelEnum.OPENAI_GPT4O)("Hello")
```

### Builder Pattern (Alternative)

```python
from ember.api.models import ModelBuilder
model = (ModelBuilder()
    .temperature(0.7)
    .max_tokens(100)
    .build("anthropic:claude-3-5-sonnet"))
response = model.generate("Explain quantum computing")
```

### Custom Contexts

```python
from ember.api.models import ModelContext, ContextConfig
context = ModelContext(config=ContextConfig(
    api_keys={"openai": "your-key"}
))
response = models.model("gpt-4o", context=context)("Hello")
```

## Operators

### Basic Operator Pattern

```python
from typing import ClassVar
from ember.api.operators import Operator, Specification, EmberModel
from ember.api import models

class InputType(EmberModel):
    query: str

class OutputType(EmberModel):
    answer: str

class MySpec(Specification):
    input_model = InputType
    structured_output = OutputType
    
class MyOperator(Operator[InputType, OutputType]):
    # Class-level specification
    specification: ClassVar[Specification] = MySpec()
    
    # Declare instance attributes
    model: object
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model = models.model(model_name)
    
    def forward(self, *, inputs: InputType) -> OutputType:
        response = self.model(inputs.query)
        return OutputType(answer=str(response))
```

### Composition Pattern

```python
class Pipeline(Operator[InputType, OutputType]):
    specification: ClassVar[Specification] = MySpec()
    
    # Declare component operators
    refiner: QueryRefiner
    answerer: AnswerGenerator
    
    def __init__(self):
        self.refiner = QueryRefiner()
        self.answerer = AnswerGenerator()
    
    def forward(self, *, inputs: InputType) -> OutputType:
        refined = self.refiner(inputs=inputs)
        return self.answerer(inputs=refined)
```

### Built-in Operators

```python
from ember.api.operators import (
    EnsembleOperator,
    MostCommonAnswerSelector,
    VerifierOperator,
    SelectorJudgeOperator,
    JudgeSynthesisOperator
)

# Ensemble of models
ensemble = EnsembleOperator(
    operators=[
        MyOperator(model="gpt-4o"),
        MyOperator(model="claude-3-5-sonnet"),
    ]
)

# Selector for aggregation
pipeline = MostCommonAnswerSelector(
    operator=ensemble
)
```

## Execution Engine (XCS)

### Basic JIT

```python
from ember.xcs import jit

@jit
class MyOperator(Operator):
    def forward(self, *, inputs):
        # Implementation
        return result
```

### JIT with Strategy Selection

```python
from ember.xcs import jit

# Auto-select optimal strategy
@jit
class MyOperator(Operator):
    # Implementation...

# Explicit strategy
@jit(mode="enhanced")
class ComplexOperator(Operator):
    # Implementation...
```

### Execution Options

```python
from ember.xcs import execution_options

# Configure execution parameters
with execution_options(scheduler="wave", max_workers=4):
    result = pipeline(query="Complex question...")
```

### Transformations

```python
from ember.xcs import vmap, pmap, compose

# Vectorized mapping for batch processing
batch_processor = vmap(my_operator)
batch_results = batch_processor(inputs={"data": [item1, item2, item3]})

# Parallel execution across multiple workers
parallel_processor = pmap(my_operator, num_workers=4)
parallel_results = parallel_processor(inputs=complex_data)

# Compose transformations
pipeline = compose(vmap(batch_size=32), pmap(num_workers=4))(my_operator)
```

## Network of Networks (NON)

### Standard API

```python
from ember.api import non

# Create ensemble of identical models
ensemble = non.UniformEnsemble(
    num_units=3,
    model_name="openai:gpt-4o",
    temperature=0.7
)

# Create judge to synthesize outputs
judge = non.JudgeSynthesis(
    model_name="anthropic:claude-3-5-sonnet",
    temperature=0.0
)

# Create sequential pipeline
pipeline = non.Sequential(operators=[ensemble, judge])

# Execute pipeline
result = pipeline(query="What causes tsunamis?")
```

### Compact Notation

```python
from ember.api import non

# Same pipeline with compact notation
pipeline = non.build_graph([
    "3:E:gpt-4o:0.7",              # Ensemble with 3 instances
    "1:J:claude-3-5-sonnet:0.0"    # Judge synthesis
])

# Execute with identical interface
result = pipeline(query="What causes tsunamis?")
```

### Component Reuse

```python
# Define reusable components
components = {
    "sub": ["2:E:gpt-4o:0.0", "1:V:gpt-4o:0.0"]  # Ensemble → Verifier
}

# Create branch architecture
nested = non.build_graph([
    "$sub",                # First branch
    "$sub",                # Second branch
    "1:J:gpt-4o:0.0"       # Final judge
], components=components)
```

### Custom Operator Types

```python
# Create a registry with custom operator types
registry = non.OpRegistry.create_standard_registry()
registry.register(
    "CE",  # Custom ensemble type
    lambda count, model, temp: non.Sequential(operators=[
        non.UniformEnsemble(num_units=count, model_name=model, temperature=temp),
        non.MostCommon()  # Auto-aggregation 
    ])
)

# Use custom operator type
pipeline = non.build_graph(["3:CE:gpt-4o:0.7"], type_registry=registry)
```

## Data System

### DatasetBuilder Pattern

```python
from ember.api.data import DatasetBuilder

# Load and transform a dataset
dataset = (DatasetBuilder()
    .from_registry("mmlu")    # Use a registered dataset
    .subset("physics")        # Select a specific subset
    .split("test")            # Choose the test split
    .sample(100)              # Random sample of 100 items
    .transform(               # Apply transformations
        lambda x: {"query": f"Question: {x['question']}"}
    )
    .build())
```

### Evaluation Pipeline

```python
from ember.api.eval import EvaluationPipeline, Evaluator

# Create an evaluation pipeline
eval_pipeline = EvaluationPipeline([
    # Standard metrics
    Evaluator.from_registry("accuracy"),
    Evaluator.from_registry("response_quality"),
    
    # Custom metrics
    Evaluator.from_function(
        lambda prediction, reference: {
            "factual_accuracy": score_factual_content(prediction, reference)
        }
    )
])

# Evaluate a model or operator
results = eval_pipeline.evaluate(my_model, dataset)
print(f"Accuracy: {results['accuracy']:.2f}")
```

## Context System

### Basic Context Usage

```python
from ember.core.context import current_context

# Get the current thread's context
ctx = current_context()

# Get a model
model = ctx.get_model("gpt4o")

# Generate text
result = model.generate("Hello, world!")
```

### Temporary Components

```python
from ember.core.context import current_context, temp_component

# Use a temporary component
with temp_component("model", "temp-model", MyModel("temporary")) as model:
    # Use the model within this scope
    result = model.generate("Hello")
```

### Configuration Access

```python
from ember.core.context import current_context
from ember.core.context.config_integration import config_override

# Access configuration through context
temperature = ctx.config.model.temperature

# Override configuration temporarily
with config_override({"model": {"temperature": 0.2}}):
    # Config value changed in this scope
    new_temp = ctx.config.model.temperature
```

## Code Style Guidelines

1. **Package Structure**:
   - Use the `ember.api` namespace for clean imports
   - Follow the layered import pattern:
     ```python
     from ember.api.models import ModelBuilder
     from ember.api.operators import Operator
     from ember.api.xcs import jit
     from ember.api import non
     ```

2. **Naming Conventions**:
   - Classes: `PascalCase`
   - Functions: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private attributes: `_leading_underscore`

3. **Typing**:
   - Use explicit type annotations
   - Leverage TypeVars for generic operators
   - Include descriptions in Field definitions

4. **Documentation**:
   - Follow Google docstring format
   - Document parameters, returns, and raises
   - Include examples for non-trivial usage

5. **Error Handling**:
   - Use specific exception types
   - Include helpful error messages
   - Handle API errors gracefully

## Examples and Templates

### Basic Operator Template

```python
from typing import ClassVar
from ember.api.operators import Operator, Specification, EmberModel, Field

class MyInput(EmberModel):
    query: str = Field(..., description="The input query")
    
class MyOutput(EmberModel):
    result: str = Field(..., description="The computed result")
    
class MySpec(Specification):
    input_model = MyInput
    structured_output = MyOutput
    
class MyOperator(Operator[MyInput, MyOutput]):
    specification: ClassVar[Specification] = MySpec()
    
    def __init__(self, param1: str):
        self.param1 = param1
        
    def forward(self, *, inputs: MyInput) -> MyOutput:
        # Implementation logic
        return MyOutput(result=f"Processed: {inputs.query}")
```

### NON Pattern Template

```python
from ember.api import non

def create_ensemble_judge_pipeline(
    ensemble_size: int = 3,
    model_name: str = "openai:gpt-4o",
    judge_model: str = "anthropic:claude-3-5-sonnet"
) -> non.Sequential:
    """Create an ensemble-judge pipeline.
    
    Args:
        ensemble_size: Number of ensemble units
        model_name: Model to use for ensemble
        judge_model: Model to use for judge
        
    Returns:
        A configured pipeline
    """
    return non.Sequential(operators=[
        non.UniformEnsemble(
            num_units=ensemble_size,
            model_name=model_name,
            temperature=0.7
        ),
        non.JudgeSynthesis(
            model_name=judge_model,
            temperature=0.0
        )
    ])
```

### Complete Application Example

```python
"""Minimal Ember application with JIT optimization."""

from typing import ClassVar
from ember.api import models, non
from ember.api.operators import Operator, Specification, EmberModel
from ember.xcs import jit

# Define I/O types
class QuestionInput(EmberModel):
    question: str
    
class AnswerOutput(EmberModel):
    answer: str
    confidence: float
    
# Define specification
class QuestionSpec(Specification):
    input_model = QuestionInput
    structured_output = AnswerOutput
    
# Define JIT-optimized operator
@jit
class QuestionAnswerer(Operator[QuestionInput, AnswerOutput]):
    specification: ClassVar[Specification] = QuestionSpec()
    
    # Declare instance attributes
    ensemble: non.UniformEnsemble
    judge: non.JudgeSynthesis
    
    def __init__(self, width: int = 3):
        self.ensemble = non.UniformEnsemble(
            num_units=width,
            model_name="gpt-4o",
            temperature=0.7
        )
        self.judge = non.JudgeSynthesis(
            model_name="claude-3-5-sonnet"
        )
    
    def forward(self, *, inputs: QuestionInput) -> AnswerOutput:
        # Get ensemble responses
        ensemble_result = self.ensemble(query=inputs.question)
        
        # Synthesize with judge
        synthesis = self.judge(
            query=inputs.question,
            responses=ensemble_result["responses"]
        )
        
        # Build response
        return AnswerOutput(
            answer=synthesis["synthesized_response"],
            confidence=float(synthesis.get("confidence", 0.8))
        )

# Main entrypoint
def main():
    # Create operator
    answerer = QuestionAnswerer(width=5)
    
    # Process question
    result = answerer(inputs=QuestionInput(
        question="What is relativity?"
    ))
    
    # Use result
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")

if __name__ == "__main__":
    main()
```