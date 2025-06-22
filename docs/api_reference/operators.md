# Operators API Reference

The operators module provides tools for building composable AI workflows.

## Import

```python
from ember.api import operators
from ember.api.operators import Operator, validate, ensemble
```

## Decorators

### @ember.op

Convert a function into an operator.

```python
from ember.api import ember

@ember.op
async def my_operator(input: str) -> str:
    return await ember.llm(f"Process: {input}")
```

**Features:**
- Automatic async handling
- Type validation from hints
- Composition support
- Metadata extraction

### @validate

Add input/output validation to operators.

```python
from ember.api import validate
from pydantic import BaseModel

class Input(BaseModel):
    text: str
    max_length: int = 100

@ember.op
@validate
async def validated_op(input: Input) -> str:
    return await ember.llm(input.text)
```

## Operator Class

For complex operators requiring state or custom behavior.

```python
from ember.api.operators import Operator

class StatefulOperator(Operator):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.state = {}
    
    async def forward(self, input: Any) -> Any:
        # Custom processing logic
        result = await ember.llm(f"{self.config['prefix']}: {input}")
        self.state['last_input'] = input
        return result
```

## Composition Functions

### ensemble()

Run multiple models and combine results.

```python
from ember.api.operators import ensemble

results = await ensemble(
    "What is machine learning?",
    models=["gpt-4", "claude-3", "gemini-pro"],
    temperature=0.7
)
```

### majority_vote()

Select most common result from a list.

```python
from ember.api.operators import majority_vote

best_answer = majority_vote([
    "Paris", "Paris", "London", "Paris"
])  # Returns "Paris"
```

### synthesize()

Combine multiple outputs into one.

```python
from ember.api.operators import synthesize

final_summary = await synthesize(
    summaries=["Summary 1", "Summary 2", "Summary 3"],
    instruction="Create a comprehensive summary"
)
```

### judge()

Evaluate and rank multiple responses.

```python
from ember.api.operators import judge

best = await judge(
    question="What causes rain?",
    responses=["Answer 1", "Answer 2", "Answer 3"],
    criteria="accuracy and clarity"
)
```

## Built-in Operators

### EnsembleOperator

```python
from ember.api.operators import EnsembleOperator

ensemble_op = EnsembleOperator(
    models=["gpt-4", "claude-3"],
    voting_strategy="weighted",
    weights=[0.6, 0.4]
)

result = await ensemble_op("Explain quantum computing")
```

### VerifierOperator

```python
from ember.api.operators import VerifierOperator

verifier = VerifierOperator(
    model="gpt-4",
    verification_prompt="Check this answer for accuracy"
)

verified = await verifier(
    question="What is 2+2?",
    answer="4"
)
```

### ChainOperator

```python
from ember.api.operators import ChainOperator

chain = ChainOperator([
    extract_entities,
    enrich_entities,
    generate_summary
])

result = await chain(document)
```

## Operator Specification

For complex operators with formal specifications.

```python
from ember.api.operators import Specification
from pydantic import BaseModel

class MyInput(BaseModel):
    query: str
    context: str

class MyOutput(BaseModel):
    answer: str
    confidence: float

spec = Specification(
    input_schema=MyInput,
    output_schema=MyOutput,
    description="Answer questions based on context"
)

class MyOperator(Operator):
    def __init__(self):
        super().__init__(spec=spec)
    
    async def forward(self, input: MyInput) -> MyOutput:
        # Implementation
        pass
```

## Operator Utilities

### compose()

Compose multiple operators into one.

```python
from ember.api.operators import compose

pipeline = compose(
    preprocess,
    analyze,
    postprocess
)

result = await pipeline(input_data)
```

### parallel_ops()

Run operators in parallel.

```python
from ember.api.operators import parallel_ops

results = await parallel_ops(
    [op1, op2, op3],
    input_data
)
```

### conditional()

Conditional operator execution.

```python
from ember.api.operators import conditional

smart_op = conditional(
    condition=lambda x: len(x) > 1000,
    if_true=detailed_analysis,
    if_false=quick_summary
)
```

## Error Handling

```python
from ember.api.operators import OperatorError, ValidationError

try:
    result = await my_operator(invalid_input)
except ValidationError as e:
    print(f"Input validation failed: {e}")
except OperatorError as e:
    print(f"Operator execution failed: {e}")
```

## Testing Operators

```python
from ember.api.operators import mock_operator

# Mock an operator for testing
with mock_operator(my_operator, returns="mocked"):
    result = await my_operator("test")
    assert result == "mocked"

# Test operator composition
async def test_pipeline():
    pipeline = compose(op1, op2, op3)
    result = await pipeline("test input")
    assert result.success
```

## Performance Optimization

### JIT Compilation

```python
from ember.api import jit

@ember.op
@jit
async def optimized_op(items: list) -> list:
    return await ember.parallel([
        process(item) for item in items
    ])
```

### Caching

```python
from ember.api.operators import cached

@ember.op
@cached(ttl=3600)
async def expensive_op(input: str) -> str:
    return await complex_computation(input)
```

## Best Practices

1. **Start Simple**: Use function operators for most cases
2. **Add Validation**: Use @validate when you need guarantees
3. **Compose Small Operators**: Build complex behavior from simple parts
4. **Type Everything**: Use type hints for automatic validation
5. **Handle Errors**: Add appropriate error handling
6. **Test Thoroughly**: Use mocks and test utilities

## See Also

- [Operators Quickstart](../quickstart/operators.md)
- [NON Patterns](../quickstart/non.md)
- [Examples](../examples/operators/)