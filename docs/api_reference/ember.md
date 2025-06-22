# ember API Reference

The core `ember` module provides the main interface for interacting with LLMs and building AI applications.

## Import

```python
from ember.api import ember
```

## Core Functions

### ember.llm()

Make LLM calls with automatic model selection and response parsing.

```python
async def llm(
    prompt: str | list[dict],
    *,
    model: str = None,
    output_type: type = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs
) -> Any
```

**Parameters:**
- `prompt`: Text prompt or conversation history
- `model`: Model name (e.g., "gpt-4", "claude-3")  
- `output_type`: Pydantic model for structured output
- `temperature`: Sampling temperature (0.0-2.0)
- `max_tokens`: Maximum response length
- `**kwargs`: Additional model-specific parameters

**Examples:**

```python
# Simple text response
response = await ember.llm("What is Python?")

# Structured output
class Analysis(BaseModel):
    summary: str
    sentiment: str
    
result = await ember.llm(
    "Analyze this review: Great product!",
    output_type=Analysis
)

# Conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"}
]
response = await ember.llm(messages)
```

### ember.op

Decorator to create operators from functions.

```python
@ember.op
async def my_operator(input: str) -> str:
    return await ember.llm(f"Process: {input}")
```

**Features:**
- Automatic async handling
- Type validation from hints
- Composability with other operators
- JIT compilation support

### ember.parallel()

Execute multiple operations in parallel.

```python
async def parallel(
    operations: list[Coroutine],
    max_workers: int = None
) -> list[Any]
```

**Example:**

```python
results = await ember.parallel([
    ember.llm(f"Summarize doc {i}")
    for i in range(10)
])
```

### ember.stream()

Process items with streaming and backpressure.

```python
async def stream(
    items: Iterable,
    processor: Callable,
    batch_size: int = 1
) -> AsyncIterator
```

**Example:**

```python
async for result in ember.stream(documents, analyze_document):
    print(f"Processed: {result.id}")
```

### ember.batch()

Create batches from an iterable.

```python
def batch(items: Iterable, size: int) -> Iterator[list]
```

**Example:**

```python
for batch in ember.batch(data, size=10):
    results = await ember.parallel([
        process(item) for item in batch
    ])
```

## Configuration Functions

### ember.set_default_model()

Set the default model for LLM calls.

```python
ember.set_default_model("gpt-4")
```

### ember.configure()

Configure Ember settings.

```python
ember.configure(
    default_model="claude-3",
    cache_enabled=True,
    retry_attempts=3
)
```

## Utility Functions

### ember.mock_llm()

Mock LLM calls for testing.

```python
with ember.mock_llm(returns="mocked response"):
    result = await my_operator("test")
    assert result == "mocked response"
```

### ember.cache()

Cache operator results.

```python
@ember.op
@ember.cache(ttl=3600)
async def expensive_operation(input: str) -> str:
    return await ember.llm(f"Complex analysis: {input}")
```

### ember.retry()

Add retry logic to operations.

```python
@ember.op
@ember.retry(max_attempts=3, backoff=True)
async def unreliable_operation(input: str) -> str:
    return await external_api.call(input)
```

## Error Handling

Ember provides clear error types:

```python
from ember.api import EmberError, ModelError, ValidationError

try:
    result = await ember.llm("prompt", output_type=MyModel)
except ValidationError as e:
    print(f"Output validation failed: {e}")
except ModelError as e:
    print(f"Model call failed: {e}")
except EmberError as e:
    print(f"General Ember error: {e}")
```

## Context Management

### ember.context()

Manage execution context.

```python
async with ember.context(model="gpt-4", temperature=0.5):
    # All LLM calls in this block use these settings
    result1 = await ember.llm("First prompt")
    result2 = await ember.llm("Second prompt")
```

## Advanced Features

### JIT Compilation

```python
from ember.api import jit

@ember.op
@jit
async def optimized_pipeline(data: list) -> list:
    # Ember optimizes execution
    return await ember.parallel([
        process(item) for item in data
    ])
```

### Custom Operators

```python
from ember.api import Operator

class CustomOperator(Operator):
    async def forward(self, input: Any) -> Any:
        # Custom processing logic
        return await ember.llm(f"Custom: {input}")
```

## Best Practices

1. **Use Type Hints**: Enable validation and better IDE support
2. **Handle Errors**: Use try/except for robust applications
3. **Batch Operations**: Use `parallel()` and `batch()` for efficiency
4. **Cache Results**: Cache expensive operations
5. **Test with Mocks**: Use `mock_llm()` for unit tests

## See Also

- [Operators Guide](../quickstart/operators.md)
- [Models Reference](./models.md)
- [Examples](../examples/)