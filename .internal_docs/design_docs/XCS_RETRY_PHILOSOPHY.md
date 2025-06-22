# XCS Retry Philosophy: Clean Separation of Concerns

## Core Principle: XCS Does NOT Retry

XCS follows JAX's philosophy of fast failure without retry logic. This is a deliberate design decision based on clean separation of concerns.

## Why No Retry Logic in XCS

### 1. Domain Knowledge Lives Elsewhere
- **Users** know when retries make sense for their use case
- **Model providers** know their API limits and retry strategies  
- **XCS** only knows about parallelization

### 2. Retry Strategies Are Domain-Specific
```python
# Financial application - might not want any retries
@jit
def process_trades(trades):
    return [execute_trade(t) for t in trades]  # Fail fast on any error

# Research application - might want aggressive retries
@jit
def analyze_documents(docs):
    return [analyze_with_backoff(doc) for doc in docs]  # User handles retries

# Production API - might want circuit breakers
@jit
def serve_requests(requests):
    return [handle_with_circuit_breaker(r) for r in requests]  # Library handles
```

### 3. Composability Over Magic
```python
# Bad: XCS has hidden retry magic
@jit(retry_count=3, backoff=True)  # NO! Hidden behavior
def process(items):
    return [model(x) for x in items]

# Good: Explicit retry logic where needed
@jit
def process(items):
    return [retry(model, x, max_attempts=3) for x in items]  # Clear and visible
```

### 4. Following the Masters

**JAX**: No retry on compilation or execution failures
**MapReduce**: Framework doesn't retry user functions
**Spark**: User functions fail fast
**TensorFlow**: Operations fail immediately

All successful frameworks separate execution from retry logic.

## Where Retry Logic Belongs

### 1. In User Code
```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def reliable_model(x):
    return model(x)

@jit
def process(items):
    return [reliable_model(x) for x in items]
```

### 2. In Model Providers
```python
# In ember.models
class ModelBinding:
    def __call__(self, *args, **kwargs):
        # Model provider handles retries based on:
        # - Rate limits
        # - Transient errors  
        # - Cost considerations
        return self._call_with_retry(*args, **kwargs)
```

### 3. In Middleware
```python
# Application-level retry policies
@with_retry_policy(ResearchRetryPolicy())
@jit
def research_pipeline(data):
    return process(data)
```

## Benefits of No Retry in XCS

1. **Predictability**: Errors surface immediately
2. **Debuggability**: Stack traces are clean
3. **Testability**: No hidden behavior to mock
4. **Performance**: No overhead from retry logic
5. **Simplicity**: Less code, fewer bugs

## Examples

### Good: Explicit Retry Handling
```python
# User controls retry logic
def safe_model_call(x):
    for attempt in range(3):
        try:
            return model(x)
        except TransientError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

@jit
def process(items):
    return [safe_model_call(x) for x in items]
```

### Bad: Hidden Retry Magic
```python
# DON'T DO THIS - XCS should not have retry options
@jit(retry=True, max_attempts=3)  # Hidden behavior!
def process(items):
    return [model(x) for x in items]
```

## Error Propagation

When an error occurs during parallel execution:

1. **Immediate Propagation**: Error surfaces immediately
2. **Clean Cancellation**: Pending work is cancelled
3. **Original Exception**: User sees their actual error
4. **Sequential Semantics**: Same error behavior as non-parallel

```python
@jit
def process(items):
    results = []
    for i, item in enumerate(items):
        # If this fails at i=50 in parallel execution,
        # behavior is identical to sequential execution
        results.append(model(item))
    return results
```

## Conclusion

XCS's job is to make parallel code run in parallel, not to handle failures. By refusing to implement retry logic, XCS:

- Stays simple and predictable
- Composes well with existing retry libraries
- Lets domain experts handle domain-specific failures
- Follows the proven patterns of successful frameworks

The rule is simple: **XCS parallelizes. Users retry.**