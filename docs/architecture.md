# Ember Architecture

This document explains Ember's architectural principles and design decisions.

## Core Principles

### 1. Simple for the 90% Case

Ember is designed around the observation that most LLM applications need the same basic operations. These should be trivial:

```python
# This covers 90% of use cases
from ember.api import ember

response = await ember.llm("What is machine learning?")
```

No configuration files, no client initialization, no session management. Just function calls.

### 2. Progressive Disclosure of Complexity

Complexity is introduced only when needed, following a clear progression:

```python
# Level 1: Simple function (90% of users)
@ember.op
async def summarize(text: str) -> str:
    return await ember.llm(f"Summarize: {text}")

# Level 2: Add validation (9% of users)
@ember.op
@validate
async def analyze(input: AnalysisInput) -> AnalysisOutput:
    return await ember.llm(input.text, output_type=AnalysisOutput)

# Level 3: Full control (1% of users)
class CustomOperator(Operator):
    def __init__(self, config):
        super().__init__(spec=custom_spec)
        self.config = config
```

### 3. Composability Over Inheritance

Instead of deep inheritance hierarchies, Ember uses functional composition:

```python
# Bad: Deep inheritance
class BaseOperator:
    pass

class ValidatedOperator(BaseOperator):
    pass

class CachedValidatedOperator(ValidatedOperator):
    pass

# Good: Composition
@ember.op
@validate
@cache(ttl=3600)
async def my_operator(input: Input) -> Output:
    pass
```

### 4. Type Safety by Default

Type hints aren't optional - they drive the system:

```python
# Types define behavior
@ember.op
async def process(data: List[str]) -> Dict[str, float]:
    # Ember knows:
    # - Input is a list of strings
    # - Output is a dict mapping strings to floats
    # - Can validate both automatically
    pass
```

### 5. Performance Without Configuration

Optimization happens automatically:

```python
# Ember detects parallelizable operations
results = await ember.parallel([
    analyze(doc) for doc in documents
])  # Runs with optimal concurrency

# JIT compilation when beneficial
@ember.op
@jit  # Traces execution for optimization
async def pipeline(data):
    pass
```

## Architectural Layers

### API Layer (`ember.api`)

The user-facing API is intentionally minimal:

```
ember.api/
├── __init__.py      # Core ember functions
├── operators.py     # Operator decorators and utilities
├── models.py        # Model management
├── data.py          # Data processing utilities
└── types.py         # Shared type definitions
```

Key decisions:
- Single import for common cases: `from ember.api import ember`
- Flat namespace - no deep module hierarchies
- Functions over classes where possible

### Core Layer (`ember.core`)

The implementation layer handles complexity:

```
ember.core/
├── operators/       # Operator implementation
├── models/          # Model provider abstraction
├── data/            # Data loading and transformation
├── xcs/             # Execution engine
└── utils/           # Shared utilities
```

Key decisions:
- Clear separation from API layer
- Internal complexity hidden from users
- Modular design for maintainability

### Execution Layer (XCS)

The execution engine provides:
- Automatic parallelization
- JIT compilation
- Graph-based execution
- Performance optimization

```python
# User writes simple code
@ember.op
async def process_batch(items):
    return await ember.parallel([
        process_item(item) for item in items
    ])

# XCS automatically:
# - Builds execution graph
# - Identifies parallelizable operations
# - Optimizes execution strategy
# - Manages resources
```

## Key Design Patterns

### 1. Registry Pattern

Central registries manage complexity while providing simple access:

```python
# Models registry
models("gpt-4", prompt)  # Registry handles provider resolution

# Data registry
datasets("mmlu")  # Registry handles dataset loading

# Operator registry
@ember.op  # Automatically registered for composition
```

### 2. Builder Pattern

For cases needing configuration:

```python
model = ModelBuilder()
    .temperature(0.7)
    .max_tokens(1000)
    .build("gpt-4")

dataset = DatasetBuilder()
    .from_registry("mmlu")
    .filter(lambda x: x.difficulty == "hard")
    .sample(100)
    .build()
```

### 3. Decorator Pattern

Cross-cutting concerns via decorators:

```python
@ember.op
@validate      # Add validation
@cache(ttl=3600)  # Add caching
@retry(attempts=3)  # Add retry logic
@trace         # Add observability
async def complex_operation(input: Input) -> Output:
    pass
```

### 4. Adapter Pattern

Uniform interface across providers:

```python
# Same interface, different providers
response1 = models("gpt-4", prompt)
response2 = models("claude-3", prompt)
response3 = models("gemini-pro", prompt)

# All return same response type
print(response1.text)
print(response2.text)
print(response3.text)
```

## Performance Architecture

### Lazy Initialization

Resources are initialized only when needed:

```python
# No initialization cost
from ember.api import ember

# First call initializes provider
response = await ember.llm("Hello")  # ~10ms overhead

# Subsequent calls are fast
response = await ember.llm("World")  # <1ms overhead
```

### Lock-Free Operations

Read-heavy operations use lock-free patterns:

```python
# Registry lookups don't require locks
model = registry.get_model("gpt-4")  # Lock-free read

# Only registration requires synchronization
registry.register_model(new_model)  # Single mutex
```

### Memory Efficiency

Single instance per resource:

```python
# Same model instance reused
model1 = models.instance("gpt-4")
model2 = models.instance("gpt-4")
assert model1 is model2  # True
```

## Extension Points

### Custom Providers

```python
from ember.api.models import BaseProvider, register_provider

@register_provider("custom")
class CustomProvider(BaseProvider):
    async def complete(self, prompt, **kwargs):
        # Custom implementation
        pass
```

### Custom Operators

```python
from ember.api.operators import Operator

class DomainSpecificOperator(Operator):
    def __init__(self, domain_config):
        super().__init__()
        self.config = domain_config
    
    async def forward(self, input):
        # Domain-specific logic
        pass
```

### Custom Data Sources

```python
from ember.api.data import register_dataset

@register_dataset("my-data")
class CustomDataset:
    def load(self, config):
        # Custom loading logic
        pass
```

## Design Trade-offs

### Simplicity vs Flexibility

We chose simplicity for common cases:
- ✅ `ember.llm(prompt)` vs ❌ `client.completions.create(...)`
- ✅ Automatic provider detection vs ❌ Explicit provider configuration
- ✅ Type-driven validation vs ❌ Schema configuration files

### Performance vs Correctness

We chose correctness with automatic optimization:
- Type checking at runtime ensures correctness
- JIT compilation recovers performance
- Profiling guides optimization decisions

### Abstraction vs Transparency

We chose transparency:
- Clear error messages over generic exceptions
- Explicit costs over hidden charges
- Visible parallelization over magic

## Future Considerations

The architecture supports evolution without breaking changes:

1. **New Providers**: Add via registry without API changes
2. **New Optimizations**: Add to XCS without user code changes
3. **New Patterns**: Add via decorators without core changes

The key is maintaining the simple interface while evolving the implementation.

## Summary

Ember's architecture reflects a deliberate choice: make simple things simple and complex things possible. By hiding complexity behind a minimal API, using progressive disclosure, and leveraging automatic optimization, Ember provides both ease of use and production-grade performance.