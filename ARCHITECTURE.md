# Ember Architecture

## Design Philosophy

Ember follows three core principles:

1. **Simple by Default** - Zero configuration, direct function calls, no boilerplate
2. **Progressive Disclosure** - Advanced features available when needed, hidden when not
3. **10x Performance** - Automatic optimization without manual tuning

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         PUBLIC API                          │
├─────────────────┬─────────────────┬────────────────────────┤
│     models()    │   operators()   │    data.stream()       │
│  Direct LLM API │ Composable AI   │  Streaming Data        │
└────────┬────────┴────────┬────────┴───────┬────────────────┘
         │                 │                 │
┌────────▼────────┬────────▼────────┬───────▼────────────────┐
│  Model Registry │ Operator System │   Data Pipeline        │
│                 │                 │                        │
│  • Provider     │ • Composition   │  • Loaders            │
│    Resolution   │ • Validation    │  • Transformers       │
│  • Cost Tracking│ • JAX Pytrees   │  • Samplers           │
└────────┬────────┴────────┬────────┴───────┬────────────────┘
         │                 │                 │
┌────────▼─────────────────▼─────────────────▼────────────────┐
│                      XCS ENGINE                              │
│                                                              │
│  • Automatic JIT Compilation                                │
│  • Parallelism Detection                                     │
│  • Execution Optimization                                    │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Context System

Centralized configuration and credential management with support for isolation:

```python
from ember.context import get_context, create_context, with_context

# Get current context
ctx = get_context()
api_key = ctx.get_credential("openai", "OPENAI_API_KEY")

# Create isolated child context
with create_context(models={"default": "gpt-4"}) as child_ctx:
    # All model calls in this block use gpt-4 by default
    response = models(None, "Hello")  # Uses gpt-4
    
# Back to original context - uses original default model

# Temporary context with overrides
with with_context(models={"temperature": 0.9}):
    # High temperature for creative tasks
    response = models("gpt-4", "Write a poem")
```

**Async Context Propagation:**

```python
import asyncio
from ember.context import create_context, get_context

async def process_with_model(text: str) -> str:
    # Context automatically propagates to async functions
    ctx = get_context()
    model = ctx.get_config("models.default")
    print(f"Processing with {model}")
    # Simulate async work
    await asyncio.sleep(0.1)
    return f"Processed: {text}"

async def main():
    # Create different contexts for concurrent tasks
    async with asyncio.TaskGroup() as tg:
        # Task 1: Uses GPT-4
        with create_context(models={"default": "gpt-4"}):
            task1 = tg.create_task(process_with_model("Hello"))
        
        # Task 2: Uses Claude
        with create_context(models={"default": "claude-3"}):
            task2 = tg.create_task(process_with_model("World"))
    
    # Each task sees its own context
    print(await task1)  # Processed with gpt-4
    print(await task2)  # Processed with claude-3
```

**Thread-Safe Context Usage:**

```python
import threading
from ember.context import create_context, get_context

def worker(name: str, model: str):
    # Each thread gets isolated context
    with create_context(models={"default": model}):
        ctx = get_context()
        print(f"{name}: Using {ctx.get_config('models.default')}")
        # Do work with thread-local model

# Launch workers with different models
threads = [
    threading.Thread(target=worker, args=("Worker1", "gpt-4")),
    threading.Thread(target=worker, args=("Worker2", "claude-3")),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

**Configuration Management:**

```python
from ember.context import get_context, with_context
from ember.api import models

# Get and modify configuration
ctx = get_context()
ctx.set_config("models.temperature", 0.7)
ctx.save()  # Persist to disk

# Temporary configuration overrides
with with_context(models={"temperature": 0.9, "max_tokens": 2000}):
    # This block uses high temperature and more tokens
    response = models("gpt-4", "Write a creative story")
    
# Back to original settings
response = models("gpt-4", "Summarize this document")  # Uses temperature=0.7
```

### 2. Models API

Direct LLM invocation without client initialization:

```python
# Simple case - no setup required
response = models("gpt-4", "Hello world")

# Advanced case - reusable configuration
gpt4 = models.instance("gpt-4", temperature=0.7)
```

**Design Decisions:**
- Registry pattern for provider management
- Lazy initialization for fast startup
- Integrated cost and usage tracking
- Thread-safe singleton implementation

### 2. Operators System

Composable building blocks for AI applications:

```python
# Function decorator approach
@operators.op
def summarize(text: str) -> str:
    return models("gpt-4", f"Summarize: {text}")

# Class-based with validation
class ValidatedOp(Operator):
    input_spec = InputModel
    output_spec = OutputModel
```

**Design Decisions:**
- JAX pytree compatibility for automatic differentiation
- Optional but powerful validation system
- Composition over inheritance
- Zero overhead when validation not used

### 3. Data Pipeline

Streaming-first data loading with progressive enhancement:

```python
# Basic streaming
for item in stream("dataset"):
    process(item)

# Chained transformations
stream("dataset").filter(valid).transform(clean).batch(32)
```

**Design Decisions:**
- Memory-efficient streaming by default
- Explicit materialization when needed
- Protocol-based extensibility
- Built-in caching layer

### 4. XCS Execution Engine

Zero-configuration optimization:

```python
@xcs.jit
def complex_workflow(data):
    # Automatically optimized
    return pipeline(data)

# Automatic parallelization
results = xcs.vmap(process)(batch)
```

**Design Decisions:**
- Tracing-based optimization
- Automatic strategy selection
- JAX backend for numerical operations
- Orchestration for I/O-bound tasks

## Key Design Patterns

### Registry Pattern
Single source of truth for each resource type:
- `ModelRegistry` - LLM provider management
- `DataRegistry` - Dataset loader management
- Thread-safe with proven single-lock approach

### Module System
JAX-compatible base class for all operators:
- Automatic parameter detection
- Pytree registration
- Clean composition semantics

### Progressive Disclosure
Three levels of API complexity:
1. **Simple Functions** - Direct calls for basic use
2. **Decorators** - Enhancement without boilerplate
3. **Classes** - Full control when needed

### Type-Driven Development
Optional but powerful when used:
- Pydantic models for validation
- Type hints guide system behavior
- Runtime validation from static types

## Performance Architecture

### JIT Compilation Strategy
1. Function decorated with `@jit`
2. First call traces execution
3. IR built from trace
4. Optimal backend selected:
   - JAX for numerical operations
   - Async orchestration for I/O
5. Subsequent calls use compiled version

### Parallelization Detection
- Automatic detection of map operations
- Data dependency analysis
- Optimal chunking for throughput
- Zero configuration required

### Memory Management
- Streaming by default for data
- Lazy evaluation where possible
- Explicit materialization points
- Automatic garbage collection hints

## Extension Points

### Adding Providers
1. Implement `BaseProvider` interface
2. Register with `ModelRegistry`
3. No core changes required

### Custom Operators
1. Inherit from `Operator` base
2. Define `call()` method
3. Optional validation specs
4. Automatic JAX integration

### Data Loaders
1. Implement loader protocol
2. Register with `DataRegistry`
3. Streaming support automatic

## Testing Strategy

### Unit Tests
- Isolated component testing
- Minimal test doubles
- Type testing utilities

### Integration Tests
- Cross-module interactions
- Real provider testing
- Performance benchmarks

### Test Patterns
- Helper modules for common setups
- Simplified imports for isolation
- Comprehensive coverage tracking

## Security Considerations

### API Key Management
- Environment variable loading
- No keys in code
- Secure defaults

### Input Validation
- Optional but recommended
- Pydantic integration
- Type-safe boundaries

### Rate Limiting
- Provider-level handling
- Automatic retry logic
- Exponential backoff

## Future Architecture

### Planned Enhancements
1. **Distributed Execution** - Multi-node XCS
2. **Model Quantization** - Automatic optimization
3. **Streaming Inference** - Token-level streaming
4. **Edge Deployment** - Browser/mobile runtime

### Design Principles Maintained
- Simple API remains simple
- Advanced features stay optional
- Performance improvements automatic
- Backward compatibility preserved

## Architecture Decision Records

### ADR-001: Direct Instantiation
**Decision**: Use direct function calls instead of dependency injection
**Rationale**: Eliminates boilerplate, improves discoverability
**Consequences**: Simpler API, easier testing, less flexibility

### ADR-002: Registry Pattern
**Decision**: Single registry per resource type
**Rationale**: Clear ownership, thread-safe, extensible
**Consequences**: Centralized management, potential bottleneck

### ADR-003: JAX Integration
**Decision**: Base operators on JAX pytrees
**Rationale**: Automatic differentiation, JIT compilation
**Consequences**: Power user features, slight complexity

### ADR-004: Streaming First
**Decision**: Default to streaming for data operations
**Rationale**: Memory efficiency, scalability
**Consequences**: Explicit materialization needed sometimes

This architecture embodies the principles of simplicity, performance, and extensibility that guide Ember's development.