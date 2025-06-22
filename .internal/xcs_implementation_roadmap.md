# XCS Implementation Roadmap

## Phase 1: Foundation - Clean IR leveraging Module System (Week 1)

### 1.1 IR Node Design (Knuth: Literate, Clean)
```python
@dataclass(frozen=True)
class IRNode:
    """Immutable IR node - no magic, just data."""
    operator: Any  # The actual operator (module)
    inputs: Tuple[str, ...]  # Input variable names
    outputs: Tuple[str, ...]  # Output variable names
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 1.2 Parallelism Discovery via Pytrees (Dean/Ghemawat: Data Locality)
- Leverage `_tree_registry` from module system
- Automatic dependency analysis from pytree structure
- No user configuration needed

### Review Point 1: Does the IR feel like Unix pipes? Simple, composable, no magic?

## Phase 2: Progressive API Layers (Week 2)

### 2.1 Layer 1: The 90% API (Jobs: Simplicity)
```python
# ember/xcs/__init__.py
from ember.xcs.simple import jit

# That's it. One import, one decorator.
```

### 2.2 Layer 2: Advanced Config (9% of users)
```python
# ember/xcs/config.py
@dataclass(frozen=True)
class Config:
    parallel: bool = True
    cache: bool = True
    profile: bool = False
    # No scheduler exposure, no strategies
```

### 2.3 Layer 3: Expert Mode (1% of users)
```python
# ember/xcs/advanced/__init__.py
# Only imported explicitly by experts
from .engine import ExecutionEngine
from .ir import IRTransform
from .scheduler import Scheduler
```

### Review Point 2: Can a beginner use XCS without seeing any complexity?

## Phase 3: Smart Execution Engine (Week 3)

### 3.1 Hidden Scheduler (Martin: Dependency Inversion)
- Schedulers are internal implementation details
- Engine automatically picks the best one
- No scheduler types in public API

### 3.2 Automatic Strategy Selection (Carmack: Embrace Constraints)
```python
def _select_strategy(ir_graph: IRGraph) -> Strategy:
    """Pick strategy based on graph characteristics."""
    if ir_graph.is_purely_sequential():
        return SequentialStrategy()
    elif ir_graph.has_independent_branches():
        return ParallelStrategy()
    else:
        return AdaptiveStrategy()
```

### Review Point 3: Is the execution engine making smart choices without user input?

## Phase 4: Measurement & Self-Tuning (Week 4)

### 4.1 Built-in Profiling (Page: Measure Everything)
```python
@jit
def my_op(x):
    return model(x)

# Automatically tracked:
# - Execution time
# - Memory usage  
# - Cache hit rate
# - Parallelism efficiency
```

### 4.2 Self-Improving System (Brockman: Platforms)
- Learn from execution patterns
- Improve strategy selection over time
- Export learnings for similar workloads

### Review Point 4: Is the system getting smarter without user intervention?

## Phase 5: Integration & Polish (Week 5)

### 5.1 Operator Integration
- Seamless integration with EmberModule
- Automatic batching via pytree protocol
- Zero-config parallelism for ensemble operators

### 5.2 Documentation & Examples
- Simple examples for 90% use case
- Advanced examples hidden in separate docs
- Expert examples require explicit opt-in

### Final Review: Would Dean, Jobs, and Knuth approve?

## Implementation Order

1. **Start with IR**: Clean, immutable, pytree-aware
2. **Then simple API**: Just `@jit`, nothing else visible
3. **Then execution**: Hidden complexity, smart defaults
4. **Then measurement**: Automatic profiling and tuning
5. **Finally polish**: Documentation, examples, integration

## Success Criteria

1. **Simplicity Test**: Can a new user write `@jit` and get parallelism?
2. **Power Test**: Can an expert achieve everything the old system could?
3. **Leakage Test**: Are implementation details hidden from users?
4. **Performance Test**: Is the new system as fast or faster?
5. **Debuggability Test**: Can users understand what went wrong?

## Anti-Patterns to Avoid

1. **No `JITMode.AUTO` in user code** - Just `@jit`
2. **No scheduler selection** - System chooses
3. **No execution options** - Smart defaults
4. **No strategy configuration** - Automatic selection
5. **No premature optimization** - Measure first

## Code Smells to Watch For

1. If users import from more than one module, we failed
2. If users see the word "scheduler", we failed
3. If users configure strategies, we failed
4. If beginners need documentation, we failed
5. If experts can't extend it, we failed