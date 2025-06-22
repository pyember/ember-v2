# Operator Tier Interoperability Analysis

## Current State: Limited Interoperability ❌

The current design has **separate implementations** that don't naturally compose:
- Simple functions are just functions
- Advanced operators are classes with protocols
- Experimental features wrap differently

This creates problems:
1. Can't use simple functions inside advanced operators easily
2. Tree protocols don't apply to simple functions
3. IR tracing can't see through class boundaries
4. Type information is lost between tiers

## Solution: Unified Operator Protocol 

### Core Design: Everything is a Callable

```python
# src/ember/operators/core/protocol.py

from typing import Protocol, TypeVar, Any, Callable, Union

T = TypeVar('T')
R = TypeVar('R')

class OperatorProtocol(Protocol[T, R]):
    """Universal protocol that all operators satisfy.
    
    Key insight: Both functions and classes with __call__ satisfy this.
    """
    def __call__(self, inputs: T) -> R: ...

# Type alias for any operator
AnyOperator = Union[
    Callable[[Any], Any],  # Simple function
    OperatorProtocol[Any, Any],  # Advanced operator
]
```

### Tier 1 → Tier 2 Adapter

```python
# src/ember/operators/advanced/adapters.py

from typing import Any, Callable, Tuple, List
from ember.operators.advanced import Operator, TreeProtocol, operator

@operator.advanced
class FunctionAdapter(Operator, TreeProtocol):
    """Adapts simple functions to advanced operators.
    
    This allows simple functions to:
    - Participate in tree transformations
    - Be used in advanced compositions
    - Maintain type information
    """
    
    func: Callable
    func_name: str
    static_config: dict = {}
    
    def __init__(self, func: Callable, **static_config):
        self.func = func
        self.func_name = getattr(func, '__name__', 'anonymous')
        self.static_config = static_config
    
    def __call__(self, *args, **kwargs):
        """Execute the wrapped function."""
        return self.func(*args, **kwargs)
    
    def tree_flatten(self) -> Tuple[List[Any], dict]:
        """Enable tree transformations on simple functions."""
        # Functions are leaves in the tree
        return [self.func], {"name": self.func_name, **self.static_config}
    
    @classmethod
    def tree_unflatten(cls, aux_data, values):
        """Reconstruct from tree."""
        func = values[0]
        config = {k: v for k, v in aux_data.items() if k != 'name'}
        return cls(func, **config)

# Convenience function
def lift(func: Callable, **config) -> FunctionAdapter:
    """Lift a simple function to advanced operator."""
    return FunctionAdapter(func, **config)
```

### Automatic Adaptation in Composition

```python
# src/ember/operators/core/composition.py

from typing import List, Any, Union, Callable
from ember.operators.advanced.adapters import lift

def ensure_operator(obj: Any) -> Any:
    """Ensure object is a proper operator.
    
    - Functions are lifted to FunctionAdapter
    - Classes with __call__ are passed through
    - Everything else raises TypeError
    """
    if hasattr(obj, '__call__'):
        # It's callable, check if it needs adaptation
        if hasattr(obj, 'tree_flatten'):
            # Already an advanced operator
            return obj
        elif callable(obj):
            # Simple function, lift it
            return lift(obj)
    
    raise TypeError(f"Object {obj} is not callable")

def chain(*operators):
    """Chain operators with automatic adaptation."""
    adapted = [ensure_operator(op) for op in operators]
    
    def chained(x):
        result = x
        for op in adapted:
            result = op(result)
        return result
    
    # Make the chain itself liftable
    chained._operators = adapted
    chained._is_chain = True
    
    return chained

def parallel(*operators):
    """Parallel execution with automatic adaptation."""
    adapted = [ensure_operator(op) for op in operators]
    
    def parallel_exec(x):
        # Could use ThreadPoolExecutor here
        return [op(x) for op in adapted]
    
    parallel_exec._operators = adapted
    parallel_exec._is_parallel = True
    
    return parallel_exec
```

### IR Tracing Across Tiers

```python
# src/ember/operators/experimental/unified_tracing.py

from typing import Any, Callable
import inspect

class UnifiedTracer:
    """Traces execution across all operator tiers."""
    
    def trace(self, operator: Any, example_inputs: Any):
        """Trace any operator type to build IR."""
        
        # Handle different operator types
        if hasattr(operator, '_is_chain'):
            # Composite operator
            return self._trace_chain(operator, example_inputs)
        
        elif hasattr(operator, 'tree_flatten'):
            # Advanced operator with tree protocol
            return self._trace_advanced(operator, example_inputs)
        
        elif hasattr(operator, '__wrapped__'):
            # Decorated function (validate, measure, etc)
            return self._trace_decorated(operator, example_inputs)
        
        elif callable(operator):
            # Simple function
            return self._trace_simple(operator, example_inputs)
        
        else:
            raise TypeError(f"Cannot trace {type(operator)}")
    
    def _trace_simple(self, func: Callable, inputs: Any):
        """Trace simple function execution."""
        # Use bytecode analysis or AST parsing
        source = inspect.getsource(func)
        # Build IR nodes...
        pass
    
    def _trace_advanced(self, operator: Any, inputs: Any):
        """Trace advanced operator with tree protocol."""
        # Flatten to get structure
        values, aux = operator.tree_flatten()
        
        # Trace each component
        for val in values:
            if callable(val):
                self.trace(val, inputs)
        
        # Build composite IR...
        pass
    
    def _trace_chain(self, chain: Any, inputs: Any):
        """Trace chain composition."""
        ir_nodes = []
        current_input = inputs
        
        for op in chain._operators:
            node = self.trace(op, current_input)
            ir_nodes.append(node)
            # Simulate execution for next input
            current_input = op(current_input)
        
        # Connect nodes in sequence...
        pass
```

## Complete Interoperability Example

```python
# All three tiers working together seamlessly

from ember.operators import validate, measure, chain
from ember.operators.advanced import Operator, TreeProtocol, operator
from ember.operators.experimental import jit_compile, trace

# Tier 1: Simple function
@measure
@validate(input=str, output=str)
def extract_keywords(text: str) -> str:
    return models("gpt-4", f"Extract keywords from: {text}")

# Tier 2: Advanced operator
@operator.advanced
class KeywordAnalyzer(Operator, TreeProtocol):
    model: str = "gpt-4"
    max_keywords: int = 5
    
    def __call__(self, keywords: str) -> dict:
        prompt = f"Analyze these keywords: {keywords}"
        analysis = models(self.model, prompt)
        return {"keywords": keywords, "analysis": analysis}
    
    def tree_flatten(self):
        return [], {"model": self.model, "max_keywords": self.max_keywords}
    
    @classmethod
    def tree_unflatten(cls, aux, values):
        return cls(**aux)

# Tier 3: Experimental optimization
@jit_compile
@trace
def optimized_pipeline(texts: List[str]) -> List[dict]:
    """This pipeline mixes all three tiers seamlessly."""
    
    # Create pipeline mixing tiers
    analyzer = KeywordAnalyzer(model="claude-3")
    
    # Chain simple function with advanced operator
    pipeline = chain(
        extract_keywords,  # Tier 1: Simple function
        analyzer,          # Tier 2: Advanced operator
        lambda x: {**x, "timestamp": time.time()}  # Tier 1: Lambda
    )
    
    # The JIT compiler can see through all layers
    return [pipeline(text) for text in texts]

# Usage is seamless
results = optimized_pipeline(["text1", "text2", "text3"])

# The IR tracer can see:
# 1. extract_keywords function body
# 2. KeywordAnalyzer tree structure  
# 3. Lambda function
# 4. Loop pattern for optimization
```

## Implementation Requirements

### 1. Unified Type System
```python
# src/ember/operators/core/types.py

from typing import TypeVar, Protocol, Union, Callable

# Universal operator protocol
class UniversalOperator(Protocol):
    """Protocol that all operators satisfy."""
    
    def __call__(self, *args, **kwargs): ...
    
    # Optional protocols
    def tree_flatten(self) -> tuple: ...
    def get_static_config(self) -> dict: ...
    def get_dependencies(self) -> list: ...

# Type aliases for clarity
SimpleOperator = Callable[[Any], Any]
AdvancedOperator = UniversalOperator
AnyOperator = Union[SimpleOperator, AdvancedOperator]
```

### 2. Automatic Protocol Detection
```python
# src/ember/operators/core/inspection.py

def get_operator_capabilities(op: Any) -> dict:
    """Detect what protocols an operator supports."""
    return {
        'callable': callable(op),
        'tree_protocol': hasattr(op, 'tree_flatten'),
        'dependencies': hasattr(op, 'get_dependencies'),
        'static_config': hasattr(op, 'get_static_config'),
        'validated': hasattr(op, '_validation_spec'),
        'measured': hasattr(op, '_measurement_enabled'),
        'jit_compiled': hasattr(op, '_jit_compiled'),
        'traced': hasattr(op, '_trace_enabled'),
    }

def adapt_operator(op: Any, target_tier: str) -> Any:
    """Adapt operator to target tier requirements."""
    capabilities = get_operator_capabilities(op)
    
    if target_tier == "advanced" and not capabilities['tree_protocol']:
        # Lift simple to advanced
        return lift(op)
    
    elif target_tier == "simple" and capabilities['tree_protocol']:
        # Wrap advanced as simple function
        return lambda *args, **kwargs: op(*args, **kwargs)
    
    # Already compatible
    return op
```

### 3. Transparent Boundaries
```python
# XCS transformations work across all tiers
from ember.xcs import vmap

# Simple function
def process(x): return x * 2

# Advanced operator  
analyzer = KeywordAnalyzer()

# Both work with vmap transparently
batch_process = vmap(process)
batch_analyze = vmap(analyzer)

# Can even vmap mixed compositions
mixed = chain(process, analyzer)
batch_mixed = vmap(mixed)  # Works seamlessly!
```

## Benefits of Full Interoperability

1. **No Artificial Boundaries**: Use any operator anywhere
2. **Gradual Complexity**: Start simple, enhance as needed
3. **Optimization Friendly**: IR can see through all layers
4. **Type Safe**: Types flow through adaptations
5. **XCS Compatible**: All transformations work
6. **Future Proof**: New tiers can be added

## Summary

Yes, with the proper adapter layer and unified protocol, all three tiers can be fully interoperable. The key insights are:

1. Everything is fundamentally callable
2. Automatic adaptation at composition boundaries
3. IR tracing that understands all tier types
4. Tree protocol adapters for simple functions
5. Unified type system across all tiers

This ensures that users can freely mix operators from different tiers and everything will "just work" from both a runtime and optimization perspective.