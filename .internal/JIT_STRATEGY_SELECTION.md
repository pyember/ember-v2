# JIT Strategy Selection Guide

This document explains how Ember's JIT system selects compilation strategies and how to debug or override the selection process.

## Strategy Overview

Ember's JIT system includes three compilation strategies, each optimized for different use cases:

### 1. structural strategy (Max Score: 55)
Best for simple functions with predictable execution paths.

**Scoring Criteria:**
- **30 points**: Simple function (< 20 lines) - trace works best on small functions
- **20 points**: Simple control flow - minimal branching means predictable traces  
- **5 points**: Base score - trace is the fallback strategy for all functions

**Ideal For:**
- Small utility functions
- Functions with minimal branching
- Data transformation functions

### 2. Structural Strategy (Max Score: 100)
Best for Ember operators and classes with well-defined structure.

**Scoring Criteria:**
- **40 points**: Has nested operators (strongest indicator)
- **30 points**: Has forward() method - indicates operator pattern
- **20 points**: Has __call__ method - callable class
- **10 points**: Has specification attribute - operator pattern

**Ideal For:**
- Ember Operator classes
- Complex nested operators
- Classes with forward() methods

### 3. Enhanced Strategy (Max Score: 100)
Best for complex functions with parallelization opportunities.

**Scoring Criteria:**
- **40 points**: Multiple loops (>1) - prime candidate for parallelization
- **30 points**: Ensemble naming pattern - designed for parallel execution
- **20 points**: Single loop - potential for optimization
- **20 points**: Iteration capability - supports parallel processing
- **10 points**: Base score - good general fallback

**Ideal For:**
- Ensemble operators
- Functions with multiple loops
- Parallel processing opportunities

## Debugging Strategy Selection

Enable debug logging to see how strategies are selected:

```python
import logging

# Enable debug logging for JIT modules
logging.getLogger('ember.xcs.jit.core').setLevel(logging.DEBUG)
logging.getLogger('ember.xcs.jit.strategies').setLevel(logging.DEBUG)
```

This will output detailed information like:

```
JIT strategy selection for EnsembleOperator:
  structural: score=60, breakdown={'forward_method': 30, 'callable_class': 20, 'specification': 10}, reason=Has 'forward' method (likely an operator); Has __call__ method; Has 'specification' attribute (operator pattern)
  enhanced: score=60, breakdown={'ensemble_pattern': 30, 'single_loop': 20, 'base_score': 10}, reason=Name suggests ensemble pattern; Contains loops; Good general-purpose option
  trace: score=55, breakdown={'simple_function': 30, 'simple_control_flow': 20, 'base_score': 5}, reason=Simple function (< 20 lines); Simple control flow; Basic fallback strategy
Selected structural strategy (highest score)
```

## Forcing a Specific Strategy

You can override automatic selection using either `mode` or `force_strategy`:

```python
# Force structural strategy
@jit(mode="structural")
def my_function(*, inputs):
    return process(inputs)

# Alternative syntax using force_strategy
@jit(force_strategy="structural")
def my_function(*, inputs):
    return process(inputs)
```

Available strategies:
- `"trace"` - Use trace-based compilation
- `"structural"` - Use structural analysis
- `"enhanced"` - Use enhanced compilation with parallelism detection
- `"auto"` (default) - Let the system choose

## When to Override Strategy Selection

Consider overriding the automatic selection when:

1. **You know your code structure better** - The heuristics might miss domain-specific patterns
2. **Performance testing** - Compare different strategies for your specific use case
3. **Debugging** - Force a simpler strategy (trace) to isolate issues
4. **Special requirements** - Your code has unique characteristics not captured by the heuristics

## Performance Considerations

- **Trace** is fastest to compile but may miss optimization opportunities
- **Structural** provides good optimization for operators with predictable structure
- **Enhanced** takes longer to analyze but can discover parallelism in sequential code

Always profile your specific use case to determine the best strategy.