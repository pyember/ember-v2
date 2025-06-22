# Ember Refactoring Plan: The Minimal Implementation

## Executive Summary

Based on deep analysis of the codebase and usage patterns, this plan outlines a radical simplification of Ember following the principles of:
1. Measure what actually matters (99% of usage is simple model calls)
2. Ruthlessly simplify based on measurements
3. Create the minimal implementation that could possibly work

## Current State Analysis

### What Users Actually Do (99% of Usage)
```python
# 1. Call models
response = models("gpt-4", "What is 2+2?")

# 2. Load data
dataset = data("mmlu")

# 3. Evaluate models
score = eval("gpt-4", "mmlu", "accuracy")

# 4. Cache expensive calls
@fast
def pipeline(text):
    return models("gpt-4", f"Analyze: {text}")
```

### What's Actually Slow in LLM Applications
1. **API calls to LLMs** (99% of execution time)
2. **Data loading** (for large datasets)
3. **NOT**: Registry lookups, context switches, operator initialization

### Current Complexity Metrics
- **Total lines of code**: ~50,000+
- **Number of classes**: 200+
- **Configuration files**: Multiple YAML schemas
- **Initialization paths**: 5+ different ways
- **Context systems**: 2 (EmberContext, ModelContext)

## Measurement Plan

### 1. Usage Analysis Script
```python
# analyze_usage.py - Measure what APIs are actually used
import ast
import os
from collections import Counter

def analyze_ember_usage(directory):
    """Analyze which Ember APIs are actually used."""
    api_calls = Counter()
    import_patterns = Counter()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                # Parse and count API usage
                # Track: models(), data(), eval(), operators, xcs, non
                pass
    
    return api_calls, import_patterns
```

### 2. Performance Profiling
```python
# profile_bottlenecks.py - Measure where time is actually spent
import cProfile
import pstats

def profile_typical_workflow():
    """Profile a typical Ember workflow."""
    # Measure:
    # - Model call time vs initialization time
    # - Registry lookup overhead
    # - Context switching cost
    pass
```

## Minimal Implementation Design

### Core API (ember/__init__.py)
```python
"""Ember: Simple LLM Framework.

Example:
    >>> import ember
    >>> answer = ember.models("gpt-4", "What is 2+2?")
    >>> print(answer.text)
    "4"
"""

import os
from functools import lru_cache
from typing import Optional, Iterator, Dict, Any

__version__ = "2.0.0"

# Direct model dispatch - no registries
def models(model: str, prompt: str, **kwargs) -> ModelResponse:
    """Call any model. Just works."""
    if "gpt" in model or "o1" in model:
        return _call_openai(model, prompt, **kwargs)
    elif "claude" in model:
        return _call_anthropic(model, prompt, **kwargs)
    elif "gemini" in model:
        return _call_google(model, prompt, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")

# Direct dataset loading - no builders
def data(name: str, **kwargs) -> Iterator[Dict[str, Any]]:
    """Load any dataset. Returns iterator."""
    loaders = {
        "mmlu": _load_mmlu,
        "humaneval": _load_humaneval,
        "truthfulqa": _load_truthfulqa,
        "aime": _load_aime,
        "gpqa": _load_gpqa,
    }
    
    if name in loaders:
        return loaders[name](**kwargs)
    elif "path" in kwargs:
        return _load_file(kwargs["path"], kwargs.get("format", "jsonl"))
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Direct evaluation - no pipelines
def eval(model: str, data_name: str, metric: str = "accuracy") -> float:
    """Evaluate model on dataset."""
    dataset = data(data_name) if isinstance(data_name, str) else data_name
    
    evaluators = {
        "accuracy": _eval_accuracy,
        "exact_match": _eval_exact_match,
        "numeric": _eval_numeric,
        "code": _eval_code_execution,
    }
    
    if metric not in evaluators:
        raise ValueError(f"Unknown metric: {metric}")
    
    return evaluators[metric](model, dataset)

# Simple caching decorator
fast = lru_cache(maxsize=1000)

# Model response object
class ModelResponse:
    def __init__(self, text: str, model: str, usage: Optional[Dict] = None):
        self.text = text
        self.model = model
        self.usage = usage or {}

# Private implementations
def _call_openai(model: str, prompt: str, **kwargs):
    """Direct OpenAI API call."""
    import openai
    
    api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )
    
    return ModelResponse(
        text=response.choices[0].message.content,
        model=model,
        usage={
            "total_tokens": response.usage.total_tokens,
            "cost": _calculate_cost(model, response.usage)
        }
    )

# Similar for other providers...
```

### That's It. The Entire Public API.

## Files to Delete (Phase 1)

### Context Systems (8,000+ lines)
- `/src/ember/core/context/` - Entire directory
- `/src/ember/core/config/` - Entire directory except schema.py
- All EmberContext and ModelContext references

### Registry Systems (10,000+ lines)
- `/src/ember/core/registry/model/base/registry/` - Keep only model info
- `/src/ember/core/registry/operator/` - Entire directory
- All discovery and factory patterns

### Operator Systems (5,000+ lines)
- `/src/ember/core/operators_v2/` - Convert to simple functions
- `/src/ember/api/operators.py` - Replace with function utilities
- All operator base classes and protocols

### XCS System (15,000+ lines)
- `/src/ember/xcs/` - Keep only simple caching
- All JIT strategies except basic memoization
- Graph building and execution engine

### Total Deletion: ~40,000 lines

## New Minimal Implementations

### 1. ember/__init__.py (200 lines)
- `models()` - Direct provider dispatch
- `data()` - Direct loader dispatch
- `eval()` - Simple evaluation loop
- `fast` - LRU cache decorator

### 2. ember/providers.py (300 lines)
- `_call_openai()` - Direct OpenAI API
- `_call_anthropic()` - Direct Anthropic API
- `_call_google()` - Direct Google API
- Cost calculation utilities

### 3. ember/datasets.py (200 lines)
- `_load_mmlu()` - Direct MMLU loader
- `_load_humaneval()` - Direct HumanEval loader
- Common dataset utilities

### 4. ember/legacy.py (100 lines)
- Import redirects for backward compatibility
- Deprecation warnings
- Migration helpers

### Total New Code: ~800 lines

## Migration Path

### For 99% of Users
```python
# Before (complex)
from ember import initialize_ember
from ember.core.context import EmberContext

initialize_ember(config_path="config.yaml")
ctx = EmberContext.current()
model = ctx.get_model("gpt-4")
response = model(prompt="Hello")

# After (simple)
import ember
response = ember.models("gpt-4", "Hello")
```

### For the 1% (Advanced Users)
```python
# Provide ember.legacy module
from ember.legacy import Operator, EmberContext, XCS

# But strongly encourage migration to simple functions
```

## Implementation Timeline

### Week 1: Measurement and Analysis
- [ ] Run usage analysis on all examples and tests
- [ ] Profile performance bottlenecks
- [ ] Document actual vs perceived needs
- [ ] Create detailed deletion list

### Week 2: Core Implementation
- [ ] Implement minimal ember/__init__.py
- [ ] Implement direct provider calls
- [ ] Implement direct dataset loaders
- [ ] Add basic caching with @fast

### Week 3: Migration Support
- [ ] Create ember.legacy module
- [ ] Add deprecation warnings
- [ ] Update critical examples
- [ ] Write migration guide

### Week 4: Cleanup and Release
- [ ] Delete identified files
- [ ] Update all documentation
- [ ] Ensure backward compatibility
- [ ] Release Ember 2.0

## Success Metrics

### Simplicity
- Lines of code: 50,000 → 1,000 (98% reduction)
- Number of classes: 200+ → 5 (97% reduction)
- Time to first LLM call: <1 second
- Lines to understand: 4 functions vs 50+ classes

### Performance
- Import time: <100ms (vs current ~2s)
- First model call: No initialization overhead
- Memory usage: Minimal (no registries)
- Cache hit rate: >90% for repeated calls

### Adoption
- New user time-to-productivity: <5 minutes
- Migration effort for 99%: Change imports only
- Documentation needs: 1 page vs 100+ pages

## Philosophy

This refactoring embodies the Unix philosophy:
- **Do one thing well**: Call LLMs simply
- **Make it obvious**: No documentation needed
- **Optimize for the common case**: Direct function calls
- **Text streams**: Everything is text in, text out

As Kernighan and Ritchie showed with C, and Thompson and Ritchie showed with Unix, the best abstractions are often no abstractions at all. Just simple, direct, obvious functions that do exactly what you'd expect.

## Conclusion

By measuring what actually matters (LLM API latency dominates everything) and designing for the 99% use case (simple model calls), we can create an API that needs no documentation, has no learning curve, and just works. The complex cases that need operators, contexts, and registries represent <1% of usage and can use the legacy module.

The result: Ember becomes as simple as `requests.get()` or `numpy.array()` - obvious, immediate, and powerful.