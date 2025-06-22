# Quick Reference: Architectural Decisions

## Models Module
**Decision**: Hide registry, expose simple API
```python
# Public API (all you see)
response = models("gpt-4", "Hello")
gpt4 = models.bind("gpt-4", temperature=0.5)  # Not .instance

# Hidden: Provider registry, routing, configuration
# Day 0 Fix: Remove get_registry() from public API
```

## Data Module  
**Decision**: Minimal essential metadata only
```python
# What matters
metadata = DatasetMetadata(
    size_bytes=1000000,
    estimated_examples=10000,
    recommended_batch_size=32,  # Measured, not configured
    example_item={"text": "Hello", "label": 1}
)

# Not: Validation schemas, statistical properties, extensive types
```

## Operators Module
**Decision**: Progressive disclosure, no forced structure
```python
# Level 1: Just functions (90% of users)
def my_op(x): return x + 1

# Level 2: Optional validation (9% of users)
from ember.api.operators_v2 import validate  # Day 0: Export this
@validate(input=str, output=str)
def my_op(x): return x + "!"

# Level 3: Full specs if needed (1% of users)
# But never required
```

## XCS Module
**Decision**: Keep IR, hide complexity, inspire from MLIR
```python
# Public API
@jit
def process(x): ...

# Hidden: IR graphs, optimization passes, strategies
# Critical: IR enables cloud scheduler optimization
# Day 0: Scaffold basic IR package to unblock work
```

## Migration Strategy
**Decision**: Clean break, no backward compatibility
- No shims
- No compatibility layers  
- Clear migration guides
- Fresh start

## The Masters' Rules

<div style="background: #f0f0f0; padding: 10px; border: 1px solid #ccc;">

### Before Adding Anything
1. What error does this prevent?
2. Can users do this themselves simply?
3. Are we optimizing the common case?
4. Would Ritchie approve?

### Before Deleting Anything  
1. Why was this added originally?
2. What use case does it serve?
3. Is there a simpler way?
4. What breaks if removed?

### The North Star Question
"If Dean, Ghemawat, Brockman, Martin, Jobs, Ritchie, Knuth, and Carmack were pair programming, would they nod in approval?"

</div>

## Success Metrics
- Each module < 1000 lines (CI enforced)
- Public API fits on one page
- 80% use simplest form
- Zero backward compatibility code
- Google L10+ quality
- Contract tests prevent leakage

## Where to Look for More

| Topic | Location |
|-------|----------|
| Design Philosophy | `docs/design/` |
| XCS Deep Dive | `docs/xcs/` |
| Migration Guides | `docs/migration/` |
| IR System Details | `src/ember/ir/README.md` |