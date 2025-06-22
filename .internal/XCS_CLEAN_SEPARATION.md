# XCS Clean Separation: @jit vs @structural analysis provides performance improvements for I/O-bound operations
@trace
@jit
class InstrumentedPipeline(Operator):
    # JIT optimizes, trace measures
    pass
```

## The Deeper Truth

Separating @jit and @trace acknowledges that:
1. **Optimization and observation are different concerns**
2. **Not everything can be made faster**
3. **Understanding execution is valuable on its own**
4. **Clear tools are better than clever tools**

This is the simplification XCS needs.