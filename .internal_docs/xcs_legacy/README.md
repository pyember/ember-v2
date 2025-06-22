# Legacy XCS Implementation

This directory contains the original complex XCS implementation that has been superseded by the simplified API in the parent directory.

## Why This Was Moved

The new simplified XCS API provides:
- Zero-configuration @jit decorator that "just works"
- Automatic parallelism detection without complex graph building
- Clean separation between tensor and orchestration operations
- 90% of functionality with 10% of the complexity

## What's Preserved Here

This legacy implementation contains:
- Complex graph building and dependency analysis
- Multiple JIT strategies (trace, structural, enhanced)
- Sophisticated scheduler implementations
- Device mesh transformations
- Extensive tracing infrastructure

These components represent significant engineering effort and contain hard-won lessons about:
- Edge cases in parallelism detection
- Complex operator composition patterns
- Performance optimization strategies
- Distributed execution patterns

## When to Reference This Code

Look at this legacy implementation when:
- Debugging complex edge cases in the new implementation
- Understanding historical design decisions
- Extracting specific optimizations that proved valuable
- Implementing advanced features for power users

## Migration Status

The following components have been successfully migrated to the simplified API:
- Basic JIT compilation → `_simple.py`
- Transformation API (vmap, pmap, etc.) → `transformations.py`
- Operation analysis → `_internal/analysis.py`
- IR representation → `_internal/ir.py`
- Parallelism detection → `_internal/parallelism.py`

## Note

This code is preserved for reference and should not be imported directly.
Use the public API in `ember.xcs` instead.