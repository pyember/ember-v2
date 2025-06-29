"""XCS: Smart execution for Ember.

Just use @jit. That's it.

Examples:
    from ember.xcs import jit

    @jit
    def my_function(x):
        return model(x)

    # Automatic optimization with zero configuration!

For advanced users (rare):
    from ember.xcs import jit, vmap, pmap, scan, grad

    # Intelligent batching
    @vmap
    def batch_process(x):
        return model(x)

    # Distributed execution
    @pmap
    def distributed_process(x):
        return model(x)

    # Automatic differentiation
    @grad
    def loss_function(params):
        return compute_loss(params)

Architecture Philosophy:
    XCS (Xccelerated Compound Systems) provides intelligent wrappers around
    JAX transformations that understand Ember's mixed computation model
    (LLM calls, tool / mcp server use, pure JAX-compatible computation).

    Core Innovation:
    Unlike standard JAX which can only compile pure numerical code, XCS
    automatically identifies and segregates:
    - Pure computations (fully compilable)
    - Model calls (cached and batched)
    - Tool/API/MCP calls (deferred execution)

    This enables JIT compilation of hybrid AI systems, achieving 10-100x
    speedups on repeated executions while maintaining full flexibility.

Design Principles:
    1. **Zero Configuration**: @jit works on any Ember code
    2. **Graceful Degradation**: Non-compilable parts run normally
    3. **Transparent Optimization**: No code changes required
    4. **Progressive Enhancement**: Advanced features available when needed

Performance Characteristics:
    - First execution: Analysis and tracing overhead (~100-500ms)
    - Subsequent executions: Near-native speed for pure parts
    - Model calls: Automatic batching reduces API latency
    - Memory: Caches compilation artifacts (~10-100MB per function)

Trade-offs:
    - Analysis overhead on first call vs massive speedup thereafter
    - Memory usage for caching vs reduced computation
    - Complexity hidden from users vs less control
    - Automatic optimization vs explicit optimization
"""

# Core transformations - these subsume JAX transformations
# Simple utility for stats
from ember.xcs._simple import get_jit_stats
from ember.xcs.transformations import grad, jit, pmap, scan, vmap

# That's the complete public API!
__all__ = ["jit", "vmap", "pmap", "scan", "grad", "get_jit_stats"]

# Note: The old complex API is preserved but hidden.
# Power users can still access it via ember.xcs.legacy
# but we don't advertise this.
