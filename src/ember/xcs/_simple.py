"""Simple JIT implementation for automatic optimization.

This module provides the core JIT decorator that automatically optimizes
Python functions by detecting parallelism opportunities and applying
compilation strategies.

The implementation follows Ember's principle of progressive disclosure:
- Zero configuration required for basic use
- Automatic parallelism detection
- Graceful fallback for non-optimizable functions
- Optional configuration for advanced use cases

Architecture notes:
    The JIT system uses runtime tracing to build an IR graph, analyzes it
    for parallelism opportunities, and executes using an optimized engine.
    Functions with orchestration operations (LLM calls) are detected and
    handled appropriately to avoid side effects during tracing.
"""

import functools
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from ember.xcs._internal.engine import ExecutionEngine
from ember.xcs._internal.ir_builder import IRBuilder
from ember.xcs._internal.parallelism import ParallelismAnalyzer
from ember.xcs._internal.profiler import Profiler

if TYPE_CHECKING:
    from ember.xcs.config import Config

# Global instances for singleton pattern
# These are module-level to ensure consistent state across all JIT calls
_engine: Optional[ExecutionEngine] = None
_profiler: Optional[Profiler] = None


def _get_engine() -> ExecutionEngine:
    """Get or create the global execution engine.

    Returns:
        ExecutionEngine: The singleton execution engine instance.
    """
    global _engine
    if _engine is None:
        _engine = ExecutionEngine()
    return _engine


def _get_profiler() -> Profiler:
    """Get or create the global profiler.

    Returns:
        Profiler: The singleton profiler instance.
    """
    global _profiler
    if _profiler is None:
        _profiler = Profiler()
    return _profiler


def jit(func: Optional[Callable] = None, *, _config: Optional["Config"] = None) -> Callable:
    """Make any function faster. No configuration needed.

    Just add @jit to your function and XCS will:
    - Automatically discover parallelism
    - Optimize execution order
    - Cache results when beneficial
    - Choose the best execution strategy

    Examples:
        @jit
        def process(data):
            return model(data)

        # That's it! Automatic optimization with zero configuration.

        # For advanced users (rare):
        from ember.xcs.config import Config

        @jit(config=Config(cache=False))
        def process_sensitive(data):
            return model(data)

    Args:
        func: Function to optimize (or None if used as @jit())
        _config: Advanced configuration (most users should ignore this)

    Returns:
        Optimized version of the function
    """
    # Handle both @jit and @jit() syntax
    if func is None:
        return functools.partial(jit, _config=_config)

    # Check if already jitted
    if hasattr(func, "_xcs_optimized"):
        return func

    # Store optimization state
    optimization_decision = None
    optimization_cache = {}
    builder = IRBuilder()
    analyzer = ParallelismAnalyzer()

    @functools.wraps(func)
    def optimized_func(*args, **kwargs):
        """Optimized version of the original function."""
        nonlocal optimization_decision

        start_time = time.time()

        # One-shot optimization decision (per design)
        if optimization_decision is None:
            # Check if we're already inside a tracing context
            # This handles recursive @jit functions
            if hasattr(builder.tracer, "tracing") and builder.tracer.tracing:
                # We're being called from within a trace - don't try to trace
                optimization_decision = False
            else:
                try:
                    # Check if this function contains LLM calls by doing a quick test trace
                    # This is a heuristic - if tracing produces too many operations,
                    # it's likely tracing into system internals (like model loading)
                    test_ops = builder.tracer.trace_function(func, args, kwargs)

                    # If we get a huge number of operations, it's tracing internals
                    # This suggests the function contains orchestration operations
                    if len(test_ops) > 100:
                        # Too complex - likely contains orchestration
                        # Don't try to optimize functions with LLM calls
                        optimization_decision = False
                    else:
                        # Try to trace and analyze the function
                        graph = builder.trace_function(func, args, kwargs)
                        analysis = analyzer.analyze_graph(graph)

                        # Decision: optimize if we found parallel opportunities
                        optimization_decision = len(analysis.parallel_groups) > 0 and any(
                            len(group) > 1 for group in analysis.parallel_groups
                        )

                        # Store the analysis for later use
                        if optimization_decision:
                            optimization_cache["graph"] = graph
                            optimization_cache["analysis"] = analysis

                except Exception:
                    # Can't trace/analyze - permanent fallback to original
                    optimization_decision = False

        # Execute based on optimization decision
        try:
            if optimization_decision:
                # Use optimized execution
                engine = _get_engine()
                graph = optimization_cache["graph"]
                analysis = optimization_cache["analysis"]

                # CRITICAL FIX: Pass runtime args, not traced args
                # The graph stores the computation pattern, not the data
                result = engine.execute(graph, args, kwargs, parallelism_info=analysis)
            else:
                # Fallback to original function
                result = func(*args, **kwargs)

            # Profile if enabled
            if _should_profile(_config):
                elapsed = time.time() - start_time
                profiler = _get_profiler()
                profiler.record(
                    func.__name__,
                    elapsed_ms=elapsed * 1000,
                    parallelism_info=optimization_cache.get("analysis"),
                    graph_size=(
                        len(optimization_cache.get("graph", {}).nodes)
                        if optimization_cache.get("graph")
                        else 1
                    ),
                )

            return result

        except Exception:
            # Re-raise - maintain exact sequential semantics
            raise

    # Mark as optimized
    optimized_func._xcs_optimized = True
    optimized_func._xcs_original = func

    # For debugging/introspection (hidden from normal users)
    optimized_func._xcs_explain = lambda: _explain_optimization(func, optimization_cache)

    # Add stats() method for introspection
    def stats() -> Dict[str, Any]:
        """Get optimization statistics for this function."""
        if optimization_decision is None:
            return {"status": "not_executed", "optimized": False}
        elif optimization_decision:
            analysis = optimization_cache.get("analysis")
            return {
                "status": "optimized",
                "optimized": True,
                "parallel_groups": len(analysis.parallel_groups) if analysis else 0,
                "estimated_speedup": analysis.estimated_speedup if analysis else 1.0,
                "graph_size": (
                    len(optimization_cache.get("graph", {}).nodes)
                    if optimization_cache.get("graph")
                    else 0
                ),
            }
        else:
            return {
                "status": "fallback",
                "optimized": False,
                "reason": "no_parallelism_found",
            }

    optimized_func.stats = stats

    return optimized_func


def _should_profile(config: Optional["Config"]) -> bool:
    """Decide whether to profile this execution."""
    if config and hasattr(config, "profile"):
        return config.profile

    # Default: profile 1% of executions for continuous learning
    import random

    return random.random() < 0.01


def _explain_optimization(func: Callable, optimization_cache: dict) -> str:
    """Explain how a function would be optimized (for debugging)."""
    if not optimization_cache:
        return f"Function {func.__name__} has not been executed yet."

    analysis = optimization_cache.get("analysis")
    if not analysis:
        return f"Function {func.__name__} could not be optimized (no parallelism found)."

    explanation = [
        f"Optimization for {func.__name__}:",
        f"  - Graph size: {len(optimization_cache.get('graph', {}).nodes)} nodes",
        f"  - Parallel groups: {len(analysis.parallel_groups)}",
        f"  - Estimated speedup: {analysis.estimated_speedup:.2f}x",
    ]

    if analysis.bottlenecks:
        explanation.append(f"  - Bottlenecks: {', '.join(analysis.bottlenecks)}")

    return "\n".join(explanation)


# Temporary stub for get_jit_stats until we implement profiler
def get_jit_stats(func: Optional[Callable] = None) -> Dict[str, Any]:
    """Get optimization statistics.

    Args:
        func: Specific function to get stats for, or None for global stats

    Returns:
        Dictionary of statistics and insights
    """
    profiler = _get_profiler()

    if func is not None:
        # Get stats for specific function
        func_name = getattr(func, "__name__", str(func))
        return profiler.get_function_stats(func_name)
    else:
        # Get global stats
        return profiler.get_global_stats()
