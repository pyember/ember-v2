"""Core JIT compilation system for XCS.

Provides Just-In-Time compilation that optimizes operators and functions
by analyzing their structure and execution patterns.
"""

import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from ember.xcs.jit.cache import JITCache, get_cache
from ember.xcs.jit.modes import JITMode
from ember.xcs.jit.strategies import Strategy

F = TypeVar("F", bound=Callable)
logger = logging.getLogger(__name__)


class JITSettings:
    """Settings for JIT compilation behavior."""

    def __init__(
        self,
        mode: Union[JITMode, str] = JITMode.AUTO,
        force_trace: bool = False,
        custom_cache: Optional[JITCache] = None,
        recursive: bool = True,
        preserve_stochasticity: bool = False) -> None:
        """Initialize JIT settings.

        Args:
            mode: JIT compilation mode (default: AUTO).
            force_trace: Whether to force retracing.
            custom_cache: Custom cache instance.
            recursive: Whether to apply JIT recursively.
            preserve_stochasticity: Whether to preserve stochastic behavior.
        """
        # Normalize mode to enum
        if isinstance(mode, str):
            try:
                self.mode = JITMode(mode.lower())
            except ValueError:
                logger.warning(f"Unknown JIT mode '{mode}', falling back to AUTO")
                self.mode = JITMode.AUTO
        else:
            self.mode = mode

        self.force_trace = force_trace
        self.custom_cache = custom_cache
        self.recursive = recursive
        self.preserve_stochasticity = preserve_stochasticity


class StrategySelector:
    """Selects optimal JIT strategy for target functions."""

    def __init__(self) -> None:
        """Initialize strategy registry."""
        # Import strategies lazily to avoid circular dependencies
        from ember.xcs.jit.strategies.enhanced import EnhancedStrategy
        from ember.xcs.jit.strategies.structural import StructuralStrategy

        # Map modes to strategy implementations
        self._strategies: Dict[JITMode, Strategy] = {
            JITMode.STRUCTURAL: StructuralStrategy(),
            JITMode.ENHANCED: EnhancedStrategy(),
        }

    def select_strategy(
        self, func: Callable[..., Any], mode: JITMode = JITMode.AUTO
    ) -> Strategy:
        """Select optimal strategy for the target.

        Args:
            func: Target function or class to optimize.
            mode: JIT mode (default: AUTO for automatic selection).

        Returns:
            Most appropriate strategy implementation.
        """
        # Use explicit strategy when specified
        if mode != JITMode.AUTO:
            logger.debug(f"Using explicitly requested {mode.value} strategy")
            return self._strategies[mode]

        # Collect and score strategies for auto-selection
        analyses = [
            (mode, strategy, strategy.analyze(func))
            for mode, strategy in self._strategies.items()
        ]

        # Sort by score in descending order
        analyses.sort(key=lambda x: x[2].get("score", 0), reverse=True)

        # Log detailed selection process for debugging
        func_name = getattr(func, "__name__", str(func))
        logger.debug(f"JIT strategy selection for {func_name}:")
        
        for mode, _, analysis in analyses:
            score_breakdown = analysis.get("score_breakdown", {})
            logger.debug(
                f"  {mode.value}: score={analysis.get('score', 0)}, "
                f"breakdown={score_breakdown}, "
                f"reason={analysis.get('rationale', 'No rationale provided')}"
            )

        # Log final selection
        selected_mode = analyses[0][0]
        logger.debug(f"Selected {selected_mode.value} strategy (highest score)")

        # Return highest-scoring strategy
        return analyses[0][1]


# Global strategy selector
_selector = StrategySelector()


def _jit_function(
    func: Callable[..., Any], strategy: Strategy, settings: JITSettings
) -> Callable[..., Any]:
    """Compiles a regular function using the chosen JIT strategy.

    Args:
        func: Function to compile
        strategy: Selected compilation strategy
        settings: JIT configuration settings

    Returns:
        Compiled function
    """
    return strategy.compile(
        func,
        force_trace=settings.force_trace,
        recursive=settings.recursive,
        cache=settings.custom_cache or get_cache(),
        preserve_stochasticity=settings.preserve_stochasticity)


def _create_operator_forward_proxy(strategy: Strategy, settings: JITSettings):
    """Creates a specialized proxy for operator's forward method.

    Instead of compiling the forward method directly, this creates a proxy function
    that correctly maintains the instance context when called.

    Args:
        strategy: JIT strategy to use
        settings: JIT configuration settings

    Returns:
        Function that creates a callable forward proxy for an operator instance
    """

    def create_forward_proxy(instance, forward_method):
        """Creates a bound method proxy that preserves instance context.

        Args:
            instance: Operator instance
            forward_method: The forward method to proxy

        Returns:
            Callable that acts like the operator's forward method
        """

        # Create a closure to capture the instance and method
        def forward_proxy(*, inputs):
            """Execute the forward method with tracing support.

            This proxy preserves the call interface of the original forward method
            while adding tracing transparency and JIT optimizations.
            """
            # Import here to avoid circular dependencies
            from ember.xcs.tracer.xcs_tracing import TracerContext

            # Check if we're in a tracing context
            tracer = TracerContext.get_current()

            # If we're in a tracing context, track the call
            if tracer and tracer.is_active:
                # Use the instance's name if available, otherwise use class name
                operator_name = getattr(instance, "name", instance.__class__.__name__)
                # Track the call with proper operator identity
                call_id = tracer.track_call(instance, inputs)

                try:
                    # Directly invoke the instance's forward method
                    result = forward_method(instance, inputs=inputs)
                    # Record the successful call completion
                    tracer.complete_call(call_id, result)
                    return result
                except Exception as e:
                    # Record the exception for observability
                    tracer.complete_call(call_id, {}, e)
                    # Re-raise the original exception to maintain behavior
                    raise
            else:
                # No tracing context - direct execution path
                return forward_method(instance, inputs=inputs)

        return forward_proxy

    return create_forward_proxy


def _jit_operator_class(cls: Type, strategy: Strategy, settings: JITSettings) -> Type:
    """Creates a JIT-optimized version of an operator class.

    This function isolates the operator-specific JIT logic, creating a new
    class that inherits from the original but with optimized execution.

    Args:
        cls: Operator class to optimize
        strategy: Selected compilation strategy
        settings: JIT configuration settings

    Returns:
        JIT-optimized operator class
    """
    class_name = cls.__name__ + "_JIT"
    strategy_name = strategy.__class__.__name__.replace("Strategy", "")

    # Verify the class has a forward method
    if not hasattr(cls, "forward"):
        raise ValueError(f"Operator class {cls.__name__} must have a forward method")

    # Get the forward method - we'll use it directly in call
    original_forward = cls.forward

    # Create a forward proxy factory - used to handle binding self properly
    create_proxy = _create_operator_forward_proxy(strategy, settings)

    # We compile the full operation inside the __call__ method, not just forward
    def jit_init(self, *args, **kwargs):
        # Filter out 'inputs' parameter which belongs to __call__, not __init__
        init_kwargs = {k: v for k, v in kwargs.items() if k != "inputs"}
        # Initialize the class normally
        cls.__init__(self, *args, **init_kwargs)
        # Create a proxy and compile it - this happens per instance
        self._forward_proxy = create_proxy(self, original_forward)
        
        # Get cache reference - needed for both compilation and registration
        cache = settings.custom_cache or get_cache()
        
        # Compile the function
        self._compiled_func = strategy.compile(
            self._forward_proxy,
            force_trace=settings.force_trace,
            recursive=settings.recursive,
            cache=cache,
            preserve_stochasticity=settings.preserve_stochasticity)
        
        # Store the strategy name for metrics reporting
        self._jit_strategy = strategy_name
        
        # Register this operator instance with the cache
        # This enables metrics lookups from operator to compiled_func
        cache._operator_registry[id(self)] = id(self._compiled_func)

    def jit_call(self, **kwargs):
        # Get required inputs - everything else is passed to the compiled function as-is
        inputs = kwargs.get("inputs", {})

        # Import here to avoid circular dependencies
        from ember.xcs.tracer.xcs_tracing import TracerContext

        # Get current tracing context to properly propagate tracing through call chain
        tracer = TracerContext.get_current()

        # Get the cache for recording metrics
        from ember.xcs.jit.cache import get_cache
        cache = get_cache()
        func_id = id(self._compiled_func)

        # If we're in a trace context, add operator information before execution
        if tracer and tracer.is_active:
            # Track the operator call in the trace context
            call_id = tracer.track_call(self, inputs)

            try:
                # Execute using compiled function
                result = self._compiled_func(inputs=inputs)

                # Record successful execution in metrics
                cache.metrics.record_cache_hit(func_id)

                # Complete the trace record with successful execution
                tracer.complete_call(call_id, result)
                return result
            except Exception as e:
                # Record the exception but don't swallow it
                tracer.complete_call(call_id, {}, e)
                raise
        else:
            # Not tracing - direct execution path
            # Record cache hit when reusing compiled function
            cache.metrics.record_cache_hit(func_id)
            return self._compiled_func(inputs=inputs)

    # Create the JIT-optimized class
    return type(
        class_name,
        (cls),
        {
            "__init__": jit_init,
            "__call__": jit_call,
            "__doc__": cls.__doc__,
        })


def jit(
    func: Optional[Callable[..., Any]] = None,
    *,
    mode: Union[str, JITMode] = JITMode.AUTO,
    force_strategy: Optional[Union[str, JITMode]] = None,
    force_trace: bool = False,
    cache: Optional[JITCache] = None,
    recursive: bool = True,
    preserve_stochasticity: bool = False) -> Any:
    """Optimizes functions and operators with Just-In-Time compilation.

    Core optimization decorator that analyzes and compiles functions or
    operator classes for efficient execution. Supports multiple compilation
    strategies with automatic selection based on target characteristics.

    Args:
        func: Target function or operator class
        mode: Compilation strategy to use (auto, trace, structural, enhanced)
        force_strategy: Alias for mode - explicitly select a strategy
        force_trace: Whether to force retracing on each call
        cache: Custom cache implementation
        recursive: Whether to recursively optimize nested functions
        preserve_stochasticity: If True, always executes the original function even
            when inputs match previous calls. This is important for LLMs where
            multiple calls with the same prompts should produce different outputs.

    Returns:
        Optimized function or operator class

    Example:
        ```python
        # Simple usage
        @jit
        class MyOperator(Operator):
            def forward(self, *, inputs):
                return process(inputs)

        # Force a specific strategy
        @jit(force_strategy="structural")
        class ComplexOperator(Operator):
            def forward(self, *, inputs):
                return self.complex_logic(inputs)

        # Advanced configuration
        @jit(
            mode="structural",
            recursive=False
        )
        def process_data(*, inputs):
            return {"result": complex_calculation(inputs["data"])}
            
        # LLM usage, preserving stochasticity
        @jit(preserve_stochasticity=True)
        class LLMOperator(Operator):
            def forward(self, *, inputs):
                # Each call will execute fresh, even with identical inputs
                return self.llm.generate(inputs["prompt"])
        ```
    """
    # Support both @jit and @jit() decorator styles
    if func is None:
        return lambda f: jit(
            f,
            mode=mode,
            force_strategy=force_strategy,
            force_trace=force_trace,
            cache=cache,
            recursive=recursive,
            preserve_stochasticity=preserve_stochasticity)

    # Handle force_strategy as an alias for mode
    if force_strategy is not None:
        mode = force_strategy
        logger.debug(f"Using force_strategy={force_strategy} as mode")

    # Prepare optimization configuration
    settings = JITSettings(
        mode=mode,
        force_trace=force_trace,
        custom_cache=cache,
        recursive=recursive,
        preserve_stochasticity=preserve_stochasticity)

    # Get optimal compilation strategy
    strategy = _selector.select_strategy(func, settings.mode)

    # Apply appropriate optimization based on target type
    if inspect.isclass(func) and hasattr(func, "forward"):
        return _jit_operator_class(func, strategy, settings)
    return _jit_function(func, strategy, settings)


def get_jit_stats(func: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Get statistics about JIT compilation and execution.

    Args:
        func: Optional function to get stats for. If None, returns overall stats.
            For JIT-decorated operator classes, automatically retrieves metrics 
            from the internal compiled function.

    Returns:
        Dictionary with compilation and execution statistics
    """
    cache = get_cache()
    
    # Handle JIT-decorated operator class instances
    if func is not None and hasattr(func, '_compiled_func'):
        return cache.get_metrics(func._compiled_func)
        
    return cache.get_metrics(func)


def explain_jit_selection(func: Callable[..., Any]) -> Dict[str, Any]:
    """Explain why a particular JIT strategy would be selected.

    Useful for understanding and debugging the auto-selection process.

    Args:
        func: Function to analyze

    Returns:
        Dictionary with detailed analysis from each strategy
    """
    from ember.xcs.jit.strategies.enhanced import EnhancedStrategy
    from ember.xcs.jit.strategies.structural import StructuralStrategy
    
    strategies = {
        "structural": StructuralStrategy(),
        "enhanced": EnhancedStrategy(),
    }

    results = {}
    for name, strategy in strategies.items():
        results[name] = strategy.analyze(func)

    return results
