"""Structure-based JIT compilation strategy.

Implements JIT compilation by analyzing the structure of operators directly
without executing them. This strategy is particularly effective for container
operators with nested sub-operators, as it can detect parallelization opportunities
based on the operator structure.
"""

import functools
import inspect
import logging
import time
from typing import Any, Callable, Dict, Optional

from ember.xcs.graph.graph_builder import StructuralGraphBuilder
from ember.xcs.jit.cache import JITCache
from ember.xcs.jit.strategies.base_strategy import BaseStrategy, JITFallbackMixin

logger = logging.getLogger(__name__)


class StructuralStrategy(BaseStrategy, JITFallbackMixin):
    """Structure-based JIT compilation strategy.

    Compiles operators by analyzing their structure directly, without
    execution tracing. This approach is particularly effective for
    container operators with nested sub-operators.
    """

    def __init__(self) -> None:
        """Initialize the structural strategy."""
        self.graph_builder = StructuralGraphBuilder()

    def analyze(self, func: Callable[..., Any]) -> Dict[str, Any]:
        """Analyze a function to determine if structural JIT is appropriate.

        Scoring rationale:
        - 40 points: Has nested operators (strongest indicator of structural pattern)
        - 30 points: Has forward() method (indicates operator pattern)
        - 20 points: Has __call__ method (callable class)
        - 10 points: Has specification attribute (operator convention)
        
        Total max score: 100 (perfect operator with composition)
        Typical operator: 60-70 (has forward + specification)
        Typical container: 90-100 (has nested operators)

        Args:
            func: Function to analyze

        Returns:
            Dictionary with analysis results
        """
        features = self._extract_common_features(func)
        score = 0
        rationale = []
        score_breakdown = {}

        # Check if it's an operator class with forward method
        if features["is_class"]:
            # Check for operator characteristics
            if hasattr(func, "forward") and callable(getattr(func, "forward", None)):
                score += 30  # Strong indicator of operator pattern
                score_breakdown["forward_method"] = 30
                rationale.append("Has 'forward' method (likely an operator)")

            # Check for nested operators - most important feature
            has_operator_fields = False
            nested_count = 0
            for attr_name in dir(func):
                if attr_name.startswith("_"):
                    continue

                attr = getattr(func, attr_name, None)
                if inspect.isclass(attr) and hasattr(attr, "forward"):
                    has_operator_fields = True
                    nested_count += 1

            if has_operator_fields:
                score += 40  # Highest score - best indicator of structural pattern
                score_breakdown["nested_operators"] = 40
                rationale.append(f"Has {nested_count} nested operator fields (container pattern)")

        # Check for specific method signatures that indicate operator composition
        if callable(func) and callable(func.__call__):
            score += 20  # Moderate indicator
            score_breakdown["callable_class"] = 20
            rationale.append("Has __call__ method")

        # Check for known operator-specific attributes
        if hasattr(func, "specification"):
            score += 10  # Weak indicator alone, but confirms operator pattern
            score_breakdown["specification"] = 10
            rationale.append("Has 'specification' attribute (operator pattern)")

        # Log score breakdown for debugging
        logger.debug(f"Structural analysis for {func.__name__ if hasattr(func, '__name__') else func}: "
                    f"total_score={score}, breakdown={score_breakdown}")

        return {
            "score": score,
            "rationale": "; ".join(rationale),
            "features": features,
            "score_breakdown": score_breakdown,  # For debugging
        }

    def compile(
        self,
        func: Callable[..., Any],
        force_trace: bool = False,
        recursive: bool = True,
        cache: Optional[JITCache] = None,
        preserve_stochasticity: bool = False) -> Callable[..., Any]:
        """Compile a function using structural JIT.

        Args:
            func: Function to compile
            force_trace: Whether to force analysis on every call
            recursive: Whether to recursively analyze nested operators
            cache: JIT cache to use
            preserve_stochasticity: When True, always executes the original function
                to maintain stochastic behavior (important for LLMs)

        Returns:
            Compiled function
        """
        # Use provided cache or get the default
        cache = self._get_cache(cache)

        @functools.wraps(func)
        def structural_function(*, inputs: Dict[str, Any]) -> Any:
            # Check if JIT is disabled, forcing trace, or preserving stochasticity
            if (getattr(structural_function, "_jit_disabled", False) or 
                    force_trace or preserve_stochasticity):
                # Execute directly
                # For LLMs, this ensures we get stochastic outputs on each call
                execution_start = time.time()
                result = func(inputs=inputs)
                execution_duration = time.time() - execution_start
                cache.metrics.record_execution(execution_duration)
                return result

            # Try to get structure signature
            state_signature = None
            if hasattr(func, "get_structure_signature") and callable(
                func.get_structure_signature
            ):
                state_signature = func.get_structure_signature()

            # Try to get cached graph
            graph = cache.get_with_state(func, state_signature)

            if graph is not None:
                # Execute cached graph
                try:
                    from ember.xcs.jit.execution_utils import execute_compiled_graph

                    return execute_compiled_graph(graph, inputs, cache, func=func)
                except Exception as e:
                    logger.warning(
                        f"Error executing graph: {e}. Falling back to direct execution."
                    )

            # Analyze structure and build graph
            compilation_start = time.time()

            # Set recursion flag if available on the graph builder
            if hasattr(self.graph_builder, "set_recursive"):
                self.graph_builder.set_recursive(recursive)

            graph = self.graph_builder.build_graph(func)
            compilation_duration = time.time() - compilation_start
            cache.metrics.record_compilation(compilation_duration)

            # Cache compiled graph
            state_signature = None
            if hasattr(func, "get_structure_signature") and callable(
                func.get_structure_signature
            ):
                state_signature = func.get_structure_signature()

            cache.set_with_state(func, graph, state_signature)

            # Execute and return with fallback
            return self.execute_with_fallback(graph, func, inputs, cache)

        # Add control methods
        self._add_control_methods(structural_function, func, cache)

        return structural_function

    def execute_with_fallback(
        self,
        graph: Any,
        original_func: Callable,
        inputs: Dict[str, Any],
        cache: JITCache) -> Any:
        """Execute a compiled graph with fallback to direct execution.

        Args:
            graph: Compiled graph
            original_func: Original function
            inputs: Input values
            cache: JIT cache

        Returns:
            Execution result
        """
        try:
            # Try to execute the compiled graph
            from ember.xcs.jit.execution_utils import execute_compiled_graph

            return execute_compiled_graph(graph, inputs, cache, func=original_func)
        except Exception as e:
            # Log error and fall back to direct execution
            logger.warning(
                f"Error executing compiled graph: {e}. Falling back to direct execution."
            )

            # Execute directly and time it
            execution_start = time.time()
            result = original_func(inputs=inputs)
            execution_duration = time.time() - execution_start
            cache.metrics.record_execution(execution_duration)

            return result
