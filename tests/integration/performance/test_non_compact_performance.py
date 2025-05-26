"""
Performance tests for Compact Network of Networks (NON) Graph Notation.

This module evaluates the performance characteristics of the compact notation
system compared to standard notation. It measures construction time, memory usage,
and execution efficiency, following the performance-focused approach of
Jeff Dean and Sanjay Ghemawat's system designs.
"""

import gc
import time
from typing import Any, Tuple

import pytest

from ember.core.non import (
    EnsembleInputs,
    JudgeSynthesis,
    MostCommon,
    Sequential,
    UniformEnsemble,
    Verifier)
from ember.core.non_compact import OpRegistry, build_graph


def time_execution(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time the execution of a function.

    Args:
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (function result, execution time in seconds)
    """
    # Force garbage collection to minimize interference
    gc.collect()

    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time

    return result, execution_time


@pytest.mark.skipif(
    not hasattr(pytest, "config") or not pytest.config.getoption("--run-perf-tests", default=False),  # type: ignore
    reason="Performance tests only run with --run-perf-tests flag")
class TestConstructionPerformance:
    """Tests for graph construction performance."""

    def test_construction_time_comparison(self):
        """Compare construction time between compact and standard notation."""
        # Measure time for compact notation
        compact_time_result = time_execution(
            build_graph, ["3:E:gpt-4o:0.7", "1:J:claude-3-5-sonnet:0.0"]
        )
        compact_pipeline, compact_time = compact_time_result

        # Measure time for standard notation
        standard_time_result = time_execution(
            Sequential,
            operators=[
                UniformEnsemble(num_units=3, model_name="gpt-4o", temperature=0.7),
                JudgeSynthesis(model_name="claude-3-5-sonnet", temperature=0.0)])
        standard_pipeline, standard_time = standard_time_result

        # Report the times (no assertion, just for information)
        print("\nConstruction time comparison:")
        print(f"Compact notation: {compact_time:.6f} seconds")
        print(f"Standard notation: {standard_time:.6f} seconds")
        print(f"Ratio (compact/standard): {compact_time/standard_time:.2f}x")

        # Assert both pipelines were constructed successfully
        assert isinstance(compact_pipeline, Sequential)
        assert isinstance(standard_pipeline, Sequential)

    def test_complex_structure_construction_time(self):
        """Measure construction time for complex nested structures."""
        # Define a component map with many references
        component_map = {
            "gpt_ensemble": "3:E:gpt-4o:0.7",
            "claude_ensemble": "3:E:claude-3-5-haiku:0.7",
            "verifier_gpt": "1:V:gpt-4o:0.0",
            "verifier_claude": "1:V:claude-3-5-haiku:0.0",
            "judge": "1:J:claude-3-5-sonnet:0.0",
            "branch1": ["$gpt_ensemble", "$verifier_gpt"],
            "branch2": ["$claude_ensemble", "$verifier_claude"],
            "verification_system": ["$branch1", "$branch2", "$judge"],
        }

        # Measure time for complex structure with references
        _, reference_time = time_execution(
            build_graph, "$verification_system", components=component_map
        )

        # Measure time for equivalent structure without references
        _, direct_time = time_execution(
            build_graph,
            [
                ["3:E:gpt-4o:0.7", "1:V:gpt-4o:0.0"],
                ["3:E:claude-3-5-haiku:0.7", "1:V:claude-3-5-haiku:0.0"],
                "1:J:claude-3-5-sonnet:0.0"])

        # Report the times
        print("\nComplex structure construction time:")
        print(f"With references: {reference_time:.6f} seconds")
        print(f"Direct structure: {direct_time:.6f} seconds")
        print(f"Ratio (reference/direct): {reference_time/direct_time:.2f}x")

        # Verify performance is within reasonable bounds
        # Allow reference resolution to be up to 3x slower due to lookups
        assert (
            reference_time < direct_time * 3.0
        ), "Reference resolution should not be more than 3x slower than direct construction"


@pytest.mark.skipif(
    not hasattr(pytest, "config") or not pytest.config.getoption("--run-perf-tests", default=False),  # type: ignore
    reason="Performance tests only run with --run-perf-tests flag")
class TestExecutionPerformance:
    """Tests for graph execution performance."""

    def test_execution_overhead(self):
        """Test if compact notation adds execution overhead."""
        # Build equivalent pipelines with different notation styles
        compact_pipeline = build_graph(["3:E:gpt-4o:0.7", "1:J:claude-3-5-sonnet:0.0"])

        standard_pipeline = Sequential(
            operators=[
                UniformEnsemble(num_units=3, model_name="gpt-4o", temperature=0.7),
                JudgeSynthesis(model_name="claude-3-5-sonnet", temperature=0.0)]
        )

        # Create test input
        test_input = EnsembleInputs(query="What is the meaning of life?")

        # Measure execution time for compact pipeline
        _, compact_exec_time = time_execution(compact_pipeline, inputs=test_input)

        # Measure execution time for standard pipeline
        _, standard_exec_time = time_execution(standard_pipeline, inputs=test_input)

        # Report the times
        print("\nExecution time comparison:")
        print(f"Compact pipeline: {compact_exec_time:.6f} seconds")
        print(f"Standard pipeline: {standard_exec_time:.6f} seconds")
        print(f"Ratio (compact/standard): {compact_exec_time/standard_exec_time:.2f}x")

        # Verify execution times are similar (within 10%)
        # This should pass because the structure should be identical at runtime
        ratio = compact_exec_time / standard_exec_time
        assert (
            0.9 <= ratio <= 1.1
        ), f"Execution times should be within 10% of each other, but ratio was {ratio:.2f}"

    def test_scaling_with_complexity(self):
        """Test how execution time scales with graph complexity."""

        def create_pipeline(depth):
            """Create a pipeline with the specified depth of ensemble+verifier pairs."""
            operators = []
            for _ in range(depth):
                operators.append("2:E:gpt-4o:0.0")
                operators.append("1:V:gpt-4o:0.0")
            operators.append("1:J:claude-3-5-sonnet:0.0")

            return build_graph(operators)

        depths = [1, 2, 3]  # Number of ensemble+verifier pairs
        execution_times = []

        # Test input
        test_input = EnsembleInputs(query="What is quantum computing?")

        # Measure execution time for different depths
        for depth in depths:
            pipeline = create_pipeline(depth)
            _, exec_time = time_execution(pipeline, inputs=test_input)
            execution_times.append(exec_time)

            print(f"\nDepth {depth} execution time: {exec_time:.6f} seconds")

        # Calculate scaling factor (average ratio between consecutive depths)
        scaling_factors = [
            execution_times[i] / execution_times[i - 1]
            for i in range(1, len(execution_times))
        ]
        avg_scaling = sum(scaling_factors) / len(scaling_factors)

        print(f"Average scaling factor: {avg_scaling:.2f}x per depth increase")

        # Verify scaling is roughly linear (may vary with mock implementations)
        assert (
            0.8 <= avg_scaling <= 2.0
        ), f"Execution time scaling should be roughly linear, but was {avg_scaling:.2f}x"


@pytest.mark.skipif(
    not hasattr(pytest, "config") or not pytest.config.getoption("--run-perf-tests", default=False),  # type: ignore
    reason="Performance tests only run with --run-perf-tests flag")
class TestCustomRegistryPerformance:
    """Tests for custom registry performance."""

    def test_registry_overhead(self):
        """Test if custom registries add significant overhead."""
        # Create standard registry
        standard_registry = OpRegistry.create_standard_registry()

        # Create custom registry with additional operators
        custom_registry = OpRegistry.create_standard_registry()
        custom_registry.register(
            "CE",  # Custom Ensemble
            lambda count, model, temp: Sequential(
                operators=[
                    UniformEnsemble(
                        num_units=count, model_name=model, temperature=temp
                    ),
                    MostCommon()]
            ))
        custom_registry.register(
            "CVE",  # Custom Verified Ensemble
            lambda count, model, temp: Sequential(
                operators=[
                    UniformEnsemble(
                        num_units=count, model_name=model, temperature=temp
                    ),
                    MostCommon(),
                    Verifier(model_name=model, temperature=temp)]
            ))

        # Test spec
        test_spec = "3:E:gpt-4o:0.7"

        # Measure parse time with standard registry
        _, standard_time = time_execution(
            build_graph, test_spec, type_registry=standard_registry
        )

        # Measure parse time with custom registry
        _, custom_time = time_execution(
            build_graph, test_spec, type_registry=custom_registry
        )

        # Report the times
        print("\nRegistry overhead comparison:")
        print(f"Standard registry: {standard_time:.6f} seconds")
        print(f"Custom registry: {custom_time:.6f} seconds")
        print(f"Ratio (custom/standard): {custom_time/standard_time:.2f}x")

        # Verify custom registry doesn't add excessive overhead
        assert (
            custom_time < standard_time * 1.5
        ), "Custom registry should not add more than 50% overhead"


if __name__ == "__main__":
    # To run performance tests: pytest --run-perf-tests -v test_non_compact_performance.py
    pytest.main(["--run-perf-tests", "-v"])
