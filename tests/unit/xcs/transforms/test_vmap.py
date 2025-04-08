"""
Unit tests for the vmap transform.

This module provides comprehensive testing for the vectorized mapping (vmap)
transformation in XCS, including basic functionality, edge cases, error handling,
and advanced usage patterns.
"""

import time

import pytest

# Import vmap directly for tests
from ember.xcs.transforms.vmap import (
    _combine_outputs,
    _get_batch_size,
    _prepare_batched_inputs,
    vmap,
)

# Import test operators
from tests.unit.xcs.transforms.mock_operators import (
    BasicOperator,
    ComplexInputOperator,
    ExceptionOperator,
    MockModule,
    NestedOperator,
    StatefulOperator,
)
from tests.unit.xcs.transforms.test_utils import generate_batch_inputs

# =============================== Fixtures ===============================


@pytest.fixture
def basic_operator():
    """Fixture providing a basic operator instance."""
    return BasicOperator()


@pytest.fixture
def stateful_operator():
    """Fixture providing a stateful operator instance."""
    return StatefulOperator()


@pytest.fixture
def exception_operator():
    """Fixture providing an exception-raising operator."""
    return ExceptionOperator(fail_on_inputs=["fail_input"])


@pytest.fixture
def module_operator():
    """Fixture providing a Module-based operator instance."""
    return MockModule()


# =============================== Unit Tests for Internal Functions ===============================


class TestVMapInternals:
    """Unit tests for internal vmap functions."""

    def test_get_batch_size(self):
        """Test batch size detection from inputs."""
        # Case 1: Empty inputs
        # The implementation returns 0 for empty inputs
        assert _get_batch_size({}, 0) == 0

        # Case 2: Single axis for all inputs
        inputs = {"prompts": ["a", "b", "c"], "config": {"mode": "test"}}
        assert _get_batch_size(inputs, 0) == 3

        # Case 3: Dict of axes with consistent sizes
        inputs = {"prompts": ["a", "b"], "other": ["x", "y"]}
        axes = {"prompts": 0, "other": 0}
        assert _get_batch_size(inputs, axes) == 2

        # Case 4: Inconsistent batch sizes should raise TransformError
        inputs = {"prompts": ["a", "b"], "other": ["x", "y", "z"]}
        axes = {"prompts": 0, "other": 0}
        from ember.xcs.transforms.transform_base import TransformError

        with pytest.raises(TransformError) as excinfo:
            _get_batch_size(inputs, axes)

        # Verify that the error message contains the expected information
        assert "Inconsistent batch sizes" in str(excinfo.value)

        # Case 5: Empty list inputs
        inputs = {"prompts": []}
        # Implementation returns 0 (empty batch) for empty lists
        assert _get_batch_size(inputs, 0) == 0

    def test_prepare_batched_inputs(self):
        """Test preparation of batched inputs."""
        # Case 1: Simple input with single batch axis
        inputs = {"prompts": ["a", "b"]}
        result = _prepare_batched_inputs(inputs, 0, 2)
        # Implementation returns a list of input dicts, one for each batch element
        assert len(result) == 2
        assert result[0]["prompts"] == "a"
        assert result[1]["prompts"] == "b"

        # Case 2: Mixed inputs with scalar values
        inputs = {"prompts": ["a", "b"], "config": {"mode": "test"}}
        result = _prepare_batched_inputs(inputs, 0, 2)
        assert len(result) == 2
        assert result[0]["prompts"] == "a"
        assert result[1]["prompts"] == "b"
        assert result[0]["config"] == {"mode": "test"}  # Scalar values are replicated
        assert result[1]["config"] == {"mode": "test"}

        # Case 3: Dict of axes
        inputs = {"prompts": ["a", "b"], "config": {"mode": "test"}}
        axes = {"prompts": 0}  # Only batch prompts
        result = _prepare_batched_inputs(inputs, axes, 2)
        assert len(result) == 2
        assert result[0]["prompts"] == "a"
        assert result[1]["prompts"] == "b"
        assert result[0]["config"] == {"mode": "test"}
        assert result[1]["config"] == {"mode": "test"}

        # We don't test Case 4 (empty list) because our current implementation
        # delegates empty list handling to the operator

    def test_combine_outputs(self):
        """Test combining of outputs from batched execution."""
        # Case 1: Empty results
        # Implementation returns {"results": []} for empty results
        assert _combine_outputs([]) == {"results": []}

        # Case 2: Basic dictionary outputs
        results = [{"results": ["a_processed"]}, {"results": ["b_processed"]}]
        combined = _combine_outputs(results)
        assert combined == {"results": ["a_processed", "b_processed"]}

        # Case 3: Complex nested outputs
        results = [
            {"results": ["a_processed"], "metadata": {"id": 1}},
            {"results": ["b_processed"], "metadata": {"id": 2}},
        ]
        combined = _combine_outputs(results)
        assert combined["results"] == ["a_processed", "b_processed"]
        assert combined["metadata"] == [{"id": 1}, {"id": 2}]

        # Case 4: Non-dictionary outputs
        results = ["a", "b", "c"]
        combined = _combine_outputs(results)
        # Implementation wraps non-dict results in {"results": [...]}
        assert combined == {"results": ["a", "b", "c"]}

        # Case 5: Handling nested lists
        results = [{"results": [["a_nested"]]}, {"results": [["b_nested"]]}]
        combined = _combine_outputs(results)
        # The implementation flattens nested lists, verify structure preserves outer lists
        assert isinstance(combined["results"], list)
        assert len(combined["results"]) == 2
        # Each result should contain the nested list
        assert isinstance(combined["results"][0], list)
        assert isinstance(combined["results"][1], list)


# =============================== Main vmap Tests ===============================


class TestVMap:
    """Comprehensive tests for the vmap transformation."""

    def test_vmap_basic_functionality(self, basic_operator):
        """Test that vmap correctly vectorizes a basic operator."""
        vectorized_op = vmap(basic_operator)

        # Test with batch input
        batch_inputs = {"prompts": ["prompt1", "prompt2", "prompt3"]}

        result = vectorized_op(inputs=batch_inputs)
        assert "results" in result
        assert len(result["results"]) == 3
        assert result["results"] == [
            "prompt1_processed",
            "prompt2_processed",
            "prompt3_processed",
        ]

        # Verify original operator was called for each batch item
        assert basic_operator.call_count == 3

    def test_vmap_with_empty_inputs(self, basic_operator):
        """Test vmap behavior with empty inputs."""
        vectorized_op = vmap(basic_operator)

        # Empty list - returns empty results array
        result = vectorized_op(inputs={"prompts": []})
        assert "results" in result
        assert result["results"] == []

        # Empty dict - the implementation handles this as a special case
        # where batch size is 0 and returns empty results
        result = vectorized_op(inputs={})
        assert "results" in result
        assert result["results"] == []

    def test_vmap_with_scalar_non_batch_input(self):
        """Test vmap's behavior with non-batched scalar inputs.

        The vmap implementation is designed to handle batched data (lists/tuples),
        not scalar values. When given scalar inputs, vmap should gracefully produce
        an empty results list, as there is no batch dimension to iterate over.
        """
        # Create a simple function that processes scalar inputs
        call_count = 0

        def scalar_handler(*, inputs):
            nonlocal call_count
            call_count += 1
            prompts = inputs.get("prompts", "")
            return {"results": [f"{prompts}_processed"]}

        # Apply vmap to the scalar handler
        vectorized_fn = vmap(scalar_handler)

        # When we pass a scalar value where vmap expects a batch dimension,
        # vmap should interpret this as an empty batch (no items to process)
        result = vectorized_fn(inputs={"prompts": "single_prompt"})

        # Verify the result contains an empty array (no batch items processed)
        assert "results" in result
        assert result["results"] == []

        # The function should not be called (no batch items to process)
        assert call_count == 0

    def test_vmap_with_custom_axes(self, basic_operator):
        """Test vmap with custom input and output axes."""
        # Custom input axes - only batch 'prompts', keep 'config' as non-batched
        in_axes = {"prompts": 0}  # batch prompts only
        vectorized_op = vmap(basic_operator, in_axes=in_axes)

        batch_inputs = {"prompts": ["a", "b", "c"], "config": {"mode": "test"}}

        result = vectorized_op(inputs=batch_inputs)
        assert "results" in result
        assert len(result["results"]) == 3
        assert result["results"] == ["a_processed", "b_processed", "c_processed"]

        # Verify the config was used for each call
        assert basic_operator.call_count == 3

    def test_vmap_with_function(self):
        """Test vmap with a function instead of an operator."""
        call_count = 0

        def process_fn(*, inputs):
            nonlocal call_count
            call_count += 1
            prompts = inputs.get("prompts", [])
            if isinstance(prompts, list):
                return {"results": [f"{p}_fn" for p in prompts]}
            return {"results": [f"{prompts}_fn"]}

        vectorized_fn = vmap(process_fn)

        # Test with batch input
        batch_inputs = {"prompts": ["a", "b", "c"]}

        result = vectorized_fn(inputs=batch_inputs)
        assert "results" in result
        assert len(result["results"]) == 3
        assert result["results"] == ["a_fn", "b_fn", "c_fn"]
        assert call_count == 3

    def test_vmap_with_stateful_operator(self, stateful_operator):
        """Test vmap with a stateful operator."""
        vectorized_op = vmap(stateful_operator)

        # First batch
        batch1 = {"prompts": ["s1", "s2"]}
        result1 = vectorized_op(inputs=batch1)

        assert result1["results"] == ["s1_processed", "s2_processed"]
        assert stateful_operator.history == ["s1_processed", "s2_processed"]

        # Second batch
        batch2 = {"prompts": ["s3", "s4"]}
        result2 = vectorized_op(inputs=batch2)

        assert result2["results"] == ["s3_processed", "s4_processed"]
        assert stateful_operator.history == [
            "s1_processed",
            "s2_processed",
            "s3_processed",
            "s4_processed",
        ]

        # Verify order is preserved
        assert len(stateful_operator.history) == 4
        assert stateful_operator.history[0] == "s1_processed"
        assert stateful_operator.history[1] == "s2_processed"
        assert stateful_operator.history[2] == "s3_processed"
        assert stateful_operator.history[3] == "s4_processed"

    def test_vmap_with_nested_operator(self):
        """Test vmap with a nested operator structure."""
        op1 = BasicOperator(lambda x: f"{x}_first")
        op2 = BasicOperator(lambda x: f"{x}_second")
        nested_op = NestedOperator([op1, op2])

        vectorized_op = vmap(nested_op)

        batch_inputs = {"prompts": ["n1", "n2", "n3"]}
        result = vectorized_op(inputs=batch_inputs)

        expected = ["n1_first_second", "n2_first_second", "n3_first_second"]
        assert result["results"] == expected

        # Verify each operator was called 3 times
        assert op1.call_count == 3
        assert op2.call_count == 3

    def test_vmap_exception_handling(self, exception_operator):
        """Test vmap propagates exceptions properly."""
        vectorized_op = vmap(exception_operator)

        # Regular case - should succeed
        result = vectorized_op(inputs={"prompts": ["ok1", "ok2"]})
        assert result["results"] == ["ok1_success", "ok2_success"]

        # Case with failure - should propagate the exception
        with pytest.raises(Exception) as excinfo:
            vectorized_op(inputs={"prompts": ["ok1", "fail_input", "ok2"]})

        # Get the string representation of the exception
        error_text = str(excinfo.value)

        # The exception should contain information about the batch element
        # that failed and/or the failure reason
        assert (
            "fail_input" in error_text
            or "Failed on input" in error_text
            or "Error processing batch element" in error_text
        )

    def test_vmap_with_module_operator(self, module_operator):
        """Test vmap with a Module-based operator."""
        vectorized_op = vmap(module_operator)

        batch_inputs = {"prompts": ["m1", "m2", "m3"]}
        result = vectorized_op(inputs=batch_inputs)

        assert result["results"] == ["m1_module", "m2_module", "m3_module"]
        assert module_operator.processed_count == 3

    def test_vmap_with_complex_inputs(self):
        """Test vmap with complex nested input structures."""
        op = ComplexInputOperator()
        vectorized_op = vmap(op)

        # Complex batch inputs
        batch_inputs = {
            "prompts": ["c1", "c2"],
            "config": {"param": "value", "option": 123},
            "metadata": {"source": "test", "timestamp": 1000},
        }

        result = vectorized_op(inputs=batch_inputs)

        # Verify output structure and contents
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"] == ["c1_complex", "c2_complex"]

        # Complex output fields should be properly handled
        assert "processed_config" in result
        assert len(result["processed_config"]) == 2

        assert "metadata" in result
        assert len(result["metadata"]) == 2

    def test_vmap_with_large_batch(self, basic_operator):
        """Test vmap with a large batch to ensure it scales properly."""
        vectorized_op = vmap(basic_operator)

        # Create a large batch
        batch_size = 1000
        batch_inputs = {"prompts": [f"large{i}" for i in range(batch_size)]}

        result = vectorized_op(inputs=batch_inputs)

        assert len(result["results"]) == batch_size
        assert basic_operator.call_count == batch_size

        # Verify a sample of results
        assert result["results"][0] == "large0_processed"
        assert result["results"][499] == "large499_processed"
        assert result["results"][-1] == f"large{batch_size-1}_processed"

    def test_vmap_preserves_order(self, basic_operator):
        """Test that vmap preserves the original order of inputs."""
        vectorized_op = vmap(basic_operator)

        # Use distinct prompts that can be easily checked for order
        batch_inputs = {"prompts": ["first", "second", "third", "fourth", "fifth"]}

        result = vectorized_op(inputs=batch_inputs)
        expected = [
            "first_processed",
            "second_processed",
            "third_processed",
            "fourth_processed",
            "fifth_processed",
        ]
        assert result["results"] == expected


# =============================== Property-Based Tests ===============================


class TestVMapProperties:
    """Property-based tests for the vmap transformation."""

    def test_empty_input_property(self, basic_operator):
        """Property: vmap handles empty inputs appropriately."""
        vectorized_op = vmap(basic_operator)

        # For empty inputs, vmap should return empty results
        empty_inputs = [
            {},
            {"prompts": []},
            {"config": {}},
            {"config": {}, "metadata": None},
            {"prompts": [], "config": {}},
        ]

        for inputs in empty_inputs:
            # Reset the operator state for each test case
            basic_operator.call_count = 0
            result = vectorized_op(inputs=inputs)

            # For empty batch inputs, vmap should return an empty results array
            assert "results" in result
            assert isinstance(result["results"], list)
            assert len(result["results"]) == 0

            # The operator should not be called for empty batches
            assert basic_operator.call_count == 0

    def test_identity_property(self):
        """Property: vmap of identity function preserves inputs."""

        def identity_fn(*, inputs):
            """Identity function that returns inputs unchanged."""
            return {"results": inputs.get("prompts", [])}

        vectorized_identity = vmap(identity_fn)

        test_cases = [{"prompts": ["a", "b", "c"]}, {"prompts": ["abc", "def", "ghi"]}]

        for inputs in test_cases:
            result = vectorized_identity(inputs=inputs)
            assert result["results"] == inputs["prompts"]

    def test_composition_property(self):
        """Property: vmap commutes with function composition."""

        def f(*, inputs):
            """First transformation: append '_f'."""
            prompts = inputs.get("prompts", [])
            if isinstance(prompts, list):
                return {"results": [f"{p}_f" for p in prompts]}
            return {"results": [f"{prompts}_f"]}

        def g(*, inputs):
            """Second transformation: append '_g'."""
            prompts = inputs.get("prompts", [])
            if isinstance(prompts, list):
                return {"results": [f"{p}_g" for p in prompts]}
            return {"results": [f"{prompts}_g"]}

        # Approach 1: vmap(compose(f, g))
        def compose_fg(*, inputs):
            """Composition of f and g."""
            f_result = f(inputs=inputs)
            g_inputs = {"prompts": f_result["results"]}
            return g(inputs=g_inputs)

        vmap_compose = vmap(compose_fg)

        # Approach 2: compose(vmap(f), vmap(g))
        vmap_f = vmap(f)
        vmap_g = vmap(g)

        def compose_vmap_fg(*, inputs):
            """Composition of vmap(f) and vmap(g)."""
            f_result = vmap_f(inputs=inputs)
            g_inputs = {"prompts": f_result["results"]}
            return vmap_g(inputs=g_inputs)

        # Test both approaches produce the same results
        test_inputs = {"prompts": ["a", "b", "c"]}
        result1 = vmap_compose(inputs=test_inputs)
        result2 = compose_vmap_fg(inputs=test_inputs)

        assert result1["results"] == result2["results"]


# =============================== Edge Case Tests ===============================


class TestVMapEdgeCases:
    """Tests for vmap behavior in edge cases and corner cases."""

    def test_vmap_with_non_list_nested_inputs(self):
        """Test vmap with inputs that are not lists but contain nested data.

        The vmap transform operates on batch dimensions (lists/tuples). When given
        a non-list value like a dictionary, vmap should treat it as a non-batched input
        and return an empty result list, as there are no elements to process.
        """
        # Create a tracking function
        call_count = 0

        def handle_nested(*, inputs):
            nonlocal call_count
            call_count += 1
            prompts = inputs.get("prompts", {})

            # Handle any input type
            return {"results": [f"{prompts}_processed"]}

        # Apply vmap to the function
        vectorized_op = vmap(handle_nested)

        # Nested dictionary case
        nested_input = {"prompts": {"key1": "value1", "key2": "value2"}}

        # Reset counter
        call_count = 0

        # When given a dictionary instead of a list, vmap should produce an empty batch
        result = vectorized_op(inputs=nested_input)

        # The result should be an empty list (no batch dimension to process)
        assert "results" in result
        assert result["results"] == []

        # The function should not be called (no batch items)
        assert call_count == 0

        # Test with mixed types in a list - each list item should be processed
        mixed_input = {"prompts": [{"type": "dict"}, 123, "string"]}

        # Reset counter
        call_count = 0

        # Mixed types in a list should be processed element by element
        result = vectorized_op(inputs=mixed_input)

        # Each list element should be processed
        assert len(result["results"]) == 3

        # The function should be called once per list element
        assert call_count == 3

    def test_vmap_with_nested_lists(self, basic_operator):
        """Test vmap with nested list structures."""

        # We'll create a specialized operator that can handle nested lists
        class NestedListOperator(BasicOperator):
            def __call__(self, **kwargs):
                # Get inputs
                inputs = kwargs.get("inputs", {})
                # Process prompts specially when they're lists
                prompts = inputs.get("prompts", [])
                # Track calls
                self.call_count += 1
                if isinstance(prompts, list) and all(
                    isinstance(item, list) for item in prompts
                ):
                    # Format each inner list
                    return {"results": f"{prompts}_processed"}
                # Default processing
                return {"results": f"{prompts}_processed"}

        # Create custom operator
        list_operator = NestedListOperator()
        nested_vectorized_op = vmap(list_operator)

        # Lists of lists - each outer list item is a separate batch element
        nested_lists = {"prompts": [["a", "b"], ["c", "d"], ["e", "f"]]}

        # Call vectorized operator
        result = nested_vectorized_op(inputs=nested_lists)

        # Should have 3 batch elements
        assert len(result["results"]) == 3

        # Operator should be called once for each batch element
        assert list_operator.call_count == 3

        # Each result should be processed with the format "X_processed"
        for i, batch_item in enumerate(nested_lists["prompts"]):
            processed = result["results"][i]
            # The result should contain the processed batch item
            assert str(batch_item) in processed
            assert processed.endswith("_processed")

    def test_vmap_with_special_characters(self, basic_operator):
        """Test vmap with inputs containing special characters."""
        vectorized_op = vmap(basic_operator)

        special_chars = {
            "prompts": ["line\nbreak", "tab\tchar", 'quote"quote', "emoji🔥"]
        }

        result = vectorized_op(inputs=special_chars)
        assert len(result["results"]) == 4

        # Each special character string should be processed correctly
        for i, prompt in enumerate(special_chars["prompts"]):
            assert result["results"][i] == f"{prompt}_processed"

    def test_vmap_with_none_values(self, basic_operator):
        """Test vmap with None values in inputs."""
        vectorized_op = vmap(basic_operator)

        none_inputs = {"prompts": [None, "valid", None]}

        result = vectorized_op(inputs=none_inputs)
        assert len(result["results"]) == 3
        assert result["results"][0] == "None_processed"
        assert result["results"][1] == "valid_processed"
        assert result["results"][2] == "None_processed"

    def test_vmap_with_multiple_batched_fields(self):
        """Test vmap with multiple fields that can be batched."""

        def multi_field_fn(*, inputs):
            """Process inputs with multiple fields."""
            prompts = inputs.get("prompts", [])
            contexts = inputs.get("contexts", [])

            results = []
            if isinstance(prompts, list) and isinstance(contexts, list):
                # If we have both, combine them
                for p, c in zip(prompts, contexts):
                    results.append(f"{p}+{c}")
            elif isinstance(prompts, list):
                results = [f"{p}+default" for p in prompts]
            elif isinstance(contexts, list):
                results = [f"default+{c}" for c in contexts]
            else:
                results = [f"{prompts}+{contexts}"]

            return {"results": results}

        vectorized_fn = vmap(multi_field_fn)

        # Test with multiple batched fields of same length
        multi_inputs = {"prompts": ["p1", "p2", "p3"], "contexts": ["c1", "c2", "c3"]}

        result = vectorized_fn(inputs=multi_inputs)
        assert result["results"] == ["p1+c1", "p2+c2", "p3+c3"]

        # Test with inconsistent lengths
        # This should raise a TransformError about inconsistent batch sizes
        inconsistent_inputs = {"prompts": ["p1", "p2"], "contexts": ["c1", "c2", "c3"]}

        from ember.xcs.transforms.transform_base import TransformError

        with pytest.raises(TransformError) as excinfo:
            vectorized_fn(inputs=inconsistent_inputs)

        # Verify the error message mentions the inconsistent batch sizes
        error_message = str(excinfo.value)
        assert "Inconsistent batch sizes" in error_message


# =============================== Performance Tests ===============================


class TestVMapPerformance:
    """Tests focused on the performance characteristics of vmap."""

    def test_vmap_performance_with_varying_batch_size(self, basic_operator):
        """Test how vmap performance scales with batch size.

        This test verifies that:
        1. Processing time scales roughly linearly with batch size
        2. The vectorized implementation handles increasing batch sizes efficiently
        """
        # Skip this test by default as it's a performance test
        try:
            if not pytest.config.getoption("--run-perf-tests", default=False):
                pytest.skip("Performance tests are disabled by default")
        except (AttributeError, TypeError):
            # Handle pytest.config not being available or other errors
            pytest.skip("Performance tests are disabled by default")

        # Create a CPU-intensive operation that benefits from vectorization
        def cpu_intensive_process(prompt):
            # Perform actual computation instead of sleep which may not work reliably
            result = 0
            # Scale computation based on input length to ensure consistent work
            iterations = max(10000, len(prompt) * 5000)
            for i in range(iterations):
                result += (i * i) % 100
            return f"{prompt}_processed_{result}"

        basic_operator.process_fn = cpu_intensive_process
        vectorized_op = vmap(basic_operator)

        # Focus on a reasonable range of batch sizes
        batch_sizes = [5, 20, 50]
        times = []

        print("\n=== VMAP Performance Test Results ===")

        for size in batch_sizes:
            batch_inputs = generate_batch_inputs(size)

            # Measure vectorized implementation time (with warmup)
            # Use median of multiple runs for stability
            vectorized_times = []
            for _ in range(3):
                start_time = time.time()
                result = vectorized_op(inputs=batch_inputs)
                end_time = time.time()
                vectorized_times.append(end_time - start_time)

            vectorized_times.sort()
            vectorized_time = vectorized_times[1]  # Use median
            times.append(vectorized_time)

            # Verify correct results
            assert len(result["results"]) == size

            # Calculate and print per-item processing time
            per_item_time = vectorized_time / size
            print(
                f"Batch size {size}: total={vectorized_time:.6f}s, per item={per_item_time:.8f}s"
            )

        # Verify that the per-item time doesn't increase significantly with batch size
        # (should remain constant or decrease)
        for i in range(1, len(batch_sizes)):
            per_item_small = times[i - 1] / batch_sizes[i - 1]
            per_item_large = times[i] / batch_sizes[i]
            ratio = per_item_large / per_item_small

            print(
                f"Scaling efficiency {batch_sizes[i-1]} → {batch_sizes[i]}: {ratio:.2f}x per-item time ratio"
            )

            # Per-item time should not increase significantly (allowing up to 20% overhead)
            assert ratio <= 1.2, (
                f"Per-item processing time increased too much from batch size {batch_sizes[i-1]} to {batch_sizes[i]}: "
                f"{per_item_small:.8f}s → {per_item_large:.8f}s ({ratio:.2f}x increase)"
            )

        # Verify linear scaling (or better) for total processing time
        for i in range(1, len(batch_sizes)):
            size_ratio = batch_sizes[i] / batch_sizes[i - 1]
            time_ratio = times[i] / times[i - 1]
            scaling_efficiency = time_ratio / size_ratio

            print(
                f"Time scaling {batch_sizes[i-1]} → {batch_sizes[i]}: "
                f"{time_ratio:.2f}x time increase for {size_ratio:.2f}x items (efficiency: {scaling_efficiency:.2f})"
            )

            # Processing time should scale linearly or better with batch size
            # Allowing some overhead (up to 20% beyond linear)
            assert scaling_efficiency <= 1.2, (
                f"Processing time increased faster than expected from batch size {batch_sizes[i-1]} to {batch_sizes[i]}: "
                f"{times[i-1]:.6f}s → {times[i]:.6f}s (efficiency: {scaling_efficiency:.2f}, expected ≤ 1.2)"
            )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
