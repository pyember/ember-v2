"""Test the Operator base class.

Following CLAUDE.md principles:
- Operators require forward() method
- Optional validation with input_spec/output_spec
- Clean JAX integration via Module
"""

import pytest

from ember._internal.types import EmberModel
from ember.operators import Operator


class TestOperatorBase:
    """Test the Operator base class behavior."""

    def test_operator_requires_forward(self):
        """Test that Operator requires forward() implementation."""
        # Base operator without forward() should raise
        op = Operator()

        with pytest.raises(NotImplementedError) as exc_info:
            op("test")

        assert "must implement forward()" in str(exc_info.value)

    def test_simple_operator(self):
        """Test creating a simple operator subclass."""

        class DoubleOperator(Operator):
            def forward(self, x):
                return x * 2

        op = DoubleOperator()
        result = op(5)
        assert result == 10

    def test_operator_with_validation(self):
        """Test operator with input/output validation."""

        # Define validation schemas
        class InputSpec(EmberModel):
            value: int
            multiplier: int = 2

        class OutputSpec(EmberModel):
            result: int

        class ValidatedOperator(Operator):
            input_spec = InputSpec
            output_spec = OutputSpec

            def forward(self, input: InputSpec) -> OutputSpec:
                result = input.value * input.multiplier
                return OutputSpec(result=result)

        op = ValidatedOperator()

        # Dict input gets validated
        result = op({"value": 5})
        assert result.result == 10

        # Can pass validated object directly
        result = op(InputSpec(value=3, multiplier=4))
        assert result.result == 12

    def test_operator_without_validation(self):
        """Test operator without validation specs."""

        class SimpleOperator(Operator):
            def forward(self, x):
                if isinstance(x, dict):
                    return x.get("value", 0) + x.get("y", 1)
                return x + 1

        op = SimpleOperator()

        # Direct calls work
        assert op(5) == 6
        assert op({"value": 5, "y": 3}) == 8
        assert op({"value": 5, "y": 10}) == 15

    def test_operator_inheritance(self):
        """Test operator inheritance and composition."""

        class BaseProcessor(Operator):
            def preprocess(self, x):
                return x.strip().lower()

            def forward(self, x):
                return self.preprocess(x)

        class ExtendedProcessor(BaseProcessor):
            def forward(self, x):
                # Use parent preprocessing
                processed = self.preprocess(x)
                # Add own logic
                return processed.replace(" ", "_")

        op = ExtendedProcessor()
        result = op("  Hello World  ")
        assert result == "hello_world"

    def test_operator_with_state(self):
        """Test operator with internal state (immutable via equinox)."""
        import jax.numpy as jnp

        class StatefulOperator(Operator):
            count: jnp.ndarray

            def __init__(self, initial_count=0):
                self.count = jnp.array(initial_count)

            def forward(self, x):
                # Note: This doesn't mutate self.count
                # In real use, state updates happen via JAX transforms
                return x + self.count

        op = StatefulOperator(10)
        result = op(5)
        assert result == 15

        # State is immutable
        op(20)  # Doesn't change count
        assert op.count == 10

    def test_operator_error_propagation(self):
        """Test that errors in forward() propagate correctly."""

        class ErrorOperator(Operator):
            def forward(self, x):
                if x < 0:
                    raise ValueError("Negative values not allowed")
                return x * 2

        op = ErrorOperator()

        # Normal operation
        assert op(5) == 10

        # Error case
        with pytest.raises(ValueError) as exc_info:
            op(-1)
        assert "Negative values not allowed" in str(exc_info.value)
