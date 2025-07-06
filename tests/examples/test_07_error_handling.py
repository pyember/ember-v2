"""Tests for examples in 07_error_handling directory."""

from .test_base import ExampleGoldenTest


class TestErrorHandlingExamples(ExampleGoldenTest):
    """Test all examples in the 07_error_handling directory."""

    def test_robust_patterns(self):
        """Test the robust_patterns.py example."""
        self.run_example_test("07_error_handling/robust_patterns.py", max_execution_time=30.0)
