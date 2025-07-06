"""Tests for examples in 02_core_concepts directory."""

from .test_base import ExampleGoldenTest


class TestCoreConceptsExamples(ExampleGoldenTest):
    """Test all examples in the 02_core_concepts directory."""

    def test_context_management(self):
        """Test the context_management.py example."""
        self.run_example_test("02_core_concepts/context_management.py", max_execution_time=10.0)

    def test_error_handling(self):
        """Test the error_handling.py example."""
        self.run_example_test("02_core_concepts/error_handling.py", max_execution_time=10.0)

    def test_operators_basics(self):
        """Test the operators_basics.py example."""
        self.run_example_test("02_core_concepts/operators_basics.py", max_execution_time=15.0)

    def test_rich_specifications(self):
        """Test the rich_specifications.py example."""
        # This example doesn't actually make API calls
        self.run_example_test("02_core_concepts/rich_specifications.py", max_execution_time=10.0)

    def test_type_safety(self):
        """Test the type_safety.py example."""
        self.run_example_test("02_core_concepts/type_safety.py", max_execution_time=10.0)
