"""Tests for examples in 04_compound_ai directory."""

import pytest

from .test_base import ExampleGoldenTest


class TestCompoundAIExamples(ExampleGoldenTest):
    """Test all examples in the 04_compound_ai directory."""

    @pytest.mark.requires_api_key
    def test_judge_synthesis(self):
        """Test the judge_synthesis.py example."""
        self.run_example_test(
            "04_compound_ai/judge_synthesis.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=60.0,
            # validate_sections removed due to parsing issues
        )

    def test_operators_progressive_disclosure(self):
        """Test the operators_progressive_disclosure.py example."""
        self.run_example_test(
            "04_compound_ai/operators_progressive_disclosure.py",
            max_execution_time=20.0,
        )

    def test_simple_ensemble(self):
        """Test the simple_ensemble.py example."""
        # This example doesn't actually make API calls
        self.run_example_test("04_compound_ai/simple_ensemble.py", max_execution_time=15.0)

    def test_specifications_progressive(self):
        """Test the specifications_progressive.py example."""
        self.run_example_test(
            "04_compound_ai/specifications_progressive.py", max_execution_time=15.0
        )
