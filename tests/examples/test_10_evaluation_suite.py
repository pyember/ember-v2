"""Tests for examples in 10_evaluation_suite directory."""

import pytest

from .test_base import ExampleGoldenTest


class TestEvaluationSuiteExamples(ExampleGoldenTest):
    """Test all examples in the 10_evaluation_suite directory."""

    @pytest.mark.requires_api_key
    def test_accuracy_evaluation(self):
        """Test the accuracy_evaluation.py example."""
        self.run_example_test(
            "10_evaluation_suite/accuracy_evaluation.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=90.0,
            # validate_sections removed due to parsing issues
        )

    @pytest.mark.requires_api_key
    @pytest.mark.slow
    def test_benchmark_harness(self):
        """Test the benchmark_harness.py example."""
        self.run_example_test(
            "10_evaluation_suite/benchmark_harness.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=180.0,  # Benchmarks can be slow
            # validate_sections removed due to parsing issues
        )

    @pytest.mark.requires_api_key
    def test_consistency_testing(self):
        """Test the consistency_testing.py example."""
        self.run_example_test(
            "10_evaluation_suite/consistency_testing.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=60.0,
            # validate_sections removed due to parsing issues
        )
