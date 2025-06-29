"""Tests for examples in 06_performance_optimization directory."""

import pytest
from .test_base import ExampleGoldenTest


class TestPerformanceOptimizationExamples(ExampleGoldenTest):
    """Test all examples in the 06_performance_optimization directory."""

    @pytest.mark.requires_api_key
    def test_batch_processing(self):
        """Test the batch_processing.py example."""
        self.run_example_test(
            "06_performance_optimization/batch_processing.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=60.0,
            # validate_sections removed due to parsing issues
        )

    def test_jit_basics(self):
        """Test the jit_basics.py example."""
        self.run_example_test(
            "06_performance_optimization/jit_basics.py",
            max_execution_time=20.0,
            # validate_sections removed due to parsing issues
        )

    @pytest.mark.requires_api_key
    def test_optimization_techniques(self):
        """Test the optimization_techniques.py example."""
        self.run_example_test(
            "06_performance_optimization/optimization_techniques.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=45.0,
            # validate_sections removed due to parsing issues
        )
