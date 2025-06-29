"""Tests for examples in 03_simplified_apis directory."""

from .test_base import ExampleGoldenTest


class TestSimplifiedApisExamples(ExampleGoldenTest):
    """Test all examples in the 03_simplified_apis directory."""

    def test_model_binding_patterns(self, no_api_keys):
        """Test the model_binding_patterns.py example."""
        self.run_example_test(
            "03_simplified_apis/model_binding_patterns.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=30.0,
            skip_real_mode=no_api_keys,
        )

    def test_natural_api_showcase(self, no_api_keys):
        """Test the natural_api_showcase.py example."""
        self.run_example_test(
            "03_simplified_apis/natural_api_showcase.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=45.0,
            skip_real_mode=no_api_keys,
        )

    def test_simplified_workflows(self, no_api_keys):
        """Test the simplified_workflows.py example."""
        self.run_example_test(
            "03_simplified_apis/simplified_workflows.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=40.0,
            skip_real_mode=no_api_keys,
        )

    def test_zero_config_jit(self, no_api_keys):
        """Test the zero_config_jit.py example."""
        self.run_example_test(
            "03_simplified_apis/zero_config_jit.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=30.0,
            skip_real_mode=no_api_keys,
        )
