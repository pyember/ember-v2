"""Tests for examples in 01_getting_started directory."""

from .test_base import ExampleGoldenTest


class TestGettingStartedExamples(ExampleGoldenTest):
    """Test all examples in the 01_getting_started directory."""

    def test_hello_world(self):
        """Test the hello_world.py example."""
        self.run_example_test("01_getting_started/hello_world.py", max_execution_time=5.0)

    def test_first_model_call(self, no_api_keys):
        """Test the first_model_call.py example."""
        self.run_example_test(
            "01_getting_started/first_model_call.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=30.0,
            skip_real_mode=no_api_keys,
        )

    def test_basic_prompt_engineering(self, no_api_keys):
        """Test the basic_prompt_engineering.py example."""
        self.run_example_test(
            "01_getting_started/basic_prompt_engineering.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=30.0,
            skip_real_mode=no_api_keys,
        )

    def test_model_comparison(self, no_api_keys):
        """Test the model_comparison.py example."""
        # This example only needs OpenAI API key
        self.run_example_test(
            "01_getting_started/model_comparison.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=60.0,
            skip_real_mode=no_api_keys,
        )
