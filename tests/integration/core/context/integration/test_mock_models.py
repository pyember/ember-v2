"""Test context system with mock models."""

import pytest

from ember.core.context.ember_context import current_context, temp_component


class MockModel:
    """Mock model for testing without real API keys."""

    def __init__(self, name):
        self.name = name

    def generate(self, prompt):
        return f"Mock response from {self.name}: {prompt}"


def test_context_with_mock_models():
    """Test context system with mock model implementations."""
    # Get current context
    ctx = current_context()

    # Register mock models
    ctx.register("model", "gpt4", MockModel("GPT-4"))
    ctx.register("model", "claude", MockModel("Claude"))

    # Use the models
    gpt_model = ctx.get_model("gpt4")
    claude_model = ctx.get_model("claude")

    assert (
        gpt_model.generate("Hello, world!") == "Mock response from GPT-4: Hello, world!"
    )
    assert (
        claude_model.generate("Testing context system")
        == "Mock response from Claude: Testing context system"
    )


def test_temporary_model_override():
    """Test temporary override of models in context."""
    # Get current context
    ctx = current_context()

    # Register mock model
    ctx.register("model", "gpt4", MockModel("GPT-4"))

    # Test temporary override
    with temp_component("model", "gpt4", MockModel("GPT-4 Turbo")) as temp_model:
        # Direct access to temp model
        assert (
            temp_model.generate("With temporary model")
            == "Mock response from GPT-4 Turbo: With temporary model"
        )
        # Context lookup
        assert (
            ctx.get_model("gpt4").generate("Using context lookup")
            == "Mock response from GPT-4 Turbo: Using context lookup"
        )

    # Verify original is restored
    assert (
        ctx.get_model("gpt4").generate("After temp context")
        == "Mock response from GPT-4: After temp context"
    )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
