"""Basic integration tests for EmberContext system."""

import pytest

from ember.core.context.ember_context import current_context, temp_component


def test_basic_context_registration():
    """Test basic component registration and retrieval."""
    # Get current context
    ctx = current_context()

    # Register a test component
    ctx.register("model", "test_model", lambda x: f"Hello {x}")

    # Retrieve and use the component
    model = ctx.get_model("test_model")
    result = model("World")
    assert result == "Hello World"


def test_temp_component():
    """Test temporary component override."""
    # Get current context
    ctx = current_context()

    # Register a test component
    ctx.register("model", "test_model", lambda x: f"Hello {x}")

    # Use temp_component context manager
    with temp_component(
        "model", "test_model", lambda x: f"Temporary {x}"
    ) as temp_model:
        # Inside the context, we should see the temporary component
        result = ctx.get_model("test_model")("World")
        assert result == "Temporary World"

    # Outside the context, original should be restored
    result = ctx.get_model("test_model")("World")
    assert result == "Hello World"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
