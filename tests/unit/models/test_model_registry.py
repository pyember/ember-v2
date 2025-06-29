"""Test the ModelRegistry internals.

Following CLAUDE.md principles:
- Focus on behavior, not implementation
- Test thread safety and caching
- Clear test structure
"""

import pytest
import threading
from unittest.mock import Mock, patch

from ember.models.registry import ModelRegistry
from ember._internal.exceptions import ModelNotFoundError, ModelProviderError


class TestModelRegistry:
    """Test the ModelRegistry class behavior."""

    def test_initialization(self):
        """Test registry initializes properly."""
        registry = ModelRegistry()

        # Should start empty
        assert len(registry._models) == 0
        assert registry.list_models() == []

    def test_get_model_creates_on_first_access(self):
        """Test lazy model instantiation."""
        registry = ModelRegistry()

        with patch.object(registry, "_create_model") as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model

            # First access creates
            model1 = registry.get_model("gpt-4")

            assert model1 is mock_model
            mock_create.assert_called_once_with("gpt-4")
            assert "gpt-4" in registry._models

    def test_get_model_returns_cached(self):
        """Test model caching on subsequent calls."""
        registry = ModelRegistry()

        with patch.object(registry, "_create_model") as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model

            # First call
            model1 = registry.get_model("gpt-4")
            # Second call
            model2 = registry.get_model("gpt-4")

            # Should be same instance
            assert model1 is model2
            # Only created once
            mock_create.assert_called_once_with("gpt-4")

    def test_thread_safety(self):
        """Test concurrent access is thread-safe."""
        registry = ModelRegistry()
        results = []
        errors = []

        def get_model_thread(model_id):
            try:
                with patch.object(registry, "_create_model") as mock_create:
                    # Simulate some work
                    mock_model = Mock()
                    mock_model.id = model_id
                    mock_create.return_value = mock_model

                    model = registry.get_model(model_id)
                    results.append(model)
            except Exception as e:
                errors.append(e)

        # Create multiple threads trying to get same model
        threads = []
        for i in range(10):
            t = threading.Thread(target=get_model_thread, args=("gpt-4",))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0
        # All results should be the same instance
        assert len(results) == 10
        first_model = results[0]
        for model in results[1:]:
            assert model is first_model

    def test_clear_cache(self):
        """Test clearing the model cache."""
        registry = ModelRegistry()

        # Add some models to cache
        registry._models["gpt-4"] = Mock()
        registry._models["claude-3"] = Mock()

        assert len(registry._models) == 2

        # Clear cache
        registry.clear_cache()

        assert len(registry._models) == 0
        assert registry.list_models() == []

    def test_list_models(self):
        """Test listing cached models."""
        registry = ModelRegistry()

        # Add models to cache
        registry._models["gpt-4"] = Mock()
        registry._models["claude-3-opus"] = Mock()
        registry._models["gemini-pro"] = Mock()

        models = registry.list_models()

        assert len(models) == 3
        assert "gpt-4" in models
        assert "claude-3-opus" in models
        assert "gemini-pro" in models

    def test_model_not_found_error(self):
        """Test error for unknown model."""
        registry = ModelRegistry()

        with patch("ember.models.providers.resolve_model_id") as mock_resolve:
            mock_resolve.return_value = ("unknown", "some-model")

            with pytest.raises(ModelNotFoundError) as exc_info:
                registry.get_model("some-model")

            assert "Cannot determine provider" in str(exc_info.value)
            assert "some-model" in str(exc_info.value)

    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        registry = ModelRegistry()

        with patch("ember.models.providers.resolve_model_id") as mock_resolve:
            with patch("ember.models.providers.get_provider_class") as mock_get_class:
                with patch.object(registry, "_get_api_key") as mock_get_key:
                    mock_resolve.return_value = ("openai", "gpt-4")
                    mock_get_class.return_value = Mock
                    mock_get_key.return_value = None  # No API key

                    with pytest.raises(ModelProviderError) as exc_info:
                        registry.get_model("gpt-4")

                    assert "No API key found" in str(exc_info.value)
                    assert "OPENAI_API_KEY" in str(exc_info.value)

    @patch("os.getenv")
    def test_api_key_retrieval(self, mock_getenv):
        """Test API key retrieval from environment."""
        registry = ModelRegistry()

        # Test standard format
        mock_getenv.side_effect = lambda x: (
            "test-key" if x == "OPENAI_API_KEY" else None
        )
        key = registry._get_api_key("openai")
        assert key == "test-key"

        # Test Ember-specific format
        mock_getenv.side_effect = lambda x: (
            "ember-key" if x == "EMBER_ANTHROPIC_API_KEY" else None
        )
        key = registry._get_api_key("anthropic")
        assert key == "ember-key"

        # Test no key found
        mock_getenv.return_value = None
        key = registry._get_api_key("unknown")
        assert key is None

    def test_invoke_model(self, mock_model_response):
        """Test invoking a model through the registry."""
        registry = ModelRegistry()

        # Mock the model
        mock_model = Mock()
        mock_model.complete.return_value = mock_model_response
        registry._models["gpt-4"] = mock_model

        # Invoke
        response = registry.invoke_model("gpt-4", "Hello", temperature=0.7)

        assert response is mock_model_response
        mock_model.complete.assert_called_once_with("Hello", "gpt-4", temperature=0.7)

    def test_invoke_model_with_cost_calculation(self):
        """Test cost calculation during invocation."""
        registry = ModelRegistry()

        from ember.models.schemas import ChatResponse, UsageStats

        # Mock model with usage - no cost_usd set initially
        mock_model = Mock()
        usage = UsageStats(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_model.complete.return_value = ChatResponse(data="Response", usage=usage)
        registry._models["gpt-4"] = mock_model

        # Mock the cost calculation to verify it's being called
        with patch.object(registry, "_calculate_cost") as mock_calc:
            mock_calc.return_value = 0.0015

            response = registry.invoke_model("gpt-4", "Hello")

            # Verify cost calculation was called
            mock_calc.assert_called_once_with("gpt-4", usage)

            # Verify the cost was set
            assert response.usage.cost_usd == 0.0015

    def test_usage_tracking(self):
        """Test usage statistics tracking."""
        registry = ModelRegistry()

        from ember.models.schemas import ChatResponse, UsageStats

        # Mock model
        mock_model = Mock()
        usage1 = UsageStats(
            prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.001
        )
        usage2 = UsageStats(
            prompt_tokens=15, completion_tokens=25, total_tokens=40, cost_usd=0.002
        )

        mock_model.complete.side_effect = [
            ChatResponse(data="Response1", usage=usage1),
            ChatResponse(data="Response2", usage=usage2),
        ]
        registry._models["gpt-4"] = mock_model

        # Mock cost calculation to not override our test costs
        with patch.object(registry, "_calculate_cost") as mock_calc:
            mock_calc.side_effect = [0.001, 0.002]

            # Make two calls
            registry.invoke_model("gpt-4", "First")
            registry.invoke_model("gpt-4", "Second")

        # Get usage summary
        summary = registry.get_usage_summary("gpt-4")

        assert summary is not None
        assert summary.total_tokens == 70  # 30 + 40
        assert summary.prompt_tokens == 25  # 10 + 15
        assert summary.completion_tokens == 45  # 20 + 25
        assert summary.cost_usd == 0.003  # 0.001 + 0.002
