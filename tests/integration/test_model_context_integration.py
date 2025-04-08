"""
Integration tests for the ModelContext refactoring.

These tests verify that the new ModelContext system correctly integrates with
existing components and properly manages dependencies.
"""

import unittest
from unittest.mock import MagicMock

from ember.api.models import (
    ModelAPI,
    ModelBuilder,
    create_context,
    get_default_context,
    model,
)
from ember.core.registry.model.base.context import ModelConfig, ModelContext


class TestModelContextIntegration(unittest.TestCase):
    """Integration tests for ModelContext with the model API."""

    def test_model_function_with_custom_context(self):
        """Test that the model function properly uses custom contexts."""
        # Create a mock context
        mock_registry = MagicMock()
        mock_service = MagicMock()
        mock_response = MagicMock()
        mock_response.data = "Test response"
        mock_service.invoke_model.return_value = mock_response

        # Create a context with our mocks
        context = ModelContext()
        context._registry = mock_registry
        context._model_service = mock_service
        context._initialized = True

        # Use the model function with our context
        test_model = model("test-model", context=context)
        response = test_model("Test prompt")

        # Verify the context was used
        mock_service.invoke_model.assert_called_once()
        self.assertEqual(str(response), "Test response")

    def test_multiple_isolated_contexts(self):
        """Test that multiple contexts are properly isolated."""
        # Create two mock contexts with properly initialized registries
        context1 = ModelContext()
        mock_registry1 = MagicMock()
        mock_registry1.is_registered.return_value = True  # Simulate model is registered
        mock_service1 = MagicMock()
        mock_response1 = MagicMock()
        mock_response1.data = "Response from context 1"
        mock_service1.invoke_model.return_value = mock_response1
        context1._registry = mock_registry1
        context1._model_service = mock_service1
        context1._initialized = True

        context2 = ModelContext()
        mock_registry2 = MagicMock()
        mock_registry2.is_registered.return_value = True  # Simulate model is registered
        mock_service2 = MagicMock()
        mock_response2 = MagicMock()
        mock_response2.data = "Response from context 2"
        mock_service2.invoke_model.return_value = mock_response2
        context2._registry = mock_registry2
        context2._model_service = mock_service2
        context2._initialized = True

        # Use both contexts
        model1 = model("test-model", context=context1)
        model2 = model("test-model", context=context2)

        response1 = model1("Test prompt")
        response2 = model2("Test prompt")

        # Verify each context was used independently
        mock_service1.invoke_model.assert_called_once()
        mock_service2.invoke_model.assert_called_once()
        self.assertEqual(str(response1), "Response from context 1")
        self.assertEqual(str(response2), "Response from context 2")

    def test_backwards_compatibility_with_legacy_api(self):
        """Test that the legacy API works with ModelContext."""
        # Create a mock context
        mock_registry = MagicMock()
        mock_service = MagicMock()
        mock_response = MagicMock()
        mock_response.data = "Legacy API response"
        mock_service.invoke_model.return_value = mock_response

        # Create a context with our mocks
        context = ModelContext()
        context._registry = mock_registry
        context._model_service = mock_service
        context._initialized = True

        # Use the legacy API with our context
        api = ModelAPI("test-model", context=context)
        response = api.generate("Test prompt")

        # Verify the context was used
        mock_service.invoke_model.assert_called_once()
        self.assertEqual(str(response), "Legacy API response")

    def test_builder_pattern_with_context(self):
        """Test that the builder pattern works with contexts."""
        # Create a mock context
        mock_registry = MagicMock()
        mock_service = MagicMock()
        mock_response = MagicMock()
        mock_response.data = "Builder response"
        mock_service.invoke_model.return_value = mock_response

        # Create a context with our mocks
        context = ModelContext()
        context._registry = mock_registry
        context._model_service = mock_service
        context._initialized = True

        # Use the builder with our context
        api = (
            ModelBuilder(context=context)
            .temperature(0.5)
            .max_tokens(100)
            .build("test-model")
        )
        response = api.generate("Test prompt")

        # Verify the context was used with correct parameters
        mock_service.invoke_model.assert_called_once()
        call_args = mock_service.invoke_model.call_args[1]
        # Model ID includes provider prefix - we only check that it ends with test-model
        self.assertTrue(call_args["model_id"].endswith("test-model"))
        self.assertEqual(call_args["prompt"], "Test prompt")
        self.assertEqual(call_args["temperature"], 0.5)
        self.assertEqual(call_args["max_tokens"], 100)
        self.assertEqual(str(response), "Builder response")

    def test_default_context_singleton(self):
        """Test that the default context is a singleton."""
        # Get the default context multiple times
        context1 = get_default_context()
        context2 = get_default_context()

        # They should be the same instance
        self.assertIs(context1, context2)

    def test_create_context_function(self):
        """Test that create_context creates a new context."""
        # Create custom configs
        config1 = ModelConfig(auto_discover=False, default_timeout=60)
        config2 = ModelConfig(auto_discover=True, default_timeout=30)

        # Create contexts with the configs
        context1 = create_context(config=config1)
        context2 = create_context(config=config2)

        # They should be different instances
        self.assertIsNot(context1, context2)

        # They should have the correct configs
        self.assertEqual(context1.config.auto_discover, False)
        self.assertEqual(context1.config.default_timeout, 60)
        self.assertEqual(context2.config.auto_discover, True)
        self.assertEqual(context2.config.default_timeout, 30)


if __name__ == "__main__":
    unittest.main()
