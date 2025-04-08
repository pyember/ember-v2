"""
Unit tests for the ModelContext class.

These tests verify that the ModelContext correctly manages dependency
initialization, provides thread safety, and properly integrates with
existing components.
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

from ember.core.registry.model.base.context import (
    ModelConfig,
    ModelContext,
    create_context,
    get_default_context,
)


class TestModelContext(unittest.TestCase):
    """Test the ModelContext class."""

    def test_lazy_initialization(self):
        """Test that initialization is lazy and only happens when needed."""
        # Patch the initializer to track calls
        with patch(
            "ember.core.registry.model.base.context.initialize_registry"
        ) as mock_init:
            # Create a context without accessing properties
            context = ModelContext()

            # Initialization shouldn't have happened yet
            mock_init.assert_not_called()

            # Access a property to trigger initialization
            _ = context.registry

            # Initialization should have happened once
            mock_init.assert_called_once()

            # Access another property
            _ = context.model_service

            # Initialization should still have been called only once
            mock_init.assert_called_once()

    def test_config_initialization(self):
        """Test initialization with custom configuration."""
        custom_config = ModelConfig(
            auto_discover=False, default_timeout=60, api_keys={"openai": "test-key"}
        )

        with patch(
            "ember.core.registry.model.base.context.initialize_registry"
        ) as mock_init:
            context = ModelContext(config=custom_config)
            _ = context.registry

            # Verify the config was passed correctly
            mock_init.assert_called_once_with(
                config_path=None, config_manager=None, auto_discover=False
            )

            # Verify config is accessible
            self.assertEqual(context.config, custom_config)
            self.assertEqual(context.config.default_timeout, 60)
            self.assertEqual(context.config.api_keys, {"openai": "test-key"})

    def test_factory_function(self):
        """Test the create_context factory function."""
        custom_config = ModelConfig(auto_discover=False)

        with patch(
            "ember.core.registry.model.base.context.ModelContext"
        ) as MockContext:
            context = create_context(config=custom_config)
            MockContext.assert_called_once_with(config=custom_config)

    def test_default_context_singleton(self):
        """Test that the default context is a singleton."""
        # Reset the default context for testing
        with patch("ember.core.registry.model.base.context._default_context", None):
            with patch(
                "ember.core.registry.model.base.context.ModelContext"
            ) as MockContext:
                # Create a mock instance to return
                mock_instance = MagicMock()
                MockContext.return_value = mock_instance

                # Get the default context multiple times
                context1 = get_default_context()
                context2 = get_default_context()

                # Constructor should be called only once
                MockContext.assert_called_once()

                # Same instance should be returned
                self.assertIs(context1, context2)

    def test_thread_safety(self):
        """Test thread safety of context initialization."""
        # Reset the default context for testing
        with patch("ember.core.registry.model.base.context._default_context", None):
            with patch(
                "ember.core.registry.model.base.context.ModelContext"
            ) as MockContext:
                # Create a mock instance to return
                mock_instance = MagicMock()
                MockContext.return_value = mock_instance

                # Define a function to get the default context
                def get_context():
                    return get_default_context()

                # Create and start multiple threads
                threads = []
                results = []
                for _ in range(10):
                    thread = threading.Thread(
                        target=lambda: results.append(get_context())
                    )
                    threads.append(thread)
                    thread.start()

                # Wait for all threads to finish
                for thread in threads:
                    thread.join()

                # Constructor should be called only once
                MockContext.assert_called_once()

                # All threads should get the same instance
                for result in results:
                    self.assertIs(result, mock_instance)

    def test_registry_property(self):
        """Test the registry property."""
        # Create a mock registry
        mock_registry = MagicMock()

        # Patch the initializer to return our mock
        with patch(
            "ember.core.registry.model.base.context.initialize_registry",
            return_value=mock_registry,
        ):
            context = ModelContext()
            # Access the registry property
            registry = context.registry

            # It should be our mock
            self.assertIs(registry, mock_registry)

    def test_model_service_property(self):
        """Test the model_service property."""
        # Create a mock registry and service
        mock_registry = MagicMock()
        mock_service = MagicMock()

        # Create a mock service class that returns our mock service
        mock_service_class = MagicMock(return_value=mock_service)

        # Patch the initializer
        with patch(
            "ember.core.registry.model.base.context.initialize_registry",
            return_value=mock_registry,
        ):
            # Create a context with our mock service class
            context = ModelContext(config=ModelConfig(service_class=mock_service_class))

            # Access the model_service property
            service = context.model_service

            # It should be our mock
            self.assertIs(service, mock_service)

            # The service class should have been called with the registry
            mock_service_class.assert_called_once()

    def test_usage_service_property(self):
        """Test the usage_service property."""
        # Create a mock usage service
        mock_usage = MagicMock()

        # Patch the initializer and usage service constructor
        with patch("ember.core.registry.model.base.context.initialize_registry"):
            with patch(
                "ember.core.registry.model.base.context.UsageService",
                return_value=mock_usage,
            ):
                context = ModelContext()
                # Access the usage_service property
                usage = context.usage_service

                # It should be our mock
                self.assertIs(usage, mock_usage)

    def test_initialize_idempotent(self):
        """Test that initialize is idempotent."""
        # Create mocks for all dependencies
        mock_reg_init = MagicMock()
        mock_usage_init = MagicMock()
        mock_service = MagicMock()
        mock_service_class = MagicMock(return_value=mock_service)

        # Patch the registry initializer
        with patch(
            "ember.core.registry.model.base.context.initialize_registry",
            return_value=mock_reg_init,
        ):
            with patch(
                "ember.core.registry.model.base.context.UsageService",
                return_value=mock_usage_init,
            ):
                # Create a context with our mock service class
                context = ModelContext(
                    config=ModelConfig(service_class=mock_service_class)
                )

                # Call initialize multiple times
                context.initialize()
                context.initialize()
                context.initialize()

                # Each initializer should have been called only once
                self.assertEqual(mock_reg_init, context._registry)
                self.assertEqual(mock_usage_init, context._usage_service)
                self.assertEqual(mock_service, context._model_service)
                mock_service_class.assert_called_once()


if __name__ == "__main__":
    unittest.main()
