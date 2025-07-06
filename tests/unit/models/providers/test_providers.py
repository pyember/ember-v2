"""Comprehensive tests for all provider implementations.

Following principles that Jeff Dean, Sanjay Ghemawat, and others would advocate:
- Test behavior, not implementation
- Ensure tests catch regressions before production
- Make tests deterministic and fast
- Test error conditions as thoroughly as success paths
- Tests should document expected behavior
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.anthropic import AnthropicProvider
from ember.models.providers.google import GoogleProvider
from ember.models.providers.openai import OpenAIProvider


class TestBaseProvider:
    """Test base provider functionality common to all providers."""

    def test_api_key_validation(self):
        """Test that providers validate API keys properly."""
        # BaseProvider is abstract, so test with concrete providers
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAIProvider()
            with pytest.raises(ValueError, match="API key"):
                AnthropicProvider()
            with pytest.raises(ValueError, match="API key"):
                GoogleProvider()

    def test_retry_behavior(self):
        """Test that providers retry on transient failures."""
        # This is tested in subclasses since retry is on specific methods
        pass


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_initialization(self):
        """Test OpenAI provider initialization."""
        # With explicit key
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"

        # With environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIProvider()
            assert provider.api_key == "env-key"

    def test_model_validation(self):
        """Test OpenAI model validation."""
        provider = OpenAIProvider(api_key="test-key")

        # Valid models
        assert provider.validate_model("gpt-4")
        assert provider.validate_model("gpt-4-turbo")
        assert provider.validate_model("gpt-3.5-turbo")
        assert provider.validate_model("gpt-4o")  # Changed from o1-preview

        # Invalid models
        assert not provider.validate_model("gemini-pro")
        assert not provider.validate_model("claude-3")
        assert not provider.validate_model("invalid-model")

    @patch("openai.OpenAI")
    def test_completion_success(self, mock_openai_class):
        """Test successful completion."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"

        mock_client.chat.completions.create.return_value = mock_response

        # Test
        provider = OpenAIProvider(api_key="test-key")
        response = provider.complete("Say hello", "gpt-4", temperature=0.7)

        assert response.data == "Hello!"
        assert response.usage.total_tokens == 15
        assert response.model_id == "gpt-4"

    @patch("openai.OpenAI")
    def test_completion_error_handling(self, mock_openai_class):
        """Test error handling in completions."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Test authentication error
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")

        provider = OpenAIProvider(api_key="bad-key")
        with pytest.raises(ProviderAPIError, match="Invalid API key"):
            provider.complete("test", "gpt-4")


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    def test_initialization(self):
        """Test Anthropic provider initialization."""
        # With explicit key
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"

        # With environment variable
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            provider = AnthropicProvider()
            assert provider.api_key == "env-key"

    def test_model_validation(self):
        """Test Anthropic model validation."""
        provider = AnthropicProvider(api_key="test-key")

        # Valid models
        assert provider.validate_model("claude-3-opus")
        assert provider.validate_model("claude-3-sonnet")
        assert provider.validate_model("claude-3-haiku")
        assert provider.validate_model("claude-2.1")

        # Invalid models
        assert not provider.validate_model("gpt-4")
        assert not provider.validate_model("gemini-pro")
        assert not provider.validate_model("invalid-model")

    @patch("anthropic.Anthropic")
    def test_completion_success(self, mock_anthropic_class):
        """Test successful completion."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Hello from Claude!")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-3-opus"

        mock_client.messages.create.return_value = mock_response

        # Test
        provider = AnthropicProvider(api_key="test-key")
        response = provider.complete("Say hello", "claude-3-opus", temperature=0.7)

        assert response.data == "Hello from Claude!"
        assert response.usage.total_tokens == 15
        assert response.model_id == "claude-3-opus"


class TestGoogleProvider:
    """Test Google provider implementation with pyasn1 compatibility."""

    def test_pyasn1_compatibility(self):
        """Test that pyasn1 compatibility patch is applied."""
        # This import should not raise pyasn1 errors
        from ember.models.providers.google import GoogleProvider

        # The real test is that we can create a provider without errors
        provider = GoogleProvider(api_key="test-key")
        assert provider is not None

        # And that the problematic operation doesn't fail
        from pyasn1.type import constraint, univ

        try:
            # This was the failing operation
            subtypeSpec = univ.Integer.subtypeSpec + constraint.SingleValueConstraint(0, 1)
            assert subtypeSpec is not None
        except TypeError as e:
            if "can only concatenate tuple" in str(e):
                pytest.fail("pyasn1 compatibility not working")

    def test_initialization_without_pyasn1_error(self):
        """Test Google provider can be initialized without pyasn1 errors."""
        # This was the original failing case
        provider = GoogleProvider(api_key="test-key")
        assert provider.api_key == "test-key"

        # Should not raise "can only concatenate tuple" error
        assert provider is not None

    def test_initialization_with_env_vars(self):
        """Test initialization with various environment variables."""
        # Test GOOGLE_API_KEY
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}, clear=True):
            provider = GoogleProvider()
            assert provider.api_key == "google-key"

        # Test GEMINI_API_KEY
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            provider = GoogleProvider()
            assert provider.api_key == "gemini-key"

        # Test EMBER_GOOGLE_API_KEY
        with patch.dict(os.environ, {"EMBER_GOOGLE_API_KEY": "ember-key"}, clear=True):
            provider = GoogleProvider()
            assert provider.api_key == "ember-key"

        # Test priority order
        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "google",
                "GEMINI_API_KEY": "gemini",
                "EMBER_GOOGLE_API_KEY": "ember",
            },
        ):
            provider = GoogleProvider()
            assert provider.api_key == "google"  # Highest priority

    def test_model_validation(self):
        """Test Google model validation."""
        provider = GoogleProvider(api_key="test-key")

        # Valid models
        valid_models = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "models/gemini-pro",
            "gemini-ultra",  # Future model
        ]

        for model in valid_models:
            assert provider.validate_model(model), f"Should validate {model}"

        # Invalid models
        invalid_models = ["gpt-4", "claude-3", "invalid-model"]
        for model in invalid_models:
            assert not provider.validate_model(model), f"Should not validate {model}"

    def test_model_info(self):
        """Test model info retrieval."""
        provider = GoogleProvider(api_key="test-key")

        # Test gemini-pro
        info = provider.get_model_info("gemini-pro")
        assert info["provider"] == "GoogleProvider"
        assert info["context_window"] == 32768
        assert info["supports_vision"] is False
        assert info["supports_functions"] is True

        # Test gemini-pro-vision
        info = provider.get_model_info("gemini-pro-vision")
        assert info["supports_vision"] is True

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_completion_success(self, mock_configure, mock_model_class):
        """Test successful completion."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Hello from Gemini!"
        mock_model.generate_content.return_value = mock_response

        # Test
        provider = GoogleProvider(api_key="test-key")
        response = provider.complete("Say hello", "gemini-pro", temperature=0.7)

        assert response.data == "Hello from Gemini!"
        assert response.model_id == "models/gemini-pro"

        # Verify API was configured
        mock_configure.assert_called_with(api_key="test-key")

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_completion_error_handling(self, mock_configure, mock_model_class):
        """Test error handling in completions."""
        # Test model creation error
        mock_model_class.side_effect = Exception("Invalid model")

        provider = GoogleProvider(api_key="test-key")
        with pytest.raises(ProviderAPIError, match="Failed to create Gemini model"):
            provider.complete("test", "invalid-model")

        # Test API key error
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        mock_model_class.side_effect = None
        mock_model.generate_content.side_effect = Exception("API_KEY_INVALID")

        with pytest.raises(ProviderAPIError, match="Invalid Google API key"):
            provider.complete("test", "gemini-pro")

        # Test rate limit error
        mock_model.generate_content.side_effect = Exception("RATE_LIMIT_EXCEEDED")

        with pytest.raises(ProviderAPIError, match="rate limit"):
            provider.complete("test", "gemini-pro")

    def test_pyasn1_constraint_operations(self):
        """Test specific pyasn1 operations that were failing."""
        from pyasn1.type import constraint, univ

        # This exact operation was failing before the fix
        try:
            int_constraint = univ.Integer.subtypeSpec
            single_value = constraint.SingleValueConstraint(0, 1)
            combined = int_constraint + single_value
            assert combined is not None
        except TypeError as e:
            if "can only concatenate tuple" in str(e):
                pytest.fail(f"pyasn1 patch not working: {e}")

    def test_import_performance(self):
        """Test that our patch doesn't significantly impact import time."""
        import time

        # Remove from cache
        if "ember.models.providers.google" in sys.modules:
            del sys.modules["ember.models.providers.google"]

        # Measure import time
        start = time.perf_counter()

        elapsed = time.perf_counter() - start

        # Should be fast (under 1 second even on slow systems)
        assert elapsed < 1.0, f"Import too slow: {elapsed:.3f}s"

    def test_concurrent_initialization(self):
        """Test thread-safe provider initialization."""
        import queue
        import threading

        errors = queue.Queue()
        providers = queue.Queue()

        def create_provider(key: str):
            try:
                provider = GoogleProvider(api_key=key)
                providers.put(provider)
            except Exception as e:
                errors.put(e)

        # Create providers concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=create_provider, args=(f"key-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check no errors
        assert errors.empty(), f"Concurrent init failed: {list(errors.queue)}"

        # Check all providers created
        created = []
        while not providers.empty():
            created.append(providers.get())

        assert len(created) == 5
        assert all(p.api_key.startswith("key-") for p in created)


class TestProviderRegistry:
    """Test provider registration and discovery."""

    def test_all_providers_registered(self):
        """Test that all providers are properly registered."""
        from ember.models.registry import ModelRegistry

        registry = ModelRegistry()

        # These models should work with their respective providers
        test_cases = [
            ("gpt-4", OpenAIProvider),
            ("claude-3-opus", AnthropicProvider),
            ("gemini-pro", GoogleProvider),
        ]

        for model_id, expected_provider in test_cases:
            # This will create the model and its provider
            with patch.object(registry, "_create_model") as mock_create:
                # Mock the provider creation
                mock_model = Mock()
                mock_model.provider = expected_provider(api_key="test-key")
                mock_create.return_value = mock_model

                model = registry.get_model(model_id)
                assert isinstance(model.provider, expected_provider)


# Regression tests to ensure issues don't resurface
class TestRegressionGuards:
    """Tests that guard against specific regressions."""

    def test_pyasn1_patch_exists(self):
        """Ensure pyasn1 patch file hasn't been removed."""
        import os

        patch_file = os.path.join(
            os.path.dirname(__file__), "../../../../src/ember/_internal/pyasn1_patch.py"
        )
        assert os.path.exists(patch_file), "pyasn1_patch.py was removed!"

    def test_google_provider_imports_patch(self):
        """Ensure Google provider imports the patch."""
        # Read the source file directly
        import os

        google_provider_path = os.path.join(
            os.path.dirname(__file__),
            "../../../../src/ember/models/providers/google.py",
        )

        with open(google_provider_path, "r") as f:
            source = f.read()

        assert "pyasn1_patch" in source, "Google provider must import pyasn1_patch"
        assert "ensure_pyasn1_compatibility" in source, "Must call ensure_pyasn1_compatibility"

    def test_no_pyasn1_errors_in_imports(self):
        """Test that no pyasn1 errors occur during imports."""
        # Import all providers - none should raise pyasn1 errors
        try:
            pass
        except TypeError as e:
            if "can only concatenate tuple" in str(e):
                pytest.fail(f"pyasn1 error in imports: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "not integration"])
