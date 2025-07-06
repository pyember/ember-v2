"""Test the models() function API.

Following CLAUDE.md principles:
- Test the contract, not the implementation
- Clear, explicit test cases
- Deterministic behavior
"""

from unittest.mock import patch

import pytest

from ember._internal.exceptions import (
    ModelNotFoundError,
    ModelProviderError,
    ProviderAPIError,
)
from ember.api.models import Response, models


class TestModelsFunction:
    """Test the main models() function interface."""

    def test_basic_invocation(self, mock_model_response):
        """Test basic model invocation: models("gpt-4", "Hello")."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("gpt-4", "Hello")

            assert isinstance(result, Response)
            assert result.text == "Test response"
            mock_registry.invoke_model.assert_called_once_with("gpt-4", "Hello")

    def test_with_parameters(self, mock_model_response):
        """Test invocation with parameters."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("gpt-4", "Hello", temperature=0.7, max_tokens=100)

            assert isinstance(result, Response)
            mock_registry.invoke_model.assert_called_once_with(
                "gpt-4", "Hello", temperature=0.7, max_tokens=100
            )

    def test_response_text_property(self, mock_model_response):
        """Test Response.text returns the generated content."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("gpt-4", "Hello")

            assert result.text == "Test response"
            assert str(result) == "Test response"  # __str__ returns text

    def test_response_usage_property(self, mock_model_response):
        """Test Response.usage returns token and cost info."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("gpt-4", "Hello")

            usage = result.usage
            assert usage["prompt_tokens"] == 10
            assert usage["completion_tokens"] == 20
            assert usage["total_tokens"] == 30
            # Cost should match actual GPT-4 pricing:
            # 10 tokens @ $0.03/1k input + 20 tokens @ $0.06/1k output = $0.0015
            assert usage["cost"] == 0.0015

    def test_response_model_id(self, mock_model_response):
        """Test Response.model_id returns the model identifier."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("gpt-4", "Hello")

            assert result.model_id == "gpt-4"

    def test_model_not_found_error(self):
        """Test error when model doesn't exist."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.side_effect = ModelNotFoundError(
                "Model 'invalid-model' not found"
            )

            with pytest.raises(ModelNotFoundError) as exc_info:
                models("invalid-model", "Hello")

            assert "invalid-model" in str(exc_info.value)

    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.side_effect = ModelProviderError(
                "No API key available for model gpt-4"
            )

            with pytest.raises(ModelProviderError) as exc_info:
                models("gpt-4", "Hello")

            assert "API key" in str(exc_info.value)

    def test_provider_api_error(self):
        """Test error from provider API (rate limits, etc)."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.side_effect = ProviderAPIError("Rate limit exceeded")

            with pytest.raises(ProviderAPIError) as exc_info:
                models("gpt-4", "Hello")

            assert "Rate limit" in str(exc_info.value)

    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        from ember.models.schemas import ChatResponse

        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            # Response with None data
            mock_registry.invoke_model.return_value = ChatResponse(data=None)

            result = models("gpt-4", "Hello")

            assert result.text == ""  # Empty string for None data

    def test_response_without_usage(self):
        """Test response when usage stats are missing."""
        from ember.models.schemas import ChatResponse

        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            # Response without usage
            mock_registry.invoke_model.return_value = ChatResponse(data="Response without usage")

            result = models("gpt-4", "Hello")

            usage = result.usage
            assert usage["prompt_tokens"] == 0
            assert usage["completion_tokens"] == 0
            assert usage["total_tokens"] == 0
            assert usage["cost"] == 0.0

    def test_explicit_provider_format(self, mock_model_response):
        """Test using explicit provider/model format."""
        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            mock_registry.invoke_model.return_value = mock_model_response

            result = models("openai/gpt-4", "Hello")

            assert isinstance(result, Response)
            mock_registry.invoke_model.assert_called_once_with("openai/gpt-4", "Hello")

    def test_unicode_handling(self, mock_api_keys):
        """Test handling of unicode in prompts and responses."""
        from ember.models.schemas import ChatResponse

        with patch("ember.api.models._global_models_api._registry") as mock_registry:
            # Unicode response
            mock_registry.invoke_model.return_value = ChatResponse(data="Hello 世界! 🌍")

            result = models("gpt-4", "你好")

            assert "世界" in result.text
            assert "🌍" in result.text
            mock_registry.invoke_model.assert_called_with("gpt-4", "你好")
