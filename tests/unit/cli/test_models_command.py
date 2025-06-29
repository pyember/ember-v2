"""Tests for models CLI command with proper mocking.

Following CLAUDE.md principles: minimal, focused, no magic.
"""

import sys
from unittest.mock import Mock, patch


from ember.cli.main import cmd_models


class TestModelsCommand:
    """Test models command with clean mocking."""

    def test_list_providers(self, capsys):
        """List providers returns correct output."""
        # Create a mock that behaves like the models function with attributes
        mock_models = Mock()
        mock_models.providers = Mock(return_value=["openai", "anthropic", "google"])

        # Temporarily replace the module in sys.modules
        original_module = sys.modules.get("ember.api")
        try:
            # Create a mock module with our mock models
            mock_module = Mock()
            mock_module.models = mock_models
            sys.modules["ember.api"] = mock_module

            # Run the command
            result = cmd_models(Mock(providers=True))

            # Verify output
            output = capsys.readouterr().out
            assert result == 0
            assert "Available providers:" in output
            assert "- openai" in output
            assert "- anthropic" in output
            assert "- google" in output

        finally:
            # Restore original module
            if original_module:
                sys.modules["ember.api"] = original_module
            else:
                sys.modules.pop("ember.api", None)

    def test_list_models(self, capsys):
        """List models returns formatted output."""
        # Mock model data
        models_data = {
            "gpt-4": {"description": "Advanced model", "provider": "openai"},
            "claude-3": {"description": "Anthropic model", "provider": "anthropic"},
        }

        # Create mock
        mock_models = Mock()
        mock_models.discover = Mock(return_value=models_data)

        original_module = sys.modules.get("ember.api")
        try:
            mock_module = Mock()
            mock_module.models = mock_models
            sys.modules["ember.api"] = mock_module

            # Run command
            result = cmd_models(Mock(providers=False, provider=None))

            # Verify
            output = capsys.readouterr().out
            assert result == 0
            assert "Available models:" in output
            assert "gpt-4" in output
            assert "Advanced model" in output
            assert "claude-3" in output
            assert "Anthropic model" in output

        finally:
            if original_module:
                sys.modules["ember.api"] = original_module
            else:
                sys.modules.pop("ember.api", None)

    def test_filter_by_provider(self, capsys):
        """Filter models by provider."""
        # Mock filtered data
        openai_models = {
            "gpt-4": {"description": "GPT-4", "provider": "openai"},
            "gpt-3.5": {"description": "GPT-3.5", "provider": "openai"},
        }

        mock_models = Mock()
        mock_models.discover = Mock(return_value=openai_models)

        original_module = sys.modules.get("ember.api")
        try:
            mock_module = Mock()
            mock_module.models = mock_models
            sys.modules["ember.api"] = mock_module

            # Run with provider filter
            result = cmd_models(Mock(providers=False, provider="openai"))

            # Verify discover was called with provider
            mock_models.discover.assert_called_once_with("openai")

            # Verify output
            output = capsys.readouterr().out
            assert result == 0
            assert "gpt-4" in output
            assert "gpt-3.5" in output

        finally:
            if original_module:
                sys.modules["ember.api"] = original_module
            else:
                sys.modules.pop("ember.api", None)


# Alternative approach using dependency injection
class TestModelsCommandDI:
    """Test with dependency injection pattern."""

    @patch("ember.cli.main.print")
    def test_providers_with_di(self, mock_print):
        """Test using print mocking instead of module mocking."""
        # This is a cleaner approach that doesn't require module manipulation
        mock_models = Mock()
        mock_models.providers = Mock(return_value=["provider1", "provider2"])

        # Patch at import location
        with patch.dict("sys.modules", {"ember.api": Mock(models=mock_models)}):
            result = cmd_models(Mock(providers=True))

        assert result == 0
        # Verify print was called correctly
        assert mock_print.call_count >= 3  # Header + 2 providers
