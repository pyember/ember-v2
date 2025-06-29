"""Configuration compatibility adapter for external AI tools.

This module provides transparent configuration compatibility between Ember
and external AI tools (OpenAI CLI, Anthropic CLI, etc.). It enables users
to use their existing configurations without modification while maintaining
full compatibility with Ember's configuration system.

Key features:
- Auto-detection of external configuration formats
- Runtime adaptation without modifying original files
- Preservation of tool-specific settings
- Support for environment variable references

The adapter follows the Adapter pattern to provide a unified interface
while supporting diverse configuration formats from the ecosystem.
"""

import os
from typing import Any, Dict, Set


class CompatibilityAdapter:
    """Transparent adapter for external AI tool configuration formats.

    Provides detection, adaptation, and migration capabilities for configurations
    from external tools. Designed to work transparently at runtime without
    requiring user intervention.

    Attributes:
        EXTERNAL_FIELDS: Set of field names specific to external tools
            used for format detection.
        DEFAULT_BASE_URLS: Known base URLs for various AI providers,
            used when migrating configurations.
    """

    # External tool-specific fields
    EXTERNAL_FIELDS: Set[str] = {"approvalMode", "fullAutoErrorMode", "notify"}

    # Default base URLs for known providers
    DEFAULT_BASE_URLS: Dict[str, str] = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "azure": "https://YOUR_PROJECT_NAME.openai.azure.com/openai",
        "openrouter": "https://openrouter.ai/api/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        "ollama": "http://localhost:11434/v1",
        "mistral": "https://api.mistral.ai/v1",
        "deepseek": "https://api.deepseek.com",
        "xai": "https://api.x.ai/v1",
        "groq": "https://api.groq.com/openai/v1",
        "arceeai": "https://conductor.arcee.ai/v1",
    }

    @staticmethod
    def needs_adaptation(config: Dict[str, Any]) -> bool:
        """Detect if configuration needs adaptation from external format.

        Examines configuration structure and field names to determine if
        it originated from an external tool. Detection is based on:
        1. Presence of tool-specific fields (approvalMode, etc.)
        2. Provider configurations using envKey without api_key

        Args:
            config: Configuration dictionary to examine.

        Returns:
            bool: True if configuration appears to be from an external tool
                and needs adaptation, False if already in Ember format.

        Examples:
            >>> config = {"approvalMode": "auto", "providers": {...}}
            >>> CompatibilityAdapter.needs_adaptation(config)
            True

            >>> ember_config = {"providers": {"openai": {"api_key": "..."}}}
            >>> CompatibilityAdapter.needs_adaptation(ember_config)
            False
        """
        # Check for external tool-specific fields
        if CompatibilityAdapter.EXTERNAL_FIELDS & set(config.keys()):
            return True

        # Check for providers with envKey but no api_key
        if "providers" in config:
            for provider in config["providers"].values():
                if (
                    isinstance(provider, dict)
                    and "envKey" in provider
                    and "api_key" not in provider
                ):
                    return True

        return False

    @staticmethod
    def adapt_provider_config(provider_config: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt external provider format to Ember format at runtime.

        Transforms provider configuration from external tool format to Ember's
        expected format. Key transformations:
        1. Resolves envKey to api_key using environment variables
        2. Maps camelCase fields to snake_case (baseURL -> base_url)
        3. Preserves all original fields for compatibility

        Args:
            provider_config: Provider configuration in external format,
                typically containing envKey and baseURL fields.

        Returns:
            Dict[str, Any]: Provider configuration in Ember format with:
                - api_key: Resolved from environment or ${VAR} placeholder
                - base_url: Mapped from baseURL
                - All original fields preserved

        Runtime Behavior:
            - Environment variables are resolved immediately if available
            - Missing env vars result in ${VAR_NAME} placeholders
            - Original structure is preserved for round-trip compatibility
        """
        adapted = provider_config.copy()

        # If has envKey but no api_key, this is external format
        if "envKey" in adapted and "api_key" not in adapted:
            env_var = adapted["envKey"]
            # Try to get from environment, otherwise use placeholder
            adapted["api_key"] = os.environ.get(env_var, f"${{{env_var}}}")

        # Map external fields to Ember fields
        field_mappings = {
            "baseURL": "base_url",  # Ember uses snake_case internally
        }

        for codex_field, ember_field in field_mappings.items():
            if codex_field in adapted and ember_field not in adapted:
                adapted[ember_field] = adapted[codex_field]

        return adapted

    @staticmethod
    def adapt_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt complete external configuration to Ember format.

        Top-level adaptation method that processes entire configuration files.
        Handles both provider adaptations and preservation of tool-specific
        settings for potential future use.

        Args:
            config: Full configuration dictionary from external tool.

        Returns:
            Dict[str, Any]: Complete configuration adapted to Ember format:
                - All providers adapted to Ember schema
                - Tool-specific fields preserved in _external_compat
                - Original structure maintained where possible

        Processing:
            1. Checks if adaptation is needed
            2. Adapts each provider configuration
            3. Preserves external fields in _external_compat section
            4. Returns original config if no adaptation needed

        Note:
            This method is idempotent - can be called multiple times safely.
        """
        if not CompatibilityAdapter.needs_adaptation(config):
            return config

        adapted = config.copy()

        # Adapt all providers
        if "providers" in adapted:
            for name, provider in adapted["providers"].items():
                if isinstance(provider, dict):
                    adapted["providers"][name] = CompatibilityAdapter.adapt_provider_config(
                        provider
                    )

        # Preserve external tool-specific fields for potential extensions
        external_fields = {}
        for field in CompatibilityAdapter.EXTERNAL_FIELDS:
            if field in adapted:
                external_fields[field] = adapted[field]

        if external_fields:
            adapted["_external_compat"] = external_fields

        return adapted

    @staticmethod
    def migrate_provider(provider: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate an external provider configuration to Ember format.

        Used for explicit migration commands (ember configure import) to
        permanently convert external configurations. Unlike adapt_provider_config,
        this creates a clean Ember configuration suitable for saving.

        Args:
            provider: Provider configuration in external format containing:
                - name: Provider display name
                - baseURL: API endpoint URL
                - envKey: Environment variable name for API key

        Returns:
            Dict[str, Any]: Clean Ember provider configuration:
                - name: Display name
                - base_url: API endpoint (snake_case)
                - api_key: ${ENV_VAR} placeholder
                - env_key: Original env var name (for reference)
                - _original: Complete original config (for debugging)

        Migration Strategy:
            - Creates ${VAR} placeholders for environment variables
            - Preserves original data in _original for round-trip needs
            - Uses snake_case for Ember consistency

        Examples:
            >>> external = {"name": "OpenAI", "baseURL": "...", "envKey": "OPENAI_API_KEY"}
            >>> migrated = CompatibilityAdapter.migrate_provider(external)
            >>> migrated["api_key"]
            '${OPENAI_API_KEY}'
        """
        ember_provider = {
            "name": provider.get("name", ""),
            "base_url": provider.get("baseURL", ""),
            "api_key": f"${{{provider.get('envKey', '')}}}",
        }

        # Add env key reference for compatibility
        if "envKey" in provider:
            ember_provider["env_key"] = provider["envKey"]

        # Preserve original for round-trip compatibility
        ember_provider["_original"] = provider

        return ember_provider
