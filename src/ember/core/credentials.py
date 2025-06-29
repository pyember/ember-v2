"""Credential management for Ember.

This module provides secure API key storage following the patterns established
by AWS CLI, gcloud, and other professional tools:

- Credentials stored in ~/.ember/credentials with 0600 permissions
- Environment variables take precedence over stored credentials
- Atomic writes prevent corruption during concurrent access
- JSON format for easy inspection and manual editing
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional


class CredentialManager:
    """Manages API key storage and retrieval with security best practices.

    Provides secure storage for API keys with atomic writes, proper file
    permissions, and support for multiple providers. Follows the credential
    storage patterns of AWS CLI and gcloud.

    Attributes:
        config_dir: Directory for Ember configuration (~/.ember).
        credentials_file: Path to credentials storage file.
        config_file: Path to legacy config file (for compatibility).
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize CredentialManager with default paths.

        Args:
            config_dir: Configuration directory. Defaults to ~/.ember.
        """
        self.config_dir = config_dir or (Path.home() / ".ember")
        self.credentials_file = self.config_dir / "credentials"
        self.config_file = self.config_dir / "config.json"
        self._config_dir = self.config_dir  # Alias for tests

    def get(self, provider: str) -> Optional[str]:
        """Get API key for provider from credentials file only.

        This method is deprecated. Use get_api_key() for proper precedence handling.
        Issues a deprecation warning when credentials file exists.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic').
                Case-sensitive identifier matching provider configuration.

        Returns:
            Optional[str]: API key if found in credentials file, None otherwise.
                Returns None on any error (permissions, invalid JSON, etc).

        Warnings:
            DeprecationWarning: When ~/.ember/credentials exists, advising migration
                to config.yaml or environment variables.
        """
        try:
            if self.credentials_file.exists():
                # Issue deprecation warning
                warnings.warn(
                    "The ~/.ember/credentials file is deprecated. "
                    "Please migrate your API keys to ~/.ember/config.yaml or use "
                    "environment variables. Run 'ember config migrate' to automatically "
                    "migrate your configuration.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                with open(self.credentials_file, "r") as f:
                    credentials = json.load(f)
                    provider_creds = credentials.get(provider, {})
                    return provider_creds.get("api_key")
        except Exception:
            # Invalid JSON or permissions issue
            pass

        return None

    def get_api_key(self, provider: str, env_var: str) -> Optional[str]:
        """Get API key with proper precedence handling.

        Retrieves API key following security best practices with clear precedence:
        1. Environment variable (highest priority)
        2. Credentials file (legacy support)

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic').
                Must match the provider key in configuration.
            env_var: Environment variable name to check first
                (e.g., 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY').

        Returns:
            Optional[str]: API key if found via any method, None otherwise.
                Environment variables override all other sources.

        Security Notes:
            - Environment variables are preferred for production use
            - Credentials file access errors are silently ignored
            - No logging of actual API keys for security
        """
        # 1. Check environment variable (highest precedence)
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key

        # 2. Check credentials file
        try:
            if self.credentials_file.exists():
                with open(self.credentials_file, "r") as f:
                    credentials = json.load(f)
                    provider_creds = credentials.get(provider, {})
                    return provider_creds.get("api_key")
        except Exception:
            # Invalid JSON or permissions issue
            pass

        return None

    def store(self, provider: str, api_key: str) -> None:
        """Store API key in credentials file (deprecated).

        This is a deprecated alias for save_api_key(). New code should use
        environment variables or config.yaml for API key storage.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic').
            api_key: API key to store securely.

        Security:
            - Creates parent directory with proper permissions
            - Uses atomic writes to prevent corruption
            - Sets file permissions to 0600 (owner read/write only)
        """
        self.save_api_key(provider, api_key)

    def delete(self, provider: str) -> bool:
        """Delete API key for a provider from credentials file.

        Removes the provider's credentials atomically. Safe to call even
        if provider doesn't exist or credentials file is missing.

        Args:
            provider: Provider name to remove (e.g., 'openai', 'anthropic').

        Returns:
            bool: True if provider was found and deleted, False otherwise.
                Returns False on any error (missing file, invalid JSON, etc).

        Note:
            Uses atomic write to ensure file consistency during deletion.
        """
        if not self.credentials_file.exists():
            return False

        try:
            with open(self.credentials_file, "r") as f:
                credentials = json.load(f)

            if provider in credentials:
                del credentials[provider]

                # Use atomic write for consistency
                self._write_credentials(credentials)
                return True

        except Exception:
            pass

        return False

    def list_providers(self) -> list[str]:
        """List all providers with stored credentials.

        Reads the credentials file and returns provider names only.
        Does not expose API keys for security.

        Returns:
            List[str]: Provider names with stored credentials, empty list if
                no credentials file exists or on any error.

        Security:
            Only returns provider names, never exposes actual API keys.
        """
        if not self.credentials_file.exists():
            return []

        try:
            with open(self.credentials_file, "r") as f:
                credentials = json.load(f)
            return list(credentials.keys())
        except Exception:
            return []

    def save_api_key(self, provider: str, api_key: str) -> None:
        """Save API key to credentials file atomically with secure permissions.

        Implements secure credential storage following industry best practices:
        - Atomic writes to prevent corruption
        - File permissions set to 0600 (owner-only access)
        - Input validation to prevent common mistakes
        - Timestamps for credential rotation tracking

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic').
                Must be non-empty string.
            api_key: API key to save. Will be stripped of whitespace.
                Must be at least 5 characters after stripping.

        Raises:
            ValueError: If provider or api_key is invalid:
                - Empty or non-string values
                - API key too short (<5 chars)
                - API key contains quotes or spaces
            OSError: If unable to write credentials file.

        Security:
            - Creates parent directory if needed
            - Uses tempfile + atomic rename for consistency
            - Sets restrictive permissions before rename
        """
        # Input validation
        if not provider or not isinstance(provider, str):
            raise ValueError("Provider name must be a non-empty string")

        if not api_key or not isinstance(api_key, str):
            raise ValueError("API key must be a non-empty string")

        # Basic API key format validation
        if len(api_key.strip()) < 5:
            raise ValueError("API key appears to be too short")

        # Warn about common mistakes
        if api_key.startswith('"') and api_key.endswith('"'):
            raise ValueError("API key should not be quoted")

        if " " in api_key.strip():
            raise ValueError("API key should not contain spaces")

        # Create directory if needed
        self.config_dir.mkdir(exist_ok=True)

        # Load existing credentials
        credentials = {}
        if self.credentials_file.exists():
            try:
                with open(self.credentials_file, "r") as f:
                    credentials = json.load(f)
            except Exception:
                pass

        # Update credentials
        from datetime import datetime

        credentials[provider] = {
            "api_key": api_key.strip(),
            "created_at": datetime.now().isoformat(),
        }

        # Write atomically
        self._write_credentials(credentials)

    def _write_credentials(self, credentials: Dict[str, Any]) -> None:
        """Write credentials atomically with secure permissions.

        Internal method implementing atomic write pattern:
        1. Write to temporary file in same directory
        2. Set secure permissions (0600)
        3. Atomic rename to final location

        This ensures credentials are never partially written or corrupted.

        Args:
            credentials: Complete credentials dictionary to write.
                Should contain provider keys with api_key and created_at.

        Raises:
            OSError: If unable to create temp file, set permissions, or rename.

        Implementation Notes:
            - Uses mkstemp for secure temp file creation
            - Temp file in same directory ensures atomic rename
            - Cleanup happens even on failure via finally block
        """
        import tempfile

        # Create temp file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.config_dir, prefix=".credentials_", suffix=".tmp"
        )

        try:
            # Write with secure permissions from the start
            with os.fdopen(temp_fd, "w") as f:
                json.dump(credentials, f, indent=2)

            # Ensure permissions are secure (belt and suspenders)
            os.chmod(temp_path, 0o600)

            # Atomic rename - either succeeds completely or fails
            Path(temp_path).replace(self.credentials_file)

        except Exception:
            # Clean up temp file on any error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def get_config(self) -> Dict[str, Any]:
        """Load general configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


# Global instance
_credential_manager = CredentialManager()


def get_api_key(provider: str, env_var: str) -> Optional[str]:
    """Get API key for a provider.

    Checks in order:
    1. Environment variable
    2. ~/.ember/credentials file

    Args:
        provider: Provider name (e.g., 'openai')
        env_var: Environment variable name (e.g., 'OPENAI_API_KEY')

    Returns:
        API key if found, None otherwise
    """
    return _credential_manager.get_api_key(provider, env_var)
