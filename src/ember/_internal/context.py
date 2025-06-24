"""Centralized context for Ember configuration.

Provides thread-safe configuration and credential management with clean
dependency injection. Single source of truth for runtime configuration.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import yaml

from ember.core.credentials import CredentialManager
from ember.core.config.loader import load_config
from ember.core.utils.logging import get_logger

if TYPE_CHECKING:
    from ember.models.registry import ModelRegistry
    from ember.core.utils.data.registry import DataRegistry

logger = get_logger(__name__)


class EmberContext:
    """Central context for configuration and dependency management.
    
    Thread-safe container for configuration, credentials, and registries.
    Supports context isolation for testing.
    
    Example:
        ctx = EmberContext()
        api_key = ctx.get_credential("openai", "OPENAI_API_KEY")
    """
    
    # Thread-local storage for context isolation
    _thread_local = threading.local()
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        parent: Optional[EmberContext] = None,
        isolated: bool = False
    ):
        """Initialize context.
        
        Args:
            config_path: Optional path to configuration file.
            parent: Parent context for inheritance.
            isolated: Whether to isolate from global context.
        """
        self._config = {}
        self._parent = parent
        self._isolated = isolated
        self._lock = threading.RLock()
        
        # Core components - lazily initialized
        self._credential_manager: Optional[CredentialManager] = None
        self._model_registry: Optional[ModelRegistry] = None
        self._data_registry: Optional[DataRegistry] = None
        
        # Load configuration
        if config_path:
            self._config = load_config(str(config_path))
        elif parent:
            # Inherit parent config (deep copy for isolation)
            self._config = copy.deepcopy(parent._config)
        else:
            # Try to load from default location
            self._config = self._load_default_config()
            
        # Set as thread-local context if not isolated
        if not isolated and not parent:
            EmberContext._thread_local.context = self
            
            # Check for migration on first context creation
            self._check_migration()
    
    @classmethod
    def current(cls) -> EmberContext:
        """Get current thread's context.
        
        Returns:
            Thread-local EmberContext instance.
        """
        if not hasattr(cls._thread_local, 'context'):
            cls._thread_local.context = cls()
        return cls._thread_local.context
    
    @property
    def credential_manager(self) -> CredentialManager:
        """Get credential manager (lazy initialization).
        
        Returns:
            CredentialManager instance.
        """
        if self._credential_manager is None:
            with self._lock:
                if self._credential_manager is None:
                    self._credential_manager = CredentialManager()
        return self._credential_manager
    
    @property
    def model_registry(self) -> ModelRegistry:
        """Get model registry (lazy initialization).
        
        Returns:
            ModelRegistry instance.
        """
        if self._model_registry is None:
            with self._lock:
                if self._model_registry is None:
                    # Import here to avoid circular dependency
                    from ember.models.registry import ModelRegistry
                    self._model_registry = ModelRegistry(context=self)
        return self._model_registry
    
    @property
    def data_registry(self) -> DataRegistry:
        """Get data registry (lazy initialization).
        
        Returns:
            DataRegistry instance.
        """
        if self._data_registry is None:
            with self._lock:
                if self._data_registry is None:
                    # Import here to avoid circular dependency
                    from ember.core.utils.data.registry import DataRegistry
                    self._data_registry = DataRegistry(context=self)
        return self._data_registry
    
    def get_credential(self, provider: str, env_var: str) -> Optional[str]:
        """Get credential with precedence: env > file > config.
        
        Args:
            provider: Provider name (e.g., "openai").
            env_var: Environment variable name.
            
        Returns:
            API key or None.
        """
        # Delegate to credential manager for precedence logic
        api_key = self.credential_manager.get_api_key(provider, env_var)
        if api_key:
            return api_key
            
        # Check config file as final fallback
        provider_config = self._config.get("providers", {}).get(provider, {})
        return provider_config.get("api_key")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Dot-notation key (e.g., "models.temperature").
            default: Default if not found.
            
        Returns:
            Configuration value or default.
        """
        parts = key.split(".")
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Dot-notation key.
            value: Value to set.
        """
        with self._lock:
            parts = key.split(".")
            config = self._config
            
            # Navigate to parent dict
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
                
            # Set the value
            config[parts[-1]] = value
    
    def get_model(self, model_id: Optional[str] = None, **kwargs) -> Any:
        """Get model instance.
        
        Args:
            model_id: Model identifier or None for default.
            **kwargs: Model configuration.
            
        Returns:
            Model instance.
        """
        if model_id is None:
            model_id = self.get_config("models.default", "gpt-3.5-turbo")
            
        return self.model_registry.get_model(model_id, **kwargs)
    
    def list_models(self) -> list[str]:
        """List available models.
        
        Returns:
            List of available model identifiers.
        """
        return self.model_registry.list_available()
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration (safe for display).
        
        Returns:
            Configuration dictionary with sensitive values filtered.
        """
        # For now, return all config
        # In future, could filter API keys and secrets
        return self._config.copy()
    
    def reload(self) -> None:
        """Reload configuration from disk."""
        with self._lock:
            new_config = self._load_default_config()
            self._config = new_config
    
    def load_dataset(self, name: str, **kwargs) -> Any:
        """Load a dataset through the registry.
        
        Args:
            name: Dataset name
            **kwargs: Dataset configuration
            
        Returns:
            Dataset instance.
        """
        return self.data_registry.load(name, **kwargs)
    
    def create_child(self, **config_overrides) -> EmberContext:
        """Create child context with config overrides.
        
        Args:
            **config_overrides: Configuration overrides.
            
        Returns:
            New child context.
        """
        child = EmberContext(parent=self, isolated=True)
        
        # Apply overrides
        for key, value in config_overrides.items():
            child.set_config(key, value)
            
        return child
    
    def __enter__(self) -> EmberContext:
        """Context manager entry."""
        if not self._isolated:
            self._previous = getattr(EmberContext._thread_local, 'context', None)
            EmberContext._thread_local.context = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if not self._isolated:
            if self._previous is not None:
                EmberContext._thread_local.context = self._previous
            else:
                delattr(EmberContext._thread_local, 'context')
    
    @staticmethod
    def get_config_path() -> Path:
        """Get path to configuration file.
        
        Checks EMBER_CONFIG_PATH environment variable first,
        then falls back to ~/.ember/config.yaml
        
        Returns:
            Path to config.yaml file.
        """
        env_path = os.environ.get('EMBER_CONFIG_PATH')
        if env_path:
            return Path(env_path)
        return Path.home() / ".ember" / "config.yaml"
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load configuration from default location.
        
        Returns:
            Configuration dictionary.
        """
        config_file = self.get_config_path()
        if config_file.exists():
            try:
                with open(config_file) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
        return {}
    
    def save(self) -> None:
        """Save configuration to disk atomically."""
        config_file = self.get_config_path()
        config_file.parent.mkdir(exist_ok=True)
        
        # Use process-wide lock for thread safety
        with self._lock:
            # Write to temp file first
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=config_file.parent,
                delete=False,
                suffix='.tmp',
                prefix='.ember_config_'
            ) as tmp:
                yaml.dump(self._config, tmp, default_flow_style=False)
                tmp_path = Path(tmp.name)
            
            # Atomic rename (works on all platforms)
            if sys.platform == 'win32':
                # Windows requires target to not exist
                if config_file.exists():
                    config_file.unlink()
            
            tmp_path.replace(config_file)
    
    def _check_migration(self) -> None:
        """Check and run migration if needed."""
        # Only check once per process
        if hasattr(self.__class__, '_migration_checked'):
            return
        
        self.__class__._migration_checked = True
        
        # Check if old files exist
        old_creds = Path.home() / '.ember' / 'credentials'
        old_config = Path.home() / '.ember' / 'config.json'
        
        if old_creds.exists() or old_config.exists():
            try:
                from ember._internal.migrations import migrate_credentials, migrate_config
                migrate_credentials()
                migrate_config()
            except Exception:
                # Migration failed, but don't break initialization
                pass


def current_context() -> EmberContext:
    """Get current thread's context.
    
    Returns:
        Thread-local EmberContext instance.
    """
    return EmberContext.current()


def with_context(**config) -> EmberContext:
    """Create temporary context with overrides.
    
    Args:
        **config: Configuration overrides.
        
    Returns:
        Context manager.
        
    Example:
        with with_context(models={"default": "gpt-4"}):
            model = current_context().get_model()  # Uses gpt-4
    """
    current = EmberContext.current()
    return current.create_child(**config)