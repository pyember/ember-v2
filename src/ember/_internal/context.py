"""Centralized context for Ember configuration.

Provides thread-safe configuration and credential management with clean
dependency injection. Single source of truth for runtime configuration.
"""

from __future__ import annotations

import contextvars
import copy
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, cast

import yaml

from ember.core.credentials import CredentialManager
from ember.core.config.loader import load_config
from ember.core.utils.logging import get_logger

if TYPE_CHECKING:
    from ember.models.registry import ModelRegistry
    from ember.core.utils.data.registry import DataRegistry
    from ember._internal.config_types import EmberConfig

logger = get_logger(__name__)


class EmberContext:
    """Central context for configuration and dependency management.
    
    Provides thread-safe and async-safe access to configuration, credentials,
    and registries with support for context isolation and inheritance. 
    Implements lazy initialization for efficient resource usage.
    
    Attributes:
        _thread_local: Thread-local storage for synchronous context isolation.
        _context_var: ContextVar for async-safe context propagation.
    
    Examples:
        Basic usage:
            >>> ctx = EmberContext()
            >>> api_key = ctx.get_credential("openai", "OPENAI_API_KEY")
        
        Child context with overrides:
            >>> with ctx.create_child(models={"default": "gpt-4"}) as child:
            ...     model = child.get_model()  # Uses gpt-4
            
        Async usage:
            >>> async def process():
            ...     ctx = EmberContext.current()
            ...     # Context propagates across async boundaries
    """
    
    # Thread-local storage for synchronous context isolation
    _thread_local = threading.local()
    
    # ContextVar for async-safe context propagation
    _context_var: contextvars.ContextVar[Optional['EmberContext']] = contextvars.ContextVar(
        'ember_context', default=None
    )
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        parent: Optional[EmberContext] = None,
        isolated: bool = False
    ):
        """Initialize EmberContext.
        
        Args:
            config_path: Path to configuration file. If None, loads from
                default location (~/.ember/config.yaml).
            parent: Parent context to inherit configuration from. Child
                contexts receive a deep copy of parent configuration.
            isolated: If True, context is not set as thread-local default.
                Useful for testing and temporary contexts.
                
        Raises:
            OSError: If config_path is provided but cannot be read.
        """
        self._config: Dict[str, Any] = {}
        self._parent = parent
        self._isolated = isolated
        self._lock = threading.RLock()
        self._config_file = self.get_config_path()
        
        # Core components - lazily initialized
        self._credential_manager: Optional[CredentialManager] = None
        self._model_registry: Optional[ModelRegistry] = None
        self._data_registry: Optional[DataRegistry] = None
        
        # Load configuration
        if config_path:
            self._config = load_config(str(config_path))
            self._config_file = config_path
        elif parent:
            # Inherit parent config (deep copy for isolation)
            self._config = copy.deepcopy(parent._config)
            self._config_file = parent._config_file
        else:
            # Try to load from default location
            self._config = self._load_default_config()
            
        # Set as current context if not isolated
        if not isolated and not parent:
            EmberContext._thread_local.context = self
            EmberContext._context_var.set(self)
            
            # Check for migration on first context creation
            self._check_migration()
    
    @classmethod
    def current(cls) -> EmberContext:
        """Get or create the current context (thread-safe and async-safe).
        
        Returns:
            Current EmberContext instance. Creates one if it doesn't exist.
            
        Note:
            In async contexts, uses ContextVar for proper context propagation.
            In sync contexts, falls back to thread-local storage.
            This ensures proper isolation in both threading and async scenarios.
        """
        # Try ContextVar first (works in both sync and async)
        ctx = cls._context_var.get()
        if ctx is not None:
            return ctx
            
        # Fall back to thread-local for pure threading scenarios
        if not hasattr(cls._thread_local, 'context'):
            ctx = cls()
            cls._thread_local.context = ctx
            cls._context_var.set(ctx)
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
                    # Pass config directory to credential manager
                    config_dir = self._config_file.parent if self._config_file else None
                    self._credential_manager = CredentialManager(config_dir)
        return self._credential_manager
    
    @property
    def _credentials(self) -> CredentialManager:
        """Alias for credential_manager for backward compatibility."""
        return self.credential_manager
    
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
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation for nested access
                (e.g., "models.temperature", "providers.openai.base_url").
            default: Value to return if key is not found. Defaults to None.
            
        Returns:
            The configuration value at the specified key, or default if not found.
            
        Examples:
            >>> ctx.get_config("models.default")
            'gpt-3.5-turbo'
            >>> ctx.get_config("missing.key", "fallback")
            'fallback'
        """
        # Handle edge cases
        if not key or not isinstance(key, str):
            return None
            
        parts = key.split(".")
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Dot-notation key (e.g., "models.temperature").
            value: Value to set.
            
        Raises:
            ValueError: If key is invalid or empty.
            TypeError: If key is not a string.
        """
        # Input validation
        if not isinstance(key, str):
            raise TypeError(f"Configuration key must be a string, got {type(key).__name__}")
        
        if not key or not key.strip():
            raise ValueError("Configuration key cannot be empty")
        
        # Basic key validation
        parts = key.split(".")
        if not all(parts):
            raise ValueError(f"Invalid configuration key: '{key}'")
        
        with self._lock:
            config = self._config
            
            # Navigate to parent dict
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                elif not isinstance(config[part], dict):
                    # Replace non-dict values with dict to allow nesting
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
            
        Note:
            In future versions, this will filter out sensitive values
            like API keys before returning. For now, returns complete config.
        """
        if TYPE_CHECKING:
            from ember._internal.config_types import EmberConfig
            return cast(EmberConfig, self._config.copy())
        return self._config.copy()
    
    def reload(self) -> None:
        """Reload configuration from disk."""
        with self._lock:
            if self._config_file and self._config_file.exists():
                try:
                    with open(self._config_file) as f:
                        self._config = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to reload config: {e}")
                    self._config = {}
            else:
                new_config = self._load_default_config()
                self._config = new_config
    
    def save(self) -> None:
        """Save configuration to disk."""
        with self._lock:
            if self._config_file:
                # Ensure directory exists
                self._config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save with atomic write
                temp_file = self._config_file.with_suffix('.tmp')
                try:
                    with open(temp_file, 'w') as f:
                        yaml.dump(self._config, f, default_flow_style=False)
                    temp_file.replace(self._config_file)
                except Exception as e:
                    logger.error(f"Failed to save config: {e}")
                    if temp_file.exists():
                        temp_file.unlink()
                    raise
    
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
            # Handle nested dict updates
            if isinstance(value, dict):
                for k, v in value.items():
                    child.set_config(f"{key}.{k}", v)
            else:
                child.set_config(key, value)
            
        return child
    
    def __enter__(self) -> EmberContext:
        """Context manager entry - sets this as current context."""
        # Always save and set context, even for isolated contexts
        # This allows with_context to work properly
        self._previous_thread = getattr(EmberContext._thread_local, 'context', None)
        self._previous_async = EmberContext._context_var.get()
        
        # Set this as current
        EmberContext._thread_local.context = self
        self._token = EmberContext._context_var.set(self)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - restores previous context."""
        # Always restore previous context
        # Restore thread-local
        if hasattr(self, '_previous_thread'):
            if self._previous_thread is not None:
                EmberContext._thread_local.context = self._previous_thread
            else:
                if hasattr(EmberContext._thread_local, 'context'):
                    delattr(EmberContext._thread_local, 'context')
        
        # Restore context var
        if hasattr(self, '_token'):
            EmberContext._context_var.reset(self._token)
    
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
            path = Path(env_path)
            # If it's a directory, append config.yaml
            if path.is_dir() or not path.suffix:
                return path / "config.yaml"
            return path
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