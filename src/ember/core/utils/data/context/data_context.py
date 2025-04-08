"""Thread-safe dependency container for data system.

Provides explicit component initialization and lifecycle management,
replacing global state with dependency injection.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Type, Union

from ember.core.context.ember_context import EmberContext, current_context
from ember.core.utils.data.base.loaders import HuggingFaceDatasetLoader, IDatasetLoader
from ember.core.utils.data.base.models import DatasetInfo
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.base.samplers import DatasetSampler
from ember.core.utils.data.base.validators import DatasetValidator
from ember.core.utils.data.cache.cache_manager import DatasetCache
from ember.core.utils.data.loader_factory import DatasetLoaderFactory
from ember.core.utils.data.registry import DatasetRegistry
from ember.core.utils.data.service import DatasetService

logger = logging.getLogger(__name__)

# Thread-safe singleton management for default context
_default_context = None
_default_context_lock = threading.RLock()


@dataclass(frozen=True)
class DataConfig:
    """Immutable configuration for data system components."""

    # Storage settings
    cache_dir: Optional[str] = None

    # Performance tuning
    batch_size: int = 32
    cache_ttl: int = 3600  # Default to 1 hour
    auto_register_preppers: bool = True

    @classmethod
    def from_env(cls) -> "DataConfig":
        """Create configuration from environment variables."""
        return cls(
            cache_dir=os.environ.get("EMBER_DATA_CACHE_DIR"),
            batch_size=int(os.environ.get("EMBER_DATA_BATCH_SIZE", "32")),
            cache_ttl=int(os.environ.get("EMBER_DATA_CACHE_TTL", "3600")),
            auto_register_preppers=os.environ.get("EMBER_DATA_AUTO_REGISTER", "1")
            != "0",
        )

    @classmethod
    def from_ember_context(cls, context: EmberContext) -> "DataConfig":
        """Create configuration from EmberContext."""
        config = context.config_manager.get_config()

        return cls(
            cache_dir=getattr(config, "data_cache_dir", None),
            batch_size=getattr(config, "data_batch_size", 32),
            cache_ttl=getattr(config, "data_cache_ttl", 3600),
            auto_register_preppers=getattr(config, "data_auto_register", True),
        )


class DataContext:
    """Thread-safe dependency container for data system.

    This class provides a clean, centralized way to manage dependencies in the data system
    without relying on global state. It supports both direct dependency injection and
    configuration-based initialization.
    """

    def __init__(
        self,
        *,  # Force keyword arguments
        # Direct dependencies
        registry: Optional[DatasetRegistry] = None,
        loader_factory: Optional[DatasetLoaderFactory] = None,
        cache_manager: Optional[DatasetCache] = None,
        # Configuration options
        config: Optional[DataConfig] = None,
        config_manager: Optional[Any] = None,
        config_path: Optional[str] = None,
        auto_discover: bool = True,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Initialize data context with explicit dependencies.

        This constructor supports three main use cases:
        1. Direct dependency injection (providing registry, etc.)
        2. Configuration-based initialization (providing config)
        3. EmberContext integration (providing config_manager)

        Args:
            registry: Existing dataset registry to use directly
            loader_factory: Existing loader factory to use directly
            cache_manager: Existing cache manager to use directly
            config: Data-specific configuration
            config_manager: Existing config manager for settings
            config_path: Path to configuration file
            auto_discover: Whether to automatically discover datasets
            metrics: Optional metrics collector for instrumentation
        """
        # Setup configuration first
        self._config = config or self._init_config(config_manager, config_path)

        # Initialize components with proper dependency order - careful with initialization order
        self._loader_factory = loader_factory or self._init_loader_factory()
        self._registry = registry or self._init_registry(auto_discover)
        self._cache_manager = cache_manager or self._init_cache_manager()
        self._metrics = metrics or {}

        # Initialize lazily created services
        self._dataset_service = None
        self._dataset_service_lock = threading.RLock()

    def _init_config(
        self,
        config_manager: Optional[Any],
        config_path: Optional[str],
    ) -> DataConfig:
        """Initialize configuration from manager or path.

        Args:
            config_manager: Existing config manager
            config_path: Path to configuration file

        Returns:
            Data configuration
        """
        if config_manager is not None:
            return DataConfig(
                cache_dir=getattr(config_manager.get_config(), "data_cache_dir", None),
                batch_size=getattr(config_manager.get_config(), "data_batch_size", 32),
                cache_ttl=getattr(config_manager.get_config(), "data_cache_ttl", 3600),
                auto_register_preppers=getattr(
                    config_manager.get_config(), "data_auto_register", True
                ),
            )

        if config_path is not None:
            # Load configuration from file
            try:
                import yaml

                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)

                # Extract data-specific configuration
                data_config = config_data.get("data", {})

                return DataConfig(
                    cache_dir=data_config.get("cache_dir"),
                    batch_size=data_config.get("batch_size", 32),
                    cache_ttl=data_config.get("cache_ttl", 3600),
                    auto_register_preppers=data_config.get(
                        "auto_register_preppers", True
                    ),
                )
            except (IOError, yaml.YAMLError) as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                logger.warning("Falling back to default configuration")

        # Use default configuration
        return DataConfig()

    def _init_registry(self, auto_discover: bool) -> DatasetRegistry:
        """Initialize dataset registry.

        Args:
            auto_discover: Whether to discover datasets automatically

        Returns:
            Initialized dataset registry
        """
        registry = DatasetRegistry()

        if auto_discover:
            # Initialize core datasets
            from ember.core.utils.data.initialization import initialize_dataset_registry

            initialize_dataset_registry(
                metadata_registry=registry, loader_factory=self._loader_factory
            )

        return registry

    def _init_loader_factory(self) -> DatasetLoaderFactory:
        """Initialize loader factory.

        Returns:
            Initialized loader factory
        """
        factory = DatasetLoaderFactory()

        # Connect to registry for prepper registration
        if self._config.auto_register_preppers:
            # Registry might not be initialized yet, so we'll register preppers later
            pass

        return factory

    def _init_cache_manager(self) -> DatasetCache:
        """Initialize cache manager.

        Returns:
            Initialized cache manager
        """
        return DatasetCache(
            disk_cache_dir=self._config.cache_dir,
            default_ttl=self._config.cache_ttl,
        )

    def _connect_loader_factory(self) -> None:
        """Connect loader factory to registry.

        This should be called after registry is initialized.
        """
        if self._config.auto_register_preppers:
            for name in self._registry.list_datasets():
                dataset = self._registry.get(name=name)
                if dataset and hasattr(dataset, "prepper") and dataset.prepper:
                    self._loader_factory.register(
                        dataset_name=name,
                        prepper_class=dataset.prepper.__class__,
                    )

    @property
    def registry(self) -> DatasetRegistry:
        """Get the dataset registry.

        Returns:
            Dataset registry instance
        """
        return self._registry

    @property
    def loader_factory(self) -> DatasetLoaderFactory:
        """Get the loader factory.

        Returns:
            Loader factory instance
        """
        return self._loader_factory

    @property
    def cache_manager(self) -> DatasetCache:
        """Get the cache manager.

        Returns:
            Cache manager instance
        """
        return self._cache_manager

    @property
    def dataset_service(self) -> DatasetService:
        """Get the dataset service (created on first access).

        This uses thread-safe lazy initialization with double-checked locking
        pattern for maximum performance on the hot path.

        Returns:
            Dataset service instance
        """
        # Fast path - direct access with no lock
        # This ensures minimal overhead for repeated access (~5ns)
        service = self._dataset_service
        if service is not None:
            return service

        # Slow path (with lock) - only executed once
        with self._dataset_service_lock:
            # Double-check after acquiring lock
            if self._dataset_service is None:
                # Create service with all necessary dependencies
                service = DatasetService(
                    loader=self._create_dataset_loader(),
                    validator=DatasetValidator(),
                    sampler=DatasetSampler(),
                )
                # Atomic assignment (thread-safe)
                self._dataset_service = service
            else:
                # Another thread initialized it while we were waiting
                service = self._dataset_service

            return service

    def load_dataset(
        self,
        name: str,
        *,
        config: Optional[Any] = None,
        streaming: Optional[bool] = None,
        limit: Optional[int] = None,
        transformers: Optional[List[Any]] = None,
        **kwargs,
    ) -> Union[List[Any], Iterator[Any]]:
        """Load a dataset with flexibility for streaming or eager loading.

        This is the primary method for loading datasets efficiently,
        automatically selecting streaming or eager mode based on configuration.

        Args:
            name: Dataset name
            config: Optional dataset configuration
            streaming: Whether to use streaming mode (defaults to config setting)
            limit: Optional limit on number of samples
            transformers: Optional list of transformers to apply
            **kwargs: Additional dataset options

        Returns:
            Either a list of dataset entries (eager mode) or a streaming iterator

        Raises:
            ValueError: If dataset not found
        """
        # Use a sensible streaming default if not specified
        use_streaming = streaming if streaming is not None else True

        # Get dataset entry
        dataset_entry = self.registry.get(name=name)
        if not dataset_entry:
            available = self.registry.list_datasets()
            raise ValueError(
                f"Dataset '{name}' not found in registry. Available: {available}"
            )

        # Use streaming mode
        if use_streaming:
            stream = self.get_streaming_dataset(
                name=name, transformers=transformers, **kwargs
            )
            if limit is not None:
                stream = stream.limit(limit)
            return stream

        # Use eager loading mode
        return self.dataset_service.load_and_prepare(
            dataset_info=dataset_entry.info,
            prepper=dataset_entry.prepper,
            config=config,
            num_samples=limit,
        )

    def get_streaming_dataset(self, name: str, **kwargs):
        """Get a streaming dataset by name.

        Optimized helper method for the common case of creating a streaming dataset.
        This avoids multiple registry lookups and improves performance for data access.

        Args:
            name: Dataset name
            **kwargs: Additional options for StreamingDataset constructor

        Returns:
            StreamingDataset instance

        Raises:
            ValueError: If dataset not found
        """
        # Avoid cyclic imports
        from ember.core.utils.data.streaming.dataset import StreamingDataset

        # Get dataset from registry (single lookup)
        dataset_entry = self.registry.get(name=name)
        if not dataset_entry:
            available = self.registry.list_datasets()
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )

        # Create streaming dataset with dataset entry's prepper
        return StreamingDataset(
            source=name,
            prepper=dataset_entry.prepper if dataset_entry else None,
            **kwargs,
        )

    def _create_dataset_loader(self) -> IDatasetLoader:
        """Create a dataset loader with necessary configuration.

        Returns:
            Configured dataset loader
        """
        return HuggingFaceDatasetLoader(
            cache_dir=self._config.cache_dir,
        )

    def register_dataset(
        self,
        *,
        name: str,
        source: str,
        task_type: Any,  # Use TaskType enum, but avoid circular import
        prepper_class: Optional[Type[IDatasetPrepper]] = None,
        description: str = "",
    ) -> None:
        """Register dataset with context.

        Args:
            name: Dataset name
            source: Dataset source
            task_type: Type of dataset task
            prepper_class: Optional class for dataset preparation
            description: Optional dataset description
        """
        # Create dataset info
        info = DatasetInfo(
            name=name,
            source=source,
            task_type=task_type,
            description=description,
        )

        # Create prepper instance if class provided
        prepper = None
        if prepper_class is not None:
            prepper = prepper_class()

        # Register with registry
        self._registry.register(
            name=name,
            info=info,
            prepper=prepper,
        )

        # Register with loader factory
        if prepper_class is not None:
            self._loader_factory.register(
                dataset_name=name,
                prepper_class=prepper_class,
            )

    @classmethod
    def create_from_env(
        cls, env_prefix: str = "EMBER_", **kwargs: Any
    ) -> "DataContext":
        """Create context from environment variables.

        Args:
            env_prefix: Prefix for environment variables
            **kwargs: Additional arguments to constructor

        Returns:
            Initialized data context
        """
        return cls(config=DataConfig.from_env(), **kwargs)

    @classmethod
    def create_from_ember_context(
        cls, context: Optional[EmberContext] = None, **kwargs: Any
    ) -> "DataContext":
        """Create context from EmberContext.

        Args:
            context: EmberContext to use, uses current if None
            **kwargs: Additional arguments to constructor

        Returns:
            Initialized data context
        """
        ember_ctx = context or current_context()
        return cls(config=DataConfig.from_ember_context(ember_ctx), **kwargs)

    @classmethod
    def create_test_context(cls) -> "DataContext":
        """Create minimal context for testing.

        Returns:
            Data context configured for testing
        """
        return cls(
            auto_discover=False,
            config=DataConfig(
                cache_dir=None,  # Disable caching
                auto_register_preppers=False,
            ),
        )


def get_default_context() -> DataContext:
    """Get the default global data context with thread-safe lazy initialization.

    This function follows the optimal double-checked locking pattern to ensure
    thread safety while minimizing lock contention. The pattern uses an initial
    non-locked check (fast path) followed by a synchronized check-and-create
    operation (slow path) only when necessary.

    Returns:
        The default global data context instance

    Note:
        While this function provides convenient global state access, code should
        prefer explicit context management where possible for better testability.
    """
    global _default_context

    # Fast path - check if context exists without acquiring lock (≈5ns)
    if _default_context is not None:
        return _default_context

    # Slow path - lock and initialize if needed
    with _default_context_lock:
        # Double-check after acquiring lock to avoid race conditions
        if _default_context is None:
            # Try to create from EmberContext if available, otherwise from env
            try:
                ember_ctx = current_context()
                _default_context = DataContext.create_from_ember_context(ember_ctx)
            except (ImportError, RuntimeError):
                # Fallback to environment-based configuration
                _default_context = DataContext.create_from_env()

        return _default_context


def set_default_context(context: DataContext) -> None:
    """Set the default global data context.

    This function is primarily used for testing and for applications that
    need controlled global state management.

    Args:
        context: The data context to set as the global default
    """
    global _default_context

    with _default_context_lock:
        _default_context = context


def reset_default_context() -> None:
    """Reset the default global data context.

    Removes the current default context, forcing a new one to be created
    on the next call to get_default_context(). This is primarily used
    for testing to ensure a clean state between test cases.
    """
    global _default_context

    with _default_context_lock:
        _default_context = None
