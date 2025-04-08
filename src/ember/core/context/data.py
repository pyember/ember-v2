"""Data management component.

This module provides a component for managing datasets and their
creation, transformation, and access with thread-safe lazy initialization.
"""

import importlib
import logging
from typing import Any, Dict, List, Optional

from .component import Component
from .registry import Registry


class DataComponent(Component):
    """Data management component.

    This component is responsible for dataset discovery, registration,
    and retrieval with efficient caching and thread safety.

    Features:
    - Lazy loading: Datasets are only initialized when first accessed
    - Thread safety: Safe for concurrent access
    - Configuration-based: Datasets can be defined in configuration
    - Transformation pipeline: Datasets can be transformed
    """

    def __init__(self, registry: Optional[Registry] = None):
        """Initialize with registry.

        Args:
            registry: Registry to use (current thread's if None)
        """
        super().__init__(registry)
        self._datasets: Dict[str, Any] = {}
        self._dataset_factories: Dict[str, Any] = {}
        self._logger = logging.getLogger("ember.data")

    def _register(self) -> None:
        """Register in registry as 'data'."""
        self._registry.register("data", self)

    def get_dataset(self, dataset_id: str) -> Optional[Any]:
        """Get dataset by ID.

        This method follows a multi-tiered lookup strategy:
        1. Look in the datasets cache
        2. Try to instantiate from registered factories
        3. Try to create from configuration

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset or None if not found
        """
        self._ensure_initialized()

        # Return cached dataset if available
        if dataset_id in self._datasets:
            return self._datasets[dataset_id]

        # Try to create from factory
        if dataset_id in self._dataset_factories:
            with self._lock:
                # Double-check after acquiring lock
                if dataset_id not in self._datasets:
                    try:
                        factory = self._dataset_factories[dataset_id]
                        dataset = factory()
                        self._datasets[dataset_id] = dataset
                    except Exception as e:
                        self._logger.error(
                            f"Error creating dataset '{dataset_id}': {e}"
                        )
                        return None
            return self._datasets.get(dataset_id)

        # Try to load from configuration
        config = self._registry.get("config")
        if config:
            dataset_config = config.get_config("datasets").get(dataset_id, {})
            if dataset_config:
                with self._lock:
                    # Double-check after acquiring lock
                    if dataset_id not in self._datasets:
                        dataset = self._create_dataset_from_config(
                            dataset_id, dataset_config
                        )
                        if dataset:
                            self._datasets[dataset_id] = dataset
                return self._datasets.get(dataset_id)

        return None

    def register_dataset(self, dataset_id: str, dataset: Any) -> None:
        """Register a dataset.

        Args:
            dataset_id: Dataset identifier
            dataset: Dataset instance
        """
        self._ensure_initialized()
        with self._lock:
            self._datasets[dataset_id] = dataset

    def register_dataset_factory(self, dataset_id: str, factory: Any) -> None:
        """Register a dataset factory function.

        Args:
            dataset_id: Dataset identifier
            factory: Factory function that creates dataset
        """
        self._ensure_initialized()
        with self._lock:
            self._dataset_factories[dataset_id] = factory

    def list_datasets(self) -> List[str]:
        """List available dataset IDs.

        Returns:
            List of registered dataset IDs
        """
        self._ensure_initialized()
        return sorted(
            list(
                set(list(self._datasets.keys()) + list(self._dataset_factories.keys()))
            )
        )

    def _initialize(self) -> None:
        """Initialize datasets from configuration.

        This loads dataset definitions from configuration and
        prepares them for lazy instantiation.
        """
        config = self._registry.get("config")
        if not config:
            return

        dataset_configs = config.get_config("datasets")
        if not dataset_configs:
            return

        for dataset_id, dataset_config in dataset_configs.items():
            # Skip datasets that are already registered
            if dataset_id in self._datasets or dataset_id in self._dataset_factories:
                continue

            # Register dataset factory for lazy instantiation
            try:
                self._register_dataset_factory(dataset_id, dataset_config)
            except Exception as e:
                self._logger.error(f"Error registering dataset '{dataset_id}': {e}")

    def _register_dataset_factory(
        self, dataset_id: str, dataset_config: Dict[str, Any]
    ) -> None:
        """Register dataset factory from configuration.

        Args:
            dataset_id: Dataset identifier
            dataset_config: Dataset configuration
        """
        dataset_type = dataset_config.get("type")
        if not dataset_type:
            self._logger.error(f"No type specified for dataset {dataset_id}")
            return

        # Create factory function
        def factory():
            return self._create_dataset_from_config(dataset_id, dataset_config)

        # Register factory
        self._dataset_factories[dataset_id] = factory

    def _create_dataset_from_config(
        self, dataset_id: str, dataset_config: Dict[str, Any]
    ) -> Optional[Any]:
        """Create a dataset from configuration.

        Args:
            dataset_id: Dataset identifier
            dataset_config: Dataset configuration

        Returns:
            Dataset or None if creation failed
        """
        dataset_type = dataset_config.get("type")
        if not dataset_type:
            self._logger.error(f"No type specified for dataset {dataset_id}")
            return None

        try:
            # Import dataset module
            module_path = dataset_config.get(
                "module", f"ember.core.utils.data.datasets_registry.{dataset_type}"
            )
            module = importlib.import_module(module_path)

            # Get dataset class
            dataset_class = getattr(module, f"{dataset_type.capitalize()}Dataset")

            # Create dataset
            dataset = dataset_class(dataset_config)
            return dataset
        except (ImportError, AttributeError) as e:
            self._logger.error(f"Failed to create dataset {dataset_id}: {e}")
            return None

    def transform_dataset(
        self, dataset: Any, transformations: List[Dict[str, Any]]
    ) -> Any:
        """Apply transformations to a dataset.

        Args:
            dataset: Source dataset
            transformations: List of transformation configurations

        Returns:
            Transformed dataset
        """
        result = dataset

        for transform_config in transformations:
            transform_type = transform_config.get("type")
            if not transform_type:
                self._logger.warning("Transformation missing type, skipping")
                continue

            try:
                # Import transformer module
                module_path = transform_config.get(
                    "module", "ember.core.utils.data.base.transformers"
                )
                module = importlib.import_module(module_path)

                # Get transformer class
                transformer_class = getattr(
                    module, f"{transform_type.capitalize()}Transformer"
                )

                # Create and apply transformer
                transformer = transformer_class(**transform_config.get("params", {}))
                result = transformer.transform(result)
            except (ImportError, AttributeError) as e:
                self._logger.error(
                    f"Failed to apply transformation {transform_type}: {e}"
                )

        return result
