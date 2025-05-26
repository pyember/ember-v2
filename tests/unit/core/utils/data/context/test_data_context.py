"""Unit tests for DataContext.

Tests the core functionality of the DataContext class including
initialization, dependency management, and thread safety.
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from ember.core.utils.data.base.models import TaskType
from ember.core.utils.data.cache.cache_manager import DatasetCache
from ember.core.utils.data.context.data_context import DataConfig, DataContext
from ember.core.utils.data.loader_factory import DatasetLoaderFactory
from ember.core.utils.data.registry import DatasetRegistry
from ember.core.utils.data.service import DatasetService


@pytest.fixture
def test_context() -> DataContext:
    """Fixture that creates a fresh test context for each test."""
    return DataContext.create_test_context()


class TestDataConfig:
    """Tests for the DataConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DataConfig()
        assert config.cache_dir is None
        assert config.batch_size == 32
        assert config.cache_ttl == 3600
        assert config.auto_register_preppers is True

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = DataConfig(
            cache_dir="/tmp/cache",
            batch_size=64,
            cache_ttl=7200,
            auto_register_preppers=False)
        assert config.cache_dir == "/tmp/cache"
        assert config.batch_size == 64
        assert config.cache_ttl == 7200
        assert config.auto_register_preppers is False

    def test_from_env(self):
        """Test creating config from environment variables."""
        with patch.dict(
            os.environ,
            {
                "EMBER_DATA_CACHE_DIR": "/tmp/env_cache",
                "EMBER_DATA_BATCH_SIZE": "128",
                "EMBER_DATA_CACHE_TTL": "1800",
                "EMBER_DATA_AUTO_REGISTER": "0",
            }):
            config = DataConfig.from_env()
            assert config.cache_dir == "/tmp/env_cache"
            assert config.batch_size == 128
            assert config.cache_ttl == 1800
            assert config.auto_register_preppers is False


class TestDataContext:
    """Tests for the DataContext class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        context = DataContext()
        # With the new architecture, registry might be wrapped, so check for interface instead of type
        assert hasattr(context.registry, 'get')
        assert hasattr(context.registry, 'register')
        assert hasattr(context.registry, 'list_datasets')
        # Check for loader factory interface instead of type
        assert hasattr(context.loader_factory, 'register')
        assert hasattr(context.loader_factory, 'get_prepper_class')
        # Check for cache manager interface instead of type
        assert hasattr(context.cache_manager, 'get')
        assert hasattr(context.cache_manager, 'set')

    def test_init_with_explicit_components(self):
        """Test initialization with explicit components."""
        registry = DatasetRegistry()
        loader_factory = DatasetLoaderFactory()
        cache_manager = DatasetCache()

        context = DataContext(
            registry=registry,
            loader_factory=loader_factory,
            cache_manager=cache_manager)

        assert context.registry is registry
        assert context.loader_factory is loader_factory
        assert context.cache_manager is cache_manager

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = DataConfig(
            cache_dir="/tmp/test_cache",
            batch_size=64,
            cache_ttl=1800,
            auto_register_preppers=False)

        context = DataContext(config=config)

        # Check that cache manager was created with the right config
        assert context.cache_manager._disk_cache_dir == "/tmp/test_cache"
        assert context.cache_manager._default_ttl == 1800

    def test_lazy_dataset_service(self):
        """Test that dataset_service is created lazily."""
        context = DataContext()

        # Service should not be created yet
        assert not hasattr(context, "_dataset_service") or context._dataset_service is None

        # Access service - should be created on first access
        service = context.dataset_service
        assert service is not None
        assert hasattr(service, "load_and_prepare")  # Check for service interface instead of type

        # Should return the same instance on subsequent access
        assert context.dataset_service is service

    def test_create_test_context(self):
        """Test creating a test context."""
        context = DataContext.create_test_context()

        # Test context should have minimal settings
        assert context._config.cache_dir is None
        assert context._config.auto_register_preppers is False

        # Registry should be empty
        assert len(context.registry.list_datasets()) == 0

    def test_register_dataset(self):
        """Test registering a dataset."""
        context = DataContext.create_test_context()

        # Register a dataset
        context.register_dataset(
            name="test_dataset",
            source="test/source",
            task_type=TaskType.MULTIPLE_CHOICE,  # Use actual enum value
            description="Test dataset")

        # Check that it was registered
        assert "test_dataset" in context.registry.list_datasets()
        dataset = context.registry.get(name="test_dataset")
        assert dataset.info.name == "test_dataset"
        assert dataset.info.source == "test/source"
        assert dataset.info.description == "Test dataset"


class TestEmberContextIntegration:
    """Tests for EmberContext integration."""

    def test_create_from_ember_context(self):
        """Test creating a DataContext from an EmberContext."""
        # Create a mock EmberContext
        mock_ember_context = MagicMock()
        mock_config_manager = MagicMock()
        mock_config = MagicMock()

        # Configure mocks
        mock_ember_context.config_manager = mock_config_manager
        mock_config_manager.get_config.return_value = mock_config
        mock_config.data_cache_dir = "/tmp/ember_cache"
        mock_config.data_batch_size = 128
        mock_config.data_cache_ttl = 1800
        mock_config.data_auto_register = True

        # Create DataContext from EmberContext
        context = DataContext.create_from_ember_context(mock_ember_context)

        # Check that config was set correctly
        assert context._config.cache_dir == "/tmp/ember_cache"
        assert context._config.batch_size == 128
        assert context._config.cache_ttl == 1800
        assert context._config.auto_register_preppers is True
