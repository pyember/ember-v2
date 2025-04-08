"""Tests for the new context registry system."""

import threading
import pytest

from ember.core.context.registry import Registry
from ember.core.context.component import Component
from ember.core.context.config import ConfigComponent
from ember.core.context.model import ModelComponent
from ember.core.context.data import DataComponent
from ember.core.context.metrics import MetricsComponent
from ember.core.context.management import scoped_registry, temp_component


class TestRegistry:
    """Test the Registry class."""

    def setup_method(self):
        """Set up test method."""
        # Clear any existing registry
        Registry.clear()

    def test_registry_isolation(self):
        """Test that registries are thread-local."""
        # Create registry in main thread
        main_registry = Registry.current()
        main_registry.register("test", "main")

        # Access from another thread
        thread_result = {}

        def thread_func():
            thread_registry = Registry.current()
            thread_registry.register("test", "thread")
            thread_result["test"] = thread_registry.get("test")

        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join()

        # Check that each thread has its own registry
        assert main_registry.get("test") == "main"
        assert thread_result["test"] == "thread"

    def test_component_registration(self):
        """Test component registration and retrieval."""
        # Create registry
        registry = Registry()

        # Register components
        registry.register("test1", "value1")
        registry.register("test2", "value2")

        # Retrieve components
        assert registry.get("test1") == "value1"
        assert registry.get("test2") == "value2"
        assert registry.get("nonexistent") is None

    def test_scoped_registry(self):
        """Test scoped registry context manager."""
        # Register in main registry
        main_registry = Registry.current()
        main_registry.register("test", "main")

        # Create scoped registry
        with scoped_registry() as registry:
            assert registry.get("test") is None
            registry.register("test", "scoped")
            assert registry.get("test") == "scoped"

        # Check that main registry is restored
        assert Registry.current() is main_registry
        assert Registry.current().get("test") == "main"

    def test_temp_component(self):
        """Test temporary component context manager."""
        # Register in main registry
        registry = Registry.current()
        registry.register("test", "original")

        # Temporarily replace
        with temp_component("test", "replacement") as component:
            assert component == "replacement"
            assert registry.get("test") == "replacement"

        # Check that original is restored
        assert registry.get("test") == "original"


class TestComponents:
    """Test the core components."""

    def setup_method(self):
        """Set up test method."""
        # Clear any existing registry
        Registry.clear()

    def test_config_component(self):
        """Test ConfigComponent."""
        # Create with direct config data
        config_data = {
            "section1": {"key1": "value1", "key2": "value2"},
            "section2": {"key3": "value3"},
        }
        config = ConfigComponent(config_data=config_data)

        # Test retrieval
        assert config.get_config("section1") == {"key1": "value1", "key2": "value2"}
        assert config.get_value("section1", "key1") == "value1"
        assert config.get_value("section1", "nonexistent", "default") == "default"

        # Test registry access
        registry = Registry.current()
        assert registry.get("config") is config

    def test_model_component(self):
        """Test ModelComponent."""
        # Create registry
        registry = Registry.current()

        # Create config with model configuration
        config_data = {"models": {"test_model": {"type": "mock", "provider": "mock"}}}
        ConfigComponent(config_data=config_data)

        # Create model component
        model = ModelComponent()

        # Register a test model
        model.register_model("direct_model", "model_instance")

        # Test retrieval
        assert model.get_model("direct_model") == "model_instance"

        # Test registry access
        assert registry.get("model") is model

    def test_component_interaction(self):
        """Test interaction between components."""
        # Create components
        config = ConfigComponent(
            config_data={"models": {"configured_model": {"type": "test"}}}
        )

        model = ModelComponent()
        model.register_model("test_model", "model_instance")

        data = DataComponent()
        data.register_dataset("test_dataset", "dataset_instance")

        metrics = MetricsComponent()

        # Test component lookup
        registry = Registry.current()
        assert registry.get("config") is config
        assert registry.get("model") is model
        assert registry.get("data") is data
        assert registry.get("metrics") is metrics

        # Test lookup by type
        assert model.get_model("test_model") == "model_instance"
        assert data.get_dataset("test_dataset") == "dataset_instance"

        # Test metrics recording
        metrics.counter("test_counter")
        metrics.gauge("test_gauge", 42.0)

        component_metrics = metrics.get_component_metrics("test_component")
        component_metrics.counter("component_counter")

        all_metrics = metrics.get_metrics()
        assert "test_counter" in all_metrics["counters"]
        assert "test_gauge" in all_metrics["gauges"]
        assert "test_component.component_counter" in all_metrics["counters"]
