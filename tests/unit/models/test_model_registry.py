"""Test the ModelRegistry behavior, not implementation.

Following CLAUDE.md principles:
- Focus on behavior, not implementation
- Test thread safety and caching
- Clear test structure
- NO PRIVATE ATTRIBUTE ACCESS
- Use parameterized tests
- Use test doubles instead of mocks
"""

import pytest
import threading
import queue
from unittest.mock import patch

from ember.models.registry import ModelRegistry
from ember._internal.exceptions import ModelNotFoundError, ModelProviderError

# Import our test infrastructure
from tests.test_constants import Models, APIKeys, ErrorPatterns
from tests.test_doubles import FakeProvider, FakeModelRegistry, create_registry_with_models
from tests.fixtures import fake_provider, fake_registry, assert_error_matches


class TestModelRegistry:
    """Test the ModelRegistry class behavior."""

    def test_initialization(self):
        """Test registry starts with no models."""
        registry = ModelRegistry()
        
        # Should start empty (test behavior, not _models attribute)
        assert registry.list_models() == []

    @pytest.mark.parametrize("model_id", [
        pytest.param(Models.GPT4, id="gpt4"),
        pytest.param(Models.CLAUDE3, id="claude3"),
        pytest.param(Models.GEMINI_PRO, id="gemini"),
    ])
    def test_get_model_creates_and_caches(self, model_id):
        """Test lazy model instantiation and caching behavior."""
        registry = ModelRegistry()
        
        # Mock the provider resolution
        with patch("ember.models.providers.resolve_model_id") as mock_resolve:
            with patch("ember.models.providers.get_provider_class") as mock_get_class:
                # Setup mocks to return a fake provider
                if "gpt" in model_id:
                    provider_name = "openai"
                elif "claude" in model_id:
                    provider_name = "anthropic"
                else:
                    provider_name = "google"
                    
                mock_resolve.return_value = (provider_name, model_id)
                mock_get_class.return_value = FakeProvider
                
                # First access
                env_key = f"{provider_name.upper()}_API_KEY"
                with patch.dict("os.environ", {env_key: "test-key"}):
                    model1 = registry.get_model(model_id)
                    
                    # Should return a model
                    assert model1 is not None
                    
                    # Second access should return same instance (caching behavior)
                    model2 = registry.get_model(model_id)
                    assert model1 is model2

    def test_thread_safety_behavior(self):
        """Test concurrent access returns consistent results."""
        registry = ModelRegistry()
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def get_model_thread():
            try:
                # Use patch to ensure consistent model creation
                with patch("ember.models.providers.resolve_model_id") as mock_resolve:
                    with patch("ember.models.providers.get_provider_class") as mock_get_class:
                        mock_resolve.return_value = ("openai", Models.GPT4)
                        mock_get_class.return_value = FakeProvider
                        
                        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                            model = registry.get_model(Models.GPT4)
                            results_queue.put(id(model))  # Store object ID to check same instance
            except Exception as e:
                errors_queue.put(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=get_model_thread) for _ in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert errors_queue.empty(), "No errors should occur"
        
        # All threads should get same model instance
        model_ids = []
        while not results_queue.empty():
            model_ids.append(results_queue.get())
            
        assert len(model_ids) == 10
        assert all(mid == model_ids[0] for mid in model_ids), "All threads should get same instance"

    def test_clear_cache_behavior(self):
        """Test that clear_cache removes cached models."""
        registry = ModelRegistry()
        
        # Setup environment
        with patch("ember.models.providers.resolve_model_id") as mock_resolve:
            with patch("ember.models.providers.get_provider_class") as mock_get_class:
                mock_resolve.return_value = ("openai", Models.GPT4)
                mock_get_class.return_value = FakeProvider
                
                with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                    # Get a model (should cache it)
                    model1 = registry.get_model(Models.GPT4)
                    
                    # Clear cache
                    registry.clear_cache()
                    
                    # Get model again - should be different instance
                    model2 = registry.get_model(Models.GPT4)
                    
                    # Different instances indicate cache was cleared
                    assert model1 is not model2

    def test_list_models_behavior(self):
        """Test listing cached models."""
        registry = ModelRegistry()
        
        # Initially empty
        assert registry.list_models() == []
        
        # Add some models by getting them
        with patch("ember.models.providers.resolve_model_id") as mock_resolve:
            with patch("ember.models.providers.get_provider_class") as mock_get_class:
                mock_get_class.return_value = FakeProvider
                
                # Get different models
                for model_id in [Models.GPT4, Models.CLAUDE3]:
                    provider = "openai" if "gpt" in model_id else "anthropic"
                    mock_resolve.return_value = (provider, model_id)
                    
                    with patch.dict("os.environ", {f"{provider.upper()}_API_KEY": "test-key"}):
                        registry.get_model(model_id)
        
        # Should list the cached models
        models = registry.list_models()
        assert len(models) == 2
        assert Models.GPT4 in models
        assert Models.CLAUDE3 in models

    @pytest.mark.parametrize("error_scenario,expected_error,error_pattern", [
        pytest.param(
            "unknown_provider",
            ModelNotFoundError,
            ErrorPatterns.INVALID_MODEL,
            id="unknown-provider"
        ),
        pytest.param(
            "missing_api_key", 
            ModelProviderError,
            ErrorPatterns.MISSING_API_KEY,
            id="missing-api-key"
        ),
    ])
    def test_error_scenarios(self, error_scenario, expected_error, error_pattern):
        """Test various error scenarios."""
        registry = ModelRegistry()
        
        if error_scenario == "unknown_provider":
            with patch("ember.models.providers.resolve_model_id") as mock_resolve:
                mock_resolve.return_value = ("unknown", "some-model")
                
                with pytest.raises(expected_error) as exc_info:
                    registry.get_model("some-model")
                    
                assert error_pattern.search(str(exc_info.value))
                
        elif error_scenario == "missing_api_key":
            # Test that missing API key raises ModelProviderError
            # Clear all env vars
            with patch.dict("os.environ", {}, clear=True):
                # Mock the interactive setup to return None (no key provided)
                with patch("ember.core.setup_launcher.launch_setup_if_needed", return_value=None):
                    # Also need to mock credential manager if registry uses it
                    with patch.object(registry, "_get_api_key", return_value=None):
                        with pytest.raises(expected_error) as exc_info:
                            registry.get_model(Models.GPT4)
                            
                        assert error_pattern.search(str(exc_info.value))

    @pytest.mark.parametrize("env_config,expected_key", [
        pytest.param(
            {"OPENAI_API_KEY": "standard-key"},
            "standard-key",
            id="standard-env-var"
        ),
        pytest.param(
            {"EMBER_OPENAI_API_KEY": "ember-key"},
            "ember-key",
            id="ember-prefixed-var"
        ),
        pytest.param(
            {"OPENAI_API_KEY": "standard", "EMBER_OPENAI_API_KEY": "ember"},
            "standard",
            id="standard-takes-precedence"
        ),
    ])
    def test_api_key_resolution(self, env_config, expected_key):
        """Test API key resolution from environment."""
        registry = ModelRegistry()
        
        with patch("ember.models.providers.resolve_model_id") as mock_resolve:
            with patch("ember.models.providers.get_provider_class") as mock_get_class:
                mock_resolve.return_value = ("openai", Models.GPT4)
                
                # Mock the provider class to capture the API key
                class CapturingProvider:
                    def __init__(self, api_key=None, **kwargs):
                        self.api_key = api_key
                
                mock_get_class.return_value = CapturingProvider
                
                with patch.dict("os.environ", env_config, clear=True):
                    model = registry.get_model(Models.GPT4)
                    
                    # Check the provider got the expected key
                    assert model.api_key == expected_key


    @pytest.mark.parametrize("num_threads", [
        pytest.param(10, id="10-threads"),
        pytest.param(20, id="20-threads"),
    ])
    def test_concurrent_different_models(self, num_threads):
        """Test concurrent access to different models."""
        registry = ModelRegistry()
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        # Use a fixed set of valid models
        test_models = [Models.GPT4, Models.GPT35, Models.CLAUDE3]
        
        def get_model_thread(model_idx):
            try:
                model_id = test_models[model_idx % len(test_models)]
                
                with patch("ember.models.providers.resolve_model_id") as mock_resolve:
                    with patch("ember.models.providers.get_provider_class") as mock_get_class:
                        provider = "openai" if "gpt" in model_id else "anthropic"
                        mock_resolve.return_value = (provider, model_id)
                        mock_get_class.return_value = FakeProvider
                        
                        env_key = f"{provider.upper()}_API_KEY"
                        with patch.dict("os.environ", {env_key: "test-key"}):
                            model = registry.get_model(model_id)
                            results_queue.put((model_id, id(model)))
            except Exception as e:
                errors_queue.put(e)
        
        # Create threads accessing different models
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=get_model_thread, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        if not errors_queue.empty():
            errors = []
            while not errors_queue.empty():
                errors.append(str(errors_queue.get()))
            pytest.fail(f"Errors occurred: {errors}")
        
        # Group results by model
        model_instances = {}
        while not results_queue.empty():
            model_id, instance_id = results_queue.get()
            if model_id not in model_instances:
                model_instances[model_id] = set()
            model_instances[model_id].add(instance_id)
        
        # Each model should have only one instance (proper caching)
        for model_id, instances in model_instances.items():
            assert len(instances) == 1, f"Model {model_id} should have single instance"