#!/usr/bin/env python3
"""Test centralized configuration system."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ember._internal.context import EmberContext, current_context
from ember._internal.config.manager import create_config_manager
from ember.models.registry import ModelRegistry


def test_context_system():
    """Test that context properly manages configuration."""
    print("Testing centralized configuration system...")
    
    # Test 1: Context creation and singleton
    ctx1 = EmberContext.current()
    ctx2 = current_context()
    assert ctx1 is ctx2, "Context should be singleton per thread"
    print("✓ Context singleton working")
    
    # Test 2: Configuration management
    ctx1.set_config("test.value", 42)
    assert ctx1.get_config("test.value") == 42
    assert ctx1.get_config("test.missing", "default") == "default"
    print("✓ Configuration get/set working")
    
    # Test 3: Credential precedence
    # Set test env var
    os.environ["TEST_PROVIDER_API_KEY"] = "env-key"
    
    # Store in credentials
    ctx1.credential_manager.store("test_provider", "file-key")
    
    # Env should take precedence
    key = ctx1.get_credential("test_provider", "TEST_PROVIDER_API_KEY")
    assert key == "env-key", f"Expected env-key, got {key}"
    
    # Remove env var, should fall back to file
    del os.environ["TEST_PROVIDER_API_KEY"]
    key = ctx1.get_credential("test_provider", "TEST_PROVIDER_API_KEY")
    assert key == "file-key", f"Expected file-key, got {key}"
    print("✓ Credential precedence working")
    
    # Test 4: Registry uses context
    registry = ctx1.model_registry
    assert isinstance(registry, ModelRegistry)
    assert registry._context is ctx1
    print("✓ Registry uses context for credentials")
    
    # Test 5: Child contexts
    child = ctx1.create_child(test={"nested": "value"})
    assert child.get_config("test.nested") == "value"
    assert child._parent is ctx1
    print("✓ Child contexts working")
    
    print("\nAll tests passed! Configuration system is properly centralized.")
    

def test_models_api_uses_context():
    """Test that models API uses context."""
    print("\nTesting models API context integration...")
    
    from ember.api import models
    from ember.api.models import _global_models_api
    
    # The global models API should use context
    assert hasattr(_global_models_api, '_context')
    assert _global_models_api._context is EmberContext.current()
    print("✓ Models API uses current context")
    
    # Registry should be from context
    ctx = EmberContext.current()
    assert _global_models_api._registry is ctx.model_registry
    print("✓ Models API registry is from context")
    
    print("\nModels API properly integrated with context system!")


if __name__ == "__main__":
    test_context_system()
    test_models_api_uses_context()
    
    # Clean up test credentials
    from ember.core.credentials import CredentialManager
    cm = CredentialManager()
    if cm.delete("test_provider"):
        print("\nCleaned up test credentials")