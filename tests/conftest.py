"""Test configuration and fixtures.

Following CLAUDE.md principles:
- Explicit fixtures (no magic)
- Deterministic behavior  
- Clear, reusable test utilities
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Configure pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Core test fixtures
@pytest.fixture
def tmp_ctx(tmp_path, monkeypatch):
    """Isolated EmberContext with temporary home directory.
    
    Provides complete isolation from user's real ~/.ember directory.
    All tests using this fixture are hermetic and parallelizable.
    """
    from ember._internal.context import EmberContext
    
    # Create fake home
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    
    # Clear any existing context
    if hasattr(EmberContext._thread_local, 'context'):
        delattr(EmberContext._thread_local, 'context')
    EmberContext._context_var.set(None)
    
    # Create fresh context
    ctx = EmberContext.current()
    yield ctx
    
    # Cleanup
    if hasattr(EmberContext._thread_local, 'context'):
        delattr(EmberContext._thread_local, 'context')
    EmberContext._context_var.set(None)


@pytest.fixture
def mock_cli_args(monkeypatch):
    """Mock sys.argv for CLI testing without subprocess."""
    def _mock_args(*args):
        monkeypatch.setattr(sys, "argv", ["ember"] + list(args))
    return _mock_args


# Model API fixtures
@pytest.fixture
def mock_model_response():
    """Standard mock response for model tests."""
    from ember.models.schemas import ChatResponse, UsageStats
    
    usage = UsageStats(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost_usd=0.0006
    )
    
    return ChatResponse(
        data="Test response",
        usage=usage,
        model_id="gpt-4"
    )


@pytest.fixture
def mock_registry():
    """Mock model registry for testing."""
    registry = Mock()
    registry._models = {}
    registry._lock = Mock()
    return registry


# Data API fixtures
@pytest.fixture
def temp_data_file(tmp_path):
    """Create temporary JSON data file."""
    data = [
        {"text": "Hello", "label": "greeting"},
        {"text": "World", "label": "noun"},
        {"text": "Test", "label": "verb"}
    ]
    file_path = tmp_path / "test_data.json"
    file_path.write_text(json.dumps(data))
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create temporary CSV data file."""
    csv_content = """text,label
Hello,greeting
World,noun
Test,verb"""
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(csv_content)
    return file_path


# Operator fixtures
@pytest.fixture
def simple_operator():
    """Simple operator for testing."""
    def double(x):
        return x * 2
    return double


@pytest.fixture
def mock_model_operator():
    """Mock model operator for testing."""
    def model_op(text):
        return Mock(text=f"Processed: {text}", usage={"tokens": 10})
    return model_op


# XCS fixtures
@pytest.fixture
def slow_function():
    """Function that simulates slow operation."""
    import time
    
    def slow_op(x):
        time.sleep(0.01)  # 10ms
        return x * 2
    
    return slow_op


# Environment fixtures
@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment without API keys."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False) 
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Set mock API keys for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")


# Seed fixtures for determinism
@pytest.fixture(autouse=True)
def fixed_seed():
    """Fix random seeds for deterministic tests."""
    import random
    import numpy as np
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    
    # JAX seed if available
    try:
        import jax
        key = jax.random.PRNGKey(42)
        return key
    except ImportError:
        return None