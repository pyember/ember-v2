"""Test configuration and fixtures - REFACTORED.

Following CLAUDE.md principles:
- Explicit fixtures (no magic)
- Deterministic behavior
- Clear, reusable test utilities
- NO PRIVATE ATTRIBUTE ACCESS
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Import our new test infrastructure
try:
    from .test_constants import APIKeys, Models, TestData
    from .test_doubles import FakeProvider, FakeModelRegistry, FakeContext
    from .fixtures import *
except ImportError:
    # Fallback for when running pytest from different locations
    from test_constants import APIKeys, Models, TestData
    from test_doubles import FakeProvider, FakeModelRegistry, FakeContext
    from fixtures import *

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


# Configure pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Core test fixtures - REFACTORED without private access
@pytest.fixture
def tmp_ctx(tmp_path, monkeypatch):
    """Isolated EmberContext with temporary home directory.

    Provides complete isolation from user's real ~/.ember directory.
    All tests using this fixture are hermetic and parallelizable.
    """
    # Import only if we need the real context
    try:
        from ember._internal.context import EmberContext
        
        # Create fake home
        home = tmp_path / "home"
        home.mkdir()
        ember_dir = home / ".ember"
        ember_dir.mkdir()
        
        # Set environment to use temp directory
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("EMBER_HOME", str(ember_dir))
        
        # Try to use public API for context reset if available
        if hasattr(EmberContext, 'reset'):
            EmberContext.reset()
            
        # Create fresh context
        ctx = EmberContext(isolated=True)
        yield ctx
        
        # Cleanup via public API if available
        if hasattr(EmberContext, 'reset'):
            EmberContext.reset()
            
    except ImportError:
        # If real context not available, use fake
        ctx = FakeContext(isolated=True)
        yield ctx


@pytest.fixture
def mock_cli_args(monkeypatch):
    """Mock sys.argv for CLI testing without subprocess."""

    def _mock_args(*args):
        monkeypatch.setattr(sys, "argv", ["ember"] + list(args))

    return _mock_args


# Model API fixtures - REFACTORED to use test doubles
@pytest.fixture
def mock_model_response():
    """Standard mock response for model tests."""
    from tests.fixtures import create_api_response
    
    return create_api_response(
        content="Test response",
        model=Models.GPT4,
        prompt_tokens=10,
        completion_tokens=20
    )


@pytest.fixture
def mock_registry():
    """Mock model registry for testing."""
    return FakeModelRegistry()


# Data API fixtures - REFACTORED to use constants
@pytest.fixture
def temp_data_file(tmp_path):
    """Create temporary JSON data file."""
    data = TestData.SAMPLE_JSON_DATA.copy()
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


# Environment fixtures - using new constants
@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment without API keys."""
    for key in [APIKeys.ENV_OPENAI, APIKeys.ENV_ANTHROPIC, APIKeys.ENV_GOOGLE]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Set mock API keys for testing."""
    monkeypatch.setenv(APIKeys.ENV_OPENAI, APIKeys.OPENAI)
    monkeypatch.setenv(APIKeys.ENV_ANTHROPIC, APIKeys.ANTHROPIC)
    monkeypatch.setenv(APIKeys.ENV_GOOGLE, APIKeys.GOOGLE)


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


# Add fixture to check for API key availability
@pytest.fixture
def no_api_keys():
    """Check if any real API keys are available."""
    import os
    
    real_keys = [
        os.environ.get(APIKeys.ENV_OPENAI, "").startswith("sk-"),
        os.environ.get(APIKeys.ENV_ANTHROPIC, "").startswith("ant-"),
        os.environ.get(APIKeys.ENV_GOOGLE, "").startswith("goog-")
    ]
    
    return not any(real_keys)