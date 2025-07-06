"""Pytest configuration for example tests.

This module configures pytest for testing Ember examples, including:
- Custom command line options
- Fixtures for common test scenarios
- Test collection and organization
"""

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command line options for example testing."""
    parser.addoption(
        "--no-api-keys",
        action="store_true",
        default=False,
        help="Run tests without requiring API keys (simulated mode only)",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden outputs from current execution",
    )
    parser.addoption(
        "--example",
        action="store",
        default=None,
        help="Run tests for a specific example only",
    )


@pytest.fixture
def no_api_keys(request):
    """Check if tests should run without API keys."""
    return request.config.getoption("--no-api-keys")


@pytest.fixture
def update_golden(request):
    """Check if golden outputs should be updated."""
    return request.config.getoption("--update-golden")


@pytest.fixture
def specific_example(request):
    """Get specific example to test if provided."""
    return request.config.getoption("--example")


@pytest.fixture
def examples_root():
    """Get the root directory of examples."""
    return Path(__file__).parent.parent.parent / "examples"


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Mock all common API keys to empty strings."""
    api_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "COHERE_API_KEY",
        "HUGGINGFACE_API_KEY",
    ]

    for key in api_keys:
        monkeypatch.setenv(key, "")


@pytest.fixture
def real_api_keys():
    """Check which real API keys are available."""
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "COHERE_API_KEY": bool(os.environ.get("COHERE_API_KEY")),
        "HUGGINGFACE_API_KEY": bool(os.environ.get("HUGGINGFACE_API_KEY")),
    }


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    no_api_keys = config.getoption("--no-api-keys")
    specific_example = config.getoption("--example")

    # Filter tests based on options
    selected = []
    deselected = []

    for item in items:
        # Check if this is an example test
        if not hasattr(item, "function") or not item.function.__name__.startswith("test_"):
            selected.append(item)
            continue

        # Filter by specific example if provided
        if specific_example:
            if specific_example not in str(item.nodeid):
                deselected.append(item)
                continue

        # Skip real mode tests if --no-api-keys
        if no_api_keys and "real_mode" in item.keywords:
            deselected.append(item)
            continue

        selected.append(item)

    # Update items
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Skip tests requiring API keys if they're not available
    if hasattr(item, "function"):
        marker = item.get_closest_marker("requires_api_keys")
        if marker:
            required_keys = marker.args[0] if marker.args else []
            missing_keys = [key for key in required_keys if not os.environ.get(key)]

            if missing_keys and "--no-api-keys" not in item.config.invocation_params.args:
                pytest.skip(f"Missing required API keys: {missing_keys}")
