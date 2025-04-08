"""
Root conftest.py for pytest configuration
"""

import importlib
import logging
import os
import sys
import warnings
from pathlib import Path

import pytest


# Configure logging to handle closed streams during Python interpreter shutdown
def _patch_logging_for_shutdown():
    """Patch logging handlers to gracefully handle closed streams at shutdown."""
    if not hasattr(logging.StreamHandler, "_ember_patched"):
        # Save the original emit method
        original_emit = logging.StreamHandler.emit

        # Create a safer version that handles closed file errors
        def safe_emit(self, record):
            try:
                original_emit(self, record)
            except (ValueError, IOError, OSError) as e:
                if "closed" not in str(e).lower() and "closed file" not in str(e).lower():
                    raise

        # Apply the patch
        logging.StreamHandler.emit = safe_emit
        logging.StreamHandler._ember_patched = True

        # Configure problematic loggers
        for name in ["httpcore.connection", "httpcore.http11"]:
            logging.getLogger(name).setLevel(logging.INFO)


# Apply logging patch
_patch_logging_for_shutdown()

# Setup paths
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"

print(f"Unit test Python path: {sys.path}")
print(f"Unit test current directory: {os.getcwd()}")

# Add src directory to path
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# Configure asyncio
pytest_plugins = ["pytest_asyncio"]

# Silence common warnings
warnings.filterwarnings("ignore", message=".*XCS functionality partially unavailable.*")

# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest-asyncio and register custom marks."""
    import pytest_asyncio
    
    # Use session-scoped event loops by default
    pytest_asyncio.LOOP_SCOPE = "session"
    
    # Register custom marks
    config.addinivalue_line("markers", "discovery: mark tests that interact with model discovery")
    config.addinivalue_line("markers", "xcs: mark tests related to XCS functionality")
    config.addinivalue_line("markers", "performance: mark tests that measure performance characteristics")

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Modify test items based on command line options."""
    run_all = config.getoption("--run-all-tests")
    run_api = config.getoption("--run-api-tests")
    run_perf = config.getoption("--run-perf-tests")
    
    for item in items:
        skip_marks = [mark for mark in item.own_markers if mark.name == "skip"]
        skipif_marks = [mark for mark in item.own_markers if mark.name == "skipif"]
        
        # Special handling for different test types
        if any(mark.name == "performance" for mark in item.own_markers):
            if run_all or run_perf:
                for mark in skip_marks:
                    item.own_markers.remove(mark)
        elif any("API_KEY" in str(mark.args) for mark in skipif_marks):
            if run_api:
                for mark in skip_marks:
                    item.own_markers.remove(mark)
        elif run_all:
            for mark in skip_marks:
                item.own_markers.remove(mark)

def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-perf-tests",
        action="store_true",
        default=False,
        help="Run performance tests that are skipped by default",
    )
    parser.addoption(
        "--run-all-tests",
        action="store_true",
        default=False,
        help="Run all tests including skipped tests (except those requiring API keys)",
    )
    parser.addoption(
        "--run-api-tests",
        action="store_true", 
        default=False,
        help="Run tests that require API keys and external services",
    )

@pytest.fixture(scope="session", autouse=True)
def _add_config_helper(request):
    """Add config attribute to pytest module for backward compatibility."""
    pytest.config = request.config

@pytest.fixture(scope="session")
def event_loop_policy():
    """Return the event loop policy to use."""
    import asyncio
    return asyncio.get_event_loop_policy()