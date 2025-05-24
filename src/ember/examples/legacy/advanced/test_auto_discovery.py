"""Test the automatic model discovery functionality.

This script tests the automatic model discovery functionality in the model registry.
It's a simple test script that directly uses the registry and discovery service.

To run:
    uv run python src/ember/examples/advanced/test_auto_discovery.py
"""

import logging
import os

# Import from simplified API
from ember.api import models

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_auto_discovery():
    """Test the automatic model discovery functionality."""
    # The new API auto-discovers on first access
    logger.info("Checking available models through simplified API")
    
    # List all available models
    available_models = models.list()
    if available_models:
        logger.info(
            f"Found {len(available_models)} models through auto-discovery:"
        )
        # Show first 10 models
        for model_id in available_models[:10]:
            logger.info(f"  - {model_id}")
    else:
        logger.warning("No models found through auto-discovery")

    # Test getting a specific model
    logger.info("\nTesting model retrieval...")
    try:
        model = models.get("openai:gpt-4")
        logger.info(f"Successfully retrieved model: {model}")
    except Exception as e:
        logger.error(f"Failed to retrieve model: {e}")
    
    # Test direct model invocation
    logger.info("\nTesting direct model invocation...")
    try:
        result = models("openai:gpt-4o-mini", "Say hello")
        logger.info("Direct model invocation successful")
    except Exception as e:
        logger.error(f"Direct invocation failed: {e}")
    
    # Show model details for a few models
    logger.info("\nShowing details for available models...")
    if available_models:
        for model_id in available_models[:5]:  # Show first 5 models
            try:
                model_info = models.get_info(model_id)
                logger.info(f"\nModel: {model_id}")
                if model_info:
                    logger.info(f"  Provider: {getattr(model_info, 'provider', 'Unknown')}")
                    logger.info(f"  Context Window: {getattr(model_info, 'context_window', 'Unknown')}")
                    
                    # Show cost info if available
                    cost = getattr(model_info, 'cost', None)
                    if cost:
                        i_cost = getattr(cost, 'input_cost_per_thousand', 'N/A')
                        o_cost = getattr(cost, 'output_cost_per_thousand', 'N/A')
                        logger.info(f"  Cost: Input=${i_cost}/1K, Output=${o_cost}/1K")
            except Exception as e:
                logger.error(f"Error getting info for {model_id}: {e}")


if __name__ == "__main__":
    logger.info("Testing automatic model discovery...")

    # Check which API keys are available
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }

    for key, value in api_keys.items():
        if value:
            logger.info(f"{key} is set - length: {len(value[:3])}...")
        else:
            logger.warning(f"{key} is NOT set")

    # Run the test
    test_auto_discovery()
