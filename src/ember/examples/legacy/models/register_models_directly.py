"""Custom Model Registration Example

This example shows how to register custom models with specific configurations
using the simplified API.

To run:
    uv run python src/ember/examples/models/register_models_directly.py

Note: The simplified API handles model discovery automatically when API keys
are set. This example shows manual registration for custom configurations.
"""

import logging
import os
from typing import List

from ember.api import models
from ember.api.models import ModelCost, ModelInfo, RateLimit

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def register_custom_models() -> List[str]:
    """Register custom model configurations.
    
    Returns:
        List of registered model IDs
    """
    registry = models.get_registry()
    registered = []
    
    # Example: Register a custom GPT-4 configuration with specific limits
    custom_gpt4 = ModelInfo(
        id="custom:gpt-4-limited",
        name="GPT-4 (Rate Limited)",
        context_window=8192,
        cost=ModelCost(
            input_cost_per_thousand=0.03,
            output_cost_per_thousand=0.06
        ),
        rate_limit=RateLimit(
            tokens_per_minute=10000,  # Lower limit for testing
            requests_per_minute=60
        ),
        provider={
            "name": "OpenAI",
            "default_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "base_url": "https://api.openai.com/v1"
        }
    )
    
    # Example: Register a local model
    local_model = ModelInfo(
        id="local:llama-7b",
        name="Local Llama 7B",
        context_window=4096,
        cost=ModelCost(
            input_cost_per_thousand=0.0,  # Free local model
            output_cost_per_thousand=0.0
        ),
        rate_limit=RateLimit(
            tokens_per_minute=100000,
            requests_per_minute=1000
        ),
        provider={
            "name": "Local",
            "base_url": "http://localhost:8080/v1"
        }
    )
    
    # Register models if not already registered
    for model in [custom_gpt4, local_model]:
        if not registry.is_registered(model.id):
            try:
                registry.register_model(model_info=model)
                logger.info(f"✅ Registered: {model.id}")
                registered.append(model.id)
            except Exception as e:
                logger.error(f"❌ Failed to register {model.id}: {e}")
        else:
            logger.info(f"ℹ️  {model.id} already registered")
            registered.append(model.id)
    
    return registered


def display_model_info(model_ids: List[str]):
    """Display information about registered models."""
    registry = models.get_registry()
    
    logger.info("\nRegistered Model Information:")
    logger.info("-" * 60)
    
    for model_id in model_ids:
        try:
            info = registry.get_model_info(model_id)
            logger.info(f"\n{model_id}:")
            logger.info(f"  Name: {info.name}")
            logger.info(f"  Context: {info.context_window:,} tokens")
            logger.info(f"  Input cost: ${info.cost.input_cost_per_thousand:.4f}/1K")
            logger.info(f"  Output cost: ${info.cost.output_cost_per_thousand:.4f}/1K")
            logger.info(f"  Rate limit: {info.rate_limit.tokens_per_minute:,} tokens/min")
        except Exception as e:
            logger.error(f"\n{model_id}: Error getting info - {e}")


def main():
    """Example demonstrating the simplified XCS architecture."""
    """Run the custom model registration example."""
    logger.info("=== Custom Model Registration Example ===\n")
    
    # Note about automatic discovery
    logger.info("Note: When API keys are set, models are discovered automatically.")
    logger.info("This example shows manual registration for custom configurations.\n")
    
    # Register custom models
    registered = register_custom_models()
    
    if registered:
        # Display information
        display_model_info(registered)
        
        # Show usage example
        logger.info("\nUsage example:")
        logger.info('response = models("custom:gpt-4-limited", "Hello!")')
        logger.info('print(response.text)')
    else:
        logger.info("\nNo models were registered.")
    
    # List all available models
    logger.info("\nAll available models:")
    try:
        all_models = models.list()
        for provider in set(m.split(":")[0] for m in all_models if ":" in m):
            provider_models = [m for m in all_models if m.startswith(f"{provider}:")]
            logger.info(f"  {provider}: {len(provider_models)} models")
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    
    logger.info("\n=== Example completed! ===")


if __name__ == "__main__":
    main()