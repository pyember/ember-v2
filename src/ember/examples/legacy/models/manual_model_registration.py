"""Manual Model Registration Example

This example demonstrates manual model registration for cases where you need
specific configurations or custom models not available through auto-discovery.

To run:
    uv run python src/ember/examples/models/manual_model_registration.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key
    ANTHROPIC_API_KEY (optional): Your Anthropic API key

Note: The simplified API handles model discovery automatically when API keys
are set. Use manual registration only for custom configurations.
"""

import logging
import os
from typing import List

from ember.api import models
from ember.api.models import ModelCost, ModelInfo, RateLimit

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def register_models_with_custom_config() -> List[str]:
    """Register models with custom configurations.
    
    Returns:
        List of registered model IDs
    """
    registry = models.get_registry()
    registered = []
    
    # Get API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_key:
        # Custom GPT-4 with specific rate limits for testing
        custom_gpt4 = ModelInfo(
            id="custom:gpt-4-test",
            name="GPT-4 (Test Config)",
            context_window=8192,
            cost=ModelCost(
                input_cost_per_thousand=0.03,
                output_cost_per_thousand=0.06
            ),
            rate_limit=RateLimit(
                tokens_per_minute=5000,  # Lower limit for testing
                requests_per_minute=10   # Very low for testing
            ),
            provider={
                "name": "OpenAI",
                "default_api_key": openai_key,
                "base_url": "https://api.openai.com/v1"
            }
        )
        
        if not registry.is_registered(custom_gpt4.id):
            try:
                registry.register_model(model_info=custom_gpt4)
                logger.info(f"✅ Registered: {custom_gpt4.id}")
                registered.append(custom_gpt4.id)
            except Exception as e:
                logger.error(f"❌ Failed: {e}")
    
    if anthropic_key:
        # Custom Claude with extended context
        custom_claude = ModelInfo(
            id="custom:claude-extended",
            name="Claude (Extended Context)",
            context_window=500000,  # Extended context window
            cost=ModelCost(
                input_cost_per_thousand=0.005,
                output_cost_per_thousand=0.025
            ),
            rate_limit=RateLimit(
                tokens_per_minute=1000000,
                requests_per_minute=100
            ),
            provider={
                "name": "Anthropic",
                "default_api_key": anthropic_key,
                "base_url": "https://api.anthropic.com/v1"
            }
        )
        
        if not registry.is_registered(custom_claude.id):
            try:
                registry.register_model(model_info=custom_claude)
                logger.info(f"✅ Registered: {custom_claude.id}")
                registered.append(custom_claude.id)
            except Exception as e:
                logger.error(f"❌ Failed: {e}")
    
    return registered


def display_model_comparison():
    """Compare custom models with standard models."""
    registry = models.get_registry()
    
    logger.info("\nModel Comparison:")
    logger.info("-" * 70)
    logger.info(f"{'Model ID':<30} {'Context':<15} {'Input $/1K':<12} {'RPM':<10}")
    logger.info("-" * 70)
    
    # Compare standard vs custom models
    model_pairs = [
        ("openai:gpt-4", "custom:gpt-4-test"),
        ("anthropic:claude-3-sonnet", "custom:claude-extended")
    ]
    
    for standard, custom in model_pairs:
        for model_id in [standard, custom]:
            try:
                if registry.is_registered(model_id):
                    info = registry.get_model_info(model_id)
                    logger.info(
                        f"{model_id:<30} "
                        f"{info.context_window:<15,} "
                        f"${info.cost.input_cost_per_thousand:<11.4f} "
                        f"{info.rate_limit.requests_per_minute:<10}"
                    )
                else:
                    logger.info(f"{model_id:<30} {'Not registered':<15}")
            except Exception as e:
                logger.error(f"{model_id:<30} Error: {e}")
        logger.info("")  # Blank line between pairs


def demonstrate_usage():
    """Show how to use custom models."""
    logger.info("\nUsage Examples:")
    logger.info("-" * 50)
    
    logger.info("\n1. Using custom model with lower rate limits:")
    logger.info('   response = models("custom:gpt-4-test", "Hello!")')
    logger.info('   # Useful for testing without hitting rate limits')
    
    logger.info("\n2. Using extended context model:")
    logger.info('   long_text = "..." * 100000  # Very long text')
    logger.info('   response = models("custom:claude-extended", long_text)')
    logger.info('   # Can handle much longer contexts')
    
    logger.info("\n3. Checking if custom model exists:")
    logger.info('   if models.get_registry().is_registered("custom:gpt-4-test"):')
    logger.info('       response = models("custom:gpt-4-test", "Test")')


def main():
    """Run the manual registration example."""
    logger.info("=== Manual Model Registration Example ===\n")
    
    # Check for API keys
    if not any([os.environ.get("OPENAI_API_KEY"), os.environ.get("ANTHROPIC_API_KEY")]):
        logger.warning("⚠️  No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        logger.info("\nThis example shows how to register custom model configurations.")
        demonstrate_usage()
        return
    
    # Register custom models
    logger.info("Registering custom models:")
    registered = register_models_with_custom_config()
    
    if registered:
        # Show comparison
        display_model_comparison()
        
        # Show usage
        demonstrate_usage()
        
        # Try to use a custom model
        if "custom:gpt-4-test" in registered:
            logger.info("\nTesting custom model:")
            try:
                response = models("custom:gpt-4-test", "Say hello!")
                logger.info(f"Response: {response.text}")
            except Exception as e:
                logger.info(f"(Skipping actual call: {e})")
    
    logger.info("\n=== Example completed! ===")


if __name__ == "__main__":
    main()