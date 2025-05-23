"""List Available Models Example

This script demonstrates how to list and inspect available models using
the simplified Ember models API.

To run:
    uv run python src/ember/examples/models/list_models.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key
    ANTHROPIC_API_KEY (optional): Your Anthropic API key
"""

import logging
import os

from ember.api import models

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run the list models example."""
    logger.info("=== List Available Models Example ===\n")
    
    # Check if API keys are set
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        logger.warning("⚠️  No API keys set. Model discovery may be limited.")
        logger.warning("Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY for full functionality.\n")
    
    # List all available models
    logger.info("Listing all available models:")
    try:
        available_models = models.list()
        
        if not available_models:
            logger.info("No models found. Make sure API keys are set.")
            return
            
        logger.info(f"Found {len(available_models)} models:\n")
        
        # Group by provider
        by_provider = {}
        for model_id in available_models:
            provider = model_id.split(":")[0] if ":" in model_id else "unknown"
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model_id)
        
        # Display by provider
        for provider, model_list in sorted(by_provider.items()):
            logger.info(f"{provider.upper()} ({len(model_list)} models):")
            for model_id in sorted(model_list):
                logger.info(f"  - {model_id}")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return
    
    # List models by specific provider
    logger.info("Listing models by provider:")
    for provider in ["openai", "anthropic"]:
        try:
            provider_models = models.list(provider=provider)
            logger.info(f"{provider}: {len(provider_models)} models")
        except Exception as e:
            logger.error(f"Error listing {provider} models: {e}")
    
    logger.info("")
    
    # Get detailed info for specific models
    logger.info("Getting model information:")
    example_models = ["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"]
    
    for model_id in example_models:
        try:
            info = models.info(model_id)
            logger.info(f"\n{model_id}:")
            logger.info(f"  Full ID: {info['id']}")
            logger.info(f"  Provider: {info['provider']}")
            
            if 'context_window' in info:
                logger.info(f"  Context window: {info['context_window']:,} tokens")
            
            if 'pricing' in info:
                pricing = info['pricing']
                logger.info(f"  Pricing:")
                logger.info(f"    Input: ${pricing.get('input', 0):.6f}/1K tokens")
                logger.info(f"    Output: ${pricing.get('output', 0):.6f}/1K tokens")
                
        except Exception as e:
            logger.info(f"\n{model_id}: Not available ({e})")
    
    logger.info("\n=== Example completed! ===")


if __name__ == "__main__":
    main()