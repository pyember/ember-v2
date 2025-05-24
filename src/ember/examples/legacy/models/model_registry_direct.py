"""Direct Model Usage Example

This example demonstrates direct model invocation using the simplified API
with API keys from environment variables.

To run:
    export OPENAI_API_KEY="your-openai-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
    uv run python src/ember/examples/models/model_registry_direct.py
"""

import logging
import os

from ember.api import models

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run direct model usage examples."""
    logger.info("=== Direct Model Usage Example ===\n")
    
    # Check API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        logger.warning("⚠️  No API keys found!")
        logger.warning("Please set OPENAI_API_KEY and/or ANTHROPIC_API_KEY")
        return
    
    # Direct usage - the API handles model discovery automatically
    if openai_key:
        logger.info("Using OpenAI GPT-4:")
        try:
            response = models("gpt-4", "What is the capital of France?")
            logger.info(f"Response: {response.text}")
            logger.info(f"Tokens used: {response.usage.get('total_tokens', 'N/A')}")
        except Exception as e:
            logger.error(f"Error: {e}")
    else:
        logger.info("Skipping OpenAI (no API key)")
    
    logger.info("")
    
    # Direct usage with Anthropic
    if anthropic_key:
        logger.info("Using Anthropic Claude:")
        try:
            response = models("claude-3-sonnet", "What is the capital of Italy?")
            logger.info(f"Response: {response.text}")
            logger.info(f"Tokens used: {response.usage.get('total_tokens', 'N/A')}")
        except Exception as e:
            logger.error(f"Error: {e}")
    else:
        logger.info("Skipping Anthropic (no API key)")
    
    logger.info("")
    
    # Show how to use with parameters
    logger.info("Using models with parameters:")
    if openai_key:
        try:
            response = models(
                "gpt-4",
                "Write a haiku about programming",
                temperature=0.7,
                max_tokens=50
            )
            logger.info(f"Haiku: {response.text}")
        except Exception as e:
            logger.error(f"Error: {e}")
    
    logger.info("\n=== Example completed! ===")


if __name__ == "__main__":
    main()