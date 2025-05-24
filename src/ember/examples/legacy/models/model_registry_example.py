"""Model Registry Usage Example

Demonstrates various patterns for using models with the simplified API.

This example shows:
1. Direct model invocation
2. Model invocation with parameters
3. Type-safe enum usage
4. Usage tracking and cost estimation
5. Batch processing with multiple models
6. Custom model registration

To run:
    uv run python src/ember/examples/models/model_registry_example.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key for OpenAI model examples
    ANTHROPIC_API_KEY (optional): Your Anthropic API key for Anthropic model examples
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from ember.api import models
from ember.api.models import ModelCost, ModelEnum, ModelInfo, RateLimit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def one_line_pattern():
    """Demonstrates the simplest one-line initialization pattern.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Direct model invocation with the simplified API
        response = models("openai:gpt-4o", "What is the capital of France?")
        print("\n=== One-line pattern result ===")
        print(response.data)
        return True
    except Exception as e:
        logger.exception("Error in one-line pattern: %s", str(e))
        return False


def standard_pattern():
    """Demonstrates the standard initialization pattern with more control.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use the models API with additional parameters
        response = models(
            "anthropic:claude-3-5-sonnet",
            "Explain quantum computing in one sentence.",
            temperature=0.7,
            max_tokens=100
        )

        print("\n=== Standard pattern result ===")
        print(f"Response: {response.data}")

        # Check usage statistics
        if response.usage:
            print(f"Total tokens used: {response.usage.total_tokens}")
            if hasattr(response.usage, 'cost'):
                print(f"Estimated cost: ${response.usage.cost:.4f}")

        return True
    except Exception as e:
        logger.exception("Error in standard pattern: %s", str(e))
        return False


def direct_model_pattern():
    """Demonstrates direct model access pattern.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Direct model access with the simplified API
        response = models("openai:gpt-4o", "What is the tallest mountain in the world?")

        print("\n=== Direct model pattern result ===")
        print(response.data)

        return True
    except Exception as e:
        logger.exception("Error in direct model pattern: %s", str(e))
        return False


def type_safe_enum_pattern() -> None:
    """Demonstrates using ModelEnum for type-safe model references."""
    try:
        # Use enum with the simplified API
        model_id = ModelEnum.gpt_4o.value
        response = models(
            model_id,
            "What's your favorite programming language and why?"
        )

        print("\n=== Type-safe enum pattern result ===")
        # Safely truncate long text
        truncated_text = (
            response.data[:150] + "..." if len(response.data) > 150 else response.data
        )
        print(f"Response: {truncated_text}")

        # Access model metadata
        registry = models.get_registry()
        model_info = registry.get_model_info(model_id=model_id)
        print("\nModel metadata:")
        print(f"Name: {model_info.name}")
        print(f"Provider: {model_info.provider.name}")
        print(
            f"Input cost per 1K tokens: ${model_info.cost.input_cost_per_thousand:.4f}"
        )
        print(
            f"Output cost per 1K tokens: ${model_info.cost.output_cost_per_thousand:.4f}"
        )
        print(f"Context window: {model_info.context_window} tokens")

        # Get version safely using get() instead of hasattr
        version = getattr(model_info, "version", "N/A")
        print(f"Version: {version}")

    except Exception as e:
        logger.exception("Error in type-safe enum pattern: %s", str(e))


def batch_processing_pattern() -> None:
    """Demonstrates batch processing with multiple models."""
    try:
        # Define prompts and models
        prompts = [
            "What is machine learning?",
            "Explain the concept of a neural network.",
            "What is transfer learning?",
            "Describe reinforcement learning.",
        ]

        model_ids = [
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-sonnet",
            "anthropic:claude-3-haiku",
        ]

        # Process in parallel with proper typing
        def process_prompt(args: tuple[str, str]) -> tuple[str, str, str, float]:
            """Process a single prompt with the specified model.

            Args:
                args: Tuple of (model_id, prompt)

            Returns:
                Tuple of (model_id, prompt, result_text, duration)
            """
            model_id, prompt = args
            try:
                start_time = time.time()
                response = models(model_id, prompt)
                duration = time.time() - start_time
                return model_id, prompt, response.data, duration
            except Exception as e:
                logger.warning(
                    "Error processing prompt with model %s: %s", model_id, str(e)
                )
                return model_id, prompt, f"Error: {str(e)}", 0.0

        print("\n=== Batch processing results ===")
        tasks = list(zip(model_ids, prompts))
        results = []

        # Use context manager for ThreadPoolExecutor for proper resource cleanup
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Collect results as they complete
            results = list(executor.map(process_prompt, tasks))

        # Print results
        for i, (model, prompt, result, duration) in enumerate(results):
            print(f"\nTask {i+1}:")
            print(f"Model: {model}")
            print(f"Prompt: {prompt}")
            print(f"Duration: {duration:.2f} seconds")
            # Safely truncate long results
            truncated_result = (
                result[:100] + "..." if result and len(result) > 100 else result
            )
            print(f"Result: {truncated_result}")

        # Calculate and show aggregate stats
        total_duration = sum(duration for _, _, _, duration in results)
        avg_duration = total_duration / len(results) if results else 0
        # Check for error state in a more robust way
        completed_tasks = sum(
            1
            for _, _, result, _ in results
            if result and not result.startswith("Error:")
        )

        print("\n=== Batch Processing Statistics ===")
        print(f"Total tasks: {len(results)}")
        print(f"Successfully completed: {completed_tasks}")
        print(f"Failed: {len(results) - completed_tasks}")
        print(f"Total processing time: {total_duration:.2f} seconds")
        print(f"Average response time: {avg_duration:.2f} seconds per task")

        if total_duration > 0:
            print(
                f"Effective throughput: {len(results) / total_duration:.2f} tasks per second"
            )
        else:
            print("Effective throughput: N/A (no time elapsed)")

        # Usage tracking
        print(
            "\nTotal usage across all batch operations can be tracked through response.usage"
        )
    except Exception as e:
        logger.exception("Error in batch processing pattern: %s", str(e))


def custom_model_pattern() -> None:
    """Demonstrates adding custom models to the registry."""
    try:
        # Get the registry using the simplified API
        registry = models.get_registry()

        # Register a custom model with realistic values
        custom_model = ModelInfo(
            id="custom:my-advanced-llm",
            name="MyOrg Advanced LLM",
            cost=ModelCost(
                input_cost_per_thousand=0.0015,  # $0.0015 per 1K input tokens
                output_cost_per_thousand=0.002,  # $0.002 per 1K output tokens
            ),
            rate_limit=RateLimit(
                tokens_per_minute=100000,  # 100K tokens per minute
                requests_per_minute=3000,  # 3K requests per minute
            ),
            context_window=128000,  # 128K context window
            provider={
                "name": "MyOrg AI",
                "default_api_key": "${MYORG_API_KEY}",
                "base_url": "https://api.myorg-ai.example.com/v1",
            },
        )

        # Check if model is already registered to avoid errors
        if not registry.is_registered(custom_model.id):
            # Register the model
            registry.register_model(model_info=custom_model)
            print(f"Registered custom model: {custom_model.id}")
        else:
            print(
                f"Model {custom_model.id} already registered ✅ - using existing registration"
            )

        # List all models
        model_ids = registry.list_models()

        print("\n=== Custom model registration ===")
        print(f"Registered models: {len(model_ids)} models found")
        print(f"Sample models: {model_ids[:5]} ... and more")

        # Check model exists and get info
        model_id = "custom:my-advanced-llm"
        exists = registry.is_registered(model_id)
        if exists:
            info = registry.get_model_info(model_id)
            print("\nCustom model details:")
            print(f"ID: {info.id}")
            print(f"Name: {info.name}")
            print(f"Provider: {info.provider.name}")
            print(f"API Base URL: {getattr(info.provider, 'api_base', 'N/A')}")
            print(f"Context window: {info.context_window} tokens")
            print(f"Input cost: ${info.cost.input_cost_per_thousand:.4f} per 1K tokens")
            print(
                f"Output cost: ${info.cost.output_cost_per_thousand:.4f} per 1K tokens"
            )
            print(
                f"Rate limits: {info.rate_limit.tokens_per_minute} tokens/min, {info.rate_limit.requests_per_minute} req/min"
            )
        else:
            print("Custom model registration failed!")
    except Exception as e:
        logger.exception("Error in custom model pattern: %s", str(e))


def main() -> None:
    """Run all example patterns."""
    print("Running Model Registry examples with the new API...\n")
    print("Make sure you have set up your API keys in environment variables:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY")
    print("  - GOOGLE_API_KEY (if using Gemini models)")

    # Run each pattern
    one_line_pattern()
    standard_pattern()
    direct_model_pattern()
    type_safe_enum_pattern()
    custom_model_pattern()

    # Only run batch processing if we have API keys configured
    keys_needed = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in keys_needed if os.environ.get(key) is None]
    if not missing_keys:
        batch_processing_pattern()
    else:
        print("\nSkipping batch processing example due to missing API keys.")

    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
