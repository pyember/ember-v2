"""
Example: Dependency Injection with ModelContext

This example demonstrates how to use the dependency injection capabilities
of the new model API to isolate different execution contexts.

Key concepts:
1. Creating multiple contexts with different configurations
2. Using models with explicit context dependencies
3. Provider namespaces with contexts
4. Temporary configuration overrides
"""

import os
from typing import Dict, List

from ember.api.models import (
    ContextConfig,
    configure,
    create_context,
    create_provider_namespace,
    model,
)


def create_isolated_contexts():
    """Create multiple isolated contexts with different configurations."""
    print("\n=== Creating Isolated Contexts ===\n")

    # Create a context for production use
    prod_config = ContextConfig(
        auto_discover=True,
        api_keys={
            "openai": os.environ.get("OPENAI_API_KEY", "sk-prod-key"),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY", "sk-prod-key"),
        },
        default_timeout=30,
    )
    prod_context = create_context(config=prod_config)

    # Create a context for testing
    test_config = ContextConfig(
        auto_discover=False,
        api_keys={
            "openai": "sk-test-key",
            "anthropic": "sk-test-key",
        },
        default_timeout=5,
    )
    test_context = create_context(config=test_config)

    # Use both contexts
    try:
        # This will use the production context (real API keys)
        prod_model = model("gpt-4o", context=prod_context)

        # This will use the test context (test API keys)
        test_model = model("gpt-4o", context=test_context)

        # Demonstrate the isolation - these will have different URLs and keys
        print(f"Production model: {prod_model.model_id}")
        print(f"Test model: {test_model.model_id}")

        # NOTE: Since we're using fake test keys, this would fail in a real setting
        # response = test_model("This is a test prompt")

    except Exception as e:
        print(f"Exception: {e}")

    print("\nContexts remain isolated - changing one doesn't affect the other")


def provider_namespaces_with_contexts():
    """Create provider namespaces with different contexts."""
    print("\n=== Provider Namespaces with Contexts ===\n")

    # Create two contexts with different configurations
    context1 = create_context(
        config=ContextConfig(api_keys={"openai": "key1", "anthropic": "key1"})
    )

    context2 = create_context(
        config=ContextConfig(api_keys={"openai": "key2", "anthropic": "key2"})
    )

    # Create provider namespaces with the contexts
    openai1 = create_provider_namespace("openai", context=context1)
    openai2 = create_provider_namespace("openai", context=context2)

    # The models from these namespaces will use different contexts
    model1 = openai1.gpt4o
    model2 = openai2.gpt4o

    print("Model 1 using context 1")
    print("Model 2 using context 2")

    print("\nEach model will use its own isolated key and configuration")


def configuration_contexts():
    """Demonstrate the use of configuration contexts."""
    print("\n=== Configuration Contexts ===\n")

    # Create a model with default config
    gpt4 = model("gpt-4")

    print("Default configuration:")
    print(f"Temperature: {gpt4.config.get('temperature', 0.7)}")

    # Temporarily override configuration
    with configure(temperature=0.2, max_tokens=100):
        print("\nWith configure() context manager:")
        print(f"Temperature: {gpt4.config.get('temperature', 0.7)}")

        # The override is only for this context
        with configure(temperature=0.9):
            print("\nNested configure() context:")
            print(f"Temperature: {gpt4.config.get('temperature', 0.7)}")

        print("\nBack to first configure() context:")
        print(f"Temperature: {gpt4.config.get('temperature', 0.7)}")

    print("\nBack to default configuration:")
    print(f"Temperature: {gpt4.config.get('temperature', 0.7)}")


def simulate_ab_testing():
    """Simulate A/B testing with different model configurations."""
    print("\n=== Simulating A/B Testing ===\n")

    # Create contexts for A/B testing
    context_a = create_context(
        config=ContextConfig(
            api_keys={"openai": "key-a", "anthropic": "key-a"}, auto_discover=True
        )
    )

    context_b = create_context(
        config=ContextConfig(
            api_keys={"openai": "key-b", "anthropic": "key-b"}, auto_discover=True
        )
    )

    # Create models with different contexts and configurations
    model_a = model("gpt-4o", context=context_a, temperature=0.5)
    model_b = model("claude-3-5-sonnet", context=context_b, temperature=0.7)

    # Function to simulate running experiments
    def run_experiment(prompt: str, models: Dict[str, callable], n_trials: int = 3):
        """Run an experiment with multiple models."""
        results: Dict[str, List[str]] = {name: [] for name in models}

        for trial in range(n_trials):
            for name, model_fn in models.items():
                print(f"Trial {trial+1}: Running experiment with {name}")
                # In a real setting, we would call the model
                # response = model_fn(prompt)
                # results[name].append(str(response))

        return results

    # Run the experiment
    experiment = run_experiment(
        prompt="Explain the benefits of quantum computing in three sentences.",
        models={"GPT-4o": model_a, "Claude-3.5": model_b},
    )

    print("\nExperiment completed, contexts remained isolated")


def main():
    """Run all examples."""
    print("=== Model Context and Dependency Injection Examples ===")

    create_isolated_contexts()
    provider_namespaces_with_contexts()
    configuration_contexts()
    simulate_ab_testing()

    print("\n=== End of Examples ===")


if __name__ == "__main__":
    main()
