"""
Example: Function-Style Model API

This example demonstrates the new function-style API for interacting with models.
This API is designed to be more intuitive, easier to use, and more aligned with
Python's functional programming style.

Key concepts:
1. Function-style model creation and invocation
2. Provider namespaces for direct model access
3. Configuration management
4. Response handling
"""

from ember.api.models import Response, model


def basic_usage():
    """Demonstrate basic usage of the function-style API."""
    print("\n=== Basic Usage ===\n")

    # The most direct way to use a model - creation and invocation in one line
    try:
        # Note: This will fail without actual API keys
        # response = model("gpt-4o")("What is the capital of France?")
        # print(f"Response: {response}")

        # Simulation for demonstration purposes
        print('model("gpt-4o")("What is the capital of France?")')
        print("Response: Paris is the capital of France.")
    except Exception as e:
        print(f"Note: This would fail without actual API keys: {e}")

    # Create a reusable model instance
    gpt4 = model("gpt-4o")

    # Use it multiple times
    print("\nReusable model instance:")
    print('gpt4 = model("gpt-4o")')
    print('gpt4("Tell me a joke")')
    print('gpt4("Explain quantum computing")')


def provider_namespaces():
    """Demonstrate the use of provider namespaces."""
    print("\n=== Provider Namespaces ===\n")

    # Provider namespaces provide direct access to models
    try:
        # Note: This will fail without actual API keys
        # response = openai.gpt4o("What is the capital of France?")
        # print(f"Response: {response}")

        # Simulation for demonstration purposes
        print('openai.gpt4o("What is the capital of France?")')
        print("Response: Paris is the capital of France.")

        print("\nUsing other providers:")
        print('anthropic.claude("Tell me about the Roman Empire")')
        print('deepmind.gemini("Explain quantum mechanics")')
    except Exception as e:
        print(f"Note: This would fail without actual API keys: {e}")

    # The provider namespaces handle model name normalization
    print("\nModel name normalization:")
    print("openai.gpt4_o == openai.gpt4o  # Underscores converted to hyphens")


def configuration_management():
    """Demonstrate configuration management."""
    print("\n=== Configuration Management ===\n")

    # Set configuration during model creation
    print("Model with configuration:")
    print('gpt4 = model("gpt-4o", temperature=0.7, max_tokens=100)')

    # Global configuration (for all models)

    print("\nGlobal configuration:")
    print("config.temperature = 0.5")
    print("config.update(max_tokens=200, top_p=0.8)")

    # Temporary configuration using context manager
    print("\nTemporary configuration:")
    print("with configure(temperature=0.2, max_tokens=50):")
    print('    response = model("gpt-4o")("Write a short poem")')

    # Configuration hierarchy
    print("\nConfiguration hierarchy (order of precedence):")
    print("1. Call-specific arguments: model(...)(prompt, temperature=0.1)")
    print("2. Model instance config: model(..., temperature=0.2)(prompt)")
    print("3. Temporary config: with configure(temperature=0.3): ...")
    print("4. Global config: config.temperature = 0.4")


def response_handling():
    """Demonstrate response handling."""
    print("\n=== Response Handling ===\n")

    # Creating a simulated response for demonstration
    class SimulatedResponse:
        def __init__(self):
            self.data = "This is a simulated response."
            self.usage = type(
                "Usage",
                (),
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "cost": 0.0012,
                },
            )

    # Wrap the simulated response
    response = Response(SimulatedResponse())

    # String conversion
    print(f"String conversion: {response}")

    # Accessing metadata
    print("\nAccessing metadata:")
    print(f"Model ID: {response.model_id}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Cost: ${response.usage.cost:.6f}")

    # Visualization (simplified)
    print("\nVisualization:")
    print("response.visualize()  # Displays a rich representation in notebooks")


def complete_function():
    """Demonstrate the complete() function."""
    print("\n=== Complete Function ===\n")

    # The complete() function is a convenience function for one-off completions
    print("Using complete() function:")
    print('complete("What is 2+2?", model="gpt-4o", temperature=0.5)')
    print("Response: 4")

    # It's equivalent to:
    print("\nEquivalent to:")
    print('model("gpt-4o", temperature=0.5)("What is 2+2?")')

    # It's useful when you want to prioritize the prompt in the code:
    print("\nUseful when prioritizing the prompt:")
    print("result = complete(")
    print('    "Explain the significance of the year 1969 in space exploration.",')
    print('    model="gpt-4o",')
    print("    temperature=0.7,")
    print("    max_tokens=200")
    print(")")


def comparing_api_styles():
    """Compare the function-style API with the object-oriented API."""
    print("\n=== Comparing API Styles ===\n")

    # Function-style API (recommended)
    print("Function-style API (recommended):")
    print("from ember.api.models import model")
    print('response = model("gpt-4o")("What is the capital of France?")')

    # Object-oriented API (for backward compatibility)
    print("\nObject-oriented API (backward compatibility):")
    print("from ember.api.models import ModelAPI")
    print('api = ModelAPI("gpt-4o")')
    print('response = api.generate("What is the capital of France?")')

    # Builder pattern (for backward compatibility)
    print("\nBuilder pattern (backward compatibility):")
    print("from ember.api.models import ModelBuilder")
    print('model = ModelBuilder().temperature(0.7).max_tokens(100).build("gpt-4o")')
    print('response = model.generate("What is the capital of France?")')

    # Function-style API benefits
    print("\nFunction-style API benefits:")
    print("1. More intuitive and natural")
    print("2. More consistent with Python's functional programming style")
    print("3. Cleaner and more concise code")
    print("4. More flexible configuration management")
    print("5. Better IDE assistance with type hints")


def main():
    """Run all examples."""
    print("=== Function-Style Model API Examples ===")

    basic_usage()
    provider_namespaces()
    configuration_management()
    response_handling()
    complete_function()
    comparing_api_styles()

    print("\n=== End of Examples ===")


if __name__ == "__main__":
    main()
