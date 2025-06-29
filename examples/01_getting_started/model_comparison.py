"""Model Comparison - Compare different language models side by side.

Learn how to compare responses from different models to understand
their strengths and make informed choices for your use cases.

Example:
    >>> from ember.api import models
    >>> gpt4_response = models("gpt-4", "Explain quantum computing")
    >>> gpt35_response = models("gpt-3.5-turbo", "Explain quantum computing")
"""

import sys
from pathlib import Path

# Add the shared utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import (
    print_section_header,
    print_example_output,
    ensure_api_key,
)
from _shared.conditional_execution import conditional_llm, SimulatedResponse


def compare_models_simple(_simulated_mode=False):
    """Compare basic responses from different models."""
    print("\n=== Simple Model Comparison ===\n")

    from ember.api import models

    prompt = "Explain machine learning in one sentence."

    # Try different models
    models_to_test = [
        "gpt-3.5-turbo",
        "gpt-4",
        # Add more models as needed
    ]

    responses = {}
    for model_name in models_to_test:
        try:
            if _simulated_mode:
                if model_name == "gpt-3.5-turbo":
                    response = SimulatedResponse(
                        text="Machine learning is a type of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
                    )
                else:
                    response = SimulatedResponse(
                        text="Machine learning is a field of AI where algorithms automatically improve their performance on tasks through exposure to data and experience."
                    )
            else:
                response = models(model_name, prompt)

            responses[model_name] = response.text.strip()
            print(f"{model_name}:")
            print(f"  {response.text.strip()}\n")
        except Exception as e:
            print(f"{model_name}: Error - {e}\n")

    return responses


def compare_models_detailed(_simulated_mode=False):
    """Compare models with different parameters."""
    print("\n=== Detailed Model Comparison ===\n")

    from ember.api import models

    prompt = "Write a haiku about programming."

    # Test with different temperatures
    configurations = [
        ("gpt-3.5-turbo", {"temperature": 0.3}),
        ("gpt-3.5-turbo", {"temperature": 0.9}),
        ("gpt-4", {"temperature": 0.3}),
        ("gpt-4", {"temperature": 0.9}),
    ]

    print(f"Prompt: {prompt}\n")

    simulated_haikus = {
        (
            0.3,
            "gpt-3.5-turbo",
        ): "Lines of code flow down\nBugs hide in the syntax tree\nDebugger finds peace",
        (
            0.9,
            "gpt-3.5-turbo",
        ): "Dancing pixels glow\nKeyboard whispers dreams in code\nAlgorithms bloom bright",
        (
            0.3,
            "gpt-4",
        ): "Logic gates align\nAbstraction layers build high\nSoftware comes to life",
        (
            0.9,
            "gpt-4",
        ): "Binary moonlight\nRecursion dreams infinite\nStack overflow love",
    }

    for model_name, params in configurations:
        try:
            if _simulated_mode:
                key = (params["temperature"], model_name)
                text = simulated_haikus.get(
                    key,
                    "Code flows like water\nThrough silicon valleys deep\nBits become meaning",
                )
                response = SimulatedResponse(text=text, usage={"total_tokens": 42})
            else:
                response = models(model_name, prompt, **params)

            print(f"{model_name} (temp={params['temperature']}):")
            print(f"  {response.text.strip()}")
            print(f"  Tokens: {response.usage.get('total_tokens', 'N/A')}\n")
        except Exception as e:
            print(f"{model_name}: Error - {e}\n")


def compare_model_costs(_simulated_mode=False):
    """Compare token usage and costs across models."""
    print("\n=== Cost Comparison ===\n")

    from ember.api import models

    # Same prompt for fair comparison
    prompt = """Summarize the key principles of object-oriented programming 
    including encapsulation, inheritance, and polymorphism."""

    models_to_compare = ["gpt-3.5-turbo", "gpt-4"]

    print(f"Prompt length: {len(prompt)} characters\n")

    for model_name in models_to_compare:
        try:
            if _simulated_mode:
                if model_name == "gpt-3.5-turbo":
                    response = SimulatedResponse(
                        text="OOP centers on three principles: encapsulation bundles data and methods, inheritance enables code reuse through hierarchies, and polymorphism allows objects to take multiple forms.",
                        usage={
                            "prompt_tokens": 28,
                            "completion_tokens": 35,
                            "total_tokens": 63,
                        },
                    )
                else:
                    response = SimulatedResponse(
                        text="Object-oriented programming is built on three fundamental principles: Encapsulation protects data by bundling it with methods that operate on it, controlling access through interfaces. Inheritance creates hierarchical relationships between classes, allowing derived classes to reuse and extend parent functionality. Polymorphism enables a single interface to represent different underlying forms, letting objects respond to the same method call in their own specific way.",
                        usage={
                            "prompt_tokens": 28,
                            "completion_tokens": 78,
                            "total_tokens": 106,
                        },
                    )
            else:
                response = models(model_name, prompt)

            usage = response.usage

            print(f"{model_name}:")
            print(f"  Response length: {len(response.text)} characters")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")

            # Estimate costs (example rates)
            if model_name == "gpt-3.5-turbo":
                cost_per_1k = 0.002  # $0.002 per 1K tokens
            else:  # gpt-4
                cost_per_1k = 0.03  # $0.03 per 1K tokens

            if isinstance(usage.get("total_tokens"), int):
                est_cost = (usage["total_tokens"] / 1000) * cost_per_1k
                print(f"  Estimated cost: ${est_cost:.4f}\n")
            else:
                print(f"  Estimated cost: N/A\n")

        except Exception as e:
            print(f"{model_name}: Error - {e}\n")


def compare_model_capabilities(_simulated_mode=False):
    """Compare different capabilities across models."""
    print("\n=== Capability Comparison ===\n")

    from ember.api import models

    # Test different types of tasks
    tasks = {
        "Reasoning": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "Code": "Write a Python function to reverse a string without using slicing.",
        "Creative": "Create a metaphor comparing software development to cooking.",
    }

    simulated_responses = {
        (
            "Reasoning",
            "gpt-3.5-turbo",
        ): "No, we cannot conclude that some roses fade quickly. The statement only tells us that some flowers fade quickly, but not which specific flowers.",
        (
            "Reasoning",
            "gpt-4",
        ): "No, we cannot make that conclusion. This is a logical fallacy. While all roses are flowers, and some flowers fade quickly, we don't know if roses are among the flowers that fade quickly. The 'some flowers' that fade quickly might be entirely different types of flowers, not roses.",
        (
            "Code",
            "gpt-3.5-turbo",
        ): "def reverse_string(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result",
        (
            "Code",
            "gpt-4",
        ): "def reverse_string(s):\n    # Using two pointers approach\n    chars = list(s)\n    left, right = 0, len(chars) - 1\n    while left < right:\n        chars[left], chars[right] = chars[right], chars[left]\n        left += 1\n        right -= 1\n    return ''.join(chars)",
        (
            "Creative",
            "gpt-3.5-turbo",
        ): "Software development is like cooking a complex meal where recipes are documentation, ingredients are libraries, and the chef is the developer combining everything to create a satisfying dish.",
        (
            "Creative",
            "gpt-4",
        ): "Software development is like being a master chef in a kitchen that never closes. Your code is the recipe, constantly refined through taste tests (debugging). Libraries are your pantry of pre-made ingredients, saving time but requiring skill to combine properly. Each feature is a dish that must harmonize with the full menu (system), and just like in cooking, the simplest solutions often produce the most elegant results.",
    }

    for task_name, task_prompt in tasks.items():
        print(f"\n{task_name} Task:")
        print(f"Prompt: {task_prompt[:50]}...\n")

        for model_name in ["gpt-3.5-turbo", "gpt-4"]:
            try:
                if _simulated_mode:
                    key = (task_name, model_name)
                    text = simulated_responses.get(
                        key, f"Simulated {task_name} response for {model_name}"
                    )
                    response = SimulatedResponse(text=text)
                else:
                    response = models(model_name, task_prompt)

                print(f"{model_name}:")
                # Truncate long responses for display
                text = response.text.strip()
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"  {text}\n")

            except Exception as e:
                print(f"{model_name}: Error - {e}\n")


def demo_mode():
    """Show example outputs without making API calls."""
    print("\n📋 Demo Mode - Showing example outputs:\n")

    print("GPT-3.5-turbo typically provides:")
    print("  • Concise, practical responses")
    print("  • Good for most general tasks")
    print("  • Fast response times")
    print("  • Lower cost per token\n")

    print("GPT-4 typically provides:")
    print("  • More detailed, nuanced responses")
    print("  • Better reasoning capabilities")
    print("  • Superior code generation")
    print("  • Higher cost but better quality")


def run_simulated_example():
    """Run the example with simulated responses."""
    compare_models_simple(_simulated_mode=True)
    compare_models_detailed(_simulated_mode=True)
    compare_model_costs(_simulated_mode=True)
    compare_model_capabilities(_simulated_mode=True)


@conditional_llm(providers=["openai"])
def main(_simulated_mode=False):
    """Run all model comparison examples."""
    print_section_header("Model Comparison Guide")

    if _simulated_mode:
        print("🔧 Running in simulated mode (no API keys detected)")
        print("To run this example with real API calls, set one of:")
        print("export OPENAI_API_KEY='your-key-here'\n")
        print("Simulated output will demonstrate the expected behavior.\n")
        return run_simulated_example()

    try:
        # Run comparisons
        compare_models_simple()
        compare_models_detailed()
        compare_model_costs()
        compare_model_capabilities()

        print("\n" + "=" * 50)
        print("✅ Key Insights")
        print("=" * 50)
        print("\n• GPT-4 provides more detailed, nuanced responses")
        print("• GPT-3.5-turbo is faster and more cost-effective")
        print("• Temperature affects creativity vs consistency")
        print("• Choose models based on your specific needs")
        print("\nNext: Learn prompt engineering in 'basic_prompt_engineering.py'")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("• Check your API key is valid")
        print("• Ensure you have API credits")
        print("• Check your internet connection")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
