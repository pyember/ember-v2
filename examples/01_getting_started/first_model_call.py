"""First Model Call - Direct LLM interaction with Ember's simple API.

Shows the key patterns for calling language models:
- Direct invocation for one-off calls
- Model binding for efficient reuse
- System prompts and parameters
- Cost tracking and usage monitoring

Example:
    >>> from ember.api import models
    >>> response = models("gpt-3.5-turbo", "What is AI?")
    >>> print(response.text)
    >>> print(f"Cost: ${response.usage['cost']:.4f}")
"""

import sys
from pathlib import Path

# Add the shared utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from _shared.conditional_execution import conditional_llm, SimulatedResponse


@conditional_llm(providers=["openai"])
def main(_simulated_mode=False):
    """Make your first LLM API call with Ember."""
    print_section_header("First Model Call")

    from ember.api import models

    # In simulated mode, we'll mock the models behavior
    if _simulated_mode:
        return run_simulated_example()

    try:

        # Method 1: Direct call - simplest approach
        print("Method 1: Direct model call")
        response = models("gpt-3.5-turbo", "What is machine learning in one sentence?")
        print_example_output("Response", response.text)
        print_example_output("Model used", response.model_id)

        # Method 2: Model binding - reusable configuration
        print("\nMethod 2: Model binding (reusable)")
        gpt = models.instance("gpt-3.5-turbo", temperature=0.7)

        response1 = gpt("Explain quantum computing to a 5-year-old")
        print_example_output("Creative response", response1.text[:100] + "...")

        # Method 3: With system prompt
        print("\nMethod 3: With system prompt")
        helpful_gpt = models.instance(
            "gpt-3.5-turbo",
            system="You are a helpful assistant who gives concise answers.",
        )

        response2 = helpful_gpt("What is the capital of France?")
        print_example_output("Concise response", response2.text)

        # Show response metadata
        print("\nResponse metadata:")
        print_example_output("Prompt tokens", response2.usage["prompt_tokens"])
        print_example_output("Completion tokens", response2.usage["completion_tokens"])
        print_example_output("Total tokens", response2.usage["total_tokens"])
        print_example_output("Estimated cost", f"${response2.usage['cost']:.4f}")

        # Bonus: Show how to use different models
        print("\nBonus: Using different models")
        # You can use any supported model
        models_to_try = [
            "gpt-4",  # More capable but more expensive
            "claude-3-opus",  # Anthropic's model
            "gemini-pro",  # Google's model
        ]
        print("Available models (with API keys):")
        for model in models_to_try:
            print(f"  - {model}")

        print("\n✅ Successfully made LLM API calls!")
        print("\nKey takeaways:")
        print("  - models() for direct calls")
        print("  - models.instance() for reusable configurations")
        print("  - Automatic cost tracking in response.usage")
        print("  - Same API works with any provider")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure your API key is set correctly")
        print("2. Check your internet connection")
        print("3. Verify your API key has credits")
        return 1

    return 0


def run_simulated_example():
    """Run example with simulated responses."""
    # Method 1: Direct call - simplest approach
    print("Method 1: Direct model call")
    response = SimulatedResponse(
        text=(
            "Machine learning is a type of artificial intelligence that "
            "enables computers to learn and improve from experience without "
            "being explicitly programmed."
        ),
        model_id="gpt-3.5-turbo",
    )
    print_example_output("Response", response.text)
    print_example_output("Model used", response.model_id)

    # Method 2: Model binding - reusable configuration
    print("\nMethod 2: Model binding (reusable)")
    response1 = SimulatedResponse(
        text=(
            "Imagine you have a magic toy that gets smarter every time you "
            "play with it. Quantum computing is like having a super magic toy "
            "that can think about many different games at the same time, "
            "instead of just one game like regular toys!"
        ),
        model_id="gpt-3.5-turbo",
    )
    print_example_output("Creative response", response1.text[:100] + "...")

    # Method 3: With system prompt
    print("\nMethod 3: With system prompt")
    response2 = SimulatedResponse(text="Paris.", model_id="gpt-3.5-turbo")
    print_example_output("Concise response", response2.text)

    # Show response metadata
    print("\nResponse metadata:")
    print_example_output("Prompt tokens", response2.usage["prompt_tokens"])
    print_example_output("Completion tokens", response2.usage["completion_tokens"])
    print_example_output("Total tokens", response2.usage["total_tokens"])
    print_example_output("Estimated cost", f"${response2.usage['cost']:.4f}")

    # Bonus: Show how to use different models
    print("\nBonus: Using different models")
    models_to_try = [
        "gpt-4",  # More capable but more expensive
        "claude-3-opus",  # Anthropic's model
        "gemini-pro",  # Google's model
    ]
    print("Available models (with API keys):")
    for model in models_to_try:
        print(f"  - {model}")

    print("\n✅ Successfully demonstrated LLM API calls!")
    print("\nKey takeaways:")
    print("  - models() for direct calls")
    print("  - models.instance() for reusable configurations")
    print("  - Automatic cost tracking in response.usage")
    print("  - Same API works with any provider")

    return 0


if __name__ == "__main__":
    sys.exit(main())
