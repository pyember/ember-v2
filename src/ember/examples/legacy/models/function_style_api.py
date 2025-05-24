"""Function-Style Model API Example

This example demonstrates the simplified models API that provides
a clean, function-based interface for working with language models.

To run:
    uv run python src/ember/examples/models/function_style_api.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key
    ANTHROPIC_API_KEY (optional): Your Anthropic API key
"""

import os
from ember.api import models


def basic_usage():
    """Demonstrate basic usage of the simplified API."""
    print("\n=== Basic Usage ===\n")
    
    # The simplest way - direct invocation
    print("Direct invocation:")
    print('response = models("gpt-4", "What is the capital of France?")')
    print("# Returns: The capital of France is Paris.")
    
    # With parameters
    print("\nWith parameters:")
    print('response = models("claude-3", "Write a haiku", temperature=0.7, max_tokens=50)')
    print("# Returns: A beautiful haiku...")
    
    # Note about API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Note: Set OPENAI_API_KEY to run actual examples")


def model_binding():
    """Demonstrate model binding for reuse."""
    print("\n=== Model Binding ===\n")
    
    # Bind a model with default parameters
    print("Binding for reuse:")
    print('gpt4 = models.instance("gpt-4", temperature=0.5)')
    print('response1 = gpt4("Explain quantum computing")')
    print('response2 = gpt4("What is machine learning?")')
    
    # Override parameters on specific calls
    print("\nOverriding bound parameters:")
    print('response = gpt4("Tell me a joke", temperature=0.9)')
    
    # Show binding representation
    try:
        gpt4 = models.instance("gpt-4", temperature=0.5)
        print(f"\nBinding representation: {gpt4}")
    except Exception:
        print("\nBinding representation: ModelBinding(model_id='gpt-4', params={'temperature': 0.5})")


def response_object():
    """Demonstrate the Response object features."""
    print("\n=== Response Object ===\n")
    
    print("Response attributes:")
    print("response.text         # The generated text")
    print("response.usage        # Token usage information")
    print("  .total_tokens       # Total tokens used")
    print("  .prompt_tokens      # Input tokens")
    print("  .completion_tokens  # Output tokens")
    print("  .cost              # Estimated cost in USD")
    print("response.model        # Model that generated the response")
    print("response.raw          # Raw API response")
    
    print("\nExample usage:")
    print("""
    response = models("gpt-4", "Hello world")
    print(response.text)
    print(f"Cost: ${response.usage['cost']:.6f}")
    print(f"Tokens: {response.usage['total_tokens']}")
    """)


def error_handling():
    """Demonstrate error handling patterns."""
    print("\n=== Error Handling ===\n")
    
    print("Exception types:")
    print("- AuthenticationError: Missing or invalid API key")
    print("- RateLimitError: Rate limit exceeded")
    print("- ModelNotFoundError: Model doesn't exist")
    print("- ModelError: Other model-related errors")
    
    print("\nError handling pattern:")
    print("""
    from ember.core.registry.model.base.errors import (
        AuthenticationError,
        RateLimitError,
        ModelNotFoundError,
        ModelError
    )
    
    try:
        response = models("gpt-4", "Hello")
    except AuthenticationError:
        print("Please set your API key")
    except RateLimitError:
        print("Rate limit hit, please wait")
    except ModelNotFoundError:
        print("Model not found")
    except ModelError as e:
        print(f"Error: {e}")
    """)


def listing_models():
    """Demonstrate model discovery and listing."""
    print("\n=== Model Discovery ===\n")
    
    print("List all models:")
    print("available = models.list()")
    print("# Returns: ['openai:gpt-4', 'anthropic:claude-3', ...]")
    
    print("\nList by provider:")
    print("openai_models = models.list(provider='openai')")
    print("# Returns: ['openai:gpt-4', 'openai:gpt-3.5-turbo', ...]")
    
    print("\nGet model info:")
    print("info = models.info('gpt-4')")
    print("# Returns: {")
    print("#   'id': 'openai:gpt-4',")
    print("#   'provider': 'openai',")
    print("#   'context_window': 8192,")
    print("#   'pricing': {'input': 0.03, 'output': 0.06}")
    print("# }")


def advanced_patterns():
    """Show advanced usage patterns."""
    print("\n=== Advanced Patterns ===\n")
    
    print("1. Structured outputs (when supported):")
    print("""
    from pydantic import BaseModel
    
    class Summary(BaseModel):
        title: str
        key_points: list[str]
        sentiment: str
    
    response = models("gpt-4", "Summarize this article...", 
                     response_model=Summary)
    # response.text is now a Summary instance
    """)
    
    print("\n2. Streaming responses:")
    print("""
    for chunk in models.stream("gpt-4", "Tell me a long story"):
        print(chunk, end="", flush=True)
    """)
    
    print("\n3. Custom context:")
    print("""
    from ember.api.models import create_context, ContextConfig
    
    # Create custom context with specific API keys
    context = create_context(
        config=ContextConfig(
            api_keys={"openai": "sk-..."}
        )
    )
    
    # Use models with custom context
    response = models("gpt-4", "Hello", context=context)
    """)


def migration_guide():
    """Show how to migrate from old patterns."""
    print("\n=== Migration Guide ===\n")
    
    print("Old pattern → New pattern:")
    print()
    
    print("1. Model initialization:")
    print("OLD:  registry = initialize_registry()")
    print("      model = registry.get_model('gpt-4')")
    print("NEW:  # Just use models() directly")
    print("      response = models('gpt-4', 'Hello')")
    print()
    
    print("2. Model service:")
    print("OLD:  service = ModelService(registry)")
    print("      response = service.invoke_model('gpt-4', 'Hello')")
    print("NEW:  response = models('gpt-4', 'Hello')")
    print()
    
    print("3. Direct model access:")
    print("OLD:  from ember.core.registry.model import lm")
    print("      response = lm('gpt-4', prompt='Hello')")
    print("NEW:  from ember.api import models")
    print("      response = models('gpt-4', 'Hello')")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Function-Style Model API Examples")
    print("=" * 60)
    
    basic_usage()
    model_binding()
    response_object()
    error_handling()
    listing_models()
    advanced_patterns()
    migration_guide()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()