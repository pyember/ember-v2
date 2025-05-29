"""
Example demonstrating the Ember Models API.

This file shows how to use the simplified models API to interact
with language models from different providers.

To run:
    uv run python src/ember/examples/models/model_api_example.py

    # Or if in an activated virtual environment
    python src/ember/examples/models/model_api_example.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key for OpenAI model examples
    ANTHROPIC_API_KEY (optional): Your Anthropic API key for Anthropic model examples
"""

import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the simplified models API
from ember.api import models


def basic_invocation_example():
    """Demonstrate basic model invocation."""
    print("\n=== Basic Invocation Example ===")
    
    try:
        # Direct invocation - the primary pattern
        print("Direct invocation:")
        print('response = models("gpt-4", "What is the capital of France?")')
        print("(Skipping actual API call to save credits)")
        
        # With parameters
        print("\nWith parameters:")
        print('response = models("claude-3", "Write a haiku", temperature=0.7, max_tokens=50)')
        print("(Skipping actual API call to save credits)")
        
    except Exception as e:
        print(f"Error in basic invocation: {e}")


def model_binding_example():
    """Demonstrate model binding for reuse."""
    print("\n=== Model Binding Example ===")
    
    try:
        # Bind a model with default parameters
        print("Binding a model for reuse:")
        gpt4 = models.instance("gpt-4", temperature=0.5)
        print(f"Created binding: {gpt4}")
        
        # Show how to use the bound model
        print("\nUsing bound model:")
        print('response1 = gpt4("Explain quantum computing")')
        print('response2 = gpt4("What is machine learning?")')
        print("(Skipping actual API calls to save credits)")
        
        # Override parameters on specific calls
        print("\nOverriding parameters:")
        print('response = gpt4("Tell me a joke", temperature=0.9)')
        print("(Skipping actual API call to save credits)")
        
    except Exception as e:
        print(f"Error in model binding: {e}")


def model_listing_example():
    """Demonstrate listing available models."""
    print("\n=== Model Listing Example ===")
    
    try:
        # List all available models
        available_models = models.list()
        print(f"Found {len(available_models)} available models:")
        
        # Show first 5 models
        for model_id in available_models[:5]:
            print(f"  - {model_id}")
        
        if len(available_models) > 5:
            print(f"  ... and {len(available_models) - 5} more")
        
        # List models by provider
        print("\nListing by provider:")
        openai_models = models.list(provider="openai")
        anthropic_models = models.list(provider="anthropic")
        
        print(f"OpenAI models: {len(openai_models)}")
        print(f"Anthropic models: {len(anthropic_models)}")
        
    except Exception as e:
        print(f"Error listing models: {e}")


def model_info_example():
    """Demonstrate getting model information."""
    print("\n=== Model Info Example ===")
    
    try:
        # Get info for specific models
        for model_id in ["gpt-4", "claude-3"]:
            print(f"\nInfo for {model_id}:")
            
            try:
                info = models.info(model_id)
                print(f"  Full ID: {info['id']}")
                print(f"  Provider: {info['provider']}")
                if 'context_window' in info:
                    print(f"  Context window: {info['context_window']} tokens")
                if 'pricing' in info:
                    pricing = info['pricing']
                    print(f"  Input cost: ${pricing.get('input', 0):.6f}/1K tokens")
                    print(f"  Output cost: ${pricing.get('output', 0):.6f}/1K tokens")
            except Exception as e:
                print(f"  Error getting info: {e}")
                
    except Exception as e:
        print(f"Error in model info: {e}")


def response_handling_example():
    """Demonstrate response object handling."""
    print("\n=== Response Handling Example ===")
    
    print("Response objects provide:")
    print("  - response.text: The generated text")
    print("  - response.usage: Token usage information")
    print("  - response.model: The model that generated the response")
    print("  - response.raw: The raw API response")
    
    print("\nExample usage:")
    print("""
    response = models("gpt-4", "Hello world")
    print(response.text)  # The generated text
    print(f"Tokens used: {response.usage['total_tokens']}")
    print(f"Cost: ${response.usage['cost']:.6f}")
    """)


def error_handling_example():
    """Demonstrate error handling."""
    print("\n=== Error Handling Example ===")
    
    print("The API raises specific exceptions:")
    print("  - AuthenticationError: Missing or invalid API key")
    print("  - RateLimitError: Rate limit exceeded")
    print("  - ModelNotFoundError: Model doesn't exist")
    print("  - ModelError: Other model-related errors")
    
    print("\nExample error handling:")
    print("""
    try:
        response = models("gpt-4", "Hello world")
    except AuthenticationError:
        print("Please set your OPENAI_API_KEY")
    except RateLimitError:
        print("Rate limit hit, please wait")
    except ModelNotFoundError:
        print("Model not found")
    except ModelError as e:
        print(f"Model error: {e}")
    """)


def main():
    """Example demonstrating the simplified XCS architecture."""
    """Run all examples in sequence."""
    print("=" * 60)
    print("Ember Models API Examples")
    print("=" * 60)
    
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set. Some examples may fail.")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set. Some examples may fail.")
    
    print("\nRunning examples...")
    
    # Run all examples
    basic_invocation_example()
    model_binding_example()
    model_listing_example()
    model_info_example()
    response_handling_example()
    error_handling_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()