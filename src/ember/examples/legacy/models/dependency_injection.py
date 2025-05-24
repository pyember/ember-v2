"""Dependency Injection and Configuration Example

This example demonstrates how to use different model configurations
for various use cases with the simplified API.

To run:
    uv run python src/ember/examples/models/dependency_injection.py

Required environment variables:
    OPENAI_API_KEY (optional): Your OpenAI API key
    ANTHROPIC_API_KEY (optional): Your Anthropic API key
"""

import os
from typing import Dict, List

from ember.api import models


def different_configurations():
    """Demonstrate using models with different configurations."""
    print("\n=== Different Configurations ===\n")
    
    print("The simplified API allows per-call configuration:")
    
    # Example 1: Low temperature for factual responses
    print("\n1. Factual response (low temperature):")
    print('models("gpt-4", "What is the capital of France?", temperature=0.1)')
    
    # Example 2: High temperature for creative responses  
    print("\n2. Creative response (high temperature):")
    print('models("gpt-4", "Write a poem about Paris", temperature=0.9)')
    
    # Example 3: Limited tokens for summaries
    print("\n3. Brief summary (limited tokens):")
    print('models("gpt-4", "Summarize War and Peace", max_tokens=50)')
    
    print("\nEach call can have independent configuration without context management.")


def model_binding_patterns():
    """Show how to create reusable model configurations."""
    print("\n=== Model Binding Patterns ===\n")
    
    print("Create reusable configurations with model binding:")
    
    # Create bound models with different configs
    print("\n# Create specialized models")
    print('factual_model = models.instance("gpt-4", temperature=0.1)')
    print('creative_model = models.instance("gpt-4", temperature=0.9)')
    print('summary_model = models.instance("gpt-4", max_tokens=100)')
    
    print("\n# Use them repeatedly")
    print('factual_model("What is gravity?")')
    print('creative_model("Write a story about gravity")')
    print('summary_model("Explain quantum mechanics")')
    
    # Show actual binding if API is available
    if os.environ.get("OPENAI_API_KEY"):
        try:
            factual = models.instance("gpt-4", temperature=0.1)
            print(f"\nActual binding created: {factual}")
        except Exception as e:
            print(f"\n(Skipping actual binding: {e})")


def ab_testing_pattern():
    """Demonstrate A/B testing with different configurations."""
    print("\n=== A/B Testing Pattern ===\n")
    
    def run_ab_test(prompt: str, configs: Dict[str, Dict]) -> Dict[str, str]:
        """Run A/B test with different model configurations."""
        results = {}
        
        for variant_name, config in configs.items():
            print(f"\nTesting {variant_name}:")
            print(f"  Model: {config['model']}")
            print(f"  Params: {config['params']}")
            
            # In real usage:
            # response = models(config['model'], prompt, **config['params'])
            # results[variant_name] = response.text
            
            # For demo:
            results[variant_name] = f"Mock response for {variant_name}"
        
        return results
    
    # Define test variants
    test_configs = {
        "Conservative": {
            "model": "gpt-4",
            "params": {"temperature": 0.2, "top_p": 0.9}
        },
        "Balanced": {
            "model": "gpt-4", 
            "params": {"temperature": 0.5, "top_p": 0.95}
        },
        "Creative": {
            "model": "gpt-4",
            "params": {"temperature": 0.8, "top_p": 1.0}
        }
    }
    
    # Run test
    results = run_ab_test(
        "Write a tagline for a new coffee shop",
        test_configs
    )
    
    print("\nResults collected for analysis")


def custom_context_example():
    """Show how to use custom contexts when needed."""
    print("\n=== Custom Context Example ===\n")
    
    print("For advanced use cases, you can create custom contexts:")
    
    print("""
from ember.api.models import create_context, ContextConfig

# Create context with specific API keys
dev_context = create_context(
    config=ContextConfig(
        api_keys={"openai": "dev-key", "anthropic": "dev-key"}
    )
)

# Use models with custom context
response = models("gpt-4", "Hello", context=dev_context)
""")
    
    print("\nThis is useful for:")
    print("- Testing with different API keys")
    print("- Isolated environments (dev/staging/prod)")
    print("- Multi-tenant applications")


def configuration_hierarchy():
    """Explain configuration hierarchy."""
    print("\n=== Configuration Hierarchy ===\n")
    
    print("Configuration precedence (highest to lowest):")
    print("\n1. Call-level parameters:")
    print('   models("gpt-4", "Hello", temperature=0.9)')
    
    print("\n2. Bound model parameters:")
    print('   gpt4 = models.instance("gpt-4", temperature=0.7)')
    print('   gpt4("Hello")  # Uses temperature=0.7')
    
    print("\n3. Context configuration:")
    print('   # When using custom contexts')
    
    print("\n4. Environment defaults:")
    print('   # System-wide defaults')
    
    print("\nThis allows fine-grained control over model behavior.")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Dependency Injection and Configuration Examples")
    print("=" * 60)
    
    different_configurations()
    model_binding_patterns()
    ab_testing_pattern()
    custom_context_example()
    configuration_hierarchy()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()