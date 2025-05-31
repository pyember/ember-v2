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

from _shared.example_utils import print_section_header, print_example_output, ensure_api_key


def compare_models_simple():
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
            response = models(model_name, prompt)
            responses[model_name] = response.text.strip()
            print(f"{model_name}:")
            print(f"  {response.text.strip()}\n")
        except Exception as e:
            print(f"{model_name}: Error - {e}\n")
    
    return responses


def compare_models_detailed():
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
    
    for model_name, params in configurations:
        try:
            response = models(model_name, prompt, **params)
            print(f"{model_name} (temp={params['temperature']}):")
            print(f"  {response.text.strip()}")
            print(f"  Tokens: {response.usage.get('total_tokens', 'N/A')}\n")
        except Exception as e:
            print(f"{model_name}: Error - {e}\n")


def compare_model_costs():
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
            response = models(model_name, prompt)
            usage = response.usage
            
            print(f"{model_name}:")
            print(f"  Response length: {len(response.text)} characters")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
            print(f"  Estimated cost: ${usage.get('cost', 0):.4f}\n")
        except Exception as e:
            print(f"{model_name}: Error - {e}\n")


def compare_model_capabilities():
    """Test different capabilities across models."""
    print("\n=== Capability Comparison ===\n")
    
    from ember.api import models
    
    # Different types of tasks
    tasks = [
        ("Reasoning", "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."),
        ("Code Generation", "Write a Python function to calculate fibonacci numbers efficiently."),
        ("Creative Writing", "Write a creative metaphor for artificial intelligence."),
        ("Data Analysis", "Given sales data: Q1: $100k, Q2: $120k, Q3: $90k, Q4: $150k. What's the trend and what might explain Q3?"),
    ]
    
    models_to_test = ["gpt-3.5-turbo", "gpt-4"]
    
    for task_name, prompt in tasks:
        print(f"\n--- {task_name} ---")
        print(f"Prompt: {prompt[:100]}...\n" if len(prompt) > 100 else f"Prompt: {prompt}\n")
        
        for model_name in models_to_test:
            try:
                response = models(model_name, prompt, max_tokens=150)
                print(f"{model_name}:")
                print(f"  {response.text.strip()[:200]}...\n" if len(response.text) > 200 else f"  {response.text.strip()}\n")
            except Exception as e:
                print(f"{model_name}: Error - {e}\n")


def demo_mode():
    """Show example comparisons without making actual API calls."""
    print("\n=== Demo Mode: Model Comparison Examples ===\n")
    
    print("1. Response Quality Comparison:")
    print("   GPT-3.5: Provides good, concise answers")
    print("   GPT-4: More detailed, nuanced responses\n")
    
    print("2. Cost Comparison (approximate):")
    print("   GPT-3.5-turbo: ~$0.002 per 1K tokens")
    print("   GPT-4: ~$0.03 per 1K tokens (15x more expensive)\n")
    
    print("3. Speed Comparison:")
    print("   GPT-3.5-turbo: Faster response times")
    print("   GPT-4: Slower but more thoughtful\n")
    
    print("4. Use Case Recommendations:")
    print("   • Simple tasks, high volume: GPT-3.5-turbo")
    print("   • Complex reasoning, quality critical: GPT-4")
    print("   • Creative writing: Both work well, GPT-4 more sophisticated")
    print("   • Code generation: GPT-4 generally better\n")


def main():
    """Run all model comparison examples."""
    print_section_header("Model Comparison Guide")
    
    # Check for API key
    if not ensure_api_key("openai"):
        print("\n⚠️  No OpenAI API key found.")
        print("This example compares different OpenAI models.\n")
        demo_mode()
        print("\n📝 To run real comparisons:")
        print("export OPENAI_API_KEY='your-key-here'")
        return 0
    
    try:
        # Run comparisons
        compare_models_simple()
        compare_models_detailed()
        compare_model_costs()
        compare_model_capabilities()
        
        print("\n" + "="*50)
        print("✅ Key Insights")
        print("="*50)
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