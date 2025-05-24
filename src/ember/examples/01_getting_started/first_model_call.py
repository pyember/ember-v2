"""
Example: First Model Call - Make Your First LLM API Call
Difficulty: Basic
Time: ~2 minutes
Prerequisites: hello_world.py

Learning Objectives:
- Make your first LLM API call
- Understand the models API
- Handle model responses

Key Concepts:
- models API
- Model responses
- Error handling
"""

import sys
from pathlib import Path

# Add the shared utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output, ensure_api_key


def main():
    """Make your first LLM API call with Ember."""
    print_section_header("First Model Call")
    
    # Check for API key
    if not ensure_api_key("openai"):
        print("\nNote: This example requires an OpenAI API key.")
        print("You can also modify it to use other providers like Anthropic or Google.")
        
        # Show mock example instead
        print("\n" + "="*50)
        print("Running in demo mode without API key...")
        print("="*50 + "\n")
        demo_mode()
        return 0
    
    try:
        from ember.api import models
        
        # Method 1: Direct call - simplest approach
        print("Method 1: Direct model call")
        response = models("gpt-3.5-turbo", "What is machine learning in one sentence?")
        print_example_output("Response", response.text)
        print_example_output("Model used", response.model)
        
        # Method 2: Model binding - reusable configuration
        print("\nMethod 2: Model binding (reusable)")
        gpt = models.instance("gpt-3.5-turbo", temperature=0.7)
        
        response1 = gpt("Explain quantum computing to a 5-year-old")
        print_example_output("Creative response", response1.text[:100] + "...")
        
        # Method 3: With system prompt
        print("\nMethod 3: With system prompt")
        helpful_gpt = models.instance(
            "gpt-3.5-turbo",
            system="You are a helpful assistant who gives concise answers."
        )
        
        response2 = helpful_gpt("What is the capital of France?")
        print_example_output("Concise response", response2.text)
        
        # Show response metadata
        print("\nResponse metadata:")
        print_example_output("Total tokens", response2.usage.get("total_tokens", "N/A"))
        print_example_output("Estimated cost", f"${response2.usage.get('cost', 0):.4f}")
        
        print("\n✅ Successfully made LLM API calls!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure your API key is set correctly")
        print("2. Check your internet connection")
        print("3. Verify your API key has credits")
        return 1
    
    return 0


def demo_mode():
    """Demo mode when no API key is available."""
    from ember.api import models
    
    print("Here's how you would use the models API:\n")
    
    print("# Direct call")
    print('response = models("gpt-3.5-turbo", "Your prompt here")')
    print('print(response.text)\n')
    
    print("# Model binding for reuse")
    print('gpt = models.instance("gpt-3.5-turbo", temperature=0.7)')
    print('response = gpt("Your prompt")\n')
    
    print("# With system prompt")
    print('assistant = models.instance(')
    print('    "gpt-3.5-turbo",')
    print('    system="You are a helpful assistant."')
    print(')')
    print('response = assistant("Help me with...")')
    
    print("\n📝 To run this example with real API calls:")
    print("export OPENAI_API_KEY='your-key-here'")


if __name__ == "__main__":
    sys.exit(main())