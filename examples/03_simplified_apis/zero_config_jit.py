"""Zero Configuration JIT - Automatic optimization without complexity.

Shows how the simplified XCS system provides automatic optimization
with just @jit decorator - no strategies, modes, or configuration needed.
"""

from ember.api import models
from ember.api.xcs import jit, get_jit_stats
import time


def example_basic_jit():
    """Basic JIT usage - just add @jit."""
    print("\n=== Basic JIT Example ===\n")
    
    # Define a simple function
    def analyze_sentiment(text: str) -> str:
        """Analyze sentiment of text."""
        prompt = f"Analyze the sentiment of this text as positive, negative, or neutral: {text}"
        response = models("gpt-3.5-turbo", prompt)
        return response.text.strip()
    
    # Make it fast with zero configuration
    fast_analyze = jit(analyze_sentiment)
    
    # First call includes tracing overhead
    start = time.time()
    result1 = fast_analyze("I love this new API design!")
    print(f"First call: {result1} (took {time.time() - start:.2f}s)")
    
    # Subsequent calls are optimized
    start = time.time()
    result2 = fast_analyze("The weather is terrible today.")
    print(f"Second call: {result2} (took {time.time() - start:.2f}s)")
    
    # Check optimization stats
    stats = get_jit_stats()
    print(f"\nJIT Stats: {stats}")


def example_jit_with_complex_function():
    """JIT works with complex functions automatically."""
    print("\n=== Complex Function JIT ===\n")
    
    @jit  # That's it - no configuration needed!
    def analyze_and_summarize(article: str, max_points: int = 3) -> dict:
        """Complex function with multiple LLM calls."""
        # Extract key points
        points_prompt = f"Extract {max_points} key points from: {article}"
        points = models("gpt-3.5-turbo", points_prompt).text
        
        # Generate summary
        summary_prompt = f"Summarize in one sentence: {article}"
        summary = models("gpt-3.5-turbo", summary_prompt).text
        
        # Determine category
        category_prompt = f"Categorize this article (tech/business/science/other): {article}"
        category = models("gpt-3.5-turbo", category_prompt).text
        
        return {
            "key_points": points,
            "summary": summary,
            "category": category
        }
    
    # Use it naturally
    article = "Recent advances in AI have shown that simpler APIs often lead to better adoption..."
    result = analyze_and_summarize(article)
    print(f"Analysis complete: {result}")


def example_conditional_optimization():
    """JIT handles conditional logic automatically."""
    print("\n=== Conditional Logic with JIT ===\n")
    
    @jit
    def smart_responder(query: str, verbose: bool = False) -> str:
        """Function with conditional paths - JIT handles it all."""
        # Classify query type
        classification = models("gpt-3.5-turbo", f"Classify as question/statement/command: {query}").text
        
        if "question" in classification.lower():
            response = models("gpt-3.5-turbo", f"Answer this question: {query}").text
            if verbose:
                explanation = models("gpt-3.5-turbo", f"Explain the answer: {response}").text
                return f"{response}\n\nExplanation: {explanation}"
            return response
        elif "command" in classification.lower():
            return models("gpt-3.5-turbo", f"Respond to command: {query}").text
        else:
            return models("gpt-3.5-turbo", f"Acknowledge: {query}").text
    
    # Different execution paths, same simple API
    print(smart_responder("What is the capital of France?"))
    print(smart_responder("Turn on the lights", verbose=True))
    print(smart_responder("The weather is nice today"))


def main():
    """Run all zero-config JIT examples."""
    import os
    if not any(os.environ.get(key) for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']):
        print("\n⚠️  No API keys found. This example requires model API access.")
        print("Run 'ember init' to configure your API keys.\n")
        print("Showing example code structure instead...\n")
        
        # Show the code structure
        print("Example: Basic JIT usage")
        print("```python")
        print("@jit  # That's all you need!")
        print("def my_function(input):")
        print("    # Your logic here")
        print("    return result")
        print("```")
        return
    
    example_basic_jit()
    example_jit_with_complex_function()
    example_conditional_optimization()
    
    print("\n✨ Key takeaway: JIT just works - no configuration needed!")


if __name__ == "__main__":
    main()