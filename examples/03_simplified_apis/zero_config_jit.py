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
    
    # Mock functions for demonstration (avoiding model calls in examples)
    def extract_points(text: str, max_points: int) -> str:
        return f"Key point 1, Key point 2, Key point {max_points}"
    
    def generate_summary(text: str) -> str:
        return f"Summary of: {text[:50]}..."
    
    def categorize_text(text: str) -> str:
        return "tech" if "AI" in text else "general"
    
    @jit  # That's it - no configuration needed!
    def analyze_and_summarize(article: str, max_points: int = 3) -> dict:
        """Complex function with multiple processing steps."""
        # Extract key points
        points = extract_points(article, max_points)
        
        # Generate summary  
        summary = generate_summary(article)
        
        # Determine category
        category = categorize_text(article)
        
        return {
            "key_points": points,
            "summary": summary,
            "category": category
        }
    
    # Use it naturally
    article = "Recent advances in AI have shown that simpler APIs often lead to better adoption..."
    result = analyze_and_summarize(article)
    print(f"Analysis complete: {result}")
    print(f"Function stats: {analyze_and_summarize.stats()}")


def example_conditional_optimization():
    """JIT handles conditional logic automatically."""
    print("\n=== Conditional Logic with JIT ===\n")
    
    # Mock functions for demonstration
    def classify_query(query: str) -> str:
        if "?" in query:
            return "question"
        elif any(cmd in query.lower() for cmd in ["turn", "set", "start", "stop"]):
            return "command"
        else:
            return "statement"
    
    def process_question(query: str) -> str:
        return f"Answer to: {query}"
    
    def process_command(query: str) -> str:
        return f"Executing: {query}"
    
    def process_statement(query: str) -> str:
        return f"Acknowledged: {query}"
    
    @jit
    def smart_responder(query: str, verbose: bool = False) -> str:
        """Function with conditional paths - JIT handles it all."""
        # Classify query type
        classification = classify_query(query)
        
        if classification == "question":
            response = process_question(query)
            if verbose:
                response += f"\n\nExplanation: This was classified as a {classification}"
            return response
        elif classification == "command":
            return process_command(query)
        else:
            return process_statement(query)
    
    # Different execution paths, same simple API
    print(smart_responder("What is the capital of France?"))
    print(smart_responder("Turn on the lights", verbose=True))
    print(smart_responder("The weather is nice today"))
    print(f"Function stats: {smart_responder.stats()}")


def main():
    """Run all zero-config JIT examples."""
    import os
    
    # Check if we have API keys for the basic example
    has_api_keys = any(os.environ.get(key) for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'])
    
    if has_api_keys:
        example_basic_jit()
    else:
        print("\n⚠️  No API keys found for model calls.")
        print("Skipping basic model example, showing JIT with regular functions...\n")
    
    # These examples work without API keys since they use mock functions
    example_jit_with_complex_function()
    example_conditional_optimization()
    
    print("\n✨ Key takeaway: JIT just works - no configuration needed!")
    
    # Show overall stats
    from ember.api.xcs import get_jit_stats
    final_stats = get_jit_stats()
    print(f"\nOverall JIT Stats: {final_stats}")


if __name__ == "__main__":
    main()