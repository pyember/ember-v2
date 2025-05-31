"""Operators Basics - Functions are operators in Ember's simplified API.

Learn how any function can be an operator for building AI systems.

Example:
    >>> def clean_text(text: str) -> dict:
    ...     return {"cleaned": text.strip().lower()}
    >>> result = clean_text("  Hello WORLD  ")
    >>> print(result["cleaned"])
    "hello world"
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import models
from ember.api.xcs import jit, vmap


def main():
    """Learn the fundamentals of Ember operators."""
    print_section_header("Understanding Operators")
    
    # Part 1: Why Operators?
    print("🎯 The Simplified Approach\n")
    print("In Ember's simplified API, ANY function is an operator!")
    print("Benefits:")
    print("  • No base classes needed")
    print("  • Natural Python code")
    print("  • Automatic optimization with @jit")
    print("  • Easy composition")
    print("  • Batch processing with vmap()\n")
    
    # Part 2: Simple Operator
    print("="*50)
    print("Part 1: A Simple Function as Operator")
    print("="*50 + "\n")
    
    # Just write a function - it's an operator!
    def clean_text(text: str) -> dict:
        """Cleans and normalizes text input."""
        cleaned = text.strip().lower()
        cleaned = " ".join(cleaned.split())  # Normalize whitespace
        
        return {
            "original": text,
            "cleaned": cleaned,
            "length": len(cleaned)
        }
    
    # Use it directly
    result = clean_text("  Hello   WORLD!  ")
    
    print("Text Cleaner Results:")
    print_example_output("Original", repr(result["original"]))
    print_example_output("Cleaned", result["cleaned"])
    print_example_output("Length", result["length"])
    
    # Part 3: Operator with Configuration
    print("\n" + "="*50)
    print("Part 2: Configurable Function")
    print("="*50 + "\n")
    
    # Use closures or partial functions for configuration
    def create_word_counter(min_length: int = 1):
        """Create a word counter with minimum length filter."""
        def count_words(text: str) -> dict:
            words = text.split()
            
            # Filter by minimum length
            filtered_words = [w for w in words if len(w) >= min_length]
            
            return {
                "total_words": len(words),
                "filtered_words": len(filtered_words),
                "words": filtered_words[:5]  # First 5 as sample
            }
        return count_words
    
    # Create with configuration
    counter = create_word_counter(min_length=3)
    result = counter("I am learning to use Ember operators")
    
    print("Word Counter Results:")
    print_example_output("Total words", result["total_words"])
    print_example_output("Words >= 3 chars", result["filtered_words"])
    print_example_output("Sample words", result["words"])
    
    # Part 4: Operator Composition
    print("\n" + "="*50)
    print("Part 3: Composing Functions")
    print("="*50 + "\n")
    
    # Compose functions naturally
    def text_pipeline(text: str) -> dict:
        """Combines multiple operations into a pipeline."""
        # Step 1: Clean the text
        cleaned_result = clean_text(text)
        
        # Step 2: Count words in cleaned text
        counter = create_word_counter(min_length=4)
        count_result = counter(cleaned_result["cleaned"])
        
        # Combine results
        return {
            "cleaned_text": cleaned_result["cleaned"],
            "stats": {
                "original_length": cleaned_result["length"],
                "total_words": count_result["total_words"],
                "significant_words": count_result["filtered_words"]
            }
        }
    
    # Use the pipeline directly
    result = text_pipeline("  The QUICK brown fox jumps!  ")
    
    print("Pipeline Results:")
    print_example_output("Cleaned", result["cleaned_text"])
    print_example_output("Stats", result["stats"])
    
    # Make it fast with JIT!
    fast_pipeline = jit(text_pipeline)
    result2 = fast_pipeline("  Another TEXT to process!  ")
    print("\nFast Pipeline Results:")
    print_example_output("Cleaned", result2["cleaned_text"])
    
    # Part 5: Real-World Pattern
    print("\n" + "="*50)
    print("Part 4: Practical Example - Question Analyzer")
    print("="*50 + "\n")
    
    def analyze_question(question: str) -> dict:
        """Analyzes questions to determine their type and complexity."""
        question = question.strip()
        question_lower = question.lower()
        
        # Determine type
        if question_lower.startswith(("what", "which")):
            q_type = "factual"
        elif question_lower.startswith(("why", "how")):
            q_type = "explanatory"
        elif question_lower.startswith(("is", "are", "do", "does")):
            q_type = "yes/no"
        else:
            q_type = "other"
        
        # Estimate complexity
        word_count = len(question.split())
        complexity = "simple" if word_count < 10 else "complex"
        
        return {
            "question": question,
            "type": q_type,
            "complexity": complexity,
            "word_count": word_count
        }
    
    # Use it directly
    questions = [
        "What is machine learning?",
        "Why does gravity exist?",
        "Is Python a good programming language?",
        "How do neural networks learn from data?"
    ]
    
    print("Question Analysis:")
    for q in questions:
        result = analyze_question(q)
        print(f"\nQ: {q}")
        print(f"   Type: {result['type']}, Complexity: {result['complexity']}")
    
    # Batch processing with vmap
    print("\n" + "="*50)
    print("Part 5: Batch Processing")
    print("="*50 + "\n")
    
    # Process multiple questions at once
    batch_analyzer = vmap(analyze_question)
    results = batch_analyzer(questions)
    
    print("Batch Analysis Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['type']} ({result['complexity']})")
    
    print("\n" + "="*50)
    print("✅ Key Takeaways")
    print("="*50)
    print("\n1. ANY function is an operator - no base classes!")
    print("2. Use @jit for automatic optimization")
    print("3. Use vmap() for batch processing")
    print("4. Compose functions naturally with Python")
    print("5. Clean, simple code that's easy to understand")
    
    print("\nNext: Explore more examples in the 03_simplified_apis directory!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())