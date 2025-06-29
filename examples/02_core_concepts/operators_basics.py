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
from ember.api import models, operators
from ember.api.xcs import jit
from ember.xcs import vmap


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
    print("=" * 50)
    print("Part 1: A Simple Function as Operator")
    print("=" * 50 + "\n")

    # Just write a function - it's an operator!
    def clean_text(text: str) -> dict:
        """Cleans and normalizes text input."""
        cleaned = text.strip().lower()
        cleaned = " ".join(cleaned.split())  # Normalize whitespace

        return {"original": text, "cleaned": cleaned, "length": len(cleaned)}

    # Use it directly
    result = clean_text("  Hello   WORLD!  ")

    print("Text Cleaner Results:")
    print_example_output("Original", repr(result["original"]))
    print_example_output("Cleaned", result["cleaned"])
    print_example_output("Length", result["length"])

    # Part 3: Operator with Configuration
    print("\n" + "=" * 50)
    print("Part 2: Configurable Function")
    print("=" * 50 + "\n")

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
                "words": filtered_words[:5],  # First 5 as sample
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
    print("\n" + "=" * 50)
    print("Part 3: Composing Functions")
    print("=" * 50 + "\n")

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
                "significant_words": count_result["filtered_words"],
            },
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

    # Part 5: Using the @op decorator
    print("\n" + "=" * 50)
    print("Part 4: The @op Decorator")
    print("=" * 50 + "\n")

    # Using @op decorator for operator features
    @operators.op
    def sentiment_analyzer(text: str) -> dict:
        """Analyze sentiment of text."""
        # Simple rule-based sentiment for demo
        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "poor"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "text": text,
            "sentiment": sentiment,
            "positive_signals": positive_count,
            "negative_signals": negative_count,
        }

    # The @op decorator gives us operator capabilities
    print("Sentiment Analysis (with @op decorator):")
    texts = [
        "This is a great product!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
    ]

    for text in texts:
        result = sentiment_analyzer(text)
        print(f"\n'{text}'")
        print(f"  → Sentiment: {result['sentiment']}")

    # Part 6: Real-World Pattern
    print("\n" + "=" * 50)
    print("Part 5: Practical Example - Question Analyzer")
    print("=" * 50 + "\n")

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
            "word_count": word_count,
        }

    # Use it directly
    questions = [
        "What is machine learning?",
        "Why does gravity exist?",
        "Is Python a good programming language?",
        "How do neural networks learn from data?",
    ]

    print("Question Analysis:")
    for q in questions:
        result = analyze_question(q)
        print(f"\nQ: {q}")
        print(f"   Type: {result['type']}, Complexity: {result['complexity']}")

    # Batch processing with vmap
    print("\n" + "=" * 50)
    print("Part 5: Batch Processing")
    print("=" * 50 + "\n")

    # Process multiple questions at once
    batch_analyzer = vmap(analyze_question)
    results = batch_analyzer(questions)

    print("Batch Analysis Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['type']} ({result['complexity']})")

    # Part 7: Integration with Models (if API key available)
    print("\n" + "=" * 50)
    print("Part 6: Integration with Language Models")
    print("=" * 50 + "\n")

    def ai_powered_analyzer(text: str) -> dict:
        """Analyze text using AI (demo mode without API key)."""
        # This shows how you'd integrate with models
        print("In real usage with API key:")
        print(f"  response = models('gpt-4', 'Analyze: {text[:30]}...')")
        print("  return {'analysis': response.text}")

        # Demo response
        return {
            "text": text,
            "analysis": "[AI analysis would appear here]",
            "confidence": 0.95,
        }

    # Show the pattern
    result = ai_powered_analyzer("The future of AI is bright")
    print("\nAI-Powered Analysis:")
    print_example_output("Input", result["text"])
    print_example_output("Analysis", result["analysis"])

    print("\n" + "=" * 50)
    print("✅ Key Takeaways")
    print("=" * 50)
    print("\n1. ANY function is an operator - no base classes!")
    print("2. Use @op decorator for extra operator features")
    print("3. Use @jit for automatic optimization")
    print("4. Use vmap() for batch processing")
    print("5. Compose functions naturally with Python")
    print("6. Integrate models seamlessly in any function")

    print("\nNext: Explore more examples in the 03_simplified_apis directory!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
