"""Batch Processing - Efficient parallel processing with vmap.

Difficulty: Intermediate
Time: ~5 minutes

Learning Objectives:
- Understand vmap for batch processing
- See parallelization benefits
- Learn batch processing patterns
- Combine vmap with @jit for maximum performance

The vmap function is inspired by JAX and provides:
- Automatic vectorization of functions
- Parallel processing of multiple inputs
- Memory-efficient batch operations
- Seamless integration with @jit
"""

import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import models
from ember.api.xcs import jit, vmap


def main():
    """Learn efficient batch processing with vmap."""
    print_section_header("Batch Processing with vmap")
    
    # Part 1: Basic vmap usage
    print("Part 1: Basic vmap Usage")
    print("=" * 50 + "\n")
    
    # Define a function that processes single items
    def classify_sentiment(text: str) -> dict:
        """Classify sentiment of a single text."""
        # Simple rule-based sentiment (in practice, use models)
        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "love"}
        negative_words = {"bad", "terrible", "awful", "horrible", "hate", "poor"}
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        positive_score = len(words & positive_words)
        negative_score = len(words & negative_words)
        
        if positive_score > negative_score:
            sentiment = "positive"
        elif negative_score > positive_score:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": abs(positive_score - negative_score) / max(len(words), 1)
        }
    
    # Create batch version with vmap
    batch_classify = vmap(classify_sentiment)
    
    # Process multiple texts at once
    texts = [
        "This product is absolutely amazing and wonderful!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special really.",
        "Love this! Great quality and excellent service!",
        "Poor design and awful customer support."
    ]
    
    print("Processing batch of texts:")
    start = time.time()
    results = batch_classify(texts)
    batch_time = time.time() - start
    
    # Show results
    for i, result in enumerate(results):
        print(f"{i+1}. {result['sentiment']} - {result['text'][:40]}...")
    print_example_output("Batch processing time", f"{batch_time:.4f}s")
    
    # Compare with sequential processing
    print("\nComparing with sequential processing:")
    start = time.time()
    sequential_results = [classify_sentiment(text) for text in texts]
    seq_time = time.time() - start
    print_example_output("Sequential time", f"{seq_time:.4f}s")
    print_example_output("Speedup", f"{seq_time/batch_time:.1f}x faster with vmap")
    
    # Part 2: vmap with Multiple Arguments
    print("\n" + "=" * 50)
    print("Part 2: Multiple Arguments")
    print("=" * 50 + "\n")
    
    def score_match(text: str, keyword: str) -> dict:
        """Score how well a text matches a keyword."""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Count occurrences
        count = text_lower.count(keyword_lower)
        
        # Calculate relevance score
        words = text_lower.split()
        relevance = count / max(len(words), 1)
        
        return {
            "text": text[:50] + "..." if len(text) > 50 else text,
            "keyword": keyword,
            "matches": count,
            "relevance": relevance
        }
    
    # Batch version
    batch_score = vmap(score_match)
    
    # Multiple texts and keywords
    texts = [
        "Machine learning is transforming how we build software",
        "Deep learning models require lots of data",
        "Ember makes AI development simple and efficient",
        "Natural language processing is a key ML application"
    ]
    keywords = ["learning", "learning", "AI", "ML"]
    
    print("Scoring keyword matches in batch:")
    results = batch_score(texts, keywords)
    
    for result in results:
        print(f"'{result['keyword']}' in '{result['text']}': {result['matches']} matches")
    
    # Part 3: Combining vmap with @jit
    print("\n" + "=" * 50)
    print("Part 3: Combining vmap and @jit")
    print("=" * 50 + "\n")
    
    @jit
    def extract_features(text: str) -> dict:
        """Extract features from text (with JIT optimization)."""
        words = text.split()
        chars = len(text)
        
        # Simulate more complex feature extraction
        features = {
            "length": chars,
            "words": len(words),
            "avg_word_length": chars / max(len(words), 1),
            "capitalized_words": sum(1 for w in words if w and w[0].isupper()),
            "punctuation_count": sum(1 for c in text if c in ".,!?;:"),
            "unique_words": len(set(w.lower() for w in words))
        }
        
        return features
    
    # Batch + JIT for maximum performance
    batch_extract = vmap(extract_features)
    
    # Large batch of texts
    large_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Ember provides powerful tools for AI development.",
        "Machine learning models can process text efficiently.",
        "Python is a great language for data science!",
        "How can we make AI systems more reliable?",
        "Natural language understanding is challenging.",
        "Deep learning has revolutionized computer vision.",
        "The future of AI is bright and full of possibilities!"
    ] * 10  # Repeat to make a larger batch
    
    print(f"Extracting features from {len(large_texts)} texts:")
    
    # Time the batch processing
    start = time.time()
    features = batch_extract(large_texts)
    duration = time.time() - start
    
    print_example_output("Total texts processed", len(features))
    print_example_output("Processing time", f"{duration:.4f}s")
    print_example_output("Texts per second", f"{len(features)/duration:.0f}")
    
    # Show sample features
    print("\nSample features from first text:")
    for key, value in features[0].items():
        print(f"  {key}: {value}")
    
    # Part 4: Practical Pattern - Batch Model Calls
    print("\n" + "=" * 50)
    print("Part 4: Practical Pattern")
    print("=" * 50 + "\n")
    
    def process_for_model(text: str) -> dict:
        """Prepare text for model processing."""
        # In practice, this might call models() API
        # For demo, we'll simulate the preparation
        
        prompt = f"Analyze the following text and identify the main topic: {text}"
        
        return {
            "original": text,
            "prompt": prompt,
            "token_estimate": len(prompt.split()) * 1.3  # Rough estimate
        }
    
    # Batch processing for model preparation
    batch_prepare = vmap(process_for_model)
    
    documents = [
        "Climate change is affecting global weather patterns",
        "New AI breakthrough in natural language understanding",
        "Stock markets react to economic policy changes",
        "Medical researchers discover new treatment method"
    ]
    
    print("Batch preparing documents for model analysis:")
    prepared = batch_prepare(documents)
    
    total_tokens = sum(p["token_estimate"] for p in prepared)
    print_example_output("Documents prepared", len(prepared))
    print_example_output("Estimated total tokens", f"{total_tokens:.0f}")
    print_example_output("Estimated cost (@$0.01/1K tokens)", f"${total_tokens * 0.01 / 1000:.4f}")
    
    # Part 5: Best Practices
    print("\n" + "=" * 50)
    print("Part 5: Best Practices")
    print("=" * 50 + "\n")
    
    print("✅ Use vmap for:")
    print("  • Processing multiple independent items")
    print("  • Parallel text/data transformations")
    print("  • Batch model inference preparation")
    print("  • Feature extraction from collections")
    
    print("\n🎯 vmap + @jit pattern:")
    print("  1. Define single-item function")
    print("  2. Apply @jit for optimization")
    print("  3. Use vmap for batch processing")
    print("  4. Get both compilation and parallelization benefits")
    
    print("\n💡 Tips:")
    print("  • vmap automatically handles different input sizes")
    print("  • Works with functions returning dicts, lists, or scalars")
    print("  • Combine with @jit for maximum performance")
    print("  • Great for preprocessing before model calls")
    
    print("\n🎉 Key Takeaways:")
    print("  1. vmap enables efficient batch processing")
    print("  2. Automatic parallelization of operations")
    print("  3. Works seamlessly with existing functions")
    print("  4. Combines perfectly with @jit")
    print("  5. Significant speedups for batch operations")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())