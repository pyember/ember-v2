"""JIT Basics - Zero-config optimization with @jit decorator.

Difficulty: Intermediate
Time: ~5 minutes

Learning Objectives:
- Understand the @jit decorator
- See automatic optimization benefits
- Learn when to use JIT compilation
- Measure performance improvements

The @jit decorator is Ember's key innovation for performance:
- Zero configuration required
- Automatic parallelization detection
- Smart caching of results
- Works with any Python function
"""

import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import models
from ember.api.xcs import jit, get_jit_stats


def main():
    """Learn how @jit provides automatic optimization."""
    print_section_header("JIT Optimization Basics")
    
    # Part 1: Simple JIT example
    print("Part 1: Basic @jit Usage")
    print("=" * 50 + "\n")
    
    # Regular function
    def process_text(text: str) -> dict:
        """Process text with multiple steps."""
        # Simulate some processing
        words = text.split()
        word_count = len(words)
        
        # Simulate expensive operation
        char_frequencies = {}
        for char in text.lower():
            if char.isalpha():
                char_frequencies[char] = char_frequencies.get(char, 0) + 1
        
        return {
            "word_count": word_count,
            "char_count": len(text),
            "unique_chars": len(char_frequencies),
            "most_common_char": max(char_frequencies, key=char_frequencies.get) if char_frequencies else None
        }
    
    # JIT-compiled version
    process_text_fast = jit(process_text)
    
    # First call (compilation happens here)
    text1 = "The quick brown fox jumps over the lazy dog"
    print("First call (includes compilation):")
    start = time.time()
    result1 = process_text_fast(text1)
    duration1 = time.time() - start
    print_example_output("Result", result1)
    print_example_output("Duration", f"{duration1:.4f}s")
    
    # Second call (uses compiled version)
    print("\nSecond call (uses compiled version):")
    start = time.time()
    result2 = process_text_fast(text1)
    duration2 = time.time() - start
    print_example_output("Duration", f"{duration2:.4f}s")
    print_example_output("Speedup", f"{duration1/duration2:.1f}x faster")
    
    # Part 2: JIT with Caching
    print("\n" + "=" * 50)
    print("Part 2: Automatic Caching")
    print("=" * 50 + "\n")
    
    @jit
    def expensive_analysis(text: str) -> dict:
        """Expensive text analysis that benefits from caching."""
        # Simulate expensive computation
        time.sleep(0.1)  # Pretend this takes time
        
        return {
            "length": len(text),
            "words": len(text.split()),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
    
    # First call
    print("First call to expensive_analysis:")
    start = time.time()
    result = expensive_analysis("Hello World!")
    print_example_output("Duration", f"{time.time() - start:.4f}s")
    
    # Repeated call (cached)
    print("\nRepeated call (cached):")
    start = time.time()
    result = expensive_analysis("Hello World!")
    print_example_output("Duration", f"{time.time() - start:.4f}s (from cache!)")
    
    # Different input (not cached)
    print("\nDifferent input:")
    start = time.time()
    result = expensive_analysis("Different text")
    print_example_output("Duration", f"{time.time() - start:.4f}s")
    
    # Part 3: JIT with Complex Functions
    print("\n" + "=" * 50)
    print("Part 3: Complex Function Optimization")
    print("=" * 50 + "\n")
    
    @jit
    def analyze_documents(documents: List[str]) -> Dict[str, any]:
        """Analyze multiple documents efficiently."""
        total_words = 0
        total_chars = 0
        all_words = []
        
        for doc in documents:
            words = doc.split()
            total_words += len(words)
            total_chars += len(doc)
            all_words.extend(words)
        
        # Calculate statistics
        avg_doc_length = total_chars / len(documents) if documents else 0
        avg_words_per_doc = total_words / len(documents) if documents else 0
        
        # Find common words (simplified)
        word_counts = {}
        for word in all_words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "num_documents": len(documents),
            "total_words": total_words,
            "avg_doc_length": avg_doc_length,
            "avg_words_per_doc": avg_words_per_doc,
            "common_words": common_words
        }
    
    # Test with multiple documents
    documents = [
        "Ember makes AI development simple and fast",
        "The JIT decorator optimizes Python functions automatically",
        "Write natural Python code and get optimization for free",
        "Ember handles the complexity so you can focus on building"
    ]
    
    print("Analyzing documents with JIT:")
    start = time.time()
    analysis = analyze_documents(documents)
    duration = time.time() - start
    
    print_example_output("Documents", analysis["num_documents"])
    print_example_output("Total words", analysis["total_words"])
    print_example_output("Common words", analysis["common_words"][:3])
    print_example_output("Analysis time", f"{duration:.4f}s")
    
    # Part 4: When to Use @jit
    print("\n" + "=" * 50)
    print("Part 4: Best Practices")
    print("=" * 50 + "\n")
    
    print("✅ Use @jit for:")
    print("  • Functions called multiple times with similar inputs")
    print("  • CPU-intensive computations")
    print("  • Functions with predictable input patterns")
    print("  • Data processing pipelines")
    
    print("\n❌ Avoid @jit for:")
    print("  • Functions with side effects (file I/O, network calls)")
    print("  • Functions that are already fast")
    print("  • One-time operations")
    print("  • Functions with highly variable inputs")
    
    # Part 5: Checking JIT statistics
    print("\n" + "=" * 50)
    print("Part 5: JIT Statistics")
    print("=" * 50 + "\n")
    
    stats = get_jit_stats()
    print("JIT compilation statistics:")
    print_example_output("Compiled functions", len(stats) if stats else "0")
    print("\nNote: Full statistics available with get_jit_stats()")
    
    print("\n🎉 Key Takeaways:")
    print("  1. @jit provides zero-config optimization")
    print("  2. First call includes compilation overhead")
    print("  3. Subsequent calls are much faster")
    print("  4. Automatic caching for repeated inputs")
    print("  5. Works with any Python function")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())