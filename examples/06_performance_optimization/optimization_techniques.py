"""Performance Optimization Techniques - Make your AI applications blazing fast.

Learn key optimization techniques:
- JIT compilation for single operations
- Batch processing with vmap
- Caching strategies
- Parallel execution patterns
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from _shared.conditional_execution import conditional_llm, SimulatedResponse
from _shared.conditional_llm_template import simulated_models
from ember.api import models
from ember.api.xcs import jit
from ember.xcs import vmap


@conditional_llm()
def example_jit_speedup(_simulated_mode=False):
    """Demonstrate JIT compilation speedup."""
    print("=" * 50)
    print("Example 1: JIT Compilation Speedup")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    # Define a function with multiple LLM calls
    def analyze_sentiment_detailed(text: str) -> dict:
        """Detailed sentiment analysis with multiple aspects."""
        # Overall sentiment
        sentiment_prompt = f"Classify sentiment (positive/negative/neutral): {text}"
        sentiment = model_fn("gpt-3.5-turbo", sentiment_prompt).text.strip()

        # Emotion detection
        emotion_prompt = f"Primary emotion (joy/anger/sadness/fear/surprise): {text}"
        emotion = model_fn("gpt-3.5-turbo", emotion_prompt).text.strip()

        # Intensity
        intensity_prompt = f"Sentiment intensity (1-10): {text}"
        intensity_text = model_fn("gpt-3.5-turbo", intensity_prompt).text

        try:
            intensity = int("".join(c for c in intensity_text if c.isdigit()))
        except:
            intensity = 5

        return {"sentiment": sentiment, "emotion": emotion, "intensity": intensity}

    # Time the regular function
    text = "I absolutely love this new feature! It's amazing!"

    start = time.time()
    result1 = analyze_sentiment_detailed(text)
    normal_time = time.time() - start

    print(f"Normal execution: {normal_time:.2f}s")
    print_example_output("Result", result1)

    # Create JIT version
    fast_analyze = jit(analyze_sentiment_detailed)

    # First call includes compilation
    start = time.time()
    result2 = fast_analyze(text)
    first_jit_time = time.time() - start

    print(f"\nFirst JIT call (includes compilation): {first_jit_time:.2f}s")

    # Subsequent calls are faster
    start = time.time()
    result3 = fast_analyze("This is terrible, I hate it!")
    optimized_time = time.time() - start

    print(f"Optimized JIT call: {optimized_time:.2f}s")
    print(f"Speedup: {normal_time/optimized_time:.1f}x faster")


@conditional_llm()
def example_batch_processing(_simulated_mode=False):
    """Demonstrate efficient batch processing."""
    print("\n" + "=" * 50)
    print("Example 2: Batch Processing with vmap")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    def classify_topic(text: str) -> str:
        """Classify text into topics."""
        prompt = f"Classify topic (tech/sports/politics/entertainment/other): {text}"
        return model_fn("gpt-3.5-turbo", prompt).text.strip()

    # Sample texts
    texts = [
        "The new iPhone features AI capabilities",
        "The team won the championship last night",
        "Congress passed the new climate bill",
        "The movie broke box office records",
        "Scientists discover new exoplanet",
    ]

    # Sequential processing
    print("Sequential processing:")
    start = time.time()
    sequential_results = []
    for text in texts:
        result = classify_topic(text)
        sequential_results.append(result)
        print(f"  - {text[:30]}... -> {result}")
    sequential_time = time.time() - start

    print(f"\nSequential time: {sequential_time:.2f}s")

    # Batch processing
    print("\nBatch processing with vmap:")
    batch_classify = vmap(classify_topic)

    start = time.time()
    batch_results = batch_classify(texts)
    batch_time = time.time() - start

    for text, result in zip(texts, batch_results):
        print(f"  - {text[:30]}... -> {result}")

    print(f"\nBatch time: {batch_time:.2f}s")
    print(f"Speedup: {sequential_time/batch_time:.1f}x faster")


@conditional_llm()
def example_caching_pattern(_simulated_mode=False):
    """Demonstrate caching for repeated queries."""
    print("\n" + "=" * 50)
    print("Example 3: Caching Pattern")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    # Simple cache implementation
    cache = {}

    def cached_translate(text: str, target_lang: str = "Spanish") -> str:
        """Translate with caching."""
        cache_key = f"{text}:{target_lang}"

        if cache_key in cache:
            print(f"  [Cache hit for '{text[:20]}...']")
            return cache[cache_key]

        print(f"  [Cache miss for '{text[:20]}...']")
        prompt = f"Translate to {target_lang}: {text}"
        result = model_fn("gpt-3.5-turbo", prompt).text.strip()

        cache[cache_key] = result
        return result

    # Demonstrate caching
    phrases = [
        "Hello, how are you?",
        "Thank you very much",
        "Hello, how are you?",  # Duplicate
        "Good morning",
        "Thank you very much",  # Duplicate
    ]

    print("Translation requests:")
    for phrase in phrases:
        translation = cached_translate(phrase)
        print(f"  '{phrase}' -> '{translation}'")

    print(f"\nCache stats: {len(cache)} unique translations cached")


@conditional_llm()
def example_optimized_pipeline(_simulated_mode=False):
    """Build an optimized processing pipeline."""
    print("\n" + "=" * 50)
    print("Example 4: Optimized Pipeline")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    # Create an optimized document processing pipeline
    @jit
    def process_document(doc: str) -> dict:
        """Process document with multiple analyses."""
        # Summarize
        summary = model_fn("gpt-3.5-turbo", f"Summarize in one sentence: {doc}").text

        # Extract key points
        points = model_fn("gpt-3.5-turbo", f"List 3 key points: {doc}").text

        # Determine category
        category = model_fn(
            "gpt-3.5-turbo", f"Category (news/blog/research): {doc}"
        ).text

        return {
            "summary": summary.strip(),
            "key_points": points.strip(),
            "category": category.strip(),
        }

    # Process multiple documents efficiently
    documents = [
        "AI research has made significant progress in natural language understanding...",
        "The stock market reached new highs today as tech companies reported earnings...",
        "A new study shows the benefits of regular exercise on mental health...",
    ]

    # Use vmap for batch processing of the JIT-optimized function
    batch_process = vmap(process_document)

    print("Processing documents in batch:")
    start = time.time()
    results = batch_process(documents)
    elapsed = time.time() - start

    for i, (doc, result) in enumerate(zip(documents, results)):
        print(f"\nDocument {i+1}:")
        print(f"  Preview: {doc[:50]}...")
        print(f"  Summary: {result['summary']}")
        print(f"  Category: {result['category']}")

    print(f"\nTotal processing time: {elapsed:.2f}s")
    print(f"Average per document: {elapsed/len(documents):.2f}s")


def example_performance_tips():
    """Share key performance optimization tips."""
    print("\n" + "=" * 50)
    print("Performance Optimization Tips")
    print("=" * 50 + "\n")

    print("1. 🚀 Use @jit for functions called multiple times")
    print("   - First call includes compilation overhead")
    print("   - Subsequent calls are significantly faster")
    print("   - Best for: Repeated operations, production APIs")

    print("\n2. 📦 Use vmap() for batch processing")
    print("   - Processes multiple inputs in parallel")
    print("   - Automatic batching optimization")
    print("   - Best for: Processing lists, datasets, bulk operations")

    print("\n3. 💾 Implement caching where appropriate")
    print("   - Cache expensive LLM calls")
    print("   - Consider TTL for dynamic content")
    print("   - Best for: Repeated queries, reference data")

    print("\n4. 🔧 Combine optimizations")
    print("   - JIT + vmap = Maximum performance")
    print("   - Cache + JIT = Fast repeated operations")
    print("   - Best for: Production systems")

    print("\n5. 📊 Profile before optimizing")
    print("   - Measure actual bottlenecks")
    print("   - Focus on hot paths")
    print("   - Best for: Real-world applications")


def main():
    """Run all performance optimization examples."""
    print_section_header("Performance Optimization Techniques")

    try:
        example_jit_speedup()
        example_batch_processing()
        example_caching_pattern()
        example_optimized_pipeline()
        example_performance_tips()

        print("\n✨ Remember: Premature optimization is the root of all evil.")
        print("   Profile first, then optimize where it matters!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
