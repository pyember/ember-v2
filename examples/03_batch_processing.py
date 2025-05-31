"""Process multiple inputs efficiently with batching.

This example shows how to handle multiple inputs efficiently.
You'll learn:
- How to process lists of inputs
- How to use vmap for automatic batching
- How to handle mixed results

Requirements:
- ember
- Models: Any supported model

Expected output:
    Manual batching:
    Classified 5 items in X.X seconds
    
    With vmap:
    Classified 5 items in X.X seconds (parallel)
    
    Results:
    "I love this product!" -> positive
    "Terrible experience" -> negative
    "It's okay I guess" -> neutral
    "Best purchase ever!!!" -> positive  
    "Waste of money" -> negative
"""

import time
from ember.api import models
from ember.xcs import vmap


# Sample data
REVIEWS = [
    "I love this product!",
    "Terrible experience",
    "It's okay I guess", 
    "Best purchase ever!!!",
    "Waste of money"
]


def classify_sentiment(text: str) -> str:
    """Classify sentiment of a single text."""
    model = models.instance("gpt-4", temperature=0.1)
    prompt = f"Classify sentiment as positive/negative/neutral: {text}"
    response = model(prompt)
    return response.text.strip().lower()


def main():
    # Method 1: Manual batching (sequential)
    print("Manual batching:")
    start = time.time()
    
    results_manual = []
    for review in REVIEWS:
        result = classify_sentiment(review)
        results_manual.append(result)
    
    manual_time = time.time() - start
    print(f"Classified {len(REVIEWS)} items in {manual_time:.1f} seconds\n")
    
    # Method 2: Using vmap (can be parallelized by XCS)
    print("With vmap:")
    start = time.time()
    
    # vmap automatically vectorizes the function
    batch_classify = vmap(classify_sentiment)
    results_vmap = batch_classify(REVIEWS)
    
    vmap_time = time.time() - start
    print(f"Classified {len(REVIEWS)} items in {vmap_time:.1f} seconds (parallel)\n")
    
    # Show results
    print("Results:")
    for review, sentiment in zip(REVIEWS, results_vmap):
        print(f'"{review}" -> {sentiment}')
    
    # Note: vmap can be JIT compiled for even better performance
    # fast_batch_classify = jit(vmap(classify_sentiment))


if __name__ == "__main__":
    main()


# Next steps:
# - Try with 100+ items to see performance difference
# - Use pmap for true parallel execution across devices
# - See examples/performance/parallel_processing.py