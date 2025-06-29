#!/usr/bin/env python3
"""Examples showcasing Ember's operator system.

This file demonstrates the key operator patterns in Ember, following
the principle of progressive disclosure - simple things are simple,
complex things are possible.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from ember.operators import (
    Operator,
    Ensemble,
    Chain,
    Router,
    LearnableRouter,
    Retry,
    Cache,
    ensemble,
    chain,
)
from ember.api import model, op

# =============================================================================
# Basic Operators
# =============================================================================

print("=== Basic Operators ===\n")


# Simple operator with forward method
class Summarizer(Operator):
    """Summarize text using an LLM."""

    def __init__(self):
        self.model = model("gpt-4", temperature=0.3)

    def forward(self, text: str) -> str:
        prompt = f"Summarize in one sentence: {text}"
        return self.model(prompt).text


# Even simpler with decorator
@op
def classify_sentiment(text: str) -> str:
    """Classify sentiment as positive, negative, or neutral."""
    response = model("gpt-4")(f"Classify sentiment: {text}")
    return response.text


# Test them
summarizer = Summarizer()
print(f"Summary: {summarizer('Ember is a framework for building AI systems.')}")
print(f"Sentiment: {classify_sentiment('This is amazing!')}")

# =============================================================================
# Composition Patterns
# =============================================================================

print("\n=== Composition Patterns ===\n")

# Chain operators sequentially
process_chain = chain(
    Summarizer(),
    classify_sentiment,  # Functions decorated with @op work too!
)

result = process_chain("Long text about how wonderful Ember is...")
print(f"Chain result: {result}")

# Ensemble for reliability
reliable_classifier = ensemble(
    classify_sentiment,
    classify_sentiment,  # Could be different models
    classify_sentiment,
    aggregator=lambda results: max(set(results), key=results.count),  # Majority vote
)

# =============================================================================
# Advanced Routing
# =============================================================================

print("\n=== Advanced Routing ===\n")

# Simple rule-based routing
simple_router = Router(
    routes={
        "question": model("gpt-4"),
        "statement": Summarizer(),
    },
    router_fn=lambda text: "question" if "?" in text else "statement",
)

print(simple_router("What is Ember?"))
print(simple_router("Ember is great."))


# Learnable routing with embeddings
def complexity_embedder(text: str) -> jax.Array:
    """Embed based on text complexity."""
    word_count = len(text.split())
    avg_word_len = sum(len(w) for w in text.split()) / max(word_count, 1)
    return jnp.array([word_count / 20, avg_word_len / 10])


smart_router = LearnableRouter(
    routes={
        "simple": model("gpt-3.5-turbo"),
        "complex": model("gpt-4"),
    },
    embedding_fn=complexity_embedder,
    embed_dim=2,
    key=jax.random.PRNGKey(42),
)

print(smart_router("Hi"))
print(smart_router("Explain the philosophical implications of quantum mechanics"))

# =============================================================================
# Reliability Patterns
# =============================================================================

print("\n=== Reliability Patterns ===\n")

# Retry on failure
reliable_api = Retry(
    operator=model("gpt-4"),
    max_attempts=3,
    should_retry=lambda e, n: "rate_limit" in str(e).lower(),
)

# Cache expensive operations
cached_analyzer = Cache(
    operator=Chain([Summarizer(), classify_sentiment]), max_size=100
)

# First call computes
result1 = cached_analyzer("Some text to analyze")
# Second call uses cache (instant)
result2 = cached_analyzer("Some text to analyze")
print(f"Cached result: {result1}")

# =============================================================================
# Complete Example: Smart Document Processor
# =============================================================================

print("\n=== Complete Example: Smart Document Processor ===\n")


class DocumentProcessor(Operator):
    """Process documents intelligently based on content."""

    def __init__(self):
        # Components
        self.classifier = model("gpt-3.5-turbo", temperature=0)
        self.summarizer = Summarizer()
        self.qa_system = model("gpt-4")

        # Smart routing based on document type
        self.router = LearnableRouter(
            routes={
                "technical": self.technical_processor,
                "general": self.general_processor,
                "question": self.qa_processor,
            },
            embedding_fn=self.compute_doc_embedding,
            embed_dim=5,
            key=jax.random.PRNGKey(123),
        )

    def compute_doc_embedding(self, text: str) -> jax.Array:
        """Compute document embedding for routing."""
        features = [
            float("?" in text),  # Has questions
            float(any(w in text.lower() for w in ["code", "algorithm", "function"])),
            len(text.split()) / 100,  # Length
            text.count(".") / max(len(text.split()), 1),  # Sentence density
            float(text.isupper()) * 0.5,  # Formatting
        ]
        return jnp.array(features)

    def technical_processor(self, text: str) -> str:
        return f"[Technical Analysis] {self.summarizer(text)}"

    def general_processor(self, text: str) -> str:
        return f"[General Summary] {self.summarizer(text)}"

    def qa_processor(self, text: str) -> str:
        return f"[Q&A Response] {self.qa_system(text).text}"

    def forward(self, document: str) -> str:
        # Route to appropriate processor
        return self.router(document)


# Use the processor
processor = DocumentProcessor()

docs = [
    "What is the time complexity of quicksort?",
    "The weather today is sunny with a chance of rain.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]

for doc in docs:
    print(f"Input: '{doc[:50]}...'")
    print(f"Output: {processor(doc)}")
    print()

print("✅ Ember operators: Simple, Composable, Powerful!")
