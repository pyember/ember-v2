#!/usr/bin/env python3
"""Example showcasing the LearnableRouter design patterns.

This example demonstrates the clean, principled design of LearnableRouter
that follows the masters' principles: simple for simple cases, flexible for
complex cases, with clear separation of concerns.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List
from ember.operators import LearnableRouter, Operator, Chain


# Mock model operators for demonstration
class QuickModel(Operator):
    """Fast but less accurate model."""

    def forward(self, text: str) -> str:
        return f"[Quick Analysis] {text[:50]}..."


class DeepModel(Operator):
    """Slower but more accurate model."""

    def forward(self, text: str) -> str:
        words = len(text.split())
        return f"[Deep Analysis] {words} words analyzed. Key insights extracted."


class CreativeModel(Operator):
    """Model optimized for creative tasks."""

    def forward(self, text: str) -> str:
        return f"[Creative Response] Reimagining: {text[:30]}... in new ways"


# =============================================================================
# Pattern 1: Simple Embedding Function (Most Common Case)
# =============================================================================


def complexity_embedder(text: str) -> jax.Array:
    """Embed text based on complexity heuristics."""
    # Simple heuristics for demonstration
    word_count = len(text.split())
    avg_word_length = sum(len(w) for w in text.split()) / max(word_count, 1)
    sentence_count = text.count(".") + text.count("!") + text.count("?")

    # Create embedding based on these features
    features = jnp.array(
        [
            word_count / 100.0,  # Normalized word count
            avg_word_length / 10.0,  # Normalized average word length
            sentence_count / 10.0,  # Normalized sentence count
            float("?" in text),  # Has questions
            float(any(w.isupper() for w in text.split())),  # Has emphasis
        ]
    )

    return features


# Create router with embedding function
simple_router = LearnableRouter(
    routes={"quick": QuickModel(), "deep": DeepModel(), "creative": CreativeModel()},
    embedding_fn=complexity_embedder,
    embed_dim=5,
    key=jax.random.PRNGKey(42),
    temperature=0.5,  # Lower temperature = more decisive routing
)

print("=== Pattern 1: Router with Embedding Function ===")
print()

# Test different types of input
inputs = [
    "Hello",
    "What is the meaning of life, the universe, and everything?",
    "The quick brown fox jumps over the lazy dog. This is a simple sentence.",
    "URGENT: Need creative solution NOW! Help redesign our approach!!!",
]

for text in inputs:
    result = simple_router(text)
    print(f"Input: '{text[:40]}...'")
    print(f"Output: {result}")
    print()

# =============================================================================
# Pattern 2: External Embeddings (Advanced Use Case)
# =============================================================================


@dataclass
class AnalysisRequest:
    """Structured input with pre-computed embeddings."""

    data: str
    embedding: jax.Array
    metadata: dict = None


# Router expecting external embeddings
advanced_router = LearnableRouter(
    routes={"technical": DeepModel(), "general": QuickModel()},
    embedding_fn=None,  # No embedding function - expects structured input
    embed_dim=10,
    key=jax.random.PRNGKey(123),
)

print("=== Pattern 2: Router with External Embeddings ===")
print()

# Simulate embeddings from an external service
# (In practice, these might come from a specialized embedding model)
technical_embedding = jnp.array([1.0, 0.8, 0.9, 0.7, 0.95, 0.85, 0.9, 0.8, 0.9, 0.85])
general_embedding = jnp.array([0.3, 0.2, 0.4, 0.3, 0.2, 0.35, 0.3, 0.25, 0.3, 0.4])

requests = [
    AnalysisRequest(
        data="Explain quantum computing algorithms",
        embedding=technical_embedding,
        metadata={"source": "academic", "priority": "high"},
    ),
    AnalysisRequest(
        data="What's the weather like today?",
        embedding=general_embedding,
        metadata={"source": "casual", "priority": "low"},
    ),
]

for req in requests:
    result = advanced_router(req)
    print(f"Request: '{req.data}'")
    print(f"Metadata: {req.metadata}")
    print(f"Output: {result}")
    print()

# =============================================================================
# Pattern 3: Composition - Router Within a Larger System
# =============================================================================


class AdaptiveAnalyzer(Operator):
    """Sophisticated analyzer that routes based on learned embeddings."""

    def __init__(self, key):
        key1, key2 = jax.random.split(key)

        # Learnable parameters for embedding computation
        self.context_weights = jax.random.normal(key1, (50, 8))

        # Router with custom embedding function
        self.router = LearnableRouter(
            routes={
                "factual": Chain([QuickModel(), SummaryOperator()]),  # Chain operators
                "analytical": DeepModel(),
                "creative": CreativeModel(),
            },
            embedding_fn=self.compute_contextual_embedding,
            embed_dim=8,
            key=key2,
        )

    def compute_contextual_embedding(self, text: str) -> jax.Array:
        """Compute embedding using learned weights."""
        # Extract simple features (in practice, would be more sophisticated)
        features = []

        # Character distribution features
        for i in range(50):
            if i < len(text):
                features.append(float(ord(text[i])) / 128.0)
            else:
                features.append(0.0)

        feature_vector = jnp.array(features)

        # Apply learned transformation
        embedding = feature_vector @ self.context_weights
        return jax.nn.tanh(embedding)  # Bounded activation

    def forward(self, text: str) -> str:
        # Add any pre/post processing here
        return self.router(text)


class SummaryOperator(Operator):
    """Post-process to create summaries."""

    def forward(self, analysis: str) -> str:
        return f"{analysis} [Summary: Key points extracted]"


print("=== Pattern 3: Composed System with Learned Routing ===")
print()

analyzer = AdaptiveAnalyzer(jax.random.PRNGKey(999))

test_inputs = [
    "List the prime numbers between 1 and 20",
    "Analyze the economic impact of renewable energy",
    "Write a haiku about machine learning",
]

for text in test_inputs:
    result = analyzer(text)
    print(f"Input: '{text}'")
    print(f"Output: {result}")
    print()

# =============================================================================
# Show Routing Probabilities (For Understanding)
# =============================================================================

print("=== Understanding Routing Decisions ===")
print()

# Check what embeddings produce what routing decisions
test_embedding = complexity_embedder("What is quantum computing?")
probs = simple_router.compute_route_probabilities(test_embedding)

print("Text: 'What is quantum computing?'")
print(f"Embedding: {test_embedding}")
print("Routing probabilities:")
for i, route in enumerate(simple_router.route_names):
    print(f"  {route}: {probs[i]:.3f}")

print("\n✅ LearnableRouter provides clean, composable, and powerful routing!")
