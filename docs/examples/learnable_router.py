"""
Example: Learnable Router
Description: Advanced operator with learnable parameters for intelligent routing
Concepts: JAX integration, learnable parameters, gradient updates, mixed static/dynamic
"""

import jax
import jax.numpy as jnp
from ember.api import ember, Operator
from ember.api.xcs import jit
import asyncio
from typing import Dict, List


class LearnableRouter(Operator):
    """
    Routes requests to different models based on learned patterns.

    This operator demonstrates:
    - Automatic detection of learnable (JAX) vs static parameters
    - Mixed differentiable and non-differentiable operations
    - Integration with XCS for optimization
    """

    def __init__(self, routes: Dict[str, str], embedding_dim: int = 64):
        super().__init__()

        # Initialize random key
        self.key = jax.random.PRNGKey(42)

        # Learnable parameters (automatically detected as dynamic)
        self.routing_weights = (
            jax.random.normal(self.key, (embedding_dim, len(routes))) * 0.1
        )
        self.temperature = jnp.array(1.0)
        self.bias = jnp.zeros(len(routes))

        # Static parameters (automatically detected as non-learnable)
        self.routes = routes
        self.route_names = list(routes.keys())
        self.models = {
            name: ember.models.instance(model) for name, model in routes.items()
        }

        # Track performance for learning
        self.route_performance = {name: [] for name in routes}

    def text_to_embedding(self, text: str) -> jnp.ndarray:
        """
        Simple text embedding (in practice, use a real embedder).
        """
        # Simplified: use character-level features
        features = jnp.zeros(64)

        # Length feature
        features = features.at[0].set(len(text) / 100.0)

        # Character distribution features
        for i, char in enumerate(text[:20]):
            features = features.at[i + 1].set(ord(char) / 128.0)

        # Question indicators
        features = features.at[21].set(1.0 if "?" in text else 0.0)
        features = features.at[22].set(1.0 if "how" in text.lower() else 0.0)
        features = features.at[23].set(1.0 if "why" in text.lower() else 0.0)

        return features

    def compute_routing_scores(self, embedding: jnp.ndarray) -> jnp.ndarray:
        """
        Compute routing scores using learnable parameters.
        This is differentiable for gradient-based learning.
        """
        # Linear transformation with learnable weights
        logits = jnp.dot(embedding, self.routing_weights) + self.bias

        # Softmax with temperature for probability distribution
        scores = jax.nn.softmax(logits / self.temperature)

        return scores

    async def forward(self, text: str) -> Dict:
        """
        Route text to appropriate model based on learned patterns.
        """
        # Convert text to embedding
        embedding = self.text_to_embedding(text)

        # Compute routing probabilities (differentiable)
        scores = self.compute_routing_scores(embedding)

        # Make routing decision (non-differentiable)
        route_idx = int(jnp.argmax(scores))
        selected_route = self.route_names[route_idx]
        confidence = float(scores[route_idx])

        # Call selected model (static operation)
        response = await self.models[selected_route](text)

        # Return results with metadata
        return {
            "response": response,
            "route": selected_route,
            "confidence": confidence,
            "scores": {
                name: float(scores[i]) for i, name in enumerate(self.route_names)
            },
        }

    def update_from_feedback(self, text: str, route_used: str, performance: float):
        """
        Update routing weights based on performance feedback.
        """
        # Store performance data
        self.route_performance[route_used].append(performance)

        # Compute gradient for weight update
        embedding = self.text_to_embedding(text)

        # Define loss: we want to increase score for good routes
        def loss_fn(weights, bias):
            logits = jnp.dot(embedding, weights) + bias
            scores = jax.nn.softmax(logits / self.temperature)

            # Negative log likelihood of the route that performed well
            route_idx = self.route_names.index(route_used)
            return -jnp.log(scores[route_idx]) * performance

        # Compute gradients
        grad_weights, grad_bias = jax.grad(loss_fn, argnums=(0, 1))(
            self.routing_weights, self.bias
        )

        # Update parameters with gradient descent
        learning_rate = 0.01
        self.routing_weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

    def get_statistics(self) -> Dict:
        """Get routing statistics."""
        stats = {}
        for route, performances in self.route_performance.items():
            if performances:
                stats[route] = {
                    "count": len(performances),
                    "avg_performance": sum(performances) / len(performances),
                }
        return stats


# Optimized version with JIT
@jit
class OptimizedLearnableRouter(LearnableRouter):
    """JIT-compiled version for better performance."""

    pass


async def main():
    print("Learnable Router Example")
    print("=" * 50)

    # Define routing configuration
    routes = {
        "fast": "gpt-3.5-turbo",  # Fast but less accurate
        "balanced": "gpt-4",  # Balanced performance
        "accurate": "claude-3-opus",  # Most accurate but expensive
    }

    # Create router
    router = LearnableRouter(routes)

    # Test queries with different characteristics
    test_queries = [
        # Simple queries - should route to "fast"
        "What is 2+2?",
        "What day is it?",
        "Hello, how are you?",
        # Medium complexity - should route to "balanced"
        "Explain the water cycle in simple terms",
        "What are the main causes of climate change?",
        "How does a computer CPU work?",
        # Complex queries - should route to "accurate"
        "Analyze the philosophical implications of artificial consciousness",
        "Explain quantum entanglement and its applications in computing",
        "Discuss the socioeconomic factors that led to the Renaissance",
    ]

    print("\n=== Initial Routing (Untrained) ===")
    for query in test_queries[:3]:
        result = await router(query)
        print(f"\nQuery: {query}")
        print(f"Route: {result['route']} (confidence: {result['confidence']:.3f})")
        print(f"All scores: {result['scores']}")

    # Simulate training with feedback
    print("\n\n=== Training Router with Feedback ===")

    # Simulate performance feedback (1.0 = perfect, 0.0 = poor)
    training_data = [
        # Simple queries perform well on fast model
        ("What is 2+2?", "fast", 0.9),
        ("What day is it?", "fast", 0.95),
        ("Hello!", "fast", 1.0),
        # But poorly on complex queries
        ("Explain quantum computing", "fast", 0.3),
        # Complex queries need accurate model
        ("Analyze consciousness", "accurate", 0.95),
        ("Explain quantum entanglement", "accurate", 0.9),
        # Medium queries work well with balanced
        ("Explain photosynthesis", "balanced", 0.85),
        ("How do airplanes fly?", "balanced", 0.8),
    ]

    for text, route, performance in training_data:
        router.update_from_feedback(text, route, performance)
        print(f"Trained on: '{text[:30]}...' -> {route} (performance: {performance})")

    # Test routing after training
    print("\n\n=== Routing After Training ===")
    for query in test_queries:
        result = await router(query)
        print(f"\nQuery: {query[:50]}...")
        print(f"Route: {result['route']} (confidence: {result['confidence']:.3f})")

    # Show statistics
    print("\n\n=== Routing Statistics ===")
    stats = router.get_statistics()
    for route, data in stats.items():
        print(
            f"{route}: {data['count']} calls, "
            f"avg performance: {data['avg_performance']:.3f}"
        )

    # Demonstrate gradient flow
    print("\n\n=== Gradient Flow Demonstration ===")

    # Get initial parameters
    initial_weights = router.routing_weights.copy()

    # Define a loss function for demonstration
    def routing_loss(weights):
        # Simplified loss: prefer "fast" for short texts
        embedding = router.text_to_embedding("Hi")
        logits = jnp.dot(embedding, weights)
        scores = jax.nn.softmax(logits)
        fast_idx = router.route_names.index("fast")
        return -jnp.log(scores[fast_idx])

    # Compute gradient
    grad = jax.grad(routing_loss)(router.routing_weights)
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient norm: {jnp.linalg.norm(grad):.6f}")
    print(
        f"Weights changed: {not jnp.allclose(initial_weights, router.routing_weights)}"
    )

    # Test JIT-compiled version
    print("\n\n=== JIT Optimization ===")
    optimized_router = OptimizedLearnableRouter(routes)

    # Time comparison (in practice, would be more significant with larger batches)
    import time

    query = "Explain the theory of relativity"

    # Regular version
    start = time.time()
    result1 = await router(query)
    regular_time = time.time() - start

    # JIT version (first call includes compilation)
    start = time.time()
    result2 = await optimized_router(query)
    jit_time_first = time.time() - start

    # JIT version (subsequent calls are faster)
    start = time.time()
    result3 = await optimized_router(query)
    jit_time_second = time.time() - start

    print(f"Regular version: {regular_time*1000:.2f}ms")
    print(f"JIT first call: {jit_time_first*1000:.2f}ms (includes compilation)")
    print(f"JIT subsequent: {jit_time_second*1000:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
