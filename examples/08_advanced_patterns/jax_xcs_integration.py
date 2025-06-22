"""JAX-XCS Integration - Seamless handling of learnable and static parameters.

Difficulty: Advanced
Time: ~10 minutes

Learning Objectives:
- Understand how JAX and XCS work together
- Learn the static-by-default, dynamic-when-JAX principle
- Build operators that mix ML parameters with API calls
- See how gradients flow through mixed operations

This example demonstrates Ember's key innovation: seamlessly mixing
learnable parameters (JAX arrays) with static components (models, tools)
in a single operator.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
from functools import partial

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import models, operators
from ember.api.xcs import jit


def main():
    """Explore JAX-XCS integration for mixed static/dynamic operations."""
    print_section_header("JAX-XCS Integration")
    
    # Part 1: Understanding Static vs Dynamic
    print("Part 1: Static vs Dynamic Detection")
    print("=" * 50 + "\n")
    
    print("Ember's automatic detection rules:")
    print("  • JAX arrays (jnp.ndarray) → Dynamic (learnable)")
    print("  • Everything else → Static (not learnable)")
    print("  • No decorators needed - it just works!\n")
    
    class MixedOperator(operators.Operator):
        """Operator mixing static and dynamic components."""
        
        def __init__(self):
            # Dynamic (automatically detected as learnable)
            self.weights = jnp.array([1.0, 2.0, 3.0])
            self.bias = jnp.array(0.5)
            
            # Static (automatically detected as non-learnable)
            self.config = {"mode": "production"}
            self.threshold = 0.8
            self.model_name = "gpt-4"
            # In real use: self.model = models.instance("gpt-4")
        
        def forward(self, x: jnp.ndarray) -> Dict:
            """Forward pass mixing static and dynamic ops."""
            # Dynamic computation (differentiable)
            score = jnp.dot(self.weights, x) + self.bias
            
            # Static decision based on dynamic result
            if score > self.threshold:
                action = "accept"
            else:
                action = "reject"
            
            return {
                "score": float(score),
                "action": action,
                "model": self.model_name
            }
    
    op = MixedOperator()
    result = op(jnp.array([0.1, 0.2, 0.3]))
    
    print("Mixed operator results:")
    print_example_output("Dynamic score", f"{result['score']:.2f}")
    print_example_output("Static action", result['action'])
    print_example_output("Static model", result['model'])
    
    # Part 2: Gradient Flow Through Mixed Operations
    print("\n" + "=" * 50)
    print("Part 2: Gradients Through Mixed Operations")
    print("=" * 50 + "\n")
    
    def loss_fn(weights: jnp.ndarray, bias: jnp.ndarray, x: jnp.ndarray, target: float) -> float:
        """Loss function for demonstration."""
        score = jnp.dot(weights, x) + bias
        return jnp.square(score - target)
    
    # Compute gradients only w.r.t. dynamic parameters
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))  # Only weights and bias
    
    x = jnp.array([0.1, 0.2, 0.3])
    target = 5.0
    grads = grad_fn(op.weights, op.bias, x, target)
    
    print("Gradient computation:")
    print_example_output("Input", x)
    print_example_output("Target", target)
    print_example_output("Weight gradients", grads[0])
    print_example_output("Bias gradient", grads[1])
    print("\nNote: Static parameters (config, model) don't get gradients!")
    
    # Part 3: Real-World Example - Learned Prompt Router
    print("\n" + "=" * 50)
    print("Part 3: Real-World Example - Learned Prompt Router")
    print("=" * 50 + "\n")
    
    class LearnedPromptRouter(operators.Operator):
        """Routes prompts to models based on learned embeddings.
        
        This demonstrates a real use case where we:
        1. Learn embeddings for different prompt types
        2. Route to different models based on similarity
        3. Call the appropriate model (static operation)
        4. Can train the routing via gradients
        """
        
        def __init__(self, routes: Dict[str, str]):
            # Static: model configurations
            self.routes = routes
            self.models = {
                name: f"MockModel[{model}]"  # Would be models.instance(model)
                for name, model in routes.items()
            }
            
            # Dynamic: learnable embeddings for each route
            num_routes = len(routes)
            embedding_dim = 16
            
            key = jax.random.PRNGKey(42)
            self.route_embeddings = jax.random.normal(key, (num_routes, embedding_dim))
            self.projection = jax.random.normal(key, (embedding_dim, embedding_dim))
            self.temperature = jnp.array(1.0)
        
        def encode_prompt(self, prompt: str) -> jnp.ndarray:
            """Simple prompt encoding (would use real embeddings)."""
            # Simulate encoding based on prompt characteristics
            features = jnp.array([
                float(len(prompt)),
                float(prompt.count(" ")),
                float("?" in prompt),
                float("explain" in prompt.lower()),
                float("create" in prompt.lower()),
                float("analyze" in prompt.lower()),
                float(prompt.count("\n")),
                float(len(prompt.split())),
            ])
            # Pad to embedding dimension
            return jnp.pad(features, (0, 8), constant_values=0.0)
        
        def compute_routing_scores(self, prompt_embedding: jnp.ndarray) -> jnp.ndarray:
            """Compute similarity scores (differentiable)."""
            # Project prompt embedding
            projected = jnp.dot(prompt_embedding, self.projection)
            
            # Compute similarities with route embeddings
            similarities = jnp.dot(self.route_embeddings, projected)
            
            # Apply temperature and softmax
            return jax.nn.softmax(similarities / self.temperature)
        
        def forward(self, prompt: str) -> Dict:
            """Route prompt to best model."""
            # Encode prompt (could be made differentiable with learned encoder)
            prompt_embedding = self.encode_prompt(prompt)
            
            # Compute routing scores (differentiable part)
            scores = self.compute_routing_scores(prompt_embedding)
            
            # Select best route (non-differentiable, but that's OK)
            route_names = list(self.routes.keys())
            best_idx = jnp.argmax(scores).item()
            selected_route = route_names[best_idx]
            
            # Call the selected model (static operation)
            model = self.models[selected_route]
            response = f"{model} handling: '{prompt[:30]}...'"
            
            return {
                "prompt": prompt,
                "selected_route": selected_route,
                "selected_model": self.routes[selected_route],
                "confidence": float(scores[best_idx]),
                "all_scores": dict(zip(route_names, scores.tolist())),
                "response": response
            }
    
    # Create router with different model specializations
    router = LearnedPromptRouter({
        "analytical": "gpt-4",
        "creative": "claude-3-opus", 
        "coding": "gpt-4-turbo",
        "simple": "gpt-3.5-turbo"
    })
    
    # Test routing
    test_prompts = [
        "Explain the theory of relativity",
        "Write a haiku about mountains",
        "Debug this Python code: def f(x): return x/0",
        "What is 2+2?"
    ]
    
    print("Learned routing results:")
    for prompt in test_prompts:
        result = router(prompt)
        print(f"\n'{prompt[:40]}...'")
        print(f"  → {result['selected_route']} ({result['confidence']:.1%})")
        print(f"  Model: {result['selected_model']}")
    
    # Part 4: Training the Router
    print("\n" + "=" * 50)
    print("Part 4: Training via Gradients")
    print("=" * 50 + "\n")
    
    def routing_loss(route_embeddings, projection, prompt, target_route_idx):
        """Loss function for training the router."""
        # This is a simplified version - real training would be more complex
        router.route_embeddings = route_embeddings
        router.projection = projection
        
        prompt_embedding = router.encode_prompt(prompt)
        scores = router.compute_routing_scores(prompt_embedding)
        
        # Cross-entropy loss
        return -jnp.log(scores[target_route_idx] + 1e-7)
    
    # Compute gradients for router parameters
    grad_fn = jax.grad(routing_loss, argnums=(0, 1))
    
    # Example gradient computation
    prompt = "Write elegant Python code"
    target_idx = 2  # Should route to "coding"
    
    grads = grad_fn(
        router.route_embeddings,
        router.projection,
        prompt,
        target_idx
    )
    
    print("Training demonstration:")
    print_example_output("Training prompt", prompt)
    print_example_output("Target route", list(router.routes.keys())[target_idx])
    print_example_output("Embedding gradients shape", grads[0].shape)
    print_example_output("Projection gradients shape", grads[1].shape)
    
    # Part 5: XCS JIT Compilation
    print("\n" + "=" * 50)
    print("Part 5: XCS JIT with Mixed Operations")
    print("=" * 50 + "\n")
    
    @jit
    def process_batch(prompts: List[str], router_params: Tuple) -> List[Dict]:
        """Process multiple prompts efficiently.
        
        XCS handles:
        - Static string operations
        - Dynamic JAX computations
        - Caching of compiled functions
        """
        route_embeddings, projection, temperature = router_params
        
        results = []
        for prompt in prompts:
            # Mix of static and dynamic operations
            result = {
                "prompt": prompt,
                "length": len(prompt),  # Static
                "score": float(jnp.sum(route_embeddings))  # Dynamic
            }
            results.append(result)
        
        return results
    
    # Test JIT compilation
    params = (router.route_embeddings, router.projection, router.temperature)
    batch_results = process_batch(test_prompts[:2], params)
    
    print("JIT-compiled batch processing:")
    for result in batch_results:
        print(f"  '{result['prompt'][:30]}...' → score: {result['score']:.2f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ JAX-XCS Integration Summary")
    print("=" * 50)
    
    print("\n🔑 Key Principles:")
    print("  1. JAX arrays are automatically learnable")
    print("  2. Everything else is automatically static")
    print("  3. Gradients flow only through JAX operations")
    print("  4. Static operations (models, tools) work seamlessly")
    print("  5. XCS JIT handles mixed static/dynamic efficiently")
    
    print("\n💡 Practical Benefits:")
    print("  • No manual static/dynamic annotations")
    print("  • Models and tools integrate naturally")
    print("  • Full differentiability where needed")
    print("  • Optimal performance via XCS")
    print("  • Clean separation of concerns")
    
    print("\n🎯 Use Cases:")
    print("  • Learned routing between models")
    print("  • Prompt optimization with model feedback")
    print("  • Hybrid symbolic-neural systems")
    print("  • Differentiable programming with LLMs")
    print("  • Neural architecture search over operators")
    
    print("\n📚 Example Pattern:")
    print("```python")
    print("class HybridOperator(operators.Operator):")
    print("    def __init__(self):")
    print("        # Learnable parameters")
    print("        self.weights = jnp.array([...])")
    print("        ")
    print("        # Static components")
    print("        self.model = models.instance('gpt-4')")
    print("        self.tool = MyTool()")
    print("    ")
    print("    def forward(self, x):")
    print("        # Mix learnable and static freely")
    print("        score = jnp.dot(self.weights, x)")
    print("        if score > threshold:")
    print("            return self.model(prompt)")
    print("        else:")
    print("            return self.tool.process(x)")
    print("```")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())