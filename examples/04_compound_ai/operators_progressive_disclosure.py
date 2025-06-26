"""Operators Progressive Disclosure - From simple functions to advanced ML systems.

Difficulty: Intermediate to Advanced
Time: ~15 minutes

Learning Objectives:
- Understand the 5 levels of operator complexity
- See how JAX and XCS integration works
- Learn to mix static (models/tools) and dynamic (learnable) parameters
- Master the tiered API design

The operators system provides progressive disclosure:
Level 1: Simple functions with @op
Level 2: Basic Operator class
Level 3: Validated operators with specs
Level 4: JAX-integrated with learnable parameters
Level 5: Complex nested systems
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import jax
import jax.numpy as jnp
import equinox as eqx

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from ember.api import models, operators
from ember.api.xcs import jit


def main():
    """Explore the progressive disclosure of Ember's operator system."""
    print_section_header("Operators Progressive Disclosure")
    
    # Level 1: Simple Functions (90% of use cases)
    print("Level 1: Simple Functions (90% Case)")
    print("=" * 50 + "\n")
    
    # The simplest possible operator
    @operators.op
    def sentiment_analyzer(text: str) -> dict:
        """Analyze sentiment - just a function!"""
        # In real use, this would call models()
        sentiment = "positive" if any(w in text.lower() for w in ["good", "great", "love"]) else "neutral"
        return {"text": text, "sentiment": sentiment}
    
    result = sentiment_analyzer("I love this new API!")
    print("Simple function operator:")
    print_example_output("Input", result["text"])
    print_example_output("Sentiment", result["sentiment"])
    
    # Level 2: Basic Operator Class (10% of use cases)
    print("\n" + "=" * 50)
    print("Level 2: Basic Operator Class")
    print("=" * 50 + "\n")
    
    class TextProcessor(operators.Operator):
        """Basic operator with initialization."""
        style: str = eqx.field(static=True)
        
        def __init__(self, style: str = "formal"):
            object.__setattr__(self, "style", style)
            # Could initialize model bindings here
            # self.model = models.instance("gpt-4", temperature=0.7)
        
        def forward(self, text: str) -> dict:
            """Process text according to style."""
            if self.style == "formal":
                processed = text.strip().title()
            elif self.style == "casual":
                processed = text.strip().lower()
            else:
                processed = text.strip()
            
            return {
                "original": text,
                "processed": processed,
                "style": self.style
            }
    
    formal_processor = TextProcessor(style="formal")
    result = formal_processor("hello world")
    print("Basic operator class:")
    print_example_output("Processed", result["processed"])
    print_example_output("Style", result["style"])
    
    # Level 3: Validated Operators (Advanced)
    print("\n" + "=" * 50)
    print("Level 3: Validated Operators (Advanced)")
    print("=" * 50 + "\n")
    
    print("Using ember.operators.advanced for validation:")
    print("```python")
    print("from ember.operators.advanced import Operator, Specification")
    print("from ember.types import EmberModel, Field")
    print("")
    print("class QueryInput(EmberModel):")
    print("    query: str = Field(..., max_length=1000)")
    print("    max_results: int = Field(10, ge=1, le=100)")
    print("")
    print("class SearchOperator(Operator):")
    print("    specification = Specification(")
    print("        input_model=QueryInput,")
    print("        output_model=SearchResults")
    print("    )")
    print("```")
    
    # Level 4: JAX-Integrated Operators (ML Use Cases)
    print("\n" + "=" * 50)
    print("Level 4: JAX-Integrated Operators (ML Systems)")
    print("=" * 50 + "\n")
    
    class LearnableRouter(operators.Operator):
        """Router with learnable parameters (JAX arrays).
        
        This demonstrates the key innovation: mixing static (models)
        and dynamic (JAX arrays) seamlessly.
        """
        
        def __init__(self, routes: Dict[str, str], embedding_dim: int = 8):
            # Static parameters (automatically detected)
            self.routes = routes  # Route names to model names
            self.embedding_dim = embedding_dim
            
            # Dynamic parameters (JAX arrays are automatically learnable)
            self.routing_weights = jax.random.normal(
                jax.random.PRNGKey(0), 
                (len(routes), embedding_dim)
            )
            self.temperature = jnp.array(1.0)
            
            # Model bindings are static (not learnable)
            self.models = {
                name: f"Model[{model_name}]"  # Would be models.instance(model_name)
                for name, model_name in routes.items()
            }
        
        def compute_scores(self, query_embedding: jnp.ndarray) -> jnp.ndarray:
            """Compute routing scores (differentiable)."""
            # This is differentiable w.r.t. routing_weights
            logits = jnp.dot(self.routing_weights, query_embedding)
            return jax.nn.softmax(logits / self.temperature)
        
        def forward(self, query: str) -> dict:
            """Route query to best model based on learned weights."""
            # Simulate query embedding
            query_embedding = jnp.ones(self.embedding_dim)
            
            # Compute scores (differentiable part)
            scores = self.compute_scores(query_embedding)
            
            # Select route (non-differentiable but that's OK)
            route_idx = jnp.argmax(scores).item()
            route_name = list(self.routes.keys())[route_idx]
            
            # Call the selected model (static operation)
            model = self.models[route_name]
            response = f"{model} response to: {query}"
            
            return {
                "query": query,
                "selected_route": route_name,
                "confidence": float(scores[route_idx]),
                "response": response,
                "all_scores": {name: float(score) 
                             for name, score in zip(self.routes.keys(), scores)}
            }
    
    # Create learnable router
    router = LearnableRouter({
        "technical": "gpt-4",
        "creative": "claude-3-opus",
        "simple": "gpt-3.5-turbo"
    })
    
    print("Learnable Router Example:")
    queries = [
        "Explain quantum computing",
        "Write a poem about the sea",
        "What is 2+2?"
    ]
    
    for query in queries:
        result = router(query)
        print(f"\nQuery: '{query}'")
        print(f"Route: {result['selected_route']} ({result['confidence']:.2%})")
    
    # Show that we can compute gradients
    print("\nKey insight: JAX integration allows gradients!")
    print("```python")
    print("# Compute gradient w.r.t. routing weights")
    print("grad_fn = jax.grad(lambda weights: loss(weights, data))")
    print("grads = grad_fn(router.routing_weights)")
    print("```")
    
    # Level 5: Complex Nested Systems
    print("\n" + "=" * 50)
    print("Level 5: Complex Nested Systems (Platform-Level)")
    print("=" * 50 + "\n")
    
    print("Using ember.operators.experimental for cutting-edge features:")
    print("```python")
    print("from ember.operators.experimental import (")
    print("    trace,           # Execution tracing")
    print("    jit_compile,     # Advanced JIT compilation")
    print("    pattern_optimize # Pattern-based optimization")
    print(")")
    print("")
    print("# Trace execution for debugging")
    print("traced_op = trace(my_complex_operator)")
    print("")
    print("# Compile entire operator graphs")
    print("compiled = jit_compile(operator_graph)")
    print("")
    print("# Pattern-based optimization")
    print("optimized = pattern_optimize(")
    print("    operator,")
    print("    patterns=['fusion', 'constant_folding']")
    print(")")
    print("```")
    
    # Demonstration of XCS Integration
    print("\n" + "=" * 50)
    print("Bonus: XCS Integration for Static/Dynamic Mix")
    print("=" * 50 + "\n")
    
    @jit  # XCS JIT handles both static and dynamic correctly
    def hybrid_processor(text: str, weights: jnp.ndarray) -> dict:
        """Mix static operations (string processing) with dynamic (JAX).
        
        XCS is smart enough to:
        - Keep string operations static
        - Make JAX operations dynamic and differentiable
        - Cache appropriately
        """
        # Static operation
        words = text.split()
        word_count = len(words)
        
        # Dynamic operation (uses JAX array)
        score = jnp.sum(weights[:word_count])
        
        # Mix both in the output
        return {
            "text": text,
            "static_count": word_count,
            "dynamic_score": float(score)
        }
    
    # Test hybrid processing
    test_weights = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = hybrid_processor("Hello world test", test_weights)
    
    print("XCS correctly handles static/dynamic mix:")
    print_example_output("Static result", result["static_count"])
    print_example_output("Dynamic result", result["dynamic_score"])
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Progressive Disclosure Summary")
    print("=" * 50)
    
    print("\n📊 Usage Distribution:")
    print("  Level 1 (@op functions): ~90% of use cases")
    print("  Level 2 (Basic Operator): ~8% of use cases")
    print("  Level 3 (Validated): ~1.5% of use cases")
    print("  Level 4 (JAX/ML): ~0.4% of use cases")
    print("  Level 5 (Platform): ~0.1% of use cases")
    
    print("\n🔑 Key Insights:")
    print("  1. Start simple - just use @op on functions")
    print("  2. JAX arrays are automatically learnable")
    print("  3. Models and tools are automatically static")
    print("  4. XCS handles the static/dynamic split perfectly")
    print("  5. Each level is self-contained - no forced complexity")
    
    print("\n🎯 Design Philosophy:")
    print("  • Simple things should be simple")
    print("  • Complex things should be possible")
    print("  • The API grows with your needs")
    print("  • No premature complexity")
    
    print("\n💡 When to use each level:")
    print("  @op → You just need a simple transformation")
    print("  Operator → You need initialization or state")
    print("  Validated → You need strict input/output contracts")
    print("  JAX-integrated → You're building ML systems")
    print("  Experimental → You're pushing boundaries")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())