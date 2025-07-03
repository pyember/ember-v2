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

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import (
    print_section_header,
    print_example_output,
    ensure_api_key,
)
from ember.api import models, operators
from ember.operators.common import ModelCall, ModelText, Chain, Ensemble
from ember.api.xcs import jit


def main():
    """Explore the progressive disclosure of Ember's operator system."""
    print_section_header("Operators Progressive Disclosure")

    # Show API key status
    has_openai = ensure_api_key("openai")
    has_anthropic = ensure_api_key("anthropic")

    if not has_openai and not has_anthropic:
        print("\n⚠️  Running in demo mode - set API keys for full examples")
    else:
        print(f"\n✓ API keys available: OpenAI={has_openai}, Anthropic={has_anthropic}")
    print()

    # Level 1: Simple Functions (90% of use cases)
    print("Level 1: Simple Functions (90% Case)")
    print("=" * 50 + "\n")

    # The simplest possible operator
    @operators.op
    def sentiment_analyzer(text: str) -> dict:
        """Analyze sentiment - just a function!"""
        # Simple rule-based sentiment for demo
        sentiment = (
            "positive"
            if any(w in text.lower() for w in ["good", "great", "love"])
            else "neutral"
        )
        return {"text": text, "sentiment": sentiment}

    # Real model version using ModelText
    @operators.op
    def model_sentiment_analyzer(text: str) -> dict:
        """Model-based sentiment analysis."""
        if ensure_api_key("openai"):
            try:
                model = ModelText("gpt-4o-mini", temperature=0.1)
                prompt = f"Analyze the sentiment of this text (positive/negative/neutral): '{text}'"
                response = model(prompt)
                # Extract sentiment from response
                sentiment = (
                    "positive"
                    if "positive" in response.lower()
                    else "negative" if "negative" in response.lower() else "neutral"
                )
                return {
                    "text": text,
                    "sentiment": sentiment,
                    "raw_response": response[:50] + "...",
                }
            except Exception as e:
                # Fallback to rule-based
                sentiment = (
                    "positive"
                    if any(w in text.lower() for w in ["good", "great", "love"])
                    else "neutral"
                )
                return {"text": text, "sentiment": sentiment, "error": str(e)}
        else:
            # Fallback to rule-based
            sentiment = (
                "positive"
                if any(w in text.lower() for w in ["good", "great", "love"])
                else "neutral"
            )
            return {"text": text, "sentiment": sentiment}

    # Test both versions
    result_simple = sentiment_analyzer("I love this new API!")
    result_model = model_sentiment_analyzer("I love this new API!")

    print("Simple rule-based operator:")
    print_example_output("Input", result_simple["text"])
    print_example_output("Sentiment", result_simple["sentiment"])

    print("\nModel-based operator:")
    print_example_output("Input", result_model["text"])
    print_example_output("Sentiment", result_model["sentiment"])
    if "raw_response" in result_model:
        print_example_output("Model response", result_model["raw_response"])

    # Level 2: Basic Operator Class (10% of use cases)
    print("\n" + "=" * 50)
    print("Level 2: Basic Operator Class")
    print("=" * 50 + "\n")

    class TextProcessor(operators.Operator):
        """Basic operator with initialization."""

        style: str
        use_model: bool
        model: object  # Union[ModelText, None]

        def __init__(self, style: str = "formal", use_model: bool = False):
            # Initialize model binding if requested
            if use_model and ensure_api_key("openai"):
                model = ModelText("gpt-4o-mini", temperature=0.7)
            else:
                model = None

            # Equinox-style initialization - use object.__setattr__ to bypass immutability during init
            object.__setattr__(self, "style", style)
            object.__setattr__(self, "use_model", use_model)
            object.__setattr__(self, "model", model)

        def forward(self, text: str) -> dict:
            """Process text according to style."""
            if self.model:
                # Use model for processing
                try:
                    prompt = f"Rewrite this text in {self.style} style: '{text}'"
                    processed = self.model(prompt)
                except Exception as e:
                    # Fallback to rule-based
                    processed = self._rule_based_processing(text)
            else:
                processed = self._rule_based_processing(text)

            return {
                "original": text,
                "processed": processed,
                "style": self.style,
                "method": "model" if self.model else "rule-based",
            }

        def _rule_based_processing(self, text: str) -> str:
            """Fallback rule-based processing."""
            if self.style == "formal":
                return text.strip().title()
            elif self.style == "casual":
                return text.strip().lower()
            else:
                return text.strip()

    # Test both rule-based and model-based processing
    formal_processor = TextProcessor(style="formal")
    model_processor = TextProcessor(style="formal", use_model=True)

    test_text = "hello world, this is a test"

    result_rules = formal_processor(test_text)
    result_model = model_processor(test_text)

    print("Rule-based operator:")
    print_example_output("Processed", result_rules["processed"])
    print_example_output("Method", result_rules["method"])

    print("\nModel-based operator:")
    print_example_output("Processed", result_model["processed"])
    print_example_output("Method", result_model["method"])

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

        routes: Dict[str, str]
        embedding_dim: int
        routing_weights: jnp.ndarray
        temperature: jnp.ndarray
        models: Dict[str, object]

        def __init__(self, routes: Dict[str, str], embedding_dim: int = 8):
            # Dynamic parameters (JAX arrays are automatically learnable)
            routing_weights = jax.random.normal(
                jax.random.PRNGKey(0), (len(routes), embedding_dim)
            )
            temperature = jnp.array(1.0)

            # Model bindings are static (not learnable)
            # Using ModelCall for full response objects
            models = {}
            for name, model_name in routes.items():
                try:
                    if model_name.startswith("gpt") and ensure_api_key("openai"):
                        models[name] = ModelCall("gpt-4o-mini")
                    elif model_name.startswith("claude") and ensure_api_key(
                        "anthropic"
                    ):
                        models[name] = ModelCall("claude-3-haiku")
                    else:
                        # Fallback to mock
                        models[name] = lambda q: f"Mock {model_name} response to: {q}"
                except Exception:
                    models[name] = lambda q: f"Mock {model_name} response to: {q}"

            # Equinox-style initialization
            object.__setattr__(self, "routes", routes)
            object.__setattr__(self, "embedding_dim", embedding_dim)
            object.__setattr__(self, "routing_weights", routing_weights)
            object.__setattr__(self, "temperature", temperature)
            object.__setattr__(self, "models", models)

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
            try:
                if callable(model) and hasattr(model, "forward"):
                    # It's a ModelCall operator
                    response_obj = model(query)
                    response = (
                        response_obj.text
                        if hasattr(response_obj, "text")
                        else str(response_obj)
                    )
                else:
                    # It's a mock function
                    response = model(query)
            except Exception as e:
                response = f"Error calling {route_name}: {e}"

            return {
                "query": query,
                "selected_route": route_name,
                "confidence": float(scores[route_idx]),
                "response": response,
                "all_scores": {
                    name: float(score)
                    for name, score in zip(self.routes.keys(), scores)
                },
            }

    # Create learnable router with real models
    router = LearnableRouter(
        {"technical": "gpt-4", "creative": "claude-3-opus", "simple": "gpt-3.5-turbo"}
    )

    print("\nRouting models initialized:")
    for route_name, model in router.models.items():
        model_type = "Real model" if hasattr(model, "forward") else "Mock model"
        print(f"  {route_name}: {model_type}")

    print("Learnable Router Example:")
    queries = [
        "Explain quantum computing",
        "Write a poem about the sea",
        "What is 2+2?",
    ]

    for query in queries:
        result = router(query)
        print(f"\nQuery: '{query}'")
        print(f"Route: {result['selected_route']} ({result['confidence']:.2%})")
        if len(result["response"]) > 100:
            print(f"Response: {result['response'][:100]}...")
        else:
            print(f"Response: {result['response']}")

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
        return {"text": text, "static_count": word_count, "dynamic_score": float(score)}

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
    print("  2. ModelCall gives full response objects, ModelText gives text")
    print("  3. Chain and Ensemble operators enable composition")
    print("  4. JAX arrays are automatically learnable")
    print("  5. Models and tools are automatically static")
    print("  6. XCS handles the static/dynamic split perfectly")
    print("  7. Each level is self-contained - no forced complexity")

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

    # Bonus: Show operator composition with new primitives
    if has_openai or has_anthropic:
        print("\n" + "=" * 50)
        print("Bonus: Operator Composition with ModelCall/ModelText")
        print("=" * 50 + "\n")

        try:
            # Create a composition pipeline
            @operators.op
            def extract_keywords(text: str) -> str:
                """Simple keyword extraction."""
                words = text.lower().split()
                keywords = [w for w in words if len(w) > 4]
                return f"Keywords: {', '.join(keywords[:3])}"

            # Create model operator
            if has_openai:
                summarizer = ModelText("gpt-4o-mini", temperature=0.3)
            else:
                summarizer = ModelText("claude-3-haiku", temperature=0.3)

            # Chain them together
            analysis_pipeline = Chain(
                [
                    extract_keywords,
                    lambda keywords: f"Analyze these keywords and provide insights: {keywords}",
                    summarizer,
                ]
            )

            test_text = "Machine learning algorithms are transforming artificial intelligence applications worldwide"

            print("Composition Pipeline Example:")
            print_example_output("Input", test_text)

            result = analysis_pipeline(test_text)
            print_example_output("Output", result[:100] + "...")

            print("\n✅ This demonstrates:")
            print("  • Mixing @op functions with ModelText operators")
            print("  • Chain operator for sequential processing")
            print("  • Lambda functions for quick transformations")
            print("  • Seamless composition of different operator types")

        except Exception as e:
            print(f"\n⚠️  Composition example error: {e}")
            print("This might be due to API limits or connectivity issues.")

    else:
        print("\n💡 With API keys, you could see:")
        print("  • Real model composition examples")
        print("  • Chain operators with ModelText")
        print("  • Ensemble patterns with multiple models")
        print("  • Hybrid rule-based + model pipelines")
    
    print("\nNext: Explore judge synthesis patterns (judge_synthesis.py)!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
