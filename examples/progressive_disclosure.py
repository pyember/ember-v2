"""Progressive disclosure examples for Ember.

This file demonstrates how Ember supports different levels of complexity,
from simple function decorators to advanced JAX-integrated systems.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List

import ember
from ember.api import model, op
from ember.operators.base import Operator
from ember.api.types import EmberModel


# =============================================================================
# Level 1: Simple Functions (The 90% Case)
# =============================================================================


@op
def classify_sentiment(text: str) -> str:
    """Simplest possible operator - just a decorated function."""
    llm = model("gpt-4")
    response = llm(f"Classify sentiment as positive/negative/neutral: {text}")
    return response.text


@op
def summarize(text: str, max_words: int = 50) -> str:
    """Function with parameters."""
    llm = model("gpt-4", temperature=0.3)
    prompt = f"Summarize in {max_words} words: {text}"
    return llm(prompt).text


# Usage is dead simple
result = classify_sentiment("This product is amazing!")
summary = summarize("Long article text...", max_words=30)


# =============================================================================
# Level 2: Basic Operators (Adding Structure)
# =============================================================================


class SimpleClassifier(Operator):
    """Basic operator with explicit model binding."""

    def __init__(self, model_name: str = "gpt-4"):
        self.model = model(model_name)

    def forward(self, text: str) -> str:
        prompt = f"Classify the following text: {text}"
        return self.model(prompt).text


class ChainedProcessor(Operator):
    """Operator that composes other operators."""

    def __init__(self):
        self.classifier = SimpleClassifier()
        self.summarizer = summarize  # Can use decorated functions too

    def forward(self, text: str) -> Dict[str, str]:
        return {"sentiment": self.classifier(text), "summary": self.summarizer(text)}


# =============================================================================
# Level 3: Validated Operators (Production Ready)
# =============================================================================


# Define data models for validation
class TextInput(EmberModel):
    text: str
    max_length: int = 1000
    language: str = "en"


class ClassificationOutput(EmberModel):
    label: str
    confidence: float
    reasoning: str = ""


class ProductionClassifier(Operator):
    """Operator with input/output validation."""

    input_spec = TextInput
    output_spec = ClassificationOutput

    def __init__(self):
        self.model = model("gpt-4", temperature=0.1)

    def forward(self, input: TextInput) -> ClassificationOutput:
        prompt = f"""
        Classify this {input.language} text:
        {input.text[:input.max_length]}
        
        Return JSON with: label, confidence (0-1), reasoning
        """

        response = self.model(prompt).text
        # Parse response and create validated output
        # (In real code, you'd parse the JSON properly)
        return ClassificationOutput(
            label="positive", confidence=0.95, reasoning="Strong positive indicators"
        )


# =============================================================================
# Level 4: JAX-Integrated Operators (Advanced)
# =============================================================================


class LearnableRouter(Operator):
    """Router with learnable parameters using JAX."""

    def __init__(self, models: Dict[str, str], embed_dim: int, key: jax.Array):
        # Static components
        self.models = {name: model(id) for name, id in models.items()}
        self.embed_dim = embed_dim

        # Dynamic components (JAX arrays are automatically detected by Ember)
        k1, k2 = jax.random.split(key)
        self.routing_weights = jax.random.normal(k1, (embed_dim, len(models)))
        self.temperature = jnp.array(1.0)
        self.bias = jnp.zeros(len(models))

    def compute_routing_scores(self, embedding: jax.Array) -> jax.Array:
        """Compute routing scores - this part is differentiable."""
        logits = embedding @ self.routing_weights + self.bias
        return jax.nn.softmax(logits / self.temperature)

    def forward(self, text: str, embedding: jax.Array) -> str:
        # Compute routing (differentiable)
        scores = self.compute_routing_scores(embedding)

        # Select model (discrete choice)
        model_idx = jnp.argmax(scores)
        model_name = list(self.models.keys())[model_idx]

        # Call selected model (non-differentiable API call)
        return self.models[model_name](text).text


# =============================================================================
# Level 5: Complex Nested Systems (Platform Power)
# =============================================================================


class QualityAssessment(Operator):
    """Assess answer quality using a model."""

    def __init__(self, key: jax.Array):
        self.model = model("gpt-4")
        self.assessment_weights = jax.random.normal(key, (10,))

    def forward(self, answer: str) -> jax.Array:
        # In real implementation, would use model to assess quality
        # For demo, return a simple score
        return jnp.array(0.85)


class AnswerRefinement(Operator):
    """Refine answers that don't meet quality threshold."""

    def __init__(self):
        self.model = model("gpt-4")

    def forward(self, initial_answer: str, question: str) -> str:
        prompt = f"Improve this answer to '{question}': {initial_answer}"
        return self.model(prompt).text


class AdaptiveQASystem(Operator):
    """Complex system with nested operators and mixed static/dynamic fields."""

    def __init__(self, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)

        # Nested operators (static, but may contain dynamic fields)
        self.router = LearnableRouter(
            models={
                "fast": "gpt-3.5-turbo",
                "accurate": "gpt-4",
                "creative": "claude-3",
            },
            embed_dim=384,
            key=k1,
        )

        self.quality_checker = QualityAssessment(key=k2)
        self.answer_refiner = AnswerRefinement()

        # System-level learnable parameters
        self.quality_threshold = jnp.array(0.7)
        self.refinement_weight = jax.random.normal(k3, (10,))

    def forward(self, question: str, context_embedding: jax.Array) -> Dict:
        # Route to appropriate model
        initial_answer = self.router(question, context_embedding)

        # Assess quality (returns score as JAX array)
        quality_score = self.quality_checker(initial_answer)

        # Conditionally refine based on learnable threshold
        if quality_score < self.quality_threshold:
            final_answer = self.answer_refiner(initial_answer, question)
        else:
            final_answer = initial_answer

        return {
            "answer": final_answer,
            "quality_score": quality_score,
            "model_used": (
                "refined" if quality_score < self.quality_threshold else "direct"
            ),
        }


# =============================================================================
# JAX Transformations Work Naturally
# =============================================================================

# Create system
qa_system = AdaptiveQASystem(jax.random.PRNGKey(0))


# Define loss function
def loss_fn(system, question, embedding, quality_target):
    result = system(question, embedding)
    # Loss based on quality score prediction
    return jnp.mean((result["quality_score"] - quality_target) ** 2)


# Create test embedding
test_embedding = jax.random.normal(jax.random.PRNGKey(1), (384,))

# Compute gradients - only JAX arrays get gradients
grads = jax.grad(loss_fn)(qa_system, "What is JAX?", test_embedding, 0.9)

# Check what got gradients
print("Fields with gradients:")
print(f"- router.routing_weights: {grads.router.routing_weights is not None}")
print(f"- router.temperature: {grads.router.temperature is not None}")
print(f"- quality_threshold: {grads.quality_threshold is not None}")
print(f"- router.models: {grads.router.models}")  # None - static field

# JIT compilation works
jitted_qa = jax.jit(qa_system.compute_routing_scores)

# Vectorization works
batched_qa = jax.vmap(qa_system, in_axes=(None, 0))

# =============================================================================
# Summary of Progressive Disclosure
# =============================================================================

print(
    """
Progressive Disclosure Levels:

1. Simple Functions (@op decorator)
   - One-liner operators
   - No classes needed
   - Perfect for prototyping

2. Basic Operators (Operator class)
   - More structure
   - Explicit initialization
   - Composition of operators

3. Validated Operators (with specs)
   - Input/output validation
   - Type safety
   - Production ready

4. JAX-Integrated (learnable parameters)
   - Automatic differentiation
   - JAX arrays as dynamic fields
   - Optimization ready

5. Complex Systems (nested operators)
   - Arbitrary nesting
   - Mixed static/dynamic
   - Platform-level power

All levels are fully compatible and composable!
"""
)
