"""Examples demonstrating XCS transformation patterns.

This module shows how to use XCS transformations for various
real-world use cases, from simple parallelization to complex
hybrid tensor/orchestration workflows.
"""

from ember.xcs import jit, vmap, pmap, scan, grad
from ember.api import model
import jax.numpy as jnp


# =============================================================================
# Pattern 1: Simple Parallelization
# =============================================================================

@jit
def analyze_documents(documents):
    """Automatically parallelize document analysis.
    
    XCS detects the list comprehension and executes in parallel.
    """
    # This runs in parallel automatically!
    summaries = [summarize_doc(doc) for doc in documents]
    return synthesize_summaries(summaries)


def summarize_doc(doc):
    """Summarize a single document using LLM."""
    llm = model("gpt-4")
    return llm(f"Summarize this document in 2 sentences: {doc}")


def synthesize_summaries(summaries):
    """Combine multiple summaries into final report."""
    llm = model("claude-3-opus")
    all_summaries = "\n".join(summaries)
    return llm(f"Synthesize these summaries into a report:\n{all_summaries}")


# =============================================================================
# Pattern 2: Batched Processing with vmap
# =============================================================================

@jit
@vmap
def classify_texts(text):
    """Classify texts in parallel using vmap.
    
    vmap automatically batches LLM calls for efficiency.
    """
    # Extract features (tensor operation)
    embedding = embed_text(text)
    
    # Classify using LLM (orchestration operation)
    llm = model("gpt-4")
    classification = llm(f"Classify this text: {text}")
    
    # Post-process (tensor operation)
    confidence = compute_confidence(embedding)
    
    return {"class": classification, "confidence": confidence}


def embed_text(text):
    """Mock text embedding (would use real embedder)."""
    # In real code, this would call an embedding model
    return jnp.array([len(text), text.count(" "), ord(text[0])])


def compute_confidence(embedding):
    """Compute classification confidence from embedding."""
    return jnp.tanh(jnp.mean(embedding))


# =============================================================================
# Pattern 3: Sequential Processing with scan
# =============================================================================

@jit
@scan
def iterative_refinement(draft, feedback):
    """Iteratively refine a draft based on feedback.
    
    Each iteration improves the draft based on user feedback.
    """
    llm = model("claude-3-opus")
    
    # Improve draft based on feedback
    improved = llm(f"""
    Current draft: {draft}
    Feedback: {feedback}
    
    Improve the draft based on the feedback.
    """)
    
    # Return both new state and output
    return improved, improved


def refine_document(initial_draft, feedback_list):
    """Refine a document through multiple rounds of feedback."""
    final_draft, revision_history = iterative_refinement(initial_draft, feedback_list)
    return {
        "final": final_draft,
        "revisions": revision_history
    }


# =============================================================================
# Pattern 4: Hybrid Tensor/Orchestration with grad
# =============================================================================

@jit
@grad
def hybrid_loss(params, inputs, targets):
    """Compute loss for hybrid model with tensor and LLM components.
    
    XCS intelligently computes gradients only for tensor operations.
    """
    # Tensor operation - differentiable
    embeddings = neural_encoder(params, inputs)
    predictions = neural_classifier(embeddings)
    tensor_loss = jnp.mean((predictions - targets) ** 2)
    
    # Orchestration operation - not differentiable
    # XCS handles this gracefully
    quality = assess_quality(predictions)
    
    # Combined loss (only tensor part gets gradients)
    return tensor_loss + quality_penalty(quality)


def neural_encoder(params, inputs):
    """Simple neural encoder (differentiable)."""
    W, b = params["encoder"]
    return jnp.tanh(inputs @ W + b)


def neural_classifier(embeddings):
    """Neural classifier (differentiable)."""
    # Simplified for example
    return jnp.mean(embeddings, axis=-1)


def assess_quality(predictions):
    """Assess prediction quality using LLM (not differentiable)."""
    # In real code, would call LLM to assess quality
    return 0.8  # Mock quality score


def quality_penalty(quality):
    """Convert quality score to penalty."""
    return max(0, 1.0 - quality)


# =============================================================================
# Pattern 5: Nested Transformations
# =============================================================================

@jit
@vmap  # Batch over users
@vmap  # Batch over documents per user
def analyze_user_documents(doc):
    """Analyze documents with nested batching.
    
    Efficiently processes multiple documents for multiple users.
    """
    # Each document is processed independently
    analysis = analyze_single_document(doc)
    return analysis


def analyze_single_document(doc):
    """Analyze a single document."""
    llm = model("gpt-4")
    return llm(f"Analyze this document: {doc}")


# =============================================================================
# Pattern 6: Complex Production Pipeline
# =============================================================================

@jit
def production_pipeline(requests):
    """Complex production pipeline with multiple stages.
    
    XCS discovers the full operator tree and optimizes globally.
    """
    # Stage 1: Validate and preprocess (parallel)
    validated = [validate_request(req) for req in requests]
    
    # Stage 2: Route to appropriate handlers (parallel within groups)
    routed = route_requests(validated)
    
    # Stage 3: Process each group (parallel)
    results = {}
    for category, items in routed.items():
        results[category] = process_category(category, items)
    
    # Stage 4: Post-process and combine
    return combine_results(results)


def validate_request(request):
    """Validate and preprocess a request."""
    # Validation logic here
    return {"valid": True, "data": request}


def route_requests(requests):
    """Route requests to appropriate handlers."""
    routed = {"simple": [], "complex": [], "special": []}
    for req in requests:
        # Routing logic
        category = determine_category(req)
        routed[category].append(req)
    return routed


def determine_category(request):
    """Determine request category."""
    # Simple routing logic for example
    return "simple"


def process_category(category, items):
    """Process items in a category."""
    processor = get_processor(category)
    # XCS parallelizes this automatically
    return [processor(item) for item in items]


def get_processor(category):
    """Get appropriate processor for category."""
    processors = {
        "simple": lambda x: f"Simple: {x}",
        "complex": lambda x: f"Complex: {x}",
        "special": lambda x: f"Special: {x}"
    }
    return processors.get(category, lambda x: x)


def combine_results(results):
    """Combine results from all categories."""
    combined = []
    for category, items in results.items():
        combined.extend(items)
    return combined


# =============================================================================
# Pattern 7: Error Handling Patterns
# =============================================================================

# Default: Fail fast
@jit
def strict_pipeline(items):
    """Pipeline that fails on first error."""
    return [process_item(item) for item in items]


# Resilient: Continue on errors
@jit(config={"on_error": "continue"})
def resilient_pipeline(items):
    """Pipeline that continues despite errors."""
    return [safe_process(item) for item in items]


def process_item(item):
    """Process a single item (may fail)."""
    if item.get("invalid"):
        raise ValueError(f"Invalid item: {item}")
    return f"Processed: {item}"


def safe_process(item):
    """Safely process item with fallback."""
    try:
        return process_item(item)
    except Exception as e:
        return f"Failed: {e}"


# =============================================================================
# Future Pattern: Distributed Ensemble (Coming Soon)
# =============================================================================

# @jit
# @pmap(mesh=model_mesh, axis_name='model')
# def distributed_ensemble(prompt):
#     """Distribute inference across multiple models.
#     
#     Coming soon: ModelMesh will enable distribution across:
#     - Multiple API keys
#     - Different model providers
#     - Geographic regions
#     - Specialized models (code, math, vision)
#     """
#     return llm(prompt)


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    # Example 1: Document analysis
    docs = ["Document 1 content...", "Document 2 content...", "Document 3 content..."]
    result = analyze_documents(docs)
    print(f"Analysis result: {result}")
    
    # Example 2: Batch classification
    texts = ["positive text", "negative text", "neutral text"]
    classifications = classify_texts(texts)
    print(f"Classifications: {classifications}")
    
    # Example 3: Iterative refinement
    draft = "Initial draft of my document."
    feedback = ["Make it more concise", "Add more details", "Improve clarity"]
    refined = refine_document(draft, feedback)
    print(f"Final draft: {refined['final']}")
    
    # Example 4: Production pipeline
    requests = [{"id": 1}, {"id": 2}, {"id": 3}]
    results = production_pipeline(requests)
    print(f"Pipeline results: {results}")