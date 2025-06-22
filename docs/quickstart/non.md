# Ember NON (Networks of Networks)

This guide shows how to build robust AI workflows using Ember's composable operator patterns.

## What is NON?

Networks of Networks (NON) is Ember's approach to building reliable AI systems through composition. Instead of relying on a single model call, NON patterns combine multiple operators to achieve better accuracy, consistency, and robustness.

## Core Patterns

### Ensemble Pattern

Run multiple models and combine their outputs:

```python
from ember.api import ember, ensemble

@ember.op
async def robust_answer(question: str) -> str:
    """Get a robust answer using multiple models."""
    # Run multiple models in parallel
    responses = await ensemble(
        question,
        models=["gpt-4", "claude-3", "gemini-pro"]
    )
    
    # Simple majority vote
    from collections import Counter
    most_common = Counter(responses).most_common(1)[0][0]
    return most_common
```

### Judge Pattern

Use a high-quality model to evaluate and synthesize responses:

```python
from ember.api import ember
from pydantic import BaseModel

class JudgedResponse(BaseModel):
    best_answer: str
    reasoning: str
    confidence: float

@ember.op
async def judge_responses(question: str, responses: list[str]) -> JudgedResponse:
    """Use a judge model to select the best response."""
    prompt = f"""Given this question: {question}
    
And these candidate responses:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(responses))}

Select the best response and explain your reasoning."""
    
    return await ember.llm(prompt, output_type=JudgedResponse, model="claude-3-opus")
```

### Verify Pattern

Check and correct answers:

```python
@ember.op
async def verify_answer(question: str, answer: str) -> dict:
    """Verify an answer and provide corrections if needed."""
    prompt = f"""Question: {question}
Answer: {answer}

Is this answer correct? If not, provide the correct answer.
Response format: {{"is_correct": bool, "corrected_answer": str or null, "explanation": str}}"""
    
    return await ember.llm(prompt, output_type=dict)
```

## Complete Pipeline Example

Here's how to combine these patterns into a robust Q&A system:

```python
from ember.api import ember
from typing import List

@ember.op
async def robust_qa_pipeline(question: str) -> dict:
    """Robust Q&A with ensemble, judge, and verification."""
    
    # Step 1: Get multiple candidate answers
    candidates = await ember.parallel([
        ember.llm(question, model="gpt-4", temperature=0.7),
        ember.llm(question, model="claude-3", temperature=0.7),
        ember.llm(question, model="gemini-pro", temperature=0.7)
    ])
    
    # Step 2: Judge selects best answer
    judged = await judge_responses(question, candidates)
    
    # Step 3: Verify the selected answer
    verification = await verify_answer(question, judged.best_answer)
    
    # Return comprehensive result
    return {
        "question": question,
        "candidates": candidates,
        "selected_answer": judged.best_answer,
        "confidence": judged.confidence,
        "is_verified": verification["is_correct"],
        "final_answer": verification.get("corrected_answer") or judged.best_answer,
        "explanation": verification["explanation"]
    }

# Use the pipeline
result = await robust_qa_pipeline("What is the capital of Australia?")
print(f"Final answer: {result['final_answer']}")
print(f"Confidence: {result['confidence']}")
```

## Built-in NON Operators

Ember provides pre-built operators for common patterns:

```python
from ember.api.operators import ensemble, majority_vote, synthesize

# Ensemble with automatic result handling
@ember.op
async def ensemble_classifier(text: str, categories: List[str]) -> str:
    """Classify using ensemble voting."""
    results = await ensemble(
        f"Classify this text into one of {categories}: {text}",
        models=["gpt-4", "claude-3", "gemini-pro"],
        temperature=0.5
    )
    return majority_vote(results)

# Synthesis operator
@ember.op
async def synthesize_summaries(documents: List[str]) -> str:
    """Create a synthesis from multiple document summaries."""
    # Get individual summaries
    summaries = await ember.parallel([
        summarize(doc) for doc in documents
    ])
    
    # Synthesize into cohesive summary
    return await synthesize(
        summaries,
        instruction="Create a comprehensive summary that captures all key points"
    )
```

## Advanced Patterns

### Self-Consistency with Reasoning

```python
@ember.op
async def self_consistent_reasoning(problem: str, num_attempts: int = 5) -> dict:
    """Solve a problem using self-consistency with chain-of-thought."""
    
    # Generate multiple reasoning chains
    reasoning_chains = await ember.parallel([
        ember.llm(
            f"Solve step by step: {problem}",
            temperature=0.8
        ) for _ in range(num_attempts)
    ])
    
    # Extract final answers from each chain
    answers = []
    for chain in reasoning_chains:
        # Simple extraction - in practice, use structured output
        answer = await ember.llm(
            f"What is the final answer in this solution: {chain}",
            temperature=0
        )
        answers.append(answer)
    
    # Find most common answer
    from collections import Counter
    answer_counts = Counter(answers)
    best_answer = answer_counts.most_common(1)[0][0]
    confidence = answer_counts[best_answer] / num_attempts
    
    return {
        "answer": best_answer,
        "confidence": confidence,
        "reasoning_chains": reasoning_chains,
        "all_answers": answers
    }
```

### Hierarchical Processing

```python
@ember.op
async def hierarchical_analysis(document: str) -> dict:
    """Analyze document at multiple levels of detail."""
    
    # Level 1: High-level summary
    summary = await ember.llm(
        f"Summarize in one sentence: {document}",
        model="gpt-4-mini"
    )
    
    # Level 2: Key points
    key_points = await ember.llm(
        f"Extract 3-5 key points: {document}",
        output_type=list[str],
        model="gpt-4"
    )
    
    # Level 3: Detailed analysis (only if needed)
    if len(document) > 1000:
        detailed = await ember.llm(
            f"Provide detailed analysis: {document}",
            model="claude-3-opus"
        )
    else:
        detailed = None
    
    return {
        "summary": summary,
        "key_points": key_points,
        "detailed_analysis": detailed
    }
```

### Dynamic Operator Selection

```python
@ember.op
async def smart_processor(task: str, complexity: str = "auto") -> str:
    """Dynamically select processing strategy based on task complexity."""
    
    # Auto-detect complexity if needed
    if complexity == "auto":
        complexity = await ember.llm(
            f"Rate the complexity of this task (simple/medium/complex): {task}",
            model="gpt-4-mini"
        )
    
    # Route to appropriate strategy
    if complexity == "simple":
        # Single fast model
        return await ember.llm(task, model="gpt-4-mini")
    
    elif complexity == "medium":
        # Small ensemble
        results = await ember.parallel([
            ember.llm(task, model="gpt-4"),
            ember.llm(task, model="claude-3-haiku")
        ])
        return majority_vote(results)
    
    else:  # complex
        # Full pipeline with verification
        return await robust_qa_pipeline(task)
```

## Performance Optimization

NON patterns can be optimized with Ember's JIT compilation:

```python
from ember.api import ember, jit

@ember.op
@jit  # Enable JIT compilation
async def optimized_ensemble(query: str, num_models: int = 3) -> str:
    """JIT-optimized ensemble processing."""
    # Ember automatically optimizes parallel execution
    results = await ember.parallel([
        ember.llm(query, model=f"model-{i}")
        for i in range(num_models)
    ])
    
    # Fast majority voting
    return majority_vote(results)
```

## Testing NON Patterns

```python
import pytest

@pytest.mark.asyncio
async def test_ensemble_consistency():
    """Test that ensemble improves consistency."""
    question = "What is 2 + 2?"
    
    # Single model might occasionally fail
    single_results = []
    for _ in range(10):
        result = await ember.llm(question, temperature=1.0)
        single_results.append(result)
    
    # Ensemble should be more consistent
    ensemble_results = []
    for _ in range(10):
        result = await robust_answer(question)
        ensemble_results.append(result)
    
    # Ensemble should have less variance
    assert len(set(ensemble_results)) < len(set(single_results))
```

## Best Practices

1. **Start Simple**: Begin with single operators, add complexity as needed
2. **Measure Impact**: Track whether ensemble/verification improves your metrics
3. **Cost vs Quality**: Balance model costs with quality requirements
4. **Async Everything**: Use async/await for efficient parallel execution
5. **Type Your Outputs**: Use structured outputs for reliable parsing
6. **Cache Results**: Cache expensive operations when possible

## Next Steps

- [Operators Guide](./operators.md) - Building custom operators
- [Performance Guide](./performance.md) - Optimizing NON patterns
- [Advanced Patterns](../advanced/patterns.md) - Complex workflows
- [Examples](../examples/) - Real-world NON implementations