# Network of Operators (NON) Implementation in Three-Tier Architecture

## Overview

The NON pattern provides high-level abstractions for common LLM application patterns:
- **Ensemble**: Multiple model responses for robustness
- **Aggregation**: Statistical (MostCommon) or reasoned (JudgeSynthesis)
- **Verification**: Quality control with correction
- **Composition**: Sequential pipelines and graphs

## Implementation Strategy

### Tier 1: Simple Functions (90% of users)

```python
# src/ember/operators/non.py - Simple functional implementations

from typing import List, Dict, Any, Callable
from ember.api import models
from ember.operators import measure, parallel
from collections import Counter


@measure
def ensemble(prompt: str, model: str = "gpt-4", n: int = 3, temperature: float = 0.7) -> List[str]:
    """Generate multiple responses using the same model configuration.
    
    Args:
        prompt: Input prompt
        model: Model identifier
        n: Number of responses to generate
        temperature: Sampling temperature
        
    Returns:
        List of model responses
    """
    # Create n instances of the same call
    calls = [lambda p=prompt: models(model, p, temperature=temperature) for _ in range(n)]
    
    # Execute in parallel
    return parallel(*calls)(prompt)


@measure
def most_common(responses: List[str]) -> str:
    """Select the most frequently occurring response.
    
    Args:
        responses: List of responses to aggregate
        
    Returns:
        Most common response (first in case of tie)
    """
    if not responses:
        raise ValueError("No responses to aggregate")
    
    counter = Counter(responses)
    return counter.most_common(1)[0][0]


@measure
def judge_synthesis(prompt: str, responses: List[str], judge_model: str = "gpt-4") -> Dict[str, str]:
    """Synthesize multiple responses using a judge model.
    
    Args:
        prompt: Original prompt
        responses: List of responses to synthesize
        judge_model: Model to use for synthesis
        
    Returns:
        Dict with 'synthesis' and 'reasoning' keys
    """
    synthesis_prompt = f"""Original question: {prompt}

Candidate responses:
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(responses))}

Synthesize these responses into a single, high-quality answer. Explain your reasoning.

Format your response as:
REASONING: <your analysis>
SYNTHESIS: <final synthesized answer>"""

    result = models(judge_model, synthesis_prompt, temperature=0.3)
    
    # Simple parsing
    parts = result.split("SYNTHESIS:")
    reasoning = parts[0].replace("REASONING:", "").strip() if len(parts) > 1 else ""
    synthesis = parts[1].strip() if len(parts) > 1 else result
    
    return {"synthesis": synthesis, "reasoning": reasoning}


@measure  
def verify(prompt: str, answer: str, verifier_model: str = "gpt-4") -> Dict[str, Any]:
    """Verify an answer and provide corrections if needed.
    
    Args:
        prompt: Original prompt
        answer: Answer to verify
        verifier_model: Model to use for verification
        
    Returns:
        Dict with 'verdict', 'explanation', and optional 'revised_answer'
    """
    verify_prompt = f"""Question: {prompt}
Answer: {answer}

Verify if this answer is correct. If incorrect, provide a revised answer.

Format your response as:
VERDICT: correct/incorrect
EXPLANATION: <detailed explanation>
REVISED: <revised answer if incorrect, or 'N/A' if correct>"""

    result = models(verifier_model, verify_prompt, temperature=0.1)
    
    # Parse response
    lines = result.split('\n')
    verdict = "unknown"
    explanation = ""
    revised = None
    
    for line in lines:
        if line.startswith("VERDICT:"):
            verdict = "correct" if "correct" in line.lower() and "incorrect" not in line.lower() else "incorrect"
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("REVISED:"):
            revised_text = line.replace("REVISED:", "").strip()
            if revised_text.lower() != "n/a":
                revised = revised_text
    
    return {
        "verdict": verdict,
        "explanation": explanation,
        "revised_answer": revised
    }


# Convenience pipeline functions
def ensemble_most_common(prompt: str, model: str = "gpt-4", n: int = 3) -> str:
    """Generate multiple responses and return the most common."""
    responses = ensemble(prompt, model, n)
    return most_common(responses)


def ensemble_judge(prompt: str, model: str = "gpt-4", judge: str = "gpt-4", n: int = 3) -> str:
    """Generate multiple responses and synthesize with a judge."""
    responses = ensemble(prompt, model, n)
    result = judge_synthesis(prompt, responses, judge)
    return result["synthesis"]


def ensemble_verify(prompt: str, model: str = "gpt-4", verifier: str = "gpt-4", n: int = 3) -> str:
    """Generate response, then verify and correct if needed."""
    # Get initial answer (could use ensemble_judge)
    initial = models(model, prompt)
    
    # Verify it
    verification = verify(prompt, initial, verifier)
    
    # Return revised if needed, otherwise original
    return verification.get("revised_answer", initial)
```

### Tier 2: Advanced Classes (9% of power users)

```python
# src/ember/operators/advanced/non.py - Advanced NON implementations

from typing import List, Dict, Any, Optional, Protocol, Generic, TypeVar
from dataclasses import dataclass
from ember.operators.advanced import Operator, TreeProtocol, operator
from ember.operators import measure
from ember.api import models

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class EnsembleInputs:
    """Typed inputs for ensemble operations."""
    query: str
    

@dataclass
class EnsembleOutputs:
    """Typed outputs from ensemble operations."""
    responses: List[str]


@dataclass
class VerifierInputs:
    """Typed inputs for verification."""
    query: str
    candidate_answer: str


@dataclass
class VerifierOutputs:
    """Typed outputs from verification."""
    verdict: str
    explanation: str
    revised_answer: Optional[str] = None


@operator.advanced
class EnsembleOperator(Operator, TreeProtocol):
    """Advanced ensemble operator with full control.
    
    Supports:
    - Multiple model configurations
    - Custom prompt templates
    - Tree protocol for XCS transformations
    - Type-safe inputs/outputs
    """
    
    models: List[Any]  # Model instances or configurations
    prompt_template: Optional[str] = None
    
    def __call__(self, inputs: EnsembleInputs) -> EnsembleOutputs:
        """Execute ensemble operation."""
        prompt = self._build_prompt(inputs)
        
        # Execute all models (could be parallelized)
        responses = []
        for model in self.models:
            if callable(model):
                response = model(prompt)
            else:
                # Assume it's a config dict
                response = models(model['name'], prompt, **model.get('params', {}))
            responses.append(response)
        
        return EnsembleOutputs(responses=responses)
    
    def _build_prompt(self, inputs: EnsembleInputs) -> str:
        """Build prompt from template or use query directly."""
        if self.prompt_template:
            return self.prompt_template.format(query=inputs.query)
        return inputs.query
    
    def tree_flatten(self):
        """Support JAX transformations."""
        return self.models, {"prompt_template": self.prompt_template}
    
    @classmethod
    def tree_unflatten(cls, aux_data, models):
        """Reconstruct from tree representation."""
        return cls(models=models, **aux_data)


@operator.advanced
@operator.hints(stateless=True, cacheable=True)
class MostCommonOperator(Operator):
    """Statistical aggregation with advanced features."""
    
    def __call__(self, responses: List[str]) -> Dict[str, Any]:
        """Select most common with statistics."""
        from collections import Counter
        
        if not responses:
            raise ValueError("No responses to aggregate")
        
        counter = Counter(responses)
        most_common = counter.most_common()
        
        # Return rich information
        return {
            "selected": most_common[0][0],
            "count": most_common[0][1],
            "total": len(responses),
            "confidence": most_common[0][1] / len(responses),
            "distribution": dict(counter)
        }


@operator.advanced
class JudgeSynthesisOperator(Operator, TreeProtocol):
    """Advanced synthesis with structured outputs."""
    
    model: Any
    output_format: str = "json"  # or "text"
    
    def __call__(self, query: str, responses: List[str]) -> Dict[str, Any]:
        """Synthesize responses with structured output."""
        prompt = self._build_synthesis_prompt(query, responses)
        
        if self.output_format == "json":
            # Use structured output
            result = self.model(prompt, response_format="json")
            return result  # Assume model returns parsed JSON
        else:
            # Text parsing fallback
            result = self.model(prompt)
            return self._parse_text_response(result)
    
    def _build_synthesis_prompt(self, query: str, responses: List[str]) -> str:
        """Build synthesis prompt."""
        return f"""Synthesize these responses to: {query}

Responses:
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(responses))}

Provide a synthesized answer with reasoning."""
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse text response into structured format."""
        # Implementation similar to simple version
        pass
    
    def tree_flatten(self):
        """JAX transformation support."""
        return [self.model], {"output_format": self.output_format}
    
    @classmethod
    def tree_unflatten(cls, aux_data, values):
        """Reconstruct from tree."""
        return cls(model=values[0], **aux_data)


# Composition utilities
@operator.advanced
class Sequential(Operator, Generic[T, U]):
    """Type-safe sequential composition."""
    
    operators: List[Operator]
    
    def __call__(self, inputs: T) -> U:
        """Execute operators in sequence."""
        current = inputs
        for op in self.operators:
            current = op(current)
        return current
    
    def tree_flatten(self):
        """Support transformations on the pipeline."""
        return self.operators, {}
    
    @classmethod
    def tree_unflatten(cls, aux_data, operators):
        """Reconstruct pipeline."""
        return cls(operators=operators)
```

### Tier 3: Experimental Graph Building (1% of users)

```python
# src/ember/operators/experimental/graph.py - Advanced graph composition

from typing import Any, Dict, List, Union
from ember.operators.experimental import trace, jit_compile, pattern_optimize


@trace
@pattern_optimize(patterns=["ensemble", "aggregation"])
def build_non_graph(spec: Union[str, List, Dict]) -> Any:
    """Build optimized NON graphs from specifications.
    
    Supports compact notation like:
    - "3:E:gpt-4:0.7 -> MC" (3 ensemble -> most common)
    - ["3:E:gpt-4:0.7", "1:J:claude:0.3"] (ensemble -> judge)
    
    The graph is automatically optimized for parallel execution.
    """
    # This would use IR tracing to build optimized graphs
    pass


@jit_compile
def optimized_ensemble_pipeline(queries: List[str]) -> List[str]:
    """JIT-compiled ensemble pipeline with automatic optimization.
    
    This demonstrates how experimental features can dramatically
    speed up repeated ensemble operations.
    """
    results = []
    
    for query in queries:
        # Pattern detected: parallel ensemble
        responses = [
            models("gpt-4", query, temperature=0.7)
            for _ in range(3)
        ]
        
        # Pattern detected: aggregation after ensemble
        synthesized = judge_synthesis(query, responses, "claude-3-opus")
        
        results.append(synthesized["synthesis"])
    
    return results
```

## Usage Examples

### Simple (Tier 1)
```python
from ember.operators.non import ensemble_judge, verify

# One-liner for ensemble + judge
answer = ensemble_judge("What causes climate change?", n=5)

# Add verification
verified = verify("What causes climate change?", answer)
if verified["verdict"] == "incorrect":
    answer = verified["revised_answer"]
```

### Advanced (Tier 2)
```python
from ember.operators.advanced.non import (
    EnsembleOperator, JudgeSynthesisOperator, Sequential
)

# Configure ensemble with different models
ensemble = EnsembleOperator(models=[
    {"name": "gpt-4", "params": {"temperature": 0.7}},
    {"name": "claude-3", "params": {"temperature": 0.5}},
    {"name": "gemini-pro", "params": {"temperature": 0.6}}
])

# Judge with structured output
judge = JudgeSynthesisOperator(
    model=models.instance("claude-3-opus"),
    output_format="json"
)

# Compose pipeline
pipeline = Sequential(operators=[ensemble, judge])
```

### Experimental (Tier 3)
```python
from ember.operators.experimental.graph import build_non_graph

# Compact graph notation
graph = build_non_graph([
    "3:E:gpt-4:0.7",      # 3-unit ensemble
    "1:J:claude-3:0.3",    # Judge synthesis
    "1:V:gpt-4:0.1"        # Verification
])

# Automatically optimized execution
result = graph("Complex question requiring verification")
```

## Migration from Current NON

Current NON users can:

1. **Continue using existing code** - It still works
2. **Gradually adopt simple functions** - Same functionality, less code
3. **Use advanced tier when needed** - For type safety and tree protocols
4. **Experiment with graph optimization** - For performance gains

## Key Design Principles

1. **Progressive Disclosure**: Start simple, add complexity when needed
2. **Measurement Built-in**: All NON operations tracked automatically  
3. **Type Safety Optional**: Use typed classes only when beneficial
4. **Performance by Default**: Parallel execution where sensible
5. **Clean Separation**: Each tier is independent

This design gives us the best of both worlds:
- Simple, functional API for most users
- Full power of the original system when needed
- Future-looking experimental features
- Smooth migration path