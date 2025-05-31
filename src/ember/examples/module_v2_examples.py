"""Examples demonstrating the simplified module system.

Shows patterns for building stateful operators with automatic immutability
and transformation support.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from ember.core.module_v2_simple import EmberModule, static_field, SignatureOperator
from ember.api import models
from ember.xcs import jit, vmap


# Example 1: Simple operator with state
class PromptedClassifier(EmberModule):
    """A classifier with a customizable prompt template."""
    model: Any  # Model binding
    prompt_template: str
    categories: List[str]
    
    def __call__(self, text: str) -> str:
        prompt = self.prompt_template.format(
            text=text,
            categories=", ".join(self.categories)
        )
        response = self.model(prompt)
        return response.text.strip()


# Example 2: Operator with typed inputs/outputs (DSPy-style signatures)
@dataclass
class ClassificationInput:
    text: str
    context: Optional[str] = None
    
@dataclass
class ClassificationOutput:
    category: str
    confidence: float
    reasoning: str


class StructuredClassifier(SignatureOperator):
    """Classifier that works with typed inputs and outputs."""
    model: Any
    categories: List[str]
    include_reasoning: bool = True
    
    def __call__(self, input: ClassificationInput) -> ClassificationOutput:
        # Build prompt from structured input
        prompt = f"Categories: {', '.join(self.categories)}\n\n"
        
        if input.context:
            prompt += f"Context: {input.context}\n\n"
            
        prompt += f"Text to classify: {input.text}\n\n"
        
        if self.include_reasoning:
            prompt += "Provide your reasoning, then classify.\n"
            prompt += "Format: REASONING: ...\nCATEGORY: ...\nCONFIDENCE: ..."
        else:
            prompt += "Format: CATEGORY: ...\nCONFIDENCE: ..."
        
        response = self.model(prompt).text
        
        # Parse structured output
        category = "unknown"
        confidence = 0.5
        reasoning = ""
        
        for line in response.split('\n'):
            if line.startswith("CATEGORY:"):
                category = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
        
        return ClassificationOutput(
            category=category,
            confidence=confidence,
            reasoning=reasoning
        )


# Example 3: Composed operators
class EnsembleClassifier(EmberModule):
    """Ensemble multiple classifiers for robustness."""
    classifiers: tuple  # Tuple of classifier operators
    aggregation: str = "majority"  # "majority" or "weighted"
    
    def __call__(self, input: Any) -> Any:
        # Get predictions from all classifiers
        predictions = [clf(input) for clf in self.classifiers]
        
        if self.aggregation == "majority":
            # For structured outputs, extract the category
            if hasattr(predictions[0], 'category'):
                categories = [p.category for p in predictions]
                from collections import Counter
                most_common = Counter(categories).most_common(1)[0][0]
                
                # Return structured output with averaged confidence
                avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
                combined_reasoning = " | ".join(p.reasoning for p in predictions if p.reasoning)
                
                return ClassificationOutput(
                    category=most_common,
                    confidence=avg_confidence,
                    reasoning=f"Ensemble decision: {combined_reasoning}"
                )
            else:
                # Simple string outputs
                from collections import Counter
                return Counter(predictions).most_common(1)[0][0]
        
        # Add more aggregation strategies as needed
        return predictions[0]


# Example 4: Operator with static configuration
class AdaptiveClassifier(EmberModule):
    """Classifier that adapts its behavior based on confidence."""
    primary_model: Any
    fallback_model: Any
    confidence_threshold: float = 0.7
    metrics: Dict[str, int] = static_field(default_factory=lambda: {"primary": 0, "fallback": 0})
    
    def __call__(self, input: ClassificationInput) -> ClassificationOutput:
        # Try primary model first
        primary_result = StructuredClassifier(
            model=self.primary_model,
            categories=["positive", "negative", "neutral"],
            include_reasoning=False
        )(input)
        
        # Check confidence
        if primary_result.confidence >= self.confidence_threshold:
            # Note: We can't mutate metrics directly (frozen dataclass)
            # In practice, you'd handle metrics separately
            return primary_result
        
        # Fall back to more powerful model
        fallback_result = StructuredClassifier(
            model=self.fallback_model,
            categories=["positive", "negative", "neutral"],
            include_reasoning=True
        )(input)
        
        return fallback_result


# Example 5: Using with transformations
def example_usage():
    """Show how these modules work with XCS transformations."""
    
    # Create models
    gpt4 = models.instance("gpt-4", temperature=0.3)
    gpt35 = models.instance("gpt-3.5-turbo", temperature=0.3)
    
    # Create classifier
    classifier = StructuredClassifier(
        model=gpt4,
        categories=["technology", "sports", "politics", "entertainment", "other"]
    )
    
    # Single classification
    input1 = ClassificationInput(
        text="Apple announces new iPhone with AI features",
        context="This was announced at their annual developer conference"
    )
    result = classifier(input1)
    print(f"Category: {result.category} (confidence: {result.confidence})")
    print(f"Reasoning: {result.reasoning}")
    
    # Batch classification with vmap
    batch_classifier = vmap(classifier)
    inputs = [
        ClassificationInput("The Lakers won the championship"),
        ClassificationInput("Congress passed the new climate bill"),
        ClassificationInput("Taylor Swift announces new album")
    ]
    results = batch_classifier(inputs)
    
    # JIT compilation for performance
    fast_classifier = jit(classifier)
    
    # Ensemble for robustness
    ensemble = EnsembleClassifier(
        classifiers=(
            StructuredClassifier(model=gpt4, categories=["tech", "non-tech"]),
            StructuredClassifier(model=gpt35, categories=["tech", "non-tech"]),
        )
    )
    
    # The ensemble is also a module, so transformations work
    fast_ensemble = jit(ensemble)
    
    # Update module attributes (creates new instance)
    classifier_v2 = classifier.replace(include_reasoning=False)
    
    return results


if __name__ == "__main__":
    # Run examples
    results = example_usage()
    print(f"\nBatch results: {len(results)} classifications completed")