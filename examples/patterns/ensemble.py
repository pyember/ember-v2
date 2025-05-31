"""Ensemble pattern - use multiple models for robustness and quality.

This example shows how to combine multiple models for better results.
You'll learn:
- How to run multiple models in parallel
- How to aggregate results effectively  
- How to balance cost vs quality

Requirements:
- ember
- Models: At least 2 different models

Expected output:
    Question: What are the three laws of robotics?
    
    Individual responses:
    Model 1: [response]
    Model 2: [response]
    Model 3: [response]
    
    Ensemble result (majority vote): [final answer]
    Agreement level: 66.7%
"""

from collections import Counter
from typing import List, Callable
from dataclasses import dataclass

from ember.api import models
from ember.core.module_v2 import EmberModule, Ensemble


@dataclass
class EnsembleResult:
    """Result from ensemble with metadata."""
    answer: str
    confidence: float
    all_responses: List[str]
    agreement: float


# Method 1: Simple function-based ensemble
def ensemble_query(prompt: str, model_names: List[str]) -> EnsembleResult:
    """Query multiple models and aggregate results."""
    # Create model instances
    model_instances = [models.instance(name) for name in model_names]
    
    # Get responses from all models
    responses = []
    for model in model_instances:
        response = model(prompt)
        responses.append(response.text.strip())
    
    # Find most common response
    counter = Counter(responses)
    most_common, count = counter.most_common(1)[0]
    
    # Calculate agreement
    agreement = count / len(responses)
    
    return EnsembleResult(
        answer=most_common,
        confidence=agreement,
        all_responses=responses,
        agreement=agreement
    )


# Method 2: Using EmberModule for stateful ensemble
class SmartEnsemble(EmberModule):
    """Ensemble that weights models by past performance."""
    models: tuple
    weights: tuple = None
    
    def __post_init__(self):
        # Initialize equal weights if not provided
        if self.weights is None:
            n = len(self.models)
            object.__setattr__(self, 'weights', tuple(1.0/n for _ in range(n)))
    
    def __call__(self, prompt: str) -> EnsembleResult:
        # Get responses
        responses = []
        for model in self.models:
            response = model(prompt)
            responses.append(response.text.strip())
        
        # Weight votes by model confidence
        vote_counts = {}
        for response, weight in zip(responses, self.weights):
            vote_counts[response] = vote_counts.get(response, 0) + weight
        
        # Find best answer
        best_answer = max(vote_counts.items(), key=lambda x: x[1])
        
        return EnsembleResult(
            answer=best_answer[0],
            confidence=best_answer[1],
            all_responses=responses,
            agreement=len(set(responses)) / len(responses)
        )


# Method 3: Advanced ensemble with reasoning
def ensemble_with_judge(prompt: str, models_list: List[Callable]) -> str:
    """Ensemble that uses a judge to select the best response."""
    # Get all responses
    ensemble = Ensemble(operators=tuple(models_list))
    responses = ensemble(prompt)
    
    # Use a judge model to select best
    judge = models.instance("gpt-4", temperature=0.1)
    judge_prompt = f"""
    Question: {prompt}
    
    I have {len(responses)} different answers:
    {chr(10).join(f'{i+1}. {r}' for i, r in enumerate(responses))}
    
    Which answer is most accurate and complete? Reply with just the number.
    """
    
    judge_response = judge(judge_prompt)
    try:
        choice = int(judge_response.text.strip()) - 1
        return responses[choice]
    except:
        # Fallback to first response if parsing fails
        return responses[0]


def main():
    # Setup question
    question = "What are the three laws of robotics?"
    
    # Method 1: Simple ensemble
    print(f"Question: {question}\n")
    
    result = ensemble_query(
        question,
        ["gpt-4", "gpt-3.5-turbo", "gpt-4"]  # Can use same model multiple times
    )
    
    print("Individual responses:")
    for i, response in enumerate(result.all_responses):
        print(f"Model {i+1}: {response[:100]}...")
    
    print(f"\nEnsemble result (majority vote): {result.answer[:100]}...")
    print(f"Agreement level: {result.agreement:.1%}\n")
    
    # Method 2: Weighted ensemble
    print("Using weighted ensemble:")
    model1 = models.instance("gpt-4", temperature=0.3)
    model2 = models.instance("gpt-3.5-turbo", temperature=0.5) 
    model3 = models.instance("gpt-4", temperature=0.7)
    
    smart_ensemble = SmartEnsemble(
        models=(model1, model2, model3),
        weights=(0.5, 0.2, 0.3)  # Weight GPT-4 variants higher
    )
    
    weighted_result = smart_ensemble(question)
    print(f"Weighted result: {weighted_result.answer[:100]}...")
    print(f"Confidence: {weighted_result.confidence:.1%}")


if __name__ == "__main__":
    main()


# When to use ensembles:
# 1. Critical decisions that need high reliability
# 2. When different models have complementary strengths
# 3. To reduce variance in outputs
# 4. For self-consistency checking

# Cost optimization tips:
# - Use cheaper models for initial responses
# - Only use expensive models for tie-breaking
# - Cache ensemble results aggressively
# - Consider async execution for better performance

# Next steps:
# - See examples/patterns/self_critique.py for validation
# - Try examples/applications/research_assistant/ for complex ensembles
# - Learn about model-specific strengths in docs/model_comparison.md