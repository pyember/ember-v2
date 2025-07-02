"""
Example demonstrating the simplified import structure.

This example shows how to use the new top-level imports for operators and model components.
"""

from ember.api import models, operators

# Create an ensemble with multiple models
@operators.op
def ensemble_inference(query: str) -> str:
    """Simple ensemble that gets responses from multiple models and returns the first one."""
    # In practice, you might want to aggregate or vote on the results
    responses = []
    
    # Use different models for diversity
    model_configs = [
        ("gpt-3.5-turbo", {"temperature": 0.7}),
        ("gpt-3.5-turbo", {"temperature": 0.9}),
        ("gpt-3.5-turbo", {"temperature": 0.3}),
    ]
    
    for model_name, params in model_configs:
        try:
            response = models(model_name, query, **params)
            responses.append(response.text)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    # Simple aggregation - return the first successful response
    return responses[0] if responses else "No responses available"

# Create a judge to synthesize outputs
@operators.op  
def judge_synthesis(query: str, responses: list) -> str:
    """Judge that synthesizes multiple responses into a final answer."""
    if not responses:
        return "No responses to synthesize"
    
    synthesis_prompt = f"""
Given the original query: "{query}"

And these responses:
{chr(10).join([f"{i+1}. {resp}" for i, resp in enumerate(responses)])}

Provide a synthesized, high-quality answer that combines the best aspects of these responses.
"""
    
    return models("gpt-4", synthesis_prompt).text

# Demo usage
if __name__ == "__main__":
    query = "What is the future of AI?"
    
    print("Testing new operators API...")
    print(f"Query: {query}")
    print()
    
    # Get ensemble response
    result = ensemble_inference(query)
    print("Ensemble result:")
    print(result[:200] + "..." if len(result) > 200 else result)
