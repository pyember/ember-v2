"""Simple Ensemble - Coordinate multiple AI agents with the new API.

Difficulty: Intermediate
Time: ~5 minutes

Learning Objectives:
- Build ensemble systems with simple functions
- Use the ensemble() helper function
- Implement custom voting strategies
- Optimize with @jit and parallel processing

Example:
    >>> from ember.api import operators, models
    >>> 
    >>> # Define expert functions
    >>> def expert1(question): return models("gpt-4", f"Answer: {question}")
    >>> def expert2(question): return models("claude-3", f"Answer: {question}")
    >>> 
    >>> # Create ensemble  
    >>> ensemble_fn = operators.ensemble(expert1, expert2)
    >>> result = ensemble_fn("How to learn programming?")
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import random
import time

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output, timer
from ember.api import operators, models
from ember.api.xcs import jit


def main():
    """Build ensemble systems with Ember's simple API."""
    print_section_header("Simple Ensemble System")
    
    # Part 1: Create Expert Functions
    print("Part 1: Expert Functions (No Classes!)")
    print("=" * 50 + "\n")
    
    # Simple expert functions - just Python!
    def detailed_expert(question: str) -> dict:
        """Expert that gives detailed answers."""
        # Simulate expert response
        answer = f"Let me provide a comprehensive analysis of '{question}'..."
        confidence = 0.85 + random.uniform(-0.1, 0.1)
        
        return {
            "expert": "Dr. Detail",
            "answer": answer,
            "confidence": min(confidence, 1.0),
            "style": "detailed"
        }
    
    def concise_expert(question: str) -> dict:
        """Expert that gives concise answers."""
        answer = f"Short answer: It depends on the context of '{question}'."
        confidence = 0.90 + random.uniform(-0.1, 0.1)
        
        return {
            "expert": "Prof. Precise",
            "answer": answer,
            "confidence": min(confidence, 1.0),
            "style": "concise"
        }
    
    def practical_expert(question: str) -> dict:
        """Expert that gives practical answers."""
        answer = f"In practice, '{question}' usually means..."
        confidence = 0.80 + random.uniform(-0.1, 0.1)
        
        return {
            "expert": "Practical Pat",
            "answer": answer,
            "confidence": min(confidence, 1.0),
            "style": "practical"
        }
    
    # Test individual experts
    question = "What is the best way to learn programming?"
    
    print("Individual Expert Responses:")
    experts = [detailed_expert, concise_expert, practical_expert]
    
    for expert in experts:
        result = expert(question)
        print(f"  {result['expert']}: {result['confidence']:.2f} confidence")
    
    # Part 2: Manual Ensemble
    print("\n" + "=" * 50)
    print("Part 2: Manual Ensemble Pattern")
    print("=" * 50 + "\n")
    
    def manual_ensemble(question: str) -> dict:
        """Simple ensemble that consults all experts."""
        # Get all expert opinions
        results = []
        for expert in experts:
            results.append(expert(question))
        
        # Simple voting: highest confidence wins
        best = max(results, key=lambda x: x["confidence"])
        
        # Calculate consensus metrics
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        
        return {
            "question": question,
            "answer": best["answer"],
            "selected_expert": best["expert"],
            "confidence": best["confidence"],
            "avg_confidence": avg_confidence,
            "num_experts": len(results)
        }
    
    with timer("Manual ensemble"):
        result = manual_ensemble(question)
    
    print("Manual Ensemble Result:")
    print_example_output("Selected Expert", result["selected_expert"])
    print_example_output("Confidence", f"{result['confidence']:.2%}")
    print_example_output("Avg Confidence", f"{result['avg_confidence']:.2%}")
    
    # Part 3: Using operators.ensemble()
    print("\n" + "=" * 50)
    print("Part 3: Using operators.ensemble()")
    print("=" * 50 + "\n")
    
    # Create ensemble with the helper function
    ensemble_fn = operators.ensemble(*experts)
    
    with timer("Built-in ensemble"):
        # Returns list of results
        results = ensemble_fn(question)
    
    print("Built-in Ensemble Results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['expert']}: {result['confidence']:.2f}")
    
    # Part 4: Custom Aggregation
    print("\n" + "=" * 50)
    print("Part 4: Custom Aggregation Strategies")
    print("=" * 50 + "\n")
    
    def weighted_vote(results: List[dict]) -> dict:
        """Aggregate results using weighted voting."""
        if not results:
            return {"answer": "No experts available", "confidence": 0.0}
        
        # Weight answers by confidence
        total_weight = sum(r["confidence"] for r in results)
        
        # For demo, just return highest confidence
        best = max(results, key=lambda x: x["confidence"])
        
        # Calculate agreement score
        confidences = [r["confidence"] for r in results]
        variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
        agreement = 1.0 - min(variance * 10, 1.0)  # Scale variance to 0-1
        
        return {
            "answer": best["answer"],
            "expert": best["expert"],
            "confidence": best["confidence"],
            "agreement": agreement,
            "method": "weighted_vote"
        }
    
    # Create ensemble with custom aggregator
    def ensemble_with_voting(question: str) -> dict:
        """Ensemble that uses custom voting."""
        # Get all results
        results = ensemble_fn(question)
        
        # Apply custom aggregation
        return weighted_vote(results)
    
    result = ensemble_with_voting(question)
    print("Custom Aggregation Result:")
    print_example_output("Method", result["method"])
    print_example_output("Expert", result["expert"])
    print_example_output("Agreement", f"{result['agreement']:.2%}")
    
    # Part 5: Optimized Ensemble with JIT
    print("\n" + "=" * 50)
    print("Part 5: Optimized Ensemble with @jit")
    print("=" * 50 + "\n")
    
    # JIT-compile individual experts
    fast_detailed = jit(detailed_expert)
    fast_concise = jit(concise_expert)
    fast_practical = jit(practical_expert)
    
    @jit
    def fast_ensemble(question: str) -> dict:
        """JIT-optimized ensemble."""
        # Consult all experts
        results = [
            fast_detailed(question),
            fast_concise(question),
            fast_practical(question)
        ]
        
        # Find best result
        best = max(results, key=lambda x: x["confidence"])
        
        return {
            "answer": best["answer"],
            "expert": best["expert"],
            "confidence": best["confidence"]
        }
    
    # Compare performance
    print("Performance Comparison:")
    
    # Regular ensemble
    start = time.time()
    for _ in range(10):
        manual_ensemble(question)
    regular_time = time.time() - start
    
    # JIT ensemble
    start = time.time()
    for _ in range(10):
        fast_ensemble(question)
    jit_time = time.time() - start
    
    print_example_output("Regular ensemble (10 calls)", f"{regular_time:.4f}s")
    print_example_output("JIT ensemble (10 calls)", f"{jit_time:.4f}s")
    print_example_output("Speedup", f"{regular_time/jit_time:.1f}x")
    
    # Part 6: Batch Processing Multiple Questions
    print("\n" + "=" * 50)
    print("Part 6: Batch Processing with vmap")
    print("=" * 50 + "\n")
    
    questions = [
        "What is the best way to learn programming?",
        "How do I debug complex systems?",
        "What makes a good software architect?",
        "How to design scalable systems?",
        "What are best practices for code review?"
    ]
    
    # Batch process with vmap
    # Process in batch using list comprehension
    with timer("Batch processing 5 questions"):
        batch_results = [fast_ensemble(q) for q in questions]
    
    print("\nBatch Results:")
    for i, (q, r) in enumerate(zip(questions, batch_results)):
        print(f"{i+1}. Q: {q[:40]}...")
        print(f"   A: {r['expert']} ({r['confidence']:.2%})")
    
    # Part 7: Real-World Pattern with Models
    print("\n" + "=" * 50)
    print("Part 7: Real-World Pattern")
    print("=" * 50 + "\n")
    
    print("In practice, you would use real models:")
    print("\n```python")
    print("def gpt4_expert(question: str) -> dict:")
    print("    response = models('gpt-4', f'Answer concisely: {question}')")
    print("    return {'answer': response.text, 'model': 'gpt-4'}")
    print("")
    print("def claude_expert(question: str) -> dict:")
    print("    response = models('claude-3', f'Answer concisely: {question}')")
    print("    return {'answer': response.text, 'model': 'claude-3'}")
    print("")
    print("# Create multi-model ensemble")
    print("ensemble = operators.ensemble(gpt4_expert, claude_expert)")
    print("```")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Key Takeaways")
    print("=" * 50)
    
    print("\n1. Ensembles are just functions consulting other functions")
    print("2. No complex base classes or specifications needed")
    print("3. Use operators.ensemble() for parallel execution")
    print("4. Add custom aggregation as simple functions")
    print("5. Optimize with @jit for repeated calls")
    print("6. Process batches with list comprehensions")
    print("7. Same patterns work with real models")
    
    print("\nNext: Explore judge synthesis patterns!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())