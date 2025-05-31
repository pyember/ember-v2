"""Simple Ensemble - Coordinate multiple AI agents.

Build an ensemble system that consults multiple experts in parallel
and uses voting to reach consensus.

Example:
    >>> ensemble = EnsemblePipeline(experts=[expert1, expert2, expert3])
    >>> result = ensemble(question="How to learn programming?")
    >>> print(f"{result['selected_expert']}: {result['consensus']}")
"""

import sys
from pathlib import Path
from typing import List
import random
import concurrent.futures

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output, timer
from ember.api import operators


def main():
    """Build your first ensemble system."""
    print_section_header("Simple Ensemble System")
    
    # Part 1: Create Expert Operators
    print("🎯 Building an Ensemble System\n")
    print("We'll create multiple 'experts' that analyze questions.\n")
    
    class ExpertOperator(operators.Operator):
        """Simulates an expert providing an answer."""
        
        specification = operators.Specification()
        
        def __init__(self, *, name: str, style: str = "neutral"):
            self.name = name
            self.style = style
        
        def forward(self, *, inputs):
            question = inputs.get("question", "")
            
            # Simulate different expert responses based on style
            if self.style == "detailed":
                answer = f"Let me provide a comprehensive analysis of '{question}'..."
                confidence = 0.85
            elif self.style == "concise":
                answer = f"Short answer: It depends on the context of '{question}'."
                confidence = 0.90
            elif self.style == "academic":
                answer = f"Research suggests multiple perspectives on '{question}'..."
                confidence = 0.75
            elif self.style == "practical":
                answer = f"In practice, '{question}' usually means..."
                confidence = 0.80
            else:  # neutral
                answer = f"Regarding '{question}', I would say..."
                confidence = 0.70
            
            # Add some randomness
            confidence *= random.uniform(0.9, 1.1)
            
            return {
                "expert": self.name,
                "answer": answer,
                "confidence": min(confidence, 1.0),
                "style": self.style
            }
    
    # Create diverse experts
    experts = [
        ExpertOperator(name="Dr. Detail", style="detailed"),
        ExpertOperator(name="Prof. Precise", style="concise"),
        ExpertOperator(name="Scholar Sam", style="academic"),
        ExpertOperator(name="Practical Pat", style="practical"),
        ExpertOperator(name="Neutral Nancy", style="neutral")]
    
    # Part 2: Sequential Execution
    print("="*50)
    print("Part 1: Sequential Expert Consultation")
    print("="*50 + "\n")
    
    question = "What is the best way to learn programming?"
    
    with timer("Sequential execution"):
        sequential_results = []
        for expert in experts:
            result = expert(question=question)
            sequential_results.append(result)
            print(f"  {result['expert']}: {result['confidence']:.2f} confidence")
    
    # Part 3: Parallel Execution
    print("\n" + "="*50)
    print("Part 2: Parallel Expert Consultation")
    print("="*50 + "\n")
    
    def consult_expert(expert, question):
        """Helper function for parallel execution."""
        return expert(question=question)
    
    with timer("Parallel execution"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(experts)) as executor:
            # Submit all expert consultations
            futures = [
                executor.submit(consult_expert, expert, question)
                for expert in experts
            ]
            
            # Gather results
            parallel_results = [future.result() for future in futures]
        
        for result in parallel_results:
            print(f"  {result['expert']}: {result['confidence']:.2f} confidence")
    
    # Part 4: Simple Voting System
    print("\n" + "="*50)
    print("Part 3: Building Consensus with Voting")
    print("="*50 + "\n")
    
    class VotingOperator(operators.Operator):
        """Aggregates expert opinions through voting."""
        
        specification = operators.Specification()
        
        def forward(self, *, inputs):
            expert_results = inputs.get("results", [])
            
            if not expert_results:
                return {
                    "consensus": "No expert opinions available",
                    "confidence": 0.0,
                    "method": "none"
                }
            
            # Simple voting: highest confidence wins
            best_expert = max(expert_results, key=lambda x: x["confidence"])
            
            # Calculate agreement level
            avg_confidence = sum(r["confidence"] for r in expert_results) / len(expert_results)
            confidence_variance = sum(
                (r["confidence"] - avg_confidence) ** 2 for r in expert_results
            ) / len(expert_results)
            
            # Low variance means high agreement
            agreement = "high" if confidence_variance < 0.01 else "moderate" if confidence_variance < 0.05 else "low"
            
            return {
                "consensus": best_expert["answer"],
                "selected_expert": best_expert["expert"],
                "confidence": best_expert["confidence"],
                "average_confidence": avg_confidence,
                "agreement_level": agreement,
                "total_experts": len(expert_results)
            }
    
    # Apply voting
    voter = VotingOperator()
    consensus = voter(results=parallel_results)
    
    print("Voting Results:")
    print_example_output("Selected Expert", consensus["selected_expert"])
    print_example_output("Confidence", f"{consensus['confidence']:.2%}")
    print_example_output("Average Confidence", f"{consensus['average_confidence']:.2%}")
    print_example_output("Agreement Level", consensus["agreement_level"])
    
    # Part 5: Complete Ensemble Pipeline
    print("\n" + "="*50)
    print("Part 4: Complete Ensemble Pipeline")
    print("="*50 + "\n")
    
    class EnsemblePipeline(operators.Operator):
        """Complete ensemble system with parallel execution and voting."""
        
        specification = operators.Specification()
        
        def __init__(self, experts: List[operators.Operator]):
            self.experts = experts
            self.voter = VotingOperator()
        
        def forward(self, *, inputs):
            question = inputs.get("question", "")
            
            # Parallel consultation
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.experts)) as executor:
                futures = [
                    executor.submit(lambda e: e(question=question), expert)
                    for expert in self.experts
                ]
                results = [future.result() for future in futures]
            
            # Build consensus
            consensus = self.voter(results=results)
            
            # Return comprehensive result
            return {
                "question": question,
                "consensus": consensus["consensus"],
                "confidence": consensus["confidence"],
                "selected_expert": consensus["selected_expert"],
                "details": {
                    "agreement": consensus["agreement_level"],
                    "expert_count": consensus["total_experts"],
                    "all_results": results
                }
            }
    
    # Create and test the pipeline
    ensemble = EnsemblePipeline(experts)
    
    test_questions = [
        "What is the best way to learn programming?",
        "How do I debug complex systems?",
        "What makes a good software architect?"
    ]
    
    print("Ensemble Pipeline Results:\n")
    for q in test_questions:
        with timer(f"Processing '{q[:30]}...'"):
            result = ensemble(question=q)
        
        print(f"\nQ: {q}")
        print(f"A: {result['consensus']}")
        print(f"   Expert: {result['selected_expert']} ({result['confidence']:.2%} confidence)")
        print(f"   Agreement: {result['details']['agreement']} among {result['details']['expert_count']} experts")
    
    # Part 6: Show the benefits
    print("\n" + "="*50)
    print("✅ Key Takeaways")
    print("="*50)
    print("\n1. Ensemble systems combine multiple operators/models")
    print("2. Parallel execution improves performance")
    print("3. Voting and consensus improve reliability")
    print("4. Operators compose naturally into pipelines")
    print("5. Simple patterns scale to complex systems")
    
    print("\nNext: Learn about data processing in '05_data_processing/'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())