"""Consistency Testing - Ensuring reliable AI outputs.

Learn how to test AI systems for consistency, reproducibility,
and reliability across different conditions and inputs.

Example:
    >>> from ember.api import models, eval
    >>> evaluator = eval.ConsistencyEvaluator()
    >>> results = evaluator.test_consistency(model, test_cases)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def example_basic_consistency():
    """Show basic consistency testing."""
    print("\n=== Basic Consistency Testing ===\n")
    
    print("Testing model consistency:")
    print("  • Same input → Same output?")
    print("  • Similar inputs → Similar outputs?")
    print("  • Deterministic behavior?")
    print("  • Stable across time?\n")
    
    # Simulate consistency test
    test_prompt = "What is the capital of France?"
    runs = 5
    
    print(f"Test: Run same prompt {runs} times")
    print(f'Prompt: "{test_prompt}"\n')
    
    print("Results:")
    responses = ["Paris", "Paris", "Paris", "The capital of France is Paris", "Paris"]
    for i, response in enumerate(responses, 1):
        print(f"  Run {i}: {response}")
    
    print("\nConsistency score: 4/5 (80%)")
    print("Issue: Response format varies")


def example_semantic_consistency():
    """Demonstrate semantic consistency testing."""
    print("\n\n=== Semantic Consistency ===\n")
    
    print("Testing semantic equivalence:\n")
    
    test_cases = [
        ("What is 2+2?", ["4", "Four", "The answer is 4", "2+2 equals 4"]),
        ("Who wrote Romeo and Juliet?", ["Shakespeare", "William Shakespeare", "It was written by Shakespeare"]),
        ("Is water wet?", ["Yes", "Yes, water is wet", "Water is indeed wet"])
    ]
    
    for question, responses in test_cases:
        print(f'Question: "{question}"')
        print("Responses:")
        for resp in responses:
            print(f"  • {resp}")
        print("Semantic consistency: ✓ (All convey same meaning)\n")


def example_input_perturbation():
    """Show input perturbation testing."""
    print("\n\n=== Input Perturbation Testing ===\n")
    
    print("Testing robustness to input variations:\n")
    
    base_prompt = "Explain photosynthesis"
    perturbations = [
        "Explain photosynthesis",
        "explain photosynthesis",  # lowercase
        "Explain photosynthesis.",  # punctuation
        "Explain  photosynthesis",  # extra space
        "Explain photosynthesis\n",  # trailing newline
        "Can you explain photosynthesis?",  # question form
    ]
    
    print("Base prompt and variations:")
    for i, prompt in enumerate(perturbations):
        print(f"  {i+1}. '{prompt}'")
    
    print("\nExpected: Similar responses for all variations")
    print("Testing for:")
    print("  • Case sensitivity")
    print("  • Whitespace handling")
    print("  • Punctuation robustness")
    print("  • Format flexibility")


def example_temporal_consistency():
    """Demonstrate temporal consistency testing."""
    print("\n\n=== Temporal Consistency ===\n")
    
    print("Testing consistency over time:\n")
    
    print("Test scenario: Ask same question daily for a week")
    print('Question: "What are the main programming paradigms?"\n')
    
    # Simulate daily responses
    daily_responses = {
        "Monday": "Object-oriented, functional, procedural",
        "Tuesday": "OOP, functional, procedural programming",
        "Wednesday": "Object-oriented, functional, procedural",
        "Thursday": "Object-oriented, functional, and procedural",
        "Friday": "OOP, FP, and procedural paradigms",
        "Saturday": "Object-oriented, functional, procedural",
        "Sunday": "The main paradigms are OOP, functional, procedural"
    }
    
    print("Daily responses:")
    for day, response in daily_responses.items():
        print(f"  {day}: {response}")
    
    print("\nAnalysis:")
    print("  • Content: Consistent ✓")
    print("  • Format: Variable ⚠")
    print("  • Terminology: Mixed (OOP vs Object-oriented)")


def example_cross_model_consistency():
    """Show cross-model consistency testing."""
    print("\n\n=== Cross-Model Consistency ===\n")
    
    print("Comparing consistency across models:\n")
    
    question = "What is machine learning?"
    
    model_responses = {
        "GPT-3.5": "Machine learning is a subset of AI that enables systems to learn from data",
        "GPT-4": "Machine learning is a field of AI where computers learn patterns from data without explicit programming",
        "Claude": "Machine learning is an AI approach where algorithms improve through experience with data"
    }
    
    print(f'Question: "{question}"\n')
    print("Model responses:")
    for model, response in model_responses.items():
        print(f"  {model}: {response}")
    
    print("\nConsistency analysis:")
    print("  • Core concept: Consistent ✓")
    print("  • Key terms: AI, data, learn ✓")
    print("  • Detail level: Varies")
    print("  • Technical accuracy: Consistent ✓")


def example_consistency_metrics():
    """Demonstrate consistency metrics."""
    print("\n\n=== Consistency Metrics ===\n")
    
    print("Measuring consistency:\n")
    
    print("1. Exact Match Rate:")
    print("   Identical responses / Total runs")
    print("   Example: 8/10 = 80%\n")
    
    print("2. Semantic Similarity:")
    print("   Average cosine similarity of embeddings")
    print("   Example: 0.95 (very similar)\n")
    
    print("3. Key Information Retention:")
    print("   Critical facts preserved / Total facts")
    print("   Example: 15/15 = 100%\n")
    
    print("4. Format Consistency:")
    print("   Responses with same structure / Total")
    print("   Example: 6/10 = 60%\n")
    
    print("5. Confidence Variance:")
    print("   Standard deviation of confidence scores")
    print("   Example: σ = 0.05 (low variance)")


def example_edge_case_consistency():
    """Show edge case consistency testing."""
    print("\n\n=== Edge Case Consistency ===\n")
    
    print("Testing consistency on edge cases:\n")
    
    edge_cases = [
        ("Empty input", "''", "Should handle gracefully"),
        ("Very long input", "[10,000 chars]", "Should truncate/summarize"),
        ("Special characters", "🎉 ñ © π", "Should process correctly"),
        ("Mixed languages", "Hello 你好 Bonjour", "Should handle appropriately"),
        ("Contradictory request", "Be brief but explain in detail", "Should clarify or choose"),
        ("Nonsense input", "Colorless green ideas sleep", "Should indicate confusion"),
    ]
    
    print("Edge case tests:")
    for name, input_desc, expected in edge_cases:
        print(f"  {name}:")
        print(f"    Input: {input_desc}")
        print(f"    Expected: {expected}")
    
    print("\nConsistency requirements:")
    print("  • Never crash or error out")
    print("  • Provide meaningful response")
    print("  • Maintain safety guidelines")
    print("  • Be predictable in handling")


def example_consistency_test_suite():
    """Show complete consistency test suite."""
    print("\n\n=== Consistency Test Suite ===\n")
    
    print("Comprehensive testing framework:\n")
    
    print("1. Setup Phase:")
    print("   • Define test prompts")
    print("   • Set consistency thresholds")
    print("   • Configure test parameters\n")
    
    print("2. Execution Phase:")
    print("   For each test case:")
    print("     • Run N times with same parameters")
    print("     • Run with parameter variations")
    print("     • Run at different times")
    print("     • Run on different models\n")
    
    print("3. Analysis Phase:")
    print("   • Calculate consistency scores")
    print("   • Identify outliers")
    print("   • Group similar responses")
    print("   • Generate report\n")
    
    print("4. Sample Report:")
    print("   ┌─────────────────────────────────┐")
    print("   │ Consistency Test Results        │")
    print("   ├─────────────────────────────────┤")
    print("   │ Total test cases: 50            │")
    print("   │ Exact match rate: 76%           │")
    print("   │ Semantic similarity: 0.92       │")
    print("   │ Format consistency: 84%         │")
    print("   │ Edge case handling: PASS        │")
    print("   │ Temporal stability: 91%         │")
    print("   └─────────────────────────────────┘")


def example_automated_monitoring():
    """Show automated consistency monitoring."""
    print("\n\n=== Automated Consistency Monitoring ===\n")
    
    print("Continuous consistency monitoring:\n")
    
    print("1. Real-time tracking:")
    print("   • Log all model inputs/outputs")
    print("   • Calculate rolling consistency metrics")
    print("   • Alert on anomalies\n")
    
    print("2. A/B testing:")
    print("   • Compare model versions")
    print("   • Track consistency changes")
    print("   • Measure drift over time\n")
    
    print("3. Dashboard example:")
    print("   Consistency Metrics (Last 24h)")
    print("   ├─ Response similarity: 94.2% ↑")
    print("   ├─ Format consistency: 87.5% →")
    print("   ├─ Semantic drift: 0.03 ↓")
    print("   └─ Error rate: 0.1% →")


def main():
    """Run all consistency testing examples."""
    print_section_header("Consistency Testing")
    
    print("🎯 Why Consistency Testing Matters:\n")
    print("• Builds user trust")
    print("• Ensures reliability")
    print("• Identifies model issues")
    print("• Validates deployments")
    print("• Maintains quality standards")
    
    example_basic_consistency()
    example_semantic_consistency()
    example_input_perturbation()
    example_temporal_consistency()
    example_cross_model_consistency()
    example_consistency_metrics()
    example_edge_case_consistency()
    example_consistency_test_suite()
    example_automated_monitoring()
    
    print("\n" + "="*50)
    print("✅ Consistency Testing Best Practices")
    print("="*50)
    print("\n1. Define clear consistency criteria")
    print("2. Test multiple dimensions (semantic, format, etc.)")
    print("3. Include edge cases and adversarial inputs")
    print("4. Monitor consistency over time")
    print("5. Set appropriate thresholds for your use case")
    print("6. Automate testing in CI/CD pipeline")
    print("7. Document expected variations")
    
    print("\n🔧 Testing Strategies:")
    print("• Use fixed random seeds when possible")
    print("• Test with temperature=0 for determinism")
    print("• Create comprehensive test suites")
    print("• Version your test cases")
    print("• Track metrics over time")
    
    print("\nNext: See 'benchmark_harness.py' for performance testing")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())