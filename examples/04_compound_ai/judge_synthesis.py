"""Judge Synthesis - Building AI systems with judging and synthesis.

Learn how to build compound AI systems that use judges to evaluate
outputs and synthesis to combine multiple perspectives.

Example:
    >>> from ember.api import models
    >>> responses = [model(prompt) for model in ["gpt-3.5", "gpt-4"]]
    >>> judge_response = judge_model("Which response is better?")
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from _shared.conditional_execution import conditional_llm
from ember.api import models


def example_basic_judge(_simulated_mode=False):
    """Show basic judge pattern."""
    print("\n=== Basic Judge Pattern ===\n")

    print("Judge pattern components:")
    print("  1. Multiple candidate responses")
    print("  2. Judge criteria/rubric")
    print("  3. Evaluation and selection")
    print("  4. Final decision\n")

    prompt = "Explain what Python is in one sentence."
    print(f"Generating candidates for: '{prompt}'")

    if _simulated_mode:
        # Simulated execution
        candidates = [
            "Python is a high-level programming language.",
            "Python is a versatile, interpreted language known for simplicity.",
            "Python: Easy syntax, vast libraries, great for beginners!",
        ]

        print("Candidate responses:")
        for i, response in enumerate(candidates, 1):
            print(f"  {i}. {response}")

        print("\nJudge evaluation:")
        print("  Selected: 2")
        print("  Reason: Most comprehensive and balanced explanation")
    else:
        # Real execution
        candidates = []
        for i, params in enumerate(
            [{"temperature": 0.3}, {"temperature": 0.7}, {"temperature": 1.0}], 1
        ):
            response = models("gpt-3.5-turbo", prompt, **params)
            candidates.append(response.text.strip())
            print(f"  {i}. {response.text.strip()}")

        # Use judge to evaluate
        judge_prompt = f"""Evaluate these three explanations of Python and select the best one.
        
Criteria:
- Completeness: Does it cover key aspects?
- Accuracy: Is the information correct?
- Clarity: Is it easy to understand?

Responses:
1. {candidates[0]}
2. {candidates[1]}
3. {candidates[2]}

Return your evaluation as:
Selected: [1, 2, or 3]
Reason: [brief explanation]"""

        judge_response = models("gpt-3.5-turbo", judge_prompt, temperature=0)
        print("\nJudge evaluation:")
        print(f"  {judge_response.text.strip()}")


def example_synthesis_pattern(_simulated_mode=False):
    """Demonstrate synthesis of multiple outputs."""
    print("\n\n=== Synthesis Pattern ===\n")

    print("Synthesis combines multiple perspectives:")
    print()

    topic = "Python programming language"

    if _simulated_mode:
        # Simulation fallback
        perspectives = {
            "Technical": "Python uses dynamic typing and automatic memory management.",
            "Practical": "Python excels at web development, data science, and automation.",
            "Historical": "Python was created by Guido van Rossum in 1991.",
        }

        print(f"Generating perspectives on: '{topic}'")
        print("\nIndividual perspectives:")
        for viewpoint, content in perspectives.items():
            print(f"  {viewpoint}: {content}")

        print("\nSynthesized response:")
        print("  Python, created in 1991, is a dynamically-typed language")
        print("  that excels at web development, data science, and automation")
        print("  through its automatic memory management and vast ecosystem.")
    else:
        # Real API execution
        # Generate different perspectives
        perspectives = {}
        viewpoints = {
            "Technical": f"Explain the technical aspects of {topic} in one sentence.",
            "Practical": f"Explain the practical applications of {topic} in one sentence.",
            "Historical": f"Explain the history of {topic} in one sentence.",
        }

        print(f"Generating perspectives on: '{topic}'")
        print("\nIndividual perspectives:")
        for viewpoint, prompt in viewpoints.items():
            response = models("gpt-3.5-turbo", prompt, temperature=0.7)
            perspectives[viewpoint] = response.text.strip()
            print(f"  {viewpoint}: {perspectives[viewpoint]}")

        # Synthesize perspectives
        synthesis_prompt = f"""Synthesize these three perspectives about Python into a single comprehensive sentence:

Technical: {perspectives['Technical']}
Practical: {perspectives['Practical']}
Historical: {perspectives['Historical']}

Create a unified description that incorporates all three viewpoints naturally."""

        synthesis_response = models("gpt-3.5-turbo", synthesis_prompt, temperature=0.5)
        print("\nSynthesized response:")
        print(f"  {synthesis_response.text.strip()}")


def example_multi_judge_consensus():
    """Show multi-judge consensus pattern."""
    print("\n\n=== Multi-Judge Consensus ===\n")

    print("Using multiple judges for robust evaluation:\n")

    # Simulate multiple judges
    judge_scores = {
        "Accuracy Judge": {"A": 0.8, "B": 0.9, "C": 0.7},
        "Clarity Judge": {"A": 0.9, "B": 0.7, "C": 0.8},
        "Relevance Judge": {"A": 0.7, "B": 0.8, "C": 0.9},
    }

    print("Judge evaluations:")
    for judge, scores in judge_scores.items():
        print(f"  {judge}:")
        for option, score in scores.items():
            print(f"    Option {option}: {score:.1f}")

    # Calculate consensus
    print("\nConsensus calculation:")
    print("  Option A: Avg = 0.80")
    print("  Option B: Avg = 0.80")
    print("  Option C: Avg = 0.80")
    print("  → Tie! Use weighted scoring or additional criteria")


def example_iterative_refinement():
    """Demonstrate iterative refinement with judging."""
    print("\n\n=== Iterative Refinement ===\n")

    print("Improving responses through iteration:\n")

    iterations = [
        ("Initial", "Python is good for programming."),
        ("After Judge 1", "Python is excellent for rapid development and prototyping."),
        (
            "After Judge 2",
            "Python excels at rapid development with its clean syntax, "
            "extensive libraries, and strong community support.",
        ),
        (
            "Final",
            "Python is ideal for rapid development due to its readable syntax, "
            "comprehensive standard library, vast third-party packages, "
            "and supportive community, making it perfect for beginners and experts alike.",
        ),
    ]

    for stage, response in iterations:
        print(f"{stage}:")
        print(f"  {response}\n")

    print("Each iteration incorporates judge feedback to improve quality.")


def example_judge_guided_generation():
    """Show judge-guided generation pattern."""
    print("\n\n=== Judge-Guided Generation ===\n")

    print("Using judges to guide generation:\n")

    print("1. Generate with constraints:")
    print("   Prompt: 'Explain recursion to a beginner'")
    print("   Constraint: Must use an analogy\n")

    print("2. Judge checks constraint:")
    print("   ✗ Response 1: 'Recursion is when a function calls itself.'")
    print(
        "   ✓ Response 2: 'Recursion is like Russian dolls - each contains a smaller version.'"
    )
    print("\n3. Continue until judge approves or max attempts reached")


def example_synthesis_strategies():
    """Demonstrate different synthesis strategies."""
    print("\n\n=== Synthesis Strategies ===\n")

    print("1. Extractive Synthesis:")
    print("   Take best parts from each response")
    print("   Response A: 'Python is interpreted...'")
    print("   Response B: '...with dynamic typing...'")
    print("   Synthesis: 'Python is interpreted with dynamic typing'\n")

    print("2. Abstractive Synthesis:")
    print("   Generate new text combining ideas")
    print("   Responses discuss: speed, ease, libraries")
    print("   Synthesis: 'Python balances performance with developer productivity'\n")

    print("3. Hierarchical Synthesis:")
    print("   Organize information by importance")
    print("   Primary: Core language features")
    print("   Secondary: Ecosystem and tools")
    print("   Tertiary: Community and resources")


def example_practical_judge_system():
    """Show a practical judge system implementation."""
    print("\n\n=== Practical Judge System ===\n")

    print("Building a code review judge:\n")

    # Simulated code review
    code_aspects = {
        "Correctness": {"score": 0.9, "feedback": "Logic is sound"},
        "Efficiency": {"score": 0.7, "feedback": "Could optimize loops"},
        "Readability": {"score": 0.8, "feedback": "Good naming, needs comments"},
        "Security": {"score": 0.9, "feedback": "No obvious vulnerabilities"},
    }

    print("Code Review Judge Results:")
    total_score = 0
    for aspect, result in code_aspects.items():
        print(f"  {aspect}: {result['score']:.1f} - {result['feedback']}")
        total_score += result["score"]

    avg_score = total_score / len(code_aspects)
    print(f"\nOverall Score: {avg_score:.2f}/1.0")
    print("Recommendation: Approved with minor revisions")


def example_ensemble_synthesis():
    """Demonstrate ensemble synthesis pattern."""
    print("\n\n=== Ensemble Synthesis ===\n")

    print("Combining diverse model outputs:\n")

    # Simulate ensemble
    models = {
        "Creative Model": "Python slithers through your code like a serpent of simplicity",
        "Technical Model": "Python is an interpreted, object-oriented, high-level language",
        "Practical Model": "Python gets things done quickly with minimal boilerplate",
    }

    print("Individual model outputs:")
    for model, output in models.items():
        print(f"  {model}: {output}")

    print("\nSynthesis approaches:")
    print("  1. Voting: Select most common themes")
    print("  2. Blending: Combine complementary aspects")
    print("  3. Ranking: Use best output per criteria")
    print("  4. Meta-model: Train synthesizer on outputs")


@conditional_llm()
def main(_simulated_mode=False):
    """Run all judge synthesis examples."""
    print_section_header("Judge Synthesis Patterns")

    print("🎯 Judge & Synthesis in Compound AI:\n")
    print("• Judges evaluate and select best outputs")
    print("• Synthesis combines multiple perspectives")
    print("• Iterative refinement improves quality")
    print("• Consensus building increases reliability")

    example_basic_judge(_simulated_mode)
    example_synthesis_pattern(_simulated_mode)
    example_multi_judge_consensus()
    example_iterative_refinement()
    example_judge_guided_generation()
    example_synthesis_strategies()
    example_practical_judge_system()
    example_ensemble_synthesis()

    print("\n" + "=" * 50)
    print("✅ Judge & Synthesis Best Practices")
    print("=" * 50)
    print("\n1. Define clear judging criteria upfront")
    print("2. Use multiple judges for important decisions")
    print("3. Balance different synthesis strategies")
    print("4. Implement iterative refinement loops")
    print("5. Monitor judge agreement/disagreement")
    print("6. Cache synthesis results for efficiency")
    print("7. Version judge criteria and track changes")

    print("\n🔧 Implementation Tips:")
    print("• Start simple with binary judges")
    print("• Use rubrics for consistent evaluation")
    print("• Combine statistical and semantic judges")
    print("• Build synthesis pipelines incrementally")
    print("• Test judge calibration regularly")

    print("\nNext: Explore data processing in '../05_data_processing/'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
