"""Judge Synthesis - Building AI systems with judging and synthesis.

Learn how to build compound AI systems that use judges to evaluate
outputs and synthesis to combine multiple perspectives.

Example:
    >>> from ember.operators.common import ModelText, Ensemble
    >>> experts = [ModelText("gpt-4o-mini"), ModelText("claude-3-haiku")]
    >>> ensemble = Ensemble(experts)
    >>> responses = ensemble("Explain quantum computing")
    >>> judge = ModelText("gpt-4o-mini")
    >>> winner = judge(f"Which response is better? A: {responses[0]} B: {responses[1]}")
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import (
    print_section_header,
    ensure_api_key,
    timer,
)
from ember.operators.common import ModelText


def example_basic_judge():
    """Show basic judge pattern."""
    print("\n=== Basic Judge Pattern ===\n")

    print("Judge pattern components:")
    print("  1. Multiple candidate responses")
    print("  2. Judge criteria/rubric")
    print("  3. Evaluation and selection")
    print("  4. Final decision\n")

    # Simulate judging
    candidates = [
        "Python is a high-level programming language.",
        "Python is a versatile, interpreted language known for simplicity.",
        "Python: Easy syntax, vast libraries, great for beginners!",
    ]

    print("Candidate responses:")
    for i, response in enumerate(candidates, 1):
        print(f"  {i}. {response}")

    print("\nJudge evaluation:")
    print("  Criteria: Completeness, accuracy, clarity")
    print("  Selected: Response 2 (most comprehensive)")


def example_synthesis_pattern():
    """Demonstrate synthesis of multiple outputs."""
    print("\n\n=== Synthesis Pattern ===\n")

    print("Synthesis combines multiple perspectives:")
    print()

    # Example perspectives
    perspectives = {
        "Technical": "Python uses dynamic typing and automatic memory management.",
        "Practical": "Python excels at web development, data science, and automation.",
        "Historical": "Python was created by Guido van Rossum in 1991.",
    }

    print("Individual perspectives:")
    for viewpoint, content in perspectives.items():
        print(f"  {viewpoint}: {content}")

    print("\nSynthesized response:")
    print("  Python, created in 1991, is a dynamically-typed language")
    print("  that excels at web development, data science, and automation")
    print("  through its automatic memory management and vast ecosystem.")


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


def example_real_judge_system(has_openai: bool, has_anthropic: bool):
    """Show real working judge system with ModelText."""
    print("\n\n=== Real Judge System with ModelText ===\n")

    try:
        # Create competing responses using different models
        if has_openai and has_anthropic:
            gpt_expert = ModelText("gpt-4o-mini", temperature=0.7)
            claude_expert = ModelText("claude-3-haiku", temperature=0.7)

            question = "What are the benefits of functional programming?"
            print(f"Question: {question}\n")

            with timer("Generating competing responses"):
                response_a = gpt_expert(question)
                response_b = claude_expert(question)

            print("Candidate Responses:")
            print(f"A (GPT): {response_a[:100]}...")
            print(f"B (Claude): {response_b[:100]}...")

            # Create judge to evaluate responses
            judge = ModelText("gpt-4o-mini", temperature=0.1)

            judge_prompt = f"""You are a judge evaluating two responses to the question: "{question}"

Response A: {response_a}

Response B: {response_b}

Evaluate based on:
1. Accuracy of information
2. Clarity of explanation
3. Completeness of answer
4. Practical relevance

Respond with ONLY "A" or "B" for the better response, followed by a brief explanation."""

            with timer("Judge evaluation"):
                judgment = judge(judge_prompt)

            print(f"\nJudge Decision: {judgment[:200]}...")

            # Multiple criteria judging
            criteria_judges = {
                "Accuracy": "Focus on factual correctness and technical accuracy",
                "Clarity": "Focus on how easy the response is to understand",
                "Completeness": "Focus on how thoroughly the question is answered",
            }

            print("\nMulti-Criteria Judging:")
            for criterion, instruction in criteria_judges.items():
                criterion_prompt = (
                    f"{judge_prompt}\n\nFocus specifically on: {instruction}"
                )
                criterion_judgment = judge(criterion_prompt)
                winner = (
                    "A" if criterion_judgment.strip().upper().startswith("A") else "B"
                )
                print(f"  {criterion}: Response {winner}")

        else:
            model_name = "gpt-4o-mini" if has_openai else "claude-3-haiku"
            single_model = ModelText(model_name, temperature=0.7)

            # Generate multiple responses from same model with different prompts
            question = "What are the benefits of functional programming?"

            prompt_a = f"As a computer science professor, {question}"
            prompt_b = f"As a practical software engineer, {question}"

            response_a = single_model(prompt_a)
            response_b = single_model(prompt_b)

            print(f"Question: {question}\n")
            print(f"Academic perspective: {response_a[:100]}...")
            print(f"Practical perspective: {response_b[:100]}...")

            # Judge which perspective is more useful
            judge = ModelText(model_name, temperature=0.1)
            judge_prompt = f"""Which response better answers "{question}" for a general audience?

Academic: {response_a}
Practical: {response_b}

Respond with "Academic" or "Practical" and explain why."""

            judgment = judge(judge_prompt)
            print(f"\nJudge Decision: {judgment[:150]}...")

    except Exception as e:
        print(f"Error in real judge system: {e}")
        print("This might be due to API rate limits or connectivity issues.")


def example_real_synthesis(has_openai: bool, has_anthropic: bool):
    """Show real synthesis of multiple model responses."""
    print("\n\n=== Real Response Synthesis ===\n")

    try:
        # Get responses from available models
        available_models = []
        if has_openai:
            available_models.append(("GPT", ModelText("gpt-4o-mini", temperature=0.8)))
        if has_anthropic:
            available_models.append(
                ("Claude", ModelText("claude-3-haiku", temperature=0.8))
            )

        if not available_models:
            print("No models available for synthesis example.")
            return

        topic = "How can AI improve healthcare?"
        print(f"Topic: {topic}\n")

        # Collect diverse responses
        responses = {}
        for name, model in available_models:
            with timer(f"Getting {name} response"):
                response = model(f"Provide 3 specific ways {topic.lower()}")
                responses[name] = response
                print(f"{name}: {response[:120]}...")

        if len(responses) > 1:
            # Synthesize responses using another model
            synthesizer = available_models[0][1]  # Use first available model

            synthesis_prompt = f"""Synthesize these different perspectives on "{topic}":

"""
            for name, response in responses.items():
                synthesis_prompt += f"{name}: {response}\n\n"

            synthesis_prompt += """Create a comprehensive answer that:
1. Combines the best insights from each perspective
2. Organizes information logically
3. Avoids repetition
4. Provides a cohesive view

Synthesized response:"""

            with timer("Synthesizing responses"):
                synthesized = synthesizer(synthesis_prompt)

            print(f"\nSynthesized Response:\n{synthesized[:300]}...")

        else:
            print("Single model available - showing self-synthesis pattern:")
            model = available_models[0][1]

            # Get multiple perspectives from same model
            perspectives = [
                "technical perspective",
                "patient perspective",
                "healthcare provider perspective",
            ]

            responses = {}
            for perspective in perspectives:
                prompt = f"From a {perspective}, {topic.lower()}"
                response = model(prompt)
                responses[perspective] = response
                print(f"{perspective.title()}: {response[:100]}...")

            # Synthesize the perspectives
            synthesis_prompt = (
                "Combine these three perspectives into one comprehensive answer:\n\n"
            )
            for perspective, response in responses.items():
                synthesis_prompt += f"{perspective.title()}: {response}\n\n"

            synthesized = model(synthesis_prompt + "Combined answer:")
            print(f"\nSynthesized: {synthesized[:200]}...")

        print("\n✅ Synthesis demonstrates:")
        print("  • Collecting diverse viewpoints")
        print("  • Combining complementary information")
        print("  • Creating coherent unified responses")
        print("  • Reducing redundancy while preserving insights")

    except Exception as e:
        print(f"Error in synthesis example: {e}")
        print("This might be due to API rate limits or connectivity issues.")


def main():
    """Run all judge synthesis examples."""
    print_section_header("Judge Synthesis Patterns")

    # Check API availability
    has_openai = ensure_api_key("openai")
    has_anthropic = ensure_api_key("anthropic")

    if not has_openai and not has_anthropic:
        print("\n⚠️  Running in demo mode - set API keys for real judge examples")
        demo_mode = True
    else:
        print(f"\n✓ API keys available: OpenAI={has_openai}, Anthropic={has_anthropic}")
        demo_mode = False

    print("\n🎯 Judge & Synthesis in Compound AI:\n")
    print("• Judges evaluate and select best outputs")
    print("• Synthesis combines multiple perspectives")
    print("• Iterative refinement improves quality")
    print("• Consensus building increases reliability")

    example_basic_judge()
    example_synthesis_pattern()
    example_multi_judge_consensus()
    example_iterative_refinement()
    example_judge_guided_generation()
    example_synthesis_strategies()
    example_practical_judge_system()
    example_ensemble_synthesis()

    # Add real working examples
    if not demo_mode:
        example_real_judge_system(has_openai, has_anthropic)
        example_real_synthesis(has_openai, has_anthropic)

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
    print("• Use ModelText for judge responses")
    print("• Use Ensemble for multiple perspectives")
    print("• Chain judges for multi-step evaluation")

    print("\n🔧 Model Primitives Used:")
    print("• ModelText for text-only responses")
    print("• ModelCall for full response objects")
    print("• Ensemble for parallel model execution")
    print("• Chain for sequential processing")
    print("• @operators.op for custom judge logic")

    print("\nNext: Explore specifications in 'specifications_progressive.py'!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
