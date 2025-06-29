"""Natural API Showcase - Write Python, get AI superpowers.

Demonstrates how Ember's natural API lets you write normal Python code
that automatically gets optimization, batching, and composition capabilities.
"""

import sys
from pathlib import Path

# Add the shared utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header
from _shared.conditional_execution import conditional_llm, SimulatedResponse
from ember.api import models, operators, data
from ember.api.xcs import jit
from ember.api.xcs import vmap


def example_functions_as_operators():
    """Any function is an operator - no inheritance needed."""
    print("\n=== Functions as Operators ===\n")

    # Just write a function - it's already an operator!
    def classify_email(email: str) -> dict:
        """Classify an email - this is a complete operator."""
        # Determine urgency
        urgency = models(
            "gpt-3.5-turbo", f"Rate urgency (high/medium/low): {email}"
        ).text

        # Detect intent
        intent = models(
            "gpt-3.5-turbo", f"What's the intent (question/request/info): {email}"
        ).text

        # Extract action items
        actions = models("gpt-3.5-turbo", f"List action items: {email}").text

        return {
            "urgency": urgency.strip(),
            "intent": intent.strip(),
            "actions": actions.strip(),
        }

    # Use it directly
    result = classify_email("URGENT: Server is down! Please investigate immediately.")
    print(f"Direct call: {result}")

    # Make it fast
    fast_classify = jit(classify_email)
    result = fast_classify("When can we schedule the meeting?")
    print(f"JIT optimized: {result}")

    # Make it work on batches
    batch_classify = vmap(classify_email)
    emails = [
        "Thanks for the update",
        "Can you review this PR?",
        "System alert: High CPU usage",
    ]
    results = batch_classify(emails)
    print(f"Batch processing: {results}")


def example_natural_composition():
    """Compose operations naturally with Python."""
    print("\n=== Natural Composition ===\n")

    # Define simple functions
    def summarize(text: str) -> str:
        return models("gpt-3.5-turbo", f"Summarize in one sentence: {text}").text

    def translate(text: str, language: str = "Spanish") -> str:
        return models("gpt-3.5-turbo", f"Translate to {language}: {text}").text

    def make_professional(text: str) -> str:
        return models("gpt-3.5-turbo", f"Rewrite professionally: {text}").text

    # Compose naturally
    def process_document(doc: str) -> dict:
        """Composite operation using natural Python."""
        summary = summarize(doc)
        spanish = translate(summary, "Spanish")
        professional = make_professional(summary)

        return {
            "original": doc,
            "summary": summary,
            "spanish": spanish,
            "professional": professional,
        }

    # Optimize the whole pipeline
    optimized_pipeline = jit(process_document)

    doc = "The new product launch was successful. Sales exceeded expectations by 200%."
    result = optimized_pipeline(doc)
    print(f"Pipeline result: {result}")


def example_dynamic_behavior():
    """Dynamic behavior with natural Python patterns."""
    print("\n=== Dynamic Behavior ===\n")

    # Create model instances with different configs
    creative_model = models.instance("gpt-3.5-turbo", temperature=0.9)
    precise_model = models.instance("gpt-3.5-turbo", temperature=0.1)

    def adaptive_writer(topic: str, style: str = "balanced") -> str:
        """Dynamically choose model based on style."""
        prompt = f"Write a paragraph about: {topic}"

        if style == "creative":
            return creative_model(prompt).text
        elif style == "precise":
            return precise_model(prompt).text
        else:
            # Balanced - use both and combine
            creative_text = creative_model(prompt).text
            precise_text = precise_model(prompt).text
            return models(
                "gpt-3.5-turbo",
                f"Combine these two paragraphs into one balanced paragraph:\n1: {creative_text}\n2: {precise_text}",
            ).text

    # Natural Python, automatic optimization
    fast_writer = jit(adaptive_writer)

    print("Creative:", fast_writer("artificial intelligence", "creative"))
    print("\nPrecise:", fast_writer("artificial intelligence", "precise"))
    print("\nBalanced:", fast_writer("artificial intelligence"))


def example_data_integration():
    """Natural integration with data API."""
    print("\n=== Natural Data Integration ===\n")

    # Define a simple evaluation function
    @jit
    def evaluate_answer(question: str, answer: str, correct: str) -> bool:
        """Evaluate if an answer is correct."""
        prompt = f"""Question: {question}
        Given answer: {answer}
        Correct answer: {correct}
        Is the given answer correct? (yes/no)"""

        response = models("gpt-3.5-turbo", prompt).text.lower()
        return "yes" in response

    # Natural iteration over datasets (if available)
    try:
        # This would work with real datasets
        dataset = data.load("truthful_qa", split="validation[:10]")

        correct = 0
        for item in dataset:
            # Generate answer
            answer = models("gpt-3.5-turbo", item.question).text

            # Evaluate
            if evaluate_answer(item.question, answer, item.correct_answer):
                correct += 1

        print(f"Accuracy: {correct}/{len(dataset)}")

    except:
        print("Dataset example - would process data naturally in loops")
        print("No special dataset classes needed - just Python iteration")


@conditional_llm(providers=["openai", "anthropic", "google"])
def main(_simulated_mode=False):
    """Run natural API examples."""
    print_section_header("Natural API Showcase")

    if _simulated_mode:
        return run_simulated_examples()

    example_functions_as_operators()
    example_natural_composition()
    example_dynamic_behavior()
    example_data_integration()

    print("\n✨ Key takeaway: Just write Python - Ember handles the rest!")


def run_simulated_examples():
    """Run examples with simulated responses."""
    # Functions as Operators
    print("\n=== Functions as Operators ===\n")

    print(
        "Direct call: {'urgency': 'high', 'intent': 'info', 'actions': 'Investigate server issue immediately'}"
    )
    print(
        "JIT optimized: {'urgency': 'low', 'intent': 'question', 'actions': 'Schedule meeting at convenient time'}"
    )
    print("Batch processing: [")
    print("  {'urgency': 'low', 'intent': 'info', 'actions': 'None'},")
    print("  {'urgency': 'medium', 'intent': 'request', 'actions': 'Review PR'},")
    print("  {'urgency': 'high', 'intent': 'info', 'actions': 'Monitor CPU usage'}")
    print("]")

    # Natural Composition
    print("\n=== Natural Composition ===\n")

    pipeline_result = {
        "original": "The new product launch was successful. Sales exceeded expectations by 200%.",
        "summary": "Product launch achieved 200% sales target.",
        "spanish": "El lanzamiento del producto logró el 200% del objetivo de ventas.",
        "professional": "The product launch demonstrated exceptional performance, surpassing projected sales targets by 200%.",
    }
    print(f"Pipeline result: {pipeline_result}")

    # Dynamic Behavior
    print("\n=== Dynamic Behavior ===\n")

    print(
        "Creative: Artificial intelligence dances at the intersection of human dreams and silicon possibilities, weaving tapestries of logic that mirror our own consciousness in ways both beautiful and bewildering."
    )
    print(
        "\nPrecise: Artificial intelligence is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence, including pattern recognition, decision-making, and natural language processing."
    )
    print(
        "\nBalanced: Artificial intelligence represents a transformative field that combines rigorous computer science with the ambitious goal of replicating human cognitive abilities, enabling machines to learn, reason, and interact in increasingly sophisticated ways."
    )

    # Data Integration
    print("\n=== Natural Data Integration ===\n")
    print("Dataset example - would process data naturally in loops")
    print("No special dataset classes needed - just Python iteration")

    print("\n✨ Key takeaway: Just write Python - Ember handles the rest!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
