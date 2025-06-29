"""Advanced AI Patterns - Sophisticated techniques for complex systems.

Learn advanced patterns:
- Streaming responses
- State management
- Dynamic routing
- Hierarchical processing
- Meta-programming patterns
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Callable
from dataclasses import dataclass
import asyncio
import json

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output
from _shared.conditional_execution import conditional_llm, SimulatedResponse
from _shared.conditional_llm_template import simulated_models
from ember.api import models
from ember.api.xcs import jit
from ember.xcs import vmap


@conditional_llm()
def example_streaming_responses(_simulated_mode=False):
    """Implement streaming for real-time responses."""
    print("=" * 50)
    print("Example 1: Streaming Responses")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    def stream_analysis(text: str) -> Generator[Dict[str, Any], None, None]:
        """Stream analysis results as they become available."""
        # Simulate streaming by breaking analysis into steps
        steps = [
            ("sentiment", f"Analyze sentiment: {text}"),
            ("entities", f"Extract entities from: {text}"),
            ("summary", f"Summarize in 10 words: {text}"),
            ("keywords", f"Extract 3 keywords from: {text}"),
        ]

        for step_name, prompt in steps:
            # Yield progress update
            yield {"type": "progress", "step": step_name, "status": "processing"}

            # Process step
            result = model_fn("gpt-3.5-turbo", prompt).text.strip()

            # Yield result
            yield {"type": "result", "step": step_name, "data": result}

    # Use streaming
    text = "Artificial intelligence is transforming how we work and live."

    print("Streaming analysis:")
    results = {}
    for update in stream_analysis(text):
        if update["type"] == "progress":
            print(f"  ⏳ {update['step']}: {update['status']}")
        elif update["type"] == "result":
            results[update["step"]] = update["data"]
            print(f"  ✅ {update['step']}: {update['data']}")

    print(f"\nFinal results: {len(results)} analyses completed")


@conditional_llm()
def example_state_management(_simulated_mode=False):
    """Implement stateful processing with context."""
    print("\n" + "=" * 50)
    print("Example 2: State Management")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    @dataclass
    class ConversationState:
        """Manages conversation state."""

        messages: List[Dict[str, str]]
        context: Dict[str, Any]

        def add_message(self, role: str, content: str):
            """Add message to history."""
            self.messages.append({"role": role, "content": content})

        def get_context_prompt(self) -> str:
            """Build context-aware prompt."""
            history = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in self.messages[-3:]  # Last 3 messages
                ]
            )
            return f"Conversation history:\n{history}\n\nUser: "

    class StatefulAssistant:
        """Assistant that maintains conversation state."""

        def __init__(self):
            self.state = ConversationState(messages=[], context={})

        @jit
        def respond(self, user_input: str) -> str:
            """Generate context-aware response."""
            # Add user message
            self.state.add_message("user", user_input)

            # Build prompt with context
            context_prompt = self.state.get_context_prompt() + user_input

            # Generate response
            system = (
                "You are a helpful assistant. Use conversation history for context."
            )
            response = model_fn("gpt-3.5-turbo", context_prompt, system=system).text

            # Add assistant response
            self.state.add_message("assistant", response)

            return response

        def get_summary(self) -> str:
            """Summarize the conversation."""
            if not self.state.messages:
                return "No conversation yet"

            conversation = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in self.state.messages]
            )

            prompt = f"Summarize this conversation in 2 sentences:\n{conversation}"
            return model_fn("gpt-3.5-turbo", prompt).text.strip()

    # Example usage
    assistant = StatefulAssistant()

    print("Stateful conversation:")
    interactions = [
        "What is machine learning?",
        "Can you give me a simple example?",
        "How is it different from traditional programming?",
    ]

    for user_input in interactions:
        print(f"\nUser: {user_input}")
        response = assistant.respond(user_input)
        print(f"Assistant: {response}")

    print(f"\nConversation summary: {assistant.get_summary()}")


@conditional_llm()
def example_dynamic_routing(_simulated_mode=False):
    """Route requests to appropriate handlers dynamically."""
    print("\n" + "=" * 50)
    print("Example 3: Dynamic Routing")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    # Define specialized handlers
    def handle_math(query: str) -> str:
        """Handle mathematical queries."""
        prompt = f"Solve this math problem step by step: {query}"
        return model_fn("gpt-3.5-turbo", prompt).text

    def handle_code(query: str) -> str:
        """Handle coding queries."""
        prompt = f"Write Python code to: {query}\nInclude comments."
        return model_fn("gpt-3.5-turbo", prompt).text

    def handle_creative(query: str) -> str:
        """Handle creative writing queries."""
        prompt = f"Write creatively about: {query}"
        return model_fn("gpt-3.5-turbo", prompt, temperature=0.8).text

    def handle_general(query: str) -> str:
        """Handle general queries."""
        return model_fn("gpt-3.5-turbo", query).text

    @jit
    def intelligent_router(query: str) -> Dict[str, Any]:
        """Route queries to appropriate handlers."""
        # Classify query type
        classification_prompt = f"""
        Classify this query into ONE category:
        - math (calculations, equations, algorithms)
        - code (programming, debugging, scripts)
        - creative (stories, poems, ideas)
        - general (everything else)
        
        Query: {query}
        Category:"""

        category = model_fn("gpt-3.5-turbo", classification_prompt).text.strip().lower()

        # Route to appropriate handler
        handlers = {
            "math": handle_math,
            "code": handle_code,
            "creative": handle_creative,
            "general": handle_general,
        }

        handler = handlers.get(category, handle_general)
        result = handler(query)

        return {
            "query": query,
            "category": category,
            "response": result,
            "handler": handler.__name__,
        }

    # Test routing
    queries = [
        "What is the derivative of x^2 + 3x?",
        "Write a function to reverse a string",
        "Create a haiku about spring",
        "What is the capital of Japan?",
    ]

    print("Dynamic routing results:")
    for query in queries:
        result = intelligent_router(query)
        print(f"\nQuery: {query}")
        print(f"Routed to: {result['handler']} (category: {result['category']})")
        print(f"Response preview: {result['response'][:100]}...")


@conditional_llm()
def example_hierarchical_processing(_simulated_mode=False):
    """Implement hierarchical multi-stage processing."""
    print("\n" + "=" * 50)
    print("Example 4: Hierarchical Processing")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    @dataclass
    class Document:
        """Document with hierarchical structure."""

        title: str
        sections: List[str]
        metadata: Dict[str, Any]

    class HierarchicalProcessor:
        """Process documents hierarchically."""

        @jit
        def process_section(self, section: str) -> Dict[str, Any]:
            """Process individual section."""
            return {
                "summary": model_fn("gpt-3.5-turbo", f"Summarize: {section}").text,
                "key_points": model_fn(
                    "gpt-3.5-turbo", f"List 2 key points: {section}"
                ).text,
                "word_count": len(section.split()),
            }

        def process_document(self, doc: Document) -> Dict[str, Any]:
            """Process entire document hierarchically."""
            # Process sections in parallel
            section_processor = vmap(self.process_section)
            section_results = section_processor(doc.sections)

            # Aggregate section summaries
            all_summaries = " ".join([r["summary"] for r in section_results])

            # Generate document-level summary
            doc_summary = model_fn(
                "gpt-3.5-turbo",
                f"Create executive summary from section summaries: {all_summaries}",
            ).text

            # Extract themes
            themes = model_fn(
                "gpt-3.5-turbo", f"Extract 3 main themes from: {all_summaries}"
            ).text

            return {
                "title": doc.title,
                "executive_summary": doc_summary,
                "themes": themes,
                "sections": section_results,
                "total_words": sum(r["word_count"] for r in section_results),
            }

    # Example usage
    doc = Document(
        title="The Future of AI",
        sections=[
            "AI has evolved rapidly in recent years, with breakthroughs in deep learning...",
            "The applications of AI span healthcare, finance, and transportation...",
            "Ethical considerations include bias, privacy, and job displacement...",
        ],
        metadata={"author": "Dr. Smith", "date": "2024"},
    )

    processor = HierarchicalProcessor()
    result = processor.process_document(doc)

    print(f"Document: {result['title']}")
    print(f"Executive Summary: {result['executive_summary'][:150]}...")
    print(f"Main Themes: {result['themes']}")
    print(f"Total Words: {result['total_words']}")
    print(f"Sections Processed: {len(result['sections'])}")


@conditional_llm()
def example_meta_programming(_simulated_mode=False):
    """Dynamic function generation and composition."""
    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models
    print("\n" + "=" * 50)
    print("Example 5: Meta-Programming Patterns")
    print("=" * 50 + "\n")

    def create_validator(rules: Dict[str, str]) -> Callable:
        """Dynamically create a validator function from rules."""

        def validator(data: Dict[str, Any]) -> Dict[str, Any]:
            errors = []

            for field, rule in rules.items():
                if field not in data:
                    errors.append(f"Missing field: {field}")
                    continue

                # Use LLM to validate based on rule
                prompt = f"""
                Check if this value follows the rule.
                Value: {data[field]}
                Rule: {rule}
                Valid? (yes/no):"""

                response = model_fn("gpt-3.5-turbo", prompt).text.lower()
                if "no" in response:
                    errors.append(f"Field '{field}' violates rule: {rule}")

            return {"valid": len(errors) == 0, "errors": errors, "data": data}

        return jit(validator)  # Optimize the generated function

    # Create custom validators
    email_validator = create_validator(
        {
            "email": "must be a valid email address",
            "name": "must be at least 2 characters",
            "age": "must be a number between 0 and 150",
        }
    )

    # Test the dynamically created validator
    test_cases = [
        {"email": "user@example.com", "name": "John Doe", "age": "25"},
        {"email": "invalid-email", "name": "J", "age": "200"},
        {"email": "test@test.com", "name": "Alice"},  # Missing age
    ]

    print("Dynamic validator results:")
    for i, test_data in enumerate(test_cases):
        result = email_validator(test_data)
        print(f"\nTest {i+1}: {test_data}")
        if result["valid"]:
            print("  ✅ Valid")
        else:
            print("  ❌ Invalid")
            for error in result["errors"]:
                print(f"     - {error}")


@conditional_llm()
def example_adaptive_system(_simulated_mode=False):
    """Build an adaptive system that learns from usage."""
    print("\n" + "=" * 50)
    print("Example 6: Adaptive System")
    print("=" * 50 + "\n")

    # Use simulated models in simulation mode
    model_fn = simulated_models if _simulated_mode else models

    class AdaptiveClassifier:
        """Classifier that adapts based on feedback."""

        def __init__(self):
            self.feedback_history = []
            self.performance_stats = {"correct": 0, "incorrect": 0, "total": 0}

        def get_adaptive_prompt(self, text: str) -> str:
            """Build prompt using feedback history."""
            base_prompt = f"Classify sentiment (positive/negative/neutral): {text}"

            if self.feedback_history:
                # Include recent corrections
                corrections = "\n".join(
                    [
                        f"- '{fb['text']}' is {fb['correct_label']} (not {fb['predicted']})"
                        for fb in self.feedback_history[-3:]
                    ]
                )

                return f"""
                Learn from these corrections:
                {corrections}
                
                Now classify: {text}
                """

            return base_prompt

        @jit
        def classify(self, text: str) -> str:
            """Classify with adaptive prompting."""
            prompt = self.get_adaptive_prompt(text)
            return model_fn("gpt-3.5-turbo", prompt).text.strip().lower()

        def add_feedback(self, text: str, predicted: str, correct_label: str):
            """Add feedback for learning."""
            self.performance_stats["total"] += 1

            if predicted == correct_label:
                self.performance_stats["correct"] += 1
            else:
                self.performance_stats["incorrect"] += 1
                self.feedback_history.append(
                    {
                        "text": text,
                        "predicted": predicted,
                        "correct_label": correct_label,
                    }
                )

        def get_accuracy(self) -> float:
            """Calculate current accuracy."""
            if self.performance_stats["total"] == 0:
                return 0.0
            return self.performance_stats["correct"] / self.performance_stats["total"]

    # Example usage
    classifier = AdaptiveClassifier()

    # Training examples with feedback
    training_data = [
        ("This product is amazing!", "positive", "positive"),
        ("Terrible service", "negative", "negative"),
        ("It's okay I guess", "positive", "neutral"),  # Wrong prediction
        ("Not bad at all", "negative", "positive"),  # Wrong prediction
    ]

    print("Adaptive learning process:")
    for text, true_label, _ in training_data:
        prediction = classifier.classify(text)
        classifier.add_feedback(text, prediction, true_label)

        correct = "✅" if prediction == true_label else "❌"
        print(f"{correct} '{text}' -> Predicted: {prediction}, True: {true_label}")

    print(f"\nAccuracy after training: {classifier.get_accuracy():.1%}")

    # Test on new examples (should perform better)
    test_texts = [
        "It's alright",  # Should learn this is neutral
        "Not terrible",  # Should learn this is positive
    ]

    print("\nTesting adaptive system:")
    for text in test_texts:
        prediction = classifier.classify(text)
        print(f"'{text}' -> {prediction}")


def main():
    """Run all advanced pattern examples."""
    print_section_header("Advanced AI Patterns")

    try:
        example_streaming_responses()
        example_state_management()
        example_dynamic_routing()
        example_hierarchical_processing()
        example_meta_programming()
        example_adaptive_system()

        print("\n" + "=" * 50)
        print("🚀 Advanced Patterns Summary")
        print("=" * 50)
        print("\n1. Use streaming for real-time user experience")
        print("2. Maintain state for context-aware processing")
        print("3. Route dynamically for specialized handling")
        print("4. Process hierarchically for complex documents")
        print("5. Generate functions dynamically when needed")
        print("6. Build adaptive systems that improve over time")
        print("7. Combine patterns for sophisticated applications")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
