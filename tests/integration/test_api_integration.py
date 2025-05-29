#!/usr/bin/env python3
"""Integration test for Ember high-level API.

This script tests the integrated functionality of different components
using minimal dependencies.
"""

import logging
import pytest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

print("Testing Ember API Integration...")

# Import the context system and base evaluator
from ember.core.context.ember_context import EmberContext
from ember.core.utils.eval.base_evaluator import IEvaluator, EvaluationResult


def test_evaluator_basic():
    """Test basic evaluator functionality."""
    print("\n1. Testing Basic Evaluator Functionality")

    # Create a simple evaluator implementation
    class ExactMatchEvaluator(IEvaluator):
        def evaluate(self, system_output, correct_answer, **kwargs):
            is_match = str(system_output).strip() == str(correct_answer).strip()
            return EvaluationResult(
                is_correct=is_match,
                score=1.0 if is_match else 0.0,
                metadata={"exact_match": is_match})

    # Create evaluator instance
    evaluator = ExactMatchEvaluator()

    # Test with correct answer
    correct_result = evaluator.evaluate("Paris", "Paris")
    assert correct_result.is_correct is True
    assert correct_result.score == 1.0
    assert correct_result.metadata["exact_match"] is True

    # Test with incorrect answer
    incorrect_result = evaluator.evaluate("London", "Paris")
    assert incorrect_result.is_correct is False
    assert incorrect_result.score == 0.0
    assert incorrect_result.metadata["exact_match"] is False

    print("Basic evaluator test successful!")


def test_context_registry():
    """Test context registry functionality."""
    print("\n2. Testing Context Registry")

    # Get the current context
    context = EmberContext.current()

    # Create a mock model
    class MockModel:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt):
            return f"Response from {self.name}: {prompt}"

    # Register model in context
    mock_model = MockModel("test_model")
    context.register("model", "test_model", mock_model)

    # Retrieve model from context
    retrieved_model = context.get_model("test_model")

    # Verify it's the same model
    assert retrieved_model is mock_model
    assert retrieved_model.name == "test_model"

    # Test model functionality
    response = retrieved_model.generate("Hello")
    assert response == "Response from test_model: Hello"

    print("Context registry test successful!")


def test_evaluation_pipeline():
    """Test a simple evaluation pipeline."""
    print("\n3. Testing Simple Evaluation Pipeline")

    # Create a custom evaluator class
    class LengthEvaluator(IEvaluator):
        def __init__(self, min_length=10):
            self.min_length = min_length

        def evaluate(self, system_output, correct_answer, **kwargs):
            output_len = len(str(system_output))
            is_long_enough = output_len >= self.min_length
            return EvaluationResult(
                is_correct=is_long_enough,
                score=min(1.0, output_len / 50),  # Normalize
                metadata={"length": output_len})

    # Create evaluator instances
    exact_match = ExactMatchEvaluator()
    length_check = LengthEvaluator(min_length=15)

    # Create a simple pipeline
    evaluators = [exact_match, length_check]

    # Function that runs all evaluators and aggregates results
    def evaluate_all(output, reference):
        is_correct_sum = 0
        score_sum = 0
        metadata = {}

        for evaluator in evaluators:
            result = evaluator.evaluate(output, reference)

            # Aggregate metrics
            is_correct_sum += result.is_correct
            score_sum += result.score

            # Combine metadata
            if result.metadata:
                metadata.update(result.metadata)

        # Calculate averages
        avg_is_correct = is_correct_sum / len(evaluators)
        avg_score = score_sum / len(evaluators)

        # Return as dict for easy testing
        return {"is_correct": avg_is_correct, "score": avg_score, "metadata": metadata}

    # Test with examples
    short_match = evaluate_all("Paris", "Paris")
    mid_match = evaluate_all(
        "The capital of France is Paris", "The capital of France is Paris"
    )

    print(f"Short match results: {short_match}")
    print(f"Mid match results: {mid_match}")

    # Validate results
    assert (
        0.4 <= short_match["is_correct"] <= 0.6
    )  # One passes, one fails (exact match passes, length check fails)
    assert (
        mid_match["is_correct"] == 1.0
    )  # Both pass (exact match passes, length check passes)
    assert short_match["metadata"].get("length") == 5
    assert mid_match["metadata"].get("length") > 15

    print("Evaluation pipeline test successful!")


# Define ExactMatchEvaluator at module level so it can be used in multiple tests
class ExactMatchEvaluator(IEvaluator):
    def evaluate(self, system_output, correct_answer, **kwargs):
        is_match = str(system_output).strip() == str(correct_answer).strip()
        return EvaluationResult(
            is_correct=is_match,
            score=1.0 if is_match else 0.0,
            metadata={"exact_match": is_match})


# Run all tests using pytest fixtures
@pytest.mark.integration
def test_api_integration():
    """Run all integration tests."""
    try:
        test_evaluator_basic()
        test_context_registry()
        test_evaluation_pipeline()

        print("\n✅ All integration tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        raise
