"""Example demonstrating the improved Ember Operators API with graceful fallbacks.

This example shows how to create and compose operators using the simplified API,
demonstrating basic operators, ensemble patterns, and advanced composition techniques.
It follows the latest API patterns and includes proper error handling for when
API keys are not available.

Run this example with:
```bash
uv run python src/ember/examples/integration/api_operators_example.py
```
"""

import logging
from typing import Any, ClassVar, List

from ember.api.models import models
from ember.api.non import non
from ember.api.operators import Operator, Specification, EmberModel, Field


# Define input/output models
class QuestionInput(EmberModel):
    """Input model for question answering."""

    question: str = Field(..., description="The question to be answered")


class AnswerOutput(EmberModel):
    """Output model for question answering."""

    answer: str = Field(..., description="The answer to the question")
    confidence: float = Field(
        default=1.0, description="Confidence score for the answer"
    )


class MultipleAnswersOutput(EmberModel):
    """Output model with multiple candidate answers."""

    answers: List[str] = Field(..., description="Multiple candidate answers")


# Helper class for simulating models when API keys aren't available
class MockLMModule:
    """Simulates a language model without requiring API access."""

    def __init__(self, model_name: str, temperature: float = 0.7):
        """Initialize mock LM with model name and temperature.

        Args:
            model_name: Name of the model to simulate
            temperature: Simulated temperature parameter
        """
        self.model_name = model_name
        self.temperature = temperature
        # Dictionary of canned responses
        self.canned_responses = {
            "What is the meaning of life?": f"This is a simulated response from {model_name}: The answer is 42."
        }

    def __call__(self, *, prompt: str) -> str:
        """Generate a simulated response based on the prompt.

        Args:
            prompt: The input prompt text

        Returns:
            A simulated model response
        """
        # Check for synthesis prompts
        if "synthesize" in prompt.lower() or "multiple advisors" in prompt.lower():
            return (
                "Reasoning: All three perspectives agree on the value '42' as the answer to the meaning of life, "
                "though they approach it from different angles. The scientific perspective presents it as a universal constant, "
                "the philosophical perspective frames it as an existential truth, and the humorous perspective references "
                "Douglas Adams' 'Hitchhiker's Guide to the Galaxy'. This convergence across different domains suggests '42' "
                "is indeed the most comprehensive answer.\n\n"
                "Final Answer: The meaning of life is 42."
            )

        # Check for prompt prefixes for diverse answers
        for prefix in [
            "Scientific perspective:",
            "Philosophical perspective:",
            "Humorous perspective:",
        ]:
            if prefix in prompt:
                if "Scientific" in prefix:
                    return f"From a {prefix} The mathematical constant 42 appears in various equations describing the universe."
                elif "Philosophical" in prefix:
                    return f"From a {prefix} The meaning of life is to find your own purpose and create meaning through authentic choices."
                elif "Humorous" in prefix:
                    return f"From a {prefix} According to the Hitchhiker's Guide to the Galaxy, the answer is definitively 42."

        # Check for known questions
        for question, answer in self.canned_responses.items():
            if question.lower() in prompt.lower():
                return answer

        # Generic response
        return f"Simulated response from {self.model_name}: I don't have a specific answer for this question."


class ModelProvider:
    """Helper class to provide real or mock models based on availability."""

    @staticmethod
    def get_model(model_name: str, temperature: float = 0.7):
        """Get a real or mocked model based on availability.

        Args:
            model_name: The model name to get
            temperature: Temperature setting for the model

        Returns:
            Either a real bound model or a mock implementation
        """
        # Try to get the real model
        try:
            available_models = models.list()
            if model_name in available_models:
                return models.instance(model_name, temperature=temperature)
        except Exception as e:
            logging.warning(f"Could not load model {model_name}: {str(e)}")

        # Fall back to mock implementation
        logging.info(f"Using mock implementation for {model_name}")
        return MockLMModule(model_name=model_name, temperature=temperature)


# Basic operator example
class SimpleQuestionAnswerer(Operator[QuestionInput, AnswerOutput]):
    """A simple operator that answers questions using a language model."""

    specification: ClassVar[Specification[QuestionInput, AnswerOutput]] = Specification(
        input_model=QuestionInput, structured_output=AnswerOutput
    )

    # Field declarations
    model: Any  # Bound model function

    def __init__(self, *, model_name: str = "gpt-4", temperature: float = 0.7):
        """Initialize the operator with model configuration.

        Args:
            model_name: The model name to use
            temperature: Sampling temperature for generation
        """
        self.model = ModelProvider.get_model(model_name, temperature)

    def forward(self, *, inputs: QuestionInput) -> AnswerOutput:
        """Generate an answer to the input question.

        Args:
            inputs: The question input model

        Returns:
            Structured answer output
        """
        # Call the model with the question
        response = self.model(inputs.question)

        # Extract text content from response
        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = str(response)

        # Return structured output
        return AnswerOutput(answer=response_text, confidence=0.95)


# Diversification operator
class DiverseAnswerGenerator(Operator[QuestionInput, MultipleAnswersOutput]):
    """Generates multiple diverse answers to a question."""

    specification: ClassVar[
        Specification[QuestionInput, MultipleAnswersOutput]
    ] = Specification(
        input_model=QuestionInput, structured_output=MultipleAnswersOutput
    )

    # Field declarations
    prefixes: List[str]
    model: Any  # Bound model function

    def __init__(self, *, prefixes: List[str], model_name: str = "gpt-4"):
        """Initialize with different prefixes to guide diverse responses.

        Args:
            prefixes: Different framing instructions to get diverse answers
            model_name: The model name to use
        """
        self.prefixes = prefixes
        self.model = ModelProvider.get_model(model_name)

    def forward(self, *, inputs: QuestionInput) -> MultipleAnswersOutput:
        """Generate multiple diverse answers using different prefixes.

        Args:
            inputs: The question input

        Returns:
            Multiple answers output
        """
        answers = []

        for prefix in self.prefixes:
            # Prepend the prefix to the question
            prompt = f"{prefix} {inputs.question}"
            response = self.model(prompt)

            # Extract text content from response
            if hasattr(response, "text"):
                response_text = response.text
            else:
                response_text = str(response)

            answers.append(response_text)

        return MultipleAnswersOutput(answers=answers)


def main():
    """Run the example pipeline to demonstrate operator composition."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Check if API keys are available
    try:
        available_models = models.list()
        if not available_models:
            logging.warning(
                "No models were discovered. This example will use mock models instead of real ones."
            )
            logging.warning(
                "To use real models, set one of these environment variables: "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
            )
    except Exception:
        logging.warning("Could not check available models. Using mock implementation.")

    # Create a question input
    question = QuestionInput(question="What is the meaning of life?")

    print(f"Question: {question.question}\n")

    # 1. Simple operator with GPT-4
    simple_answerer = SimpleQuestionAnswerer(model_name="gpt-4")
    result1 = simple_answerer(inputs=question)
    print("1. Simple Operator:")
    print(f"   Answer: {result1.answer}")
    print(f"   Confidence: {result1.confidence}\n")

    # 2. Using ensemble operator from non API
    ensemble = non.UniformEnsemble(
        num_units=3,
        model_name="gpt-4",
        temperature=0.7
    )

    result2 = ensemble(inputs={"query": question.question})

    print("2. Ensemble Operator:")
    for i, response in enumerate(result2["responses"], 1):
        print(f"   Model {i}: {response}")
    print()

    # 3. Ensemble with answer selection using non API
    selector = non.MostCommon()
    result3 = selector(inputs={"responses": result2["responses"]})
    print("3. Ensemble with Most Common Answer Selector:")
    print(f"   Selected Answer: {result3['final_answer']}\n")

    # 4. Diverse answers with synthesis
    diverse_generator = DiverseAnswerGenerator(
        prefixes=[
            "Scientific perspective:",
            "Philosophical perspective:",
            "Humorous perspective:",
        ],
        model_enum=ModelEnum.gpt_4,
    )

    # Generate diverse answers
    diverse_results = diverse_generator(inputs=question)

    # Use JudgeSynthesisOperator from non API
    synthesizer = non.JudgeSynthesisOperator(
        model_name="gpt-4",
        temperature=0.3
    )

    # Prepare the input for the synthesizer
    synthesis_input = {"query": question.question, "responses": diverse_results.answers}

    # Run the synthesizer
    result4 = synthesizer(inputs=synthesis_input)

    print("4. Diverse Answers with Synthesis:")
    print(f"   Synthesized Answer: {result4['final_answer']}")
    print(f"   Reasoning: {result4['reasoning']}\n")


if __name__ == "__main__":
    main()
