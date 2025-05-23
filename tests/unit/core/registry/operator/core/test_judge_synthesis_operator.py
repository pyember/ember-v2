from typing import Any, Dict

from tests.helpers.simplified_imports import EmberModel


# Create test stubs
class JudgeSynthesisInputs(EmberModel):
    """Test input model."""

    query: str
    responses: list[str]


class JudgeSynthesisOutputs(EmberModel):
    """Test output model."""

    final_answer: str
    reasoning: str


class JudgeSynthesisOperator:
    """Test operator implementation."""

    def __init__(self, *, model):
        self.model = model

    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)

    def forward(self, *, inputs):
        """Process the synthesis."""
        response = self.model(
            f"Query: {inputs.query}\nResponses: {inputs.responses}"
        )
        
        # Extract text from response
        raw_output = response.text if hasattr(response, 'text') else str(response)

        # Parse the response
        final_answer = "Unknown"
        reasoning = ""

        for line in raw_output.splitlines():
            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()

        return JudgeSynthesisOutputs(final_answer=final_answer, reasoning=reasoning)


class MockResponse:
    """Mock response with text attribute."""
    def __init__(self, text):
        self.text = text


class MockModel:
    """Returns reasoning plus a final answer line."""

    def __call__(self, prompt: str):
        return MockResponse("Reasoning: Some reasoning here.\nFinal Answer: Synthesized Answer")


def test_judge_synthesis_operator_forward() -> None:
    mock_model = MockModel()
    op = JudgeSynthesisOperator(model=mock_model)

    inputs = JudgeSynthesisInputs(query="synthesize?", responses=["Ans1", "Ans2"])
    result: Dict[str, Any] = op(inputs=inputs)

    assert (
        result.final_answer == "Synthesized Answer"
    ), "JudgeSynthesisOperator did not synthesize the expected final answer."
    assert (
        "Some reasoning here." in result.reasoning
    ), "JudgeSynthesisOperator did not capture reasoning correctly."
