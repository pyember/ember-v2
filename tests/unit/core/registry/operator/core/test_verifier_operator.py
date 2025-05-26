from typing import Optional

from tests.helpers.simplified_imports import EmberModel


# Create test stubs
class VerifierOperatorInputs(EmberModel):
    """Test input model."""

    query: str
    candidate_answer: str


class VerifierOperatorOutputs(EmberModel):
    """Test output model."""

    verdict: int
    explanation: str
    revised_answer: Optional[str] = None


class VerifierOperator:
    """Test operator implementation."""

    def __init__(self, *, model):
        self.model = model

    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)

    def forward(self, *, inputs):
        """Process the verification."""
        response = self.model(
            f"Query: {inputs.query}\nCandidate: {inputs.candidate_answer}"
        )
        raw_output = response.text

        # Initialize default values
        verdict = 0
        explanation = ""
        revised_answer = None

        # Parse the response
        for line in raw_output.splitlines():
            if line.startswith("Verdict:"):
                verdict_text = line.replace("Verdict:", "").strip()
                verdict = (
                    1 if "1" in verdict_text or "correct" in verdict_text.lower() else 0
                )
            elif line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()
            elif line.startswith("Revised Answer:"):
                revised_answer = line.replace("Revised Answer:", "").strip()

        return VerifierOperatorOutputs(
            verdict=verdict,
            explanation=explanation,
            revised_answer=revised_answer or None)


class MockResponse:
    """Mock response object with text attribute."""
    def __init__(self, text: str):
        self.text = text


class CustomVerifierModel:
    """Mimics a verifier model that outputs verdict, explanation, optional revised answer lines."""

    def __call__(self, prompt: str):
        """Model call that returns a verification output."""
        return MockResponse(
            "Verdict: 1\n"
            "Explanation: The answer is correct because...\n"
            "Revised Answer: \n"
        )


def test_verifier_operator_forward() -> None:
    custom_model = CustomVerifierModel()
    op = VerifierOperator(model=custom_model)

    inputs = VerifierOperatorInputs(query="Verify this", candidate_answer="Answer")
    result: VerifierOperatorOutputs = op(inputs=inputs)

    # Verdict is numeric: 1 means correct.
    assert (
        result.verdict == 1
    ), "VerifierOperator did not return the expected verdict (1 for correct)."
    assert (
        result.explanation == "The answer is correct because..."
    ), "VerifierOperator did not return the expected explanation."
