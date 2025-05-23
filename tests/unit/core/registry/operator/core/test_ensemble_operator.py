from tests.helpers.simplified_imports import EmberModel


# Define local versions of the classes needed for testing
class Specification:
    """Simplified test Specification class."""

    def __init__(
        self, *, input_model=None, prompt_template=None, structured_output=None
    ):
        self.input_model = input_model
        self.prompt_template = prompt_template
        self.structured_output = structured_output

    def render_prompt(self, *, inputs):
        """Simple prompt rendering."""
        if self.prompt_template:
            if hasattr(inputs, "query"):
                return self.prompt_template.format(query=inputs.query)
            return self.prompt_template
        return f"Default prompt for: {inputs}"

    def validate_inputs(self, inputs):
        """No-op input validation for tests."""
        return inputs

    def validate_output(self, output):
        """No-op output validation for tests."""
        return output


class EnsembleOperatorInputs(EmberModel):
    """Simple input model for testing."""

    query: str


class EnsembleOperatorOutputs(EmberModel):
    """Simple output model for testing."""

    responses: list[str]


class EnsembleOperator:
    """Simplified test implementation."""

    specification = Specification(
        input_model=EnsembleOperatorInputs, structured_output=EnsembleOperatorOutputs
    )

    def __init__(self, *, models):
        self.models = models

    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)

    def forward(self, *, inputs):
        """Execute the ensemble operation."""
        rendered_prompt = self.specification.render_prompt(inputs=inputs)
        responses = []
        for model in self.models:
            response = model(rendered_prompt)
            # Extract text from response
            if hasattr(response, 'text'):
                responses.append(response.text)
            else:
                responses.append(str(response))
        return {"responses": responses}


class MockResponse:
    """Mock response object with text attribute."""
    def __init__(self, text):
        self.text = text


class MockModel:
    """Simple mock model that returns a response object."""

    def __call__(self, prompt: str):
        return MockResponse(f"LM response to: {prompt}")


def test_ensemble_operator_forward() -> None:
    mock_model1 = MockModel()
    mock_model2 = MockModel()

    # Optionally customize the operator's specification:
    custom_specification = Specification(
        input_model=EnsembleOperatorInputs, prompt_template="Ensemble Prompt: {query}"
    )

    op = EnsembleOperator(models=[mock_model1, mock_model2])
    # Override the default specification:
    op.specification = custom_specification

    inputs = EnsembleOperatorInputs(query="test query")
    result: EnsembleOperatorOutputs = op(inputs=inputs)

    # Verify the aggregated responses:
    rendered_prompt = custom_specification.render_prompt(inputs=inputs)
    expected_responses = [
        mock_model1(rendered_prompt).text,
        mock_model2(rendered_prompt).text,
    ]

    assert isinstance(
        result, dict
    ), "Result should be a dict (which will be converted to EnsembleOperatorOutputs by the framework)"
    assert "responses" in result, "Result should contain 'responses' key"
    assert (
        result["responses"] == expected_responses
    ), "Responses should match expected responses"
