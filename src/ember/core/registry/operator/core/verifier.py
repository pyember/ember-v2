from __future__ import annotations

from typing import Optional, Type, Any

from pydantic import Field

from ember.core.exceptions import MissingLMModuleError
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel


class VerifierOperatorInputs(EmberModel):
    """Input model for VerifierOperator.

    Attributes:
        query (str): The query string.
        candidate_answer (str): The candidate answer to verify.
    """

    query: str
    candidate_answer: str


class VerifierOperatorOutputs(EmberModel):
    """Typed output model for VerifierOperator.

    Attributes:
        verdict (int): 1 if correct, 0 if incorrect.
        explanation (str): Explanation for the verdict.
        revised_answer (Optional[str]): Optional corrected answer.
    """

    verdict: int = Field(..., description="1 for correct, 0 for incorrect")
    explanation: str
    revised_answer: Optional[str]


class VerifierSpecification(Specification):
    """Specification for VerifierOperator defining the verification prompt."""

    prompt_template: str = (
        "You are a verifier of correctness.\n"
        "Question: {query}\n"
        "Candidate Answer: {candidate_answer}\n"
        "Please decide if this is correct. Provide:\n"
        "Verdict: <1 for correct, 0 for incorrect>\n"
        "Explanation: <Your reasoning>\n"
        "Revised Answer (optional): <If you can and want to provide a corrected version>\n"
    )
    input_model: Type[EmberModel] = VerifierOperatorInputs
    structured_output: Type[EmberModel] = VerifierOperatorOutputs


class VerifierOperator(Operator[VerifierOperatorInputs, VerifierOperatorOutputs]):
    """Operator to verify a candidate answer and optionally suggest revisions.
    
    This operator now supports both string model IDs and ModelBinding instances,
    optimizing for the common case of just passing a model name.
    
    Examples:
        # Simple usage with model name
        verifier = VerifierOperator(model="gpt-4")
        
        # With custom parameters
        verifier = VerifierOperator(model="gpt-4", temperature=0.3)
        
        # With pre-configured binding
        custom_model = models.instance("gpt-4", temperature=0.1, max_tokens=500)
        verifier = VerifierOperator(model=custom_model)
    """

    specification: Specification = VerifierSpecification()

    def __init__(
        self, 
        *, 
        model: Any,  # Callable model
        **kwargs
    ) -> None:
        """Initialize the verifier with a model.
        
        Args:
            model: A callable model that accepts a prompt and returns a response
            **kwargs: Additional parameters passed to the operator
        """
        super().__init__(**kwargs)
        self.model = model

    def forward(self, *, inputs: VerifierOperatorInputs) -> VerifierOperatorOutputs:
        if not self.model:
            raise MissingLMModuleError("No model attached to VerifierOperator.")
        
        rendered_prompt: str = self.specification.render_prompt(inputs=inputs)
        response = self.model(rendered_prompt)
        # Handle both old LMModule (returns string) and new models API (returns object with .text)
        response_text = response.text if hasattr(response, 'text') else response
        raw_output: str = response_text.strip()

        # Initialize default values
        verdict = 0
        explanation = ""
        revised_answer = None

        # Process each line with more robust parsing
        in_explanation_section = False
        in_revised_answer_section = False
        explanation_lines = []
        revised_answer_lines = []

        for line in raw_output.split("\n"):
            clean_line = line.strip()

            # Parse verdict
            if clean_line.startswith("Verdict:"):
                verdict_value = clean_line.replace("Verdict:", "").strip()
                try:
                    verdict_num = int(verdict_value)
                    verdict = 1 if verdict_num == 1 else 0
                except ValueError:
                    # Handle text verdicts like "correct" or "incorrect"
                    verdict = 1 if "correct" in verdict_value.lower() else 0

            # Parse explanation
            elif clean_line.startswith("Explanation:"):
                in_explanation_section = True
                in_revised_answer_section = False
                explanation_part = clean_line.replace("Explanation:", "").strip()
                if explanation_part:
                    explanation_lines.append(explanation_part)

            # Parse revised answer
            elif clean_line.startswith("Revised Answer:"):
                in_explanation_section = False
                in_revised_answer_section = True
                revised_part = clean_line.replace("Revised Answer:", "").strip()
                if revised_part:
                    revised_answer_lines.append(revised_part)

            # Continue parsing multi-line sections
            elif in_explanation_section:
                explanation_lines.append(clean_line)
            elif in_revised_answer_section:
                revised_answer_lines.append(clean_line)

        # Finalize parsing
        if explanation_lines:
            explanation = "\n".join(explanation_lines)
        if revised_answer_lines:
            revised_answer = "\n".join(revised_answer_lines)

        # Return as dictionary - the operator __call__ will properly convert to model
        # This maintains flexibility while ensuring type safety
        return {
            "verdict": verdict,
            "explanation": explanation,
            "revised_answer": revised_answer,
        }
