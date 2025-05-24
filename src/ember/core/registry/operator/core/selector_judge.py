from __future__ import annotations

from typing import List, Optional, Type, Any

from pydantic import Field

from ember.core.exceptions import MissingLMModuleError
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types import EmberModel


class SelectorJudgeInputs(EmberModel):
    """Input model for SelectorJudgeOperator."""

    query: str
    responses: List[str] = Field(..., description="Aggregated ensemble responses.")


class SelectorJudgeOutputs(EmberModel):
    """Output model for SelectorJudgeOperator."""

    final_answer: str


class SelectorJudgeOperatorOutputs(EmberModel):
    """Output model for SelectorJudgeOperator.

    Attributes:
        final_answer (str): The selected best final answer.
        reasoning (str): Reasoning behind the selection.
    """

    final_answer: str
    reasoning: str


class SelectorJudgeSpecification(Specification):
    """Specification for SelectorJudgeOperator defining the synthesis prompt."""

    prompt_template: str = (
        "We have multiple advisors who proposed different answers:\n"
        "{responses}\n"
        "Now, we want to select the best, final answer to:\n"
        "{query}\n"
        "Explain your reasoning concisely, then provide the single best final answer.\n"
        "Format:\n"
        "Reasoning: <your reasoning for selecting this answer>\n"
        "Final Answer: <the single best answer>\n"
    )
    structured_output: Optional[Type[EmberModel]] = SelectorJudgeOutputs
    input_model: Type[EmberModel] = SelectorJudgeInputs


class SelectorJudgeOperator(
    Operator[SelectorJudgeInputs, SelectorJudgeOperatorOutputs]
):
    """Operator to select the best, final answer from multiple responses.
    
    Unlike the synthesis judge which creates a new answer, this operator
    selects one of the provided answers as the best option.
    
    Examples:
        # Simple usage with model name
        selector = SelectorJudgeOperator(model="gpt-4")
        
        # With lower temperature for more consistent selection
        selector = SelectorJudgeOperator(model="gpt-4", temperature=0.3)
    """

    specification: Specification = SelectorJudgeSpecification()

    def __init__(
        self,
        *,
        model: Any,  # Callable model
        **kwargs
    ) -> None:
        """Initialize the selector judge with a model.
        
        Args:
            model: A callable model that accepts a prompt and returns a response
            **kwargs: Additional parameters passed to the operator
        """
        super().__init__(**kwargs)
        self.model = model

    def forward(self, *, inputs: SelectorJudgeInputs) -> SelectorJudgeOperatorOutputs:
        rendered_prompt: str = self.specification.render_prompt(inputs=inputs)
        if not self.model:
            raise MissingLMModuleError(
                "No model attached to SelectorJudgeOperator."
            )
        response = self.model(rendered_prompt)
        # Handle both old LMModule (returns string) and new models API (returns object with .text)
        response_text = response.text if hasattr(response, 'text') else response
        raw_output: str = response_text.strip()

        # Parse the output for the final answer and reasoning, respectively
        final_answer: str = "Unknown"
        reasoning_lines: List[str] = []
        for line in raw_output.split("\n"):
            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            reasoning_lines.append(line)
        reasoning: str = "\n".join(reasoning_lines)
        return {"final_answer": final_answer, "reasoning": reasoning}
