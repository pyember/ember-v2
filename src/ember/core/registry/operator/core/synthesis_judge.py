from __future__ import annotations

from typing import List, Optional, Type, Any

from pydantic import Field

from ember.core.exceptions import MissingLMModuleError
from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel


class JudgeSynthesisInputs(EmberModel):
    """Input model for JudgeSynthesisOperator.

    Attributes:
        query (str): The query text.
        responses (List[str]): Aggregated ensemble responses.
    """

    query: str
    responses: List[str] = Field(..., description="Aggregated ensemble responses.")


class JudgeSynthesisOutputs(EmberModel):
    """Output model for JudgeSynthesisOperator.

    Attributes:
        final_answer (str): Synthetically combined best final answer.
        reasoning (str): Rationale behind the combined answer choice.
    """

    final_answer: str
    reasoning: str


class JudgeSynthesisSpecification(Specification):
    """Specification for JudgeSynthesisOperator defining the synthesis prompt."""

    prompt_template: str = (
        "We have multiple advisors who proposed different answers:\n"
        "{responses}\n"
        "Now, we want to synthesize a single best, final answer to:\n"
        "{query}\n"
        "Explain your reasoning concisely, then provide the single best final answer.\n"
        "Format:\n"
        "Reasoning: <your reasoning for synthesizing this answer in this way>\n"
        "Final Answer: <the single best answer>\n"
    )
    structured_output: Optional[Type[EmberModel]] = JudgeSynthesisOutputs
    input_model: Type[EmberModel] = JudgeSynthesisInputs


class JudgeSynthesisOperator(Operator[JudgeSynthesisInputs, JudgeSynthesisOutputs]):
    """Operator to synthesize a final answer and reasoning from multiple responses.
    
    This operator takes multiple responses and synthesizes them into a single,
    well-reasoned final answer. It's typically used after an ensemble operator
    to combine diverse perspectives.
    
    Examples:
        # Simple usage with model name
        synthesizer = JudgeSynthesisOperator(model="gpt-4")
        
        # With custom temperature for creative synthesis
        synthesizer = JudgeSynthesisOperator(model="gpt-4", temperature=0.7)
    """

    specification: Specification = JudgeSynthesisSpecification()

    def __init__(
        self,
        *,
        model: Any,  # Callable model
        **kwargs
    ) -> None:
        """Initialize the synthesis judge with a model.
        
        Args:
            model: A callable model that accepts a prompt and returns a response
            **kwargs: Additional parameters passed to the operator
        """
        super().__init__(**kwargs)
        self.model = model

    def forward(self, *, inputs: JudgeSynthesisInputs) -> JudgeSynthesisOutputs:
        if not self.model:
            raise MissingLMModuleError(
                "No model attached to JudgeSynthesisOperator."
            )

        rendered_prompt: str = self.specification.render_prompt(inputs=inputs)
        response = self.model(rendered_prompt)
        # Handle both old LMModule (returns string) and new models API (returns object with .text)
        response_text = response.text if hasattr(response, 'text') else response
        raw_output: str = response_text.strip()

        # Parse the response to extract reasoning and final answer
        final_answer = "Unknown"
        reasoning_lines: List[str] = []
        in_reasoning_section = False

        for line in raw_output.splitlines():
            line = line.strip()

            if line.startswith("Final Answer:"):
                final_answer = line.replace("Final Answer:", "").strip()
                break
            elif line.startswith("Reasoning:"):
                in_reasoning_section = True
                reasoning_part = line.replace("Reasoning:", "").strip()
                if reasoning_part:
                    reasoning_lines.append(reasoning_part)
            elif in_reasoning_section:
                reasoning_lines.append(line)

        reasoning = "\n".join(reasoning_lines)

        return {"final_answer": final_answer, "reasoning": reasoning}
