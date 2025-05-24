"""Operator Composition Example.

This example demonstrates clean operator composition using functional patterns.
It shows how to build a pipeline that:
1. Refines a user's question for clarity
2. Gets multiple LLM responses via ensemble
3. Aggregates responses to find consensus

The example uses functional composition for clean, testable code.

To run:
    uv run python src/ember/examples/operators/composition_example.py
"""

import logging
import time
from typing import Any, Callable, ClassVar, Dict, List, Type, TypeVar

from prettytable import PrettyTable

# ember API imports
from ember.api import models, non
from ember.api.operators import Operator, Specification, EmberModel
from ember.api.xcs import jit, execution_options

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


###############################################################################
# Composition Utilities
###############################################################################
def compose(f: Callable[[U], V], g: Callable[[T], U]) -> Callable[[T], V]:
    """Compose two functions: f ∘ g.

    Args:
        f: Function that takes output of g
        g: Function that takes initial input

    Returns:
        Composed function (f ∘ g)(x) = f(g(x))
    """

    def composed(x: T) -> V:
        return f(g(x))

    return composed


###############################################################################
# Custom Operators
###############################################################################
class QuestionRefinementInputs(EmberModel):
    """Input model for QuestionRefinement operator."""

    query: str


class QuestionRefinementOutputs(EmberModel):
    """Output model for QuestionRefinement operator."""

    refined_query: str


class QuestionRefinementSpecification(Specification):
    """Specification for QuestionRefinement operator."""

    input_model: Type[EmberModel] = QuestionRefinementInputs
    structured_output: Type[EmberModel] = QuestionRefinementOutputs
    prompt_template: str = (
        "You are an expert at refining questions to make them clearer and more precise.\n"
        "Please refine the following question:\n\n"
        "{query}\n\n"
        "Provide a refined version that is more specific and answerable."
    )


@jit()
class QuestionRefinement(Operator[QuestionRefinementInputs, QuestionRefinementOutputs]):
    """Operator that refines a user question to make it more precise."""

    specification: ClassVar[Specification] = QuestionRefinementSpecification()
    model_name: str
    temperature: float

    def __init__(self, *, model_name: str, temperature: float = 0.3) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def forward(self, *, inputs: QuestionRefinementInputs) -> QuestionRefinementOutputs:
        prompt = self.specification.render_prompt(inputs=inputs)

        try:
            # Use models API directly
            response = models(self.model_name, prompt, temperature=self.temperature)

            # Get text from response
            refined_query = response.text.strip() if hasattr(response, 'text') else str(response).strip()

            return QuestionRefinementOutputs(refined_query=refined_query)
        except Exception as e:
            # Graceful error handling for model failures
            logging.warning(f"Error invoking model {self.model_name}: {str(e)}")
            # Return a fallback refinement that doesn't fail the pipeline
            fallback_query = f"Refined: {inputs.query}"
            return QuestionRefinementOutputs(refined_query=fallback_query)


###############################################################################
# Pipeline Pattern 1: Functional Composition
###############################################################################
def create_functional_pipeline(*, model_name: str) -> Callable[[Dict[str, Any]], Any]:
    """Create a pipeline using functional composition.

    Args:
        model_name: Name of the LLM to use

    Returns:
        A callable pipeline function
    """
    # Create individual operators
    refiner = QuestionRefinement(model_name=model_name)
    ensemble = non.UniformEnsemble(num_units=3, model_name=model_name, temperature=0.7)
    aggregator = non.MostCommon()

    # Create adapter functions for functional composition
    def adapt_refiner(inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = refiner(inputs=QuestionRefinementInputs(**inputs))
        return {"query": result.refined_query}

    def adapt_ensemble(inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = ensemble(inputs=inputs)
        return {"query": inputs["query"], "responses": result["responses"]}

    # Compose the pipeline
    pipeline = compose(aggregator, compose(adapt_ensemble, adapt_refiner))

    return pipeline


###############################################################################
# Pipeline Pattern 2: Container Class with Nested Operators
###############################################################################
class PipelineInput(EmberModel):
    """Input for NestedPipeline."""

    query: str


class PipelineOutput(EmberModel):
    """Output for NestedPipeline."""

    final_answer: str


class PipelineSpecification(Specification):
    """Specification for NestedPipeline."""

    input_model: Type[EmberModel] = PipelineInput
    structured_output: Type[EmberModel] = PipelineOutput


@jit(sample_input={"query": "What is the speed of light?"})
class NestedPipeline(Operator[PipelineInput, PipelineOutput]):
    """Pipeline implemented as a container class with nested operators."""

    specification: ClassVar[Specification] = PipelineSpecification()
    refiner: QuestionRefinement
    ensemble: non.UniformEnsemble
    aggregator: non.MostCommon

    def __init__(self, *, model_name: str) -> None:
        self.refiner = QuestionRefinement(model_name=model_name)
        self.ensemble = non.UniformEnsemble(
            num_units=3, model_name=model_name, temperature=0.7
        )
        self.aggregator = non.MostCommon()

    def forward(self, *, inputs: PipelineInput) -> PipelineOutput:
        # Step 1: Refine the question
        refined = self.refiner(inputs=QuestionRefinementInputs(query=inputs.query))

        # Step 2: Generate ensemble of answers
        ensemble_result = self.ensemble(inputs={"query": refined.refined_query})

        # Step 3: Aggregate results
        final_result = self.aggregator(
            inputs={
                "query": refined.refined_query,
                "responses": ensemble_result["responses"],
            }
        )

        return PipelineOutput(final_answer=final_result["final_answer"])


###############################################################################
# Pipeline Pattern 3: Sequential Chaining
###############################################################################
def create_sequential_pipeline(*, model_name: str) -> Callable[[Dict[str, Any]], Any]:
    """Create a pipeline by explicitly chaining operators.

    Args:
        model_name: Name of the LLM to use

    Returns:
        A callable pipeline function
    """
    # Create individual operators
    refiner = QuestionRefinement(model_name=model_name)
    ensemble = non.UniformEnsemble(num_units=3, model_name=model_name, temperature=0.7)
    aggregator = non.MostCommon()

    # Create the chained function
    def pipeline(inputs: Dict[str, Any]) -> Any:
        refined = refiner(inputs=QuestionRefinementInputs(**inputs))
        ensemble_result = ensemble(inputs={"query": refined.refined_query})
        final_result = aggregator(
            inputs={
                "query": refined.refined_query,
                "responses": ensemble_result["responses"],
            }
        )
        return final_result

    return pipeline


###############################################################################
# Main Demonstration
###############################################################################
def main() -> None:
    """Run demonstration of different composition patterns."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define configuration parameters
    model_name: str = "openai:gpt-3.5-turbo"

    # Create pipelines using different patterns
    functional_pipeline = create_functional_pipeline(model_name=model_name)
    nested_pipeline = NestedPipeline(model_name=model_name)
    sequential_pipeline = create_sequential_pipeline(model_name=model_name)

    # List of questions to process
    questions: List[str] = [
        "How does gravity work?",
        "Tell me about the history of Rome",
        "What's the difference between DNA and RNA?",
    ]

    # Prepare table for results comparison
    table = PrettyTable()
    table.field_names = ["Pipeline", "Time (s)", "Result"]
    table.align = "l"

    # Process questions with each pipeline
    print("\n=== Functional Composition Pipeline ===")
    for question in questions[:1]:  # Use first question only for brevity
        print(f"\nProcessing: {question}")
        start_time = time.perf_counter()
        result = functional_pipeline({"query": question})
        elapsed = time.perf_counter() - start_time

        # Show details of pipeline execution
        print(f'Original query: "{question}"')
        final_answer = result.get("final_answer", "")
        if isinstance(result, dict) and "refined_query" in result:
            print(f"Refined query: \"{result['refined_query']}\"")
        print(
            f'Final answer: "{final_answer[:150]}..."'
            if len(final_answer) > 150
            else f'Final answer: "{final_answer}"'
        )
        print(f"Time: {elapsed:.4f}s")

        # Store in table
        table.add_row(
            [
                "Functional",
                f"{elapsed:.4f}",
                (final_answer[:50] + "..." if len(final_answer) > 50 else final_answer),
            ]
        )

    print("\n=== Nested Pipeline ===")
    for question in questions[:1]:
        print(f"\nProcessing: {question}")
        start_time = time.perf_counter()
        result = nested_pipeline(inputs={"query": question})
        elapsed = time.perf_counter() - start_time

        # Show details of pipeline execution
        final_answer = (
            result.final_answer if hasattr(result, "final_answer") else str(result)
        )
        print(
            f'Final answer: "{final_answer[:150]}..."'
            if len(final_answer) > 150
            else f'Final answer: "{final_answer}"'
        )
        print(f"Time: {elapsed:.4f}s")

        # Store in table
        table.add_row(
            [
                "Nested",
                f"{elapsed:.4f}",
                (final_answer[:50] + "..." if len(final_answer) > 50 else final_answer),
            ]
        )

    print("\n=== Sequential Pipeline ===")
    for question in questions[:1]:
        print(f"\nProcessing: {question}")
        start_time = time.perf_counter()
        result = sequential_pipeline({"query": question})
        elapsed = time.perf_counter() - start_time

        # Show details of pipeline execution
        final_answer = (
            result.get("final_answer", "") if isinstance(result, dict) else str(result)
        )
        print(
            f'Final answer: "{final_answer[:150]}..."'
            if len(final_answer) > 150
            else f'Final answer: "{final_answer}"'
        )
        print(f"Time: {elapsed:.4f}s")

        # Store in table
        table.add_row(
            [
                "Sequential",
                f"{elapsed:.4f}",
                (final_answer[:50] + "..." if len(final_answer) > 50 else final_answer),
            ]
        )

    # Demonstrate execution options with the nested pipeline
    print("\n=== Nested Pipeline with Sequential Execution ===")
    with execution_options(scheduler="sequential"):
        for question in questions[:1]:
            print(f"\nProcessing: {question}")
            start_time = time.perf_counter()
            result = nested_pipeline(inputs={"query": question})
            elapsed = time.perf_counter() - start_time

            # Show details of pipeline execution
            final_answer = (
                result.final_answer if hasattr(result, "final_answer") else str(result)
            )
            print(
                f'Final answer: "{final_answer[:150]}..."'
                if len(final_answer) > 150
                else f'Final answer: "{final_answer}"'
            )
            print(f"Time: {elapsed:.4f}s")
            print("Execution mode: Sequential scheduler")

            # Store in table
            table.add_row(
                [
                    "Nested (Sequential)",
                    f"{elapsed:.4f}",
                    (
                        final_answer[:50] + "..."
                        if len(final_answer) > 50
                        else final_answer
                    ),
                ]
            )

    # Display performance comparison
    print("\n=== Performance Comparison ===")
    print(table)

    print("\n=== Pipeline Pattern Comparison ===")
    print("1. Functional Composition:")
    print("   • Advantages: Clean separation of concerns, explicit data flow")
    print("   • Use cases: When components need to be reused separately")

    print("\n2. Nested Pipeline:")
    print("   • Advantages: Benefits from JIT optimization, cleaner code structure")
    print("   • Use cases: Complex pipelines where performance matters")

    print("\n3. Sequential Pipeline:")
    print("   • Advantages: Simplicity, flexibility for custom logic")
    print("   • Use cases: Prototyping or simpler workflows")


if __name__ == "__main__":
    main()
