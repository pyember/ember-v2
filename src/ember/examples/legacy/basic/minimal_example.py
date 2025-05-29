"""Minimal Example

This module demonstrates the simplest possible Ember operator.

This example illustrates how the Ember `Operator` construct provides a minimal
foundation that can be used for language-model transformations or to wrap
arbitrary function logic.

To run:
    uv run python src/ember/examples/basic/minimal_example.py
    uv run python src/ember/examples/basic/minimal_example.py --verbose
    uv run python src/ember/examples/basic/minimal_example.py --quiet
"""

from typing import Any, Dict, List, Optional, Type

from ember.api.operators import Operator, Specification, EmberModel, Field
from ember.core.utils.output import print_header, print_summary, print_success
from ember.core.utils.verbosity import create_argument_parser, setup_verbosity_from_args, vprint


class MinimalInput(EmberModel):
    """Input model for MinimalOperator.

    Attributes:
        value: The integer value to be processed
        options: Optional configuration parameters
    """

    value: int = Field(description="The integer value to be processed")
    options: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional configuration parameters"
    )


class MinimalOutput(EmberModel):
    """Output model for MinimalOperator.

    Attributes:
        value: The processed value
        steps: List of processing steps applied
    """

    value: int = Field(description="The processed value")
    steps: List[str] = Field(
        default_factory=list, description="List of processing steps applied"
    )

    def summarize(self) -> str:
        """Get a summary of the processing result.

        Returns:
            A formatted summary string
        """
        step_text = "\n".join(f"- {step}" for step in self.steps)
        return f"Result: {self.value}\nSteps applied:\n{step_text}"


class MinimalSpecification(Specification):
    """Specification for MinimalOperator.

    Defines the input/output contract that the operator fulfills.
    """

    input_model: Type[EmberModel] = MinimalInput
    structured_output: Type[EmberModel] = MinimalOutput


class MinimalOperator(Operator[MinimalInput, MinimalOutput]):
    """A minimal operator that performs configurable numeric operations.

    This operator demonstrates the clean patterns for creating Ember operators
    with proper typing, immutability, and clean interfaces.

    Attributes:
        specification: The operator's input/output contract
        increment: The value to add to the input
        multiplier: The value to multiply by after incrementing
    """

    # Class-level specification (immutable)
    specification: Specification = MinimalSpecification()

    # Instance attributes with type hints
    increment: int
    multiplier: int

    def __init__(self, *, increment: int = 1, multiplier: int = 1) -> None:
        """Initialize with transformation parameters.

        Args:
            increment: The value to add to the input
            multiplier: The value to multiply by after incrementing
        """
        self.increment = increment
        self.multiplier = multiplier

    def forward(self, *, inputs: MinimalInput) -> MinimalOutput:
        """Process the input value with configured transformations.

        Args:
            inputs: Validated input data containing the value to process

        Returns:
            Structured output containing the processed value and steps
        """
        # Track processing steps
        steps = []

        # Apply increment
        result = inputs.value + self.increment
        steps.append(f"Added {self.increment} to {inputs.value} = {result}")

        # Apply multiplier
        if self.multiplier != 1:
            prev = result
            result *= self.multiplier
            steps.append(f"Multiplied {prev} by {self.multiplier} = {result}")

        # Apply any custom options if provided
        if inputs.options:
            if "square" in inputs.options and inputs.options["square"]:
                prev = result
                result = result**2
                steps.append(f"Squared {prev} = {result}")

            if "add" in inputs.options:
                value = inputs.options["add"]
                prev = result
                result += value
                steps.append(f"Added {value} to {prev} = {result}")

        # Return structured output
        return MinimalOutput(value=result, steps=steps)


def main() -> None:
    """Example demonstrating the simplified XCS architecture."""
    """Run a simple demonstration of the MinimalOperator."""
    # Set up argument parser with verbosity controls
    parser = create_argument_parser("Demonstrate the MinimalOperator")
    args = parser.parse_args()
    setup_verbosity_from_args(args)
    
    print_header("Minimal Operator Example")

    # Create the operator
    op = MinimalOperator(increment=5, multiplier=2)
    vprint("Created MinimalOperator with increment=5, multiplier=2")

    # Create input with basic value
    basic_input = MinimalInput(value=10)

    # Process with only the basic configuration
    basic_result = op(inputs=basic_input)

    # Display results using clean formatting
    print_summary({
        "Input": basic_input.value,
        "Result": basic_result.value,
        "Operations": len(basic_result.steps)
    }, title="Basic Example")
    
    # Show steps in verbose mode
    if args.verbose:
        print("Processing steps:")
        for i, step in enumerate(basic_result.steps, 1):
            print(f"  {i}. {step}")
        print()

    # Create input with advanced options
    advanced_input = MinimalInput(value=7, options={"square": True, "add": 3})

    # Process with advanced options
    advanced_result = op(inputs=advanced_input)

    print_summary({
        "Input": advanced_input.value,
        "Options": str(advanced_input.options),
        "Result": advanced_result.value,
        "Operations": len(advanced_result.steps)
    }, title="Advanced Example")
    
    # Show steps in verbose mode
    if args.verbose:
        print("Processing steps:")
        for i, step in enumerate(advanced_result.steps, 1):
            print(f"  {i}. {step}")
        print()

    # Alternative invocation patterns
    vprint("\nTesting alternative invocation patterns...")
    
    # Using dict input
    dict_result = op(inputs={"value": 3})
    vprint(f"Dict input result: {dict_result.value}")

    # Using keyword arguments
    kwargs_result = op(value=4)
    vprint(f"Keyword args result: {kwargs_result.value}")

    # Final summary
    if not args.quiet:
        print("\nFinal Summary:")
        print(advanced_result.summarize())
    
    print_success("Example completed successfully!")


if __name__ == "__main__":
    main()
