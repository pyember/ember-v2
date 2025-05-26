"""
Compact NON Graph Notation Example

This example demonstrates Ember's compact graph notation for creating
complex operator graphs with minimal syntax, following the same design
principles favored by Jeff Dean and Sanjay Ghemawat: clean, precise,
orthogonal abstractions with maximal composability.
"""

from ember.api import non


# Create custom operator registry for demonstrations
def create_custom_registry():
    """Create a custom operator registry with extended operator types."""
    # Create a new registry with standard operators
    registry = non.OpRegistry.create_standard_registry()

    # Register a custom operator type
    registry.register(
        "CE",  # Custom Ensemble
        lambda count, model, temp: non.Sequential(
            operators=[
                non.UniformEnsemble(
                    num_units=count, model_name=model, temperature=temp
                ),
                non.MostCommon(),  # Automatically add MostCommon to every ensemble
            ]
        ))

    return registry


def main() -> None:
    """Example demonstrating the simplified XCS architecture."""
    """Demonstrates various ways to build NON pipelines using compact notation."""

    # Example 1: Basic ensemble with judge
    print("\n==== Example 1: Basic Ensemble + Judge ====")

    # Using compact notation
    compact_pipeline = non.build_graph(
        [
            "3:E:gpt-4o:0.7",  # Ensemble with 3 GPT-4o instances at temp=0.7
            "1:J:claude-3-5-sonnet:0.0",  # Judge using Claude with temp=0
        ]
    )

    # Equivalent pipeline using standard API
    standard_pipeline = non.Sequential(
        operators=[
            non.UniformEnsemble(num_units=3, model_name="gpt-4o", temperature=0.7),
            non.JudgeSynthesis(model_name="claude-3-5-sonnet", temperature=0.0)]
    )

    print("Compact notation pipeline created.")
    print("Both pipelines are functionally equivalent.")

    # Example 2: Complex verification pipeline
    print("\n==== Example 2: Complex Verification Pipeline ====")

    verification_pipeline = non.build_graph(
        [
            "3:E:gpt-4o:0.7",  # Generate 3 candidate answers
            "1:J:claude-3-5-sonnet:0.0",  # Synthesize into one answer
            "1:V:gpt-4o:0.0",  # Verify the synthesized answer
        ]
    )

    print("Verification pipeline created.")

    # Example 3: Nested pipeline structure
    print("\n==== Example 3: Nested Architecture ====")

    # Build a nested architecture similar to the SubNetwork/NestedNetwork example
    nested_pipeline = non.build_graph(
        [
            # First branch - GPT ensemble + verification
            ["3:E:gpt-4o:0.7", "1:V:gpt-4o:0.0"],
            # Second branch - Claude ensemble + verification
            ["3:E:claude-3-5-haiku:0.7", "1:V:claude-3-5-haiku:0.0"],
            # Final synthesis judge
            "1:J:claude-3-5-sonnet:0.0"]
    )

    print("Nested architecture pipeline created.")

    # Example 4: Recursive References
    print("\n==== Example 4: Recursive Component References ====")

    # Define component map with nested references
    component_map = {
        # Basic building blocks
        "gpt_ensemble": "3:E:gpt-4o:0.7",
        "claude_ensemble": "3:E:claude-3-5-haiku:0.7",
        # Reference other components
        "verification_pipeline": ["$gpt_ensemble", "1:V:gpt-4o:0.0"],
        # Complex compositions referencing other references
        "double_verification": [
            "$verification_pipeline",  # First verification
            ["$claude_ensemble", "1:V:claude-3-5-haiku:0.0"],  # Second verification
        ],
    }

    # Create a complex graph with multiple levels of references
    reference_pipeline = non.build_graph(
        [
            "$double_verification",  # Two parallel verification branches
            "1:J:claude-3-5-sonnet:0.0",  # Final synthesis
        ],
        components=component_map)

    print("Recursive reference pipeline created.")

    # Example 5: Custom Operator Types
    print("\n==== Example 5: Custom Operator Types ====")

    # Create a custom registry with extended operator types
    custom_registry = create_custom_registry()

    # Use the custom operator type in a specification with the custom registry
    custom_pipeline = non.build_graph(
        [
            "5:CE:gpt-4o:0.7",  # Custom ensemble with built-in MostCommon
            "1:J:claude-3-5-sonnet:0.0",  # Judge to synthesize
        ],
        type_registry=custom_registry)

    print("Custom operator type pipeline created using custom type registry.")

    # Example 6: Exact Structural Match to NestedNetwork in example_architectures.py
    print("\n==== Example 6: NestedNetwork Equivalent ====")

    # Define the SubNetwork component exactly as in example_architectures.py
    # SubNetwork: ensemble → verifier pipeline (forward flow)
    subnetwork = [
        "2:E:gpt-4o:0.0",  # Ensemble with 2 identical models at temp=0
        "1:V:gpt-4o:0.0",  # Verify first response from ensemble
    ]

    # Create components map with the SubNetwork component
    component_map = {
        "sub": subnetwork,  # Reusable SubNetwork definition
    }

    # Build the NestedNetwork equivalent using the exact structure from example_architectures.py:
    # - Two parallel SubNetwork instances (identical configuration)
    # - Judge to synthesize results from both branches
    nested_network = non.build_graph(
        [
            # Two parallel branches, both using the same SubNetwork structure
            "$sub",  # First branch: SubNetwork instance
            "$sub",  # Second branch: SubNetwork instance
            "1:J:gpt-4o:0.0",  # Final Judge synthesizes results from both branches
        ],
        components=component_map)

    print(
        "NestedNetwork equivalent created with identical structure to example_architectures.py"
    )
    # Example query to demonstrate usage
    print("\n==== Using the Pipeline ====")
    print("Example query: 'What causes the northern lights?'")
    print("(Not executing to avoid actual API calls)")
    print("Usage would be:")
    print("  result = pipeline(query='What causes the northern lights?')")
    print("  print(f'Answer: {result.answer}')")


if __name__ == "__main__":
    main()
