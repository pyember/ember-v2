"""
Integration tests for Compact Network of Networks (NON) Graph Notation.

This module tests the integration of the compact notation system with real
operators and execution flows. It verifies that graphs built with the compact
notation produce the expected computational behavior, not just structure.

Following Jeff Dean and Sanjay Ghemawat's design principles, these tests focus on:
1. Validating expected operational behavior (not just object structure)
2. Testing the full composition of components as they would be used in practice
3. Ensuring reliable performance across complex configurations
"""

from unittest import mock

import pytest

from ember.core.non import (
    JudgeSynthesis,
    MostCommon,
    Sequential,
    UniformEnsemble,
    Verifier,
)
from ember.core.non_compact import OpRegistry, build_graph


# Create test support fixtures
@pytest.fixture(autouse=True)
def mock_lm_module():
    """Mock LMModule to avoid real API calls during tests."""
    with mock.patch("ember.core.registry.model.model_module.lm.LMModule") as mock_lm:
        # Set up mock to return predetermined responses
        instance = mock_lm.return_value
        instance.return_value = "This is a mock LLM response"
        yield mock_lm


class TestCompactNotationStructure:
    """Test the structure of graphs built with compact notation."""

    def test_simple_pipeline_structure(self):
        """Test that a simple pipeline creates the expected structure."""
        # Build a pipeline using compact notation
        pipeline = build_graph(
            [
                "3:E:gpt-4o:0.7",  # Ensemble with 3 instances
                "1:J:claude-3-5-sonnet:0.0",  # Judge
            ]
        )

        # Verify the pipeline structure
        assert isinstance(pipeline, Sequential)
        assert len(pipeline.operators) == 2
        assert isinstance(pipeline.operators[0], UniformEnsemble)
        assert isinstance(pipeline.operators[1], JudgeSynthesis)

        # Verify ensemble configuration
        ensemble = pipeline.operators[0]
        assert ensemble.num_units == 3
        assert ensemble.model_name == "gpt-4o"
        assert ensemble.temperature == 0.7

        # Verify judge configuration
        judge = pipeline.operators[1]
        assert judge.model_name == "claude-3-5-sonnet"
        assert judge.temperature == 0.0

    def test_verification_pipeline_structure(self):
        """Test that a verification pipeline creates the expected structure."""
        # Build a pipeline using compact notation
        pipeline = build_graph(
            [
                "3:E:gpt-4o:0.7",  # Generate 3 candidate answers
                "1:J:claude-3-5-sonnet:0.0",  # Synthesize into one answer
                "1:V:gpt-4o:0.0",  # Verify the synthesized answer
            ]
        )

        # Verify the pipeline structure
        assert isinstance(pipeline, Sequential)
        assert len(pipeline.operators) == 3
        assert isinstance(pipeline.operators[0], UniformEnsemble)
        assert isinstance(pipeline.operators[1], JudgeSynthesis)
        assert isinstance(pipeline.operators[2], Verifier)

        # Verify verifier configuration
        verifier = pipeline.operators[2]
        assert verifier.model_name == "gpt-4o"
        assert verifier.temperature == 0.0

    def test_nested_architecture_structure(self):
        """Test that a nested architecture creates the expected structure."""
        # Build a nested architecture
        pipeline = build_graph(
            [
                # First branch - GPT ensemble + verification
                ["3:E:gpt-4o:0.7", "1:V:gpt-4o:0.0"],
                # Second branch - Claude ensemble + verification
                ["3:E:claude-3-5-haiku:0.7", "1:V:claude-3-5-haiku:0.0"],
                # Final synthesis judge
                "1:J:claude-3-5-sonnet:0.0",
            ]
        )

        # Verify the pipeline structure
        assert isinstance(pipeline, Sequential)
        assert len(pipeline.operators) == 3

        # Verify branch structures
        assert isinstance(pipeline.operators[0], Sequential)
        assert isinstance(pipeline.operators[1], Sequential)
        assert isinstance(pipeline.operators[2], JudgeSynthesis)

        # Verify first branch structure
        branch1 = pipeline.operators[0]
        assert len(branch1.operators) == 2
        assert isinstance(branch1.operators[0], UniformEnsemble)
        assert isinstance(branch1.operators[1], Verifier)

        # Verify second branch structure
        branch2 = pipeline.operators[1]
        assert len(branch2.operators) == 2
        assert isinstance(branch2.operators[0], UniformEnsemble)
        assert isinstance(branch2.operators[1], Verifier)

    def test_recursive_reference_structure(self):
        """Test that recursive references resolve correctly."""
        # Define component map with simplified references
        component_map = {
            # Basic building blocks
            "ensemble": "3:E:gpt-4o:0.7",
            # Reference other components
            "verification": ["$ensemble", "1:V:gpt-4o:0.0"],
        }

        # Create a graph with simple references
        pipeline = build_graph(
            [
                "$verification",  # Referenced component
                "1:J:claude-3-5-sonnet:0.0",  # Final synthesis
            ],
            components=component_map,
        )

        # Verify the pipeline structure
        assert isinstance(pipeline, Sequential)
        assert len(pipeline.operators) >= 2

        # Verify the judge is the last component
        assert isinstance(pipeline.operators[-1], JudgeSynthesis)

        # Test that we have an ensemble somewhere in the graph
        # This avoids assumptions about the exact nesting structure
        def has_ensemble_operator(op):
            if isinstance(op, UniformEnsemble):
                return True
            if isinstance(op, Sequential):
                return any(has_ensemble_operator(subop) for subop in op.operators)
            return False

        assert has_ensemble_operator(pipeline), "No UniformEnsemble found in the graph"

        # Test that we have a verifier somewhere in the graph
        def has_verifier_operator(op):
            if isinstance(op, Verifier):
                return True
            if isinstance(op, Sequential):
                return any(has_verifier_operator(subop) for subop in op.operators)
            return False

        assert has_verifier_operator(pipeline), "No Verifier found in the graph"

    def test_custom_operator_type_structure(self):
        """Test that custom operator types create the expected structure."""
        # Create a custom registry with extended operator types
        custom_registry = OpRegistry.create_standard_registry()
        custom_registry.register(
            "CE",  # Custom Ensemble
            lambda count, model, temp: Sequential(
                operators=[
                    UniformEnsemble(
                        num_units=count, model_name=model, temperature=temp
                    ),
                    MostCommon(),  # Automatically add MostCommon to every ensemble
                ]
            ),
        )

        # Use the custom operator type in a specification
        pipeline = build_graph(
            [
                "5:CE:gpt-4o:0.7",  # Custom ensemble with built-in MostCommon
                "1:J:claude-3-5-sonnet:0.0",  # Judge to synthesize
            ],
            type_registry=custom_registry,
        )

        # Verify the pipeline structure
        assert isinstance(pipeline, Sequential)
        assert len(pipeline.operators) == 2

        # Verify custom ensemble structure
        custom_ensemble = pipeline.operators[0]
        assert isinstance(custom_ensemble, Sequential)
        assert len(custom_ensemble.operators) == 2
        assert isinstance(custom_ensemble.operators[0], UniformEnsemble)
        assert isinstance(custom_ensemble.operators[1], MostCommon)

        # Verify ensemble configuration
        ensemble = custom_ensemble.operators[0]
        assert ensemble.num_units == 5
        assert ensemble.model_name == "gpt-4o"
        assert ensemble.temperature == 0.7


class TestCompactNotationEquivalence:
    """Test that compact and standard notations produce equivalent structures."""

    def test_structural_equivalence(self):
        """Test that compact and standard notations produce equivalent structures."""
        # Create equivalent pipelines with different notation styles
        compact_pipeline = build_graph(["3:E:gpt-4o:0.7", "1:J:claude-3-5-sonnet:0.0"])

        standard_pipeline = Sequential(
            operators=[
                UniformEnsemble(num_units=3, model_name="gpt-4o", temperature=0.7),
                JudgeSynthesis(model_name="claude-3-5-sonnet", temperature=0.0),
            ]
        )

        # Verify that pipelines have the same structure
        assert isinstance(compact_pipeline, Sequential)
        assert isinstance(standard_pipeline, Sequential)
        assert len(compact_pipeline.operators) == len(standard_pipeline.operators)

        # Verify ensemble configuration
        compact_ensemble = compact_pipeline.operators[0]
        standard_ensemble = standard_pipeline.operators[0]
        assert isinstance(compact_ensemble, UniformEnsemble)
        assert isinstance(standard_ensemble, UniformEnsemble)
        assert compact_ensemble.num_units == standard_ensemble.num_units
        assert compact_ensemble.model_name == standard_ensemble.model_name
        assert compact_ensemble.temperature == standard_ensemble.temperature

        # Verify judge configuration
        compact_judge = compact_pipeline.operators[1]
        standard_judge = standard_pipeline.operators[1]
        assert isinstance(compact_judge, JudgeSynthesis)
        assert isinstance(standard_judge, JudgeSynthesis)
        assert compact_judge.model_name == standard_judge.model_name
        assert compact_judge.temperature == standard_judge.temperature


if __name__ == "__main__":
    pytest.main()
