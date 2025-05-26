"""
Property-style tests for Compact Network of Networks (NON) Graph Notation.

This module provides comprehensive testing of the compact notation system with
a focus on systematically testing a wide range of inputs and combinations.
"""

import pytest

from ember.core.non import (
    JudgeSynthesis,
    MostCommon,
    Sequential,
    UniformEnsemble,
    Verifier)
from ember.core.non_compact import build_graph, parse_spec, resolve_refs

# =============================================================================
# Test Classes with Property-Based Test Structure
# =============================================================================


class TestPropertyBasedParsing:
    """Tests for parse_spec with a wide range of inputs."""

    def test_counts_vary(self):
        """Test that different counts work correctly."""
        counts = [1, 2, 3, 5, 10]
        for count in counts:
            spec = f"{count}:E:gpt-4o:0.7"
            op = parse_spec(spec)
            assert isinstance(op, UniformEnsemble)
            assert op.num_units == count
            assert op.model_name == "gpt-4o"
            assert op.temperature == 0.7

    def test_type_codes_vary(self):
        """Test that different type codes work correctly."""
        type_codes_and_expected_types = [
            ("E", UniformEnsemble),
            ("UE", UniformEnsemble),
            ("J", JudgeSynthesis),
            ("JF", JudgeSynthesis),
            ("V", Verifier),
            ("MC", MostCommon)]

        for code, expected_type in type_codes_and_expected_types:
            # Skip MC with count != 1 (only needs 1 instance)
            count = 1 if code == "MC" else 3
            spec = f"{count}:{code}:gpt-4o:0.7"
            op = parse_spec(spec)
            assert isinstance(
                op, expected_type
            ), f"Expected {expected_type} for code {code}"

            # Verify model name and temperature for model-based operators
            if expected_type != MostCommon:
                assert op.model_name == "gpt-4o"
                assert op.temperature == 0.7

    def test_model_names_vary(self):
        """Test that different model names work correctly."""
        model_names = [
            "gpt-4o",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "gpt-4-turbo",
            "gemini-pro",
            "llama-3-70b",
            "",  # Empty model name should work
            "model with spaces",  # Spaces in model name should work
        ]

        for model in model_names:
            spec = f"1:E:{model}:0.7"
            op = parse_spec(spec)
            assert isinstance(op, UniformEnsemble)
            assert op.model_name == model

    def test_temperatures_vary(self):
        """Test that different temperatures work correctly."""
        # Test a full range of temperatures including extended values
        # Validation happens at the provider level, not in the parser
        temperatures = [0.0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0]

        for temp in temperatures:
            spec = f"1:E:gpt-4o:{temp}"
            op = parse_spec(spec)
            assert isinstance(op, UniformEnsemble)
            assert op.temperature == temp


class TestPropertyBasedResolveRefs:
    """Tests for resolve_refs with a wide range of inputs."""

    def test_nested_references_depth(self):
        """Test resolving nested references with different depths."""
        # Test depths 1 to 3
        for depth in range(1, 4):
            components = {}

            # Create a chain of nested references
            for i in range(depth):
                if i == 0:
                    # Base reference points to a concrete spec
                    components[f"ref_{i}"] = "3:E:gpt-4o:0.7"
                else:
                    # Each subsequent reference points to the previous one
                    components[f"ref_{i}"] = f"$ref_{i-1}"

            # Resolve the deepest reference
            deepest_ref = f"$ref_{depth-1}"
            op = resolve_refs(deepest_ref, components)

            # Should resolve to a UniformEnsemble
            assert isinstance(op, UniformEnsemble)
            assert op.num_units == 3
            assert op.model_name == "gpt-4o"
            assert op.temperature == 0.7


class TestPropertyBasedBuildGraph:
    """Tests for build_graph with a wide range of inputs."""

    def test_varying_component_counts(self):
        """Test building graphs with varying numbers of components."""
        # Test with 1, 2, and 3 components
        for component_count in range(1, 4):
            # Create component map
            components = {}
            for i in range(component_count):
                components[f"comp_{i}"] = f"{i+1}:E:gpt-4o:{i/10}"

            # Create pipeline with references to all components
            refs = [f"$comp_{i}" for i in range(component_count)]

            # Add a final judge
            refs.append("1:J:claude-3-5-sonnet:0.0")

            # Build the graph
            graph = build_graph(refs, components)

            # Verify structure
            assert isinstance(graph, Sequential)
            assert len(graph.operators) == component_count + 1

            # Check components
            for i in range(component_count):
                comp = graph.operators[i]
                assert isinstance(comp, UniformEnsemble)
                assert comp.num_units == i + 1
                assert comp.model_name == "gpt-4o"
                assert comp.temperature == i / 10

            # Check judge
            assert isinstance(graph.operators[-1], JudgeSynthesis)
            assert graph.operators[-1].model_name == "claude-3-5-sonnet"
            assert graph.operators[-1].temperature == 0.0

    def test_varying_nesting_depth(self):
        """Test building graphs with varying nesting depth."""
        # Test with nesting depths 1 to 3
        for depth in range(1, 4):
            # Create a structure with the specified depth
            structure = ["1:E:gpt-4o:0.7"]
            current = structure

            # Create nested structure
            for _ in range(depth):
                new_level = ["1:E:gpt-4o:0.7"]
                current.append(new_level)
                current = new_level

            # Add final element
            current.append("1:J:claude-3-5-sonnet:0.0")

            # Build the graph
            graph = build_graph(structure)

            # Verify it built successfully
            assert isinstance(graph, Sequential)

            # Count total operators (not exact, as Sequential flattens)
            operators_count = 0

            def count_operators(op):
                nonlocal operators_count
                if isinstance(op, Sequential):
                    for sub_op in op.operators:
                        count_operators(sub_op)
                else:
                    operators_count += 1

            count_operators(graph)

            # We should have at least depth+2 operators (depth ensembles + 1 judge)
            assert operators_count >= depth + 2


if __name__ == "__main__":
    pytest.main()
