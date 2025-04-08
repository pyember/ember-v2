"""
Tests for the Compact Network of Networks (NON) Graph Notation.

This module tests the compact notation system that enables concise expression
of complex NON architectures. Following design principles favored by Jeff Dean
and Sanjay Ghemawat, these tests verify the core abstractions with clear,
thorough, and minimal test cases.
"""

from typing import cast

import pytest

from ember.core.non import (
    JudgeSynthesis,
    MostCommon,
    Sequential,
    UniformEnsemble,
    Verifier,
)
from ember.core.non_compact import (
    OpRegistry,
    build_graph,
    get_default_registry,
    parse_spec,
    resolve_refs,
)

# =============================================================================
# OpRegistry Tests
# =============================================================================


class TestOpRegistry:
    """Tests for the OpRegistry class."""

    def test_create_empty_registry(self):
        """Test creating an empty registry."""
        registry = OpRegistry()
        assert registry.get_codes() == []
        assert not registry.has_type("E")

    def test_register_factory(self):
        """Test registering a factory function."""
        registry = OpRegistry()
        registry.register(
            "TEST",
            lambda count, model, temp: UniformEnsemble(
                num_units=count, model_name=model, temperature=temp
            ),
        )

        assert registry.has_type("TEST")
        assert registry.get_codes() == ["TEST"]

    def test_create_operator(self):
        """Test creating an operator from the registry."""
        registry = OpRegistry()
        registry.register(
            "TEST",
            lambda count, model, temp: UniformEnsemble(
                num_units=count, model_name=model, temperature=temp
            ),
        )

        op = registry.create("TEST", 3, "model-name", 0.5)
        assert isinstance(op, UniformEnsemble)
        assert op.num_units == 3
        assert op.model_name == "model-name"
        assert op.temperature == 0.5

    def test_create_unknown_type(self):
        """Test attempting to create an operator with an unknown type code."""
        registry = OpRegistry()
        with pytest.raises(ValueError, match=r"Unknown operator type code: 'UNKNOWN'"):
            registry.create("UNKNOWN", 1, "model", 0.5)

    def test_standard_registry(self):
        """Test the standard registry contains expected operator types."""
        registry = OpRegistry.create_standard_registry()
        expected_types = {"E", "UE", "J", "JF", "V", "MC"}

        for code in expected_types:
            assert registry.has_type(code), f"Registry should have type code '{code}'"

        # Verify operators are created successfully
        ensemble = registry.create("E", 3, "gpt-4o", 0.7)
        assert isinstance(ensemble, UniformEnsemble)
        assert ensemble.num_units == 3

        judge = registry.create("J", 1, "claude-3-5-sonnet", 0.0)
        assert isinstance(judge, JudgeSynthesis)
        assert judge.model_name == "claude-3-5-sonnet"

        verifier = registry.create("V", 1, "gpt-4o", 0.0)
        assert isinstance(verifier, Verifier)
        assert verifier.model_name == "gpt-4o"

        most_common = registry.create("MC", 1, "model", 0.5)
        assert isinstance(most_common, MostCommon)

    def test_default_registry(self):
        """Test the default registry is properly initialized."""
        registry = get_default_registry()
        assert registry is not None
        # Verify it's a standard registry with the expected types
        assert isinstance(registry, OpRegistry)
        assert set(registry.get_codes()) >= {"E", "UE", "J", "JF", "V", "MC"}


# =============================================================================
# parse_spec Tests
# =============================================================================


class TestParseSpec:
    """Tests for the parse_spec function."""

    def test_parse_valid_spec(self):
        """Test parsing a valid operator specification."""
        op = parse_spec("3:E:gpt-4o:0.7")
        assert isinstance(op, UniformEnsemble)
        assert op.num_units == 3
        assert op.model_name == "gpt-4o"
        assert op.temperature == 0.7

    def test_parse_reference(self):
        """Test parsing a reference (should return None)."""
        op = parse_spec("$ref_name")
        assert op is None

    def test_parse_invalid_format(self):
        """Test parsing an invalid format."""
        with pytest.raises(ValueError, match=r"Invalid operator specification"):
            parse_spec("invalid:spec")

        with pytest.raises(ValueError, match=r"Invalid operator specification"):
            parse_spec("3:E:gpt-4o")  # Missing temperature

        with pytest.raises(ValueError, match=r"Invalid operator specification"):
            parse_spec("3:E:gpt-4o:0.7:extra")  # Extra part

    def test_parse_invalid_numeric_values(self):
        """Test parsing with invalid numeric values."""
        with pytest.raises(ValueError, match=r"Invalid operator specification"):
            parse_spec("abc:E:gpt-4o:0.7")  # Invalid count

        with pytest.raises(ValueError, match=r"Invalid operator specification"):
            parse_spec("3:E:gpt-4o:xyz")  # Invalid temperature

    def test_parse_custom_registry(self):
        """Test parsing with a custom registry."""
        custom_registry = OpRegistry()
        custom_registry.register(
            "CE",
            lambda count, model, temp: Sequential(
                operators=[
                    UniformEnsemble(
                        num_units=count, model_name=model, temperature=temp
                    ),
                    MostCommon(),
                ]
            ),
        )

        op = parse_spec("5:CE:gpt-4o:0.7", registry=custom_registry)
        assert isinstance(op, Sequential)
        assert len(op.operators) == 2
        assert isinstance(op.operators[0], UniformEnsemble)
        assert isinstance(op.operators[1], MostCommon)
        assert op.operators[0].num_units == 5


# =============================================================================
# resolve_refs Tests
# =============================================================================


class TestResolveRefs:
    """Tests for the resolve_refs function."""

    def test_resolve_operator_instance(self):
        """Test resolving an operator instance (should pass through)."""
        op = UniformEnsemble(num_units=3, model_name="gpt-4o", temperature=0.7)
        resolved = resolve_refs(op)
        assert resolved is op

    def test_resolve_spec_string(self):
        """Test resolving a specification string."""
        resolved = resolve_refs("3:E:gpt-4o:0.7")
        assert isinstance(resolved, UniformEnsemble)
        assert resolved.num_units == 3
        assert resolved.model_name == "gpt-4o"
        assert resolved.temperature == 0.7

    def test_resolve_reference(self):
        """Test resolving a reference to a component."""
        components = {"my_ensemble": "3:E:gpt-4o:0.7"}
        resolved = resolve_refs("$my_ensemble", components)
        assert isinstance(resolved, UniformEnsemble)
        assert resolved.num_units == 3

    def test_resolve_nested_reference(self):
        """Test resolving a nested reference."""
        components = {"base_ensemble": "3:E:gpt-4o:0.7", "nested_ref": "$base_ensemble"}
        resolved = resolve_refs("$nested_ref", components)
        assert isinstance(resolved, UniformEnsemble)
        assert resolved.num_units == 3

    def test_resolve_missing_reference(self):
        """Test resolving a missing reference."""
        components = {"known_ref": "3:E:gpt-4o:0.7"}
        with pytest.raises(
            KeyError, match=r"Referenced component 'unknown_ref' not found"
        ):
            resolve_refs("$unknown_ref", components)

    def test_resolve_list(self):
        """Test resolving a list of nodes."""
        resolved = resolve_refs(["3:E:gpt-4o:0.7", "1:J:claude-3-5-sonnet:0.0"])
        assert isinstance(resolved, Sequential)
        assert len(resolved.operators) == 2
        assert isinstance(resolved.operators[0], UniformEnsemble)
        assert isinstance(resolved.operators[1], JudgeSynthesis)

    def test_resolve_nested_list(self):
        """Test resolving a nested list structure."""
        components = {"verifier": "1:V:gpt-4o:0.0"}

        resolved = resolve_refs(
            [
                ["3:E:gpt-4o:0.7", "$verifier"],
                ["3:E:claude-3-5-haiku:0.7", "$verifier"],
                "1:J:claude-3-5-sonnet:0.0",
            ],
            components,
        )

        assert isinstance(resolved, Sequential)
        assert len(resolved.operators) == 3

        # First branch
        assert isinstance(resolved.operators[0], Sequential)
        assert len(resolved.operators[0].operators) == 2
        assert isinstance(resolved.operators[0].operators[0], UniformEnsemble)
        assert isinstance(resolved.operators[0].operators[1], Verifier)

        # Second branch
        assert isinstance(resolved.operators[1], Sequential)
        assert len(resolved.operators[1].operators) == 2
        assert isinstance(resolved.operators[1].operators[0], UniformEnsemble)
        assert isinstance(resolved.operators[1].operators[1], Verifier)

        # Judge
        assert isinstance(resolved.operators[2], JudgeSynthesis)

    def test_resolve_invalid_type(self):
        """Test resolving an invalid node type."""
        with pytest.raises(TypeError, match=r"Unsupported node type"):
            resolve_refs(123)  # type: ignore


# =============================================================================
# build_graph Tests
# =============================================================================


class TestBuildGraph:
    """Tests for the build_graph function."""

    def test_build_simple_pipeline(self):
        """Test building a simple pipeline."""
        pipeline = build_graph(["3:E:gpt-4o:0.7", "1:J:claude-3-5-sonnet:0.0"])

        assert isinstance(pipeline, Sequential)
        assert len(pipeline.operators) == 2
        assert isinstance(pipeline.operators[0], UniformEnsemble)
        assert isinstance(pipeline.operators[1], JudgeSynthesis)

        assert pipeline.operators[0].num_units == 3
        assert pipeline.operators[0].model_name == "gpt-4o"
        assert pipeline.operators[0].temperature == 0.7

        assert pipeline.operators[1].model_name == "claude-3-5-sonnet"
        assert pipeline.operators[1].temperature == 0.0

    def test_build_complex_nested_architecture(self):
        """Test building a complex nested architecture."""
        architecture = build_graph(
            [
                # First branch - GPT ensemble + verification
                ["3:E:gpt-4o:0.7", "1:V:gpt-4o:0.0"],
                # Second branch - Claude ensemble + verification
                ["3:E:claude-3-5-haiku:0.7", "1:V:claude-3-5-haiku:0.0"],
                # Final synthesis judge
                "1:J:claude-3-5-sonnet:0.0",
            ]
        )

        assert isinstance(architecture, Sequential)
        assert len(architecture.operators) == 3

        # First branch
        assert isinstance(architecture.operators[0], Sequential)
        assert len(architecture.operators[0].operators) == 2
        assert isinstance(architecture.operators[0].operators[0], UniformEnsemble)
        assert architecture.operators[0].operators[0].model_name == "gpt-4o"
        assert isinstance(architecture.operators[0].operators[1], Verifier)

        # Second branch
        assert isinstance(architecture.operators[1], Sequential)
        assert len(architecture.operators[1].operators) == 2
        assert isinstance(architecture.operators[1].operators[0], UniformEnsemble)
        assert architecture.operators[1].operators[0].model_name == "claude-3-5-haiku"
        assert isinstance(architecture.operators[1].operators[1], Verifier)

        # Judge
        assert isinstance(architecture.operators[2], JudgeSynthesis)
        assert architecture.operators[2].model_name == "claude-3-5-sonnet"

    def test_build_with_component_references(self):
        """Test building a graph with component references."""
        components = {
            "gpt_ensemble": "3:E:gpt-4o:0.7",
            "claude_ensemble": "3:E:claude-3-5-haiku:0.7",
            "verifier": "1:V:gpt-4o:0.0",
            # Reference other components
            "verification_pipeline": ["$gpt_ensemble", "$verifier"],
        }

        pipeline = build_graph(
            [
                "$verification_pipeline",
                ["$claude_ensemble", "$verifier"],
                "1:J:claude-3-5-sonnet:0.0",
            ],
            components=components,
        )

        assert isinstance(pipeline, Sequential)
        assert len(pipeline.operators) == 3

        # First branch (verification_pipeline)
        assert isinstance(pipeline.operators[0], Sequential)
        assert len(pipeline.operators[0].operators) == 2
        assert isinstance(pipeline.operators[0].operators[0], UniformEnsemble)
        assert pipeline.operators[0].operators[0].model_name == "gpt-4o"

        # Second branch
        assert isinstance(pipeline.operators[1], Sequential)
        assert len(pipeline.operators[1].operators) == 2
        assert isinstance(pipeline.operators[1].operators[0], UniformEnsemble)
        assert pipeline.operators[1].operators[0].model_name == "claude-3-5-haiku"

        # Judge
        assert isinstance(pipeline.operators[2], JudgeSynthesis)

    def test_build_with_custom_registry(self):
        """Test building a graph with a custom registry."""
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

        pipeline = build_graph(
            [
                "5:CE:gpt-4o:0.7",  # Custom ensemble with built-in MostCommon
                "1:J:claude-3-5-sonnet:0.0",  # Judge to synthesize
            ],
            type_registry=custom_registry,
        )

        assert isinstance(pipeline, Sequential)
        assert len(pipeline.operators) == 2

        # Custom ensemble
        assert isinstance(pipeline.operators[0], Sequential)
        assert len(pipeline.operators[0].operators) == 2
        assert isinstance(pipeline.operators[0].operators[0], UniformEnsemble)
        assert pipeline.operators[0].operators[0].num_units == 5
        assert isinstance(pipeline.operators[0].operators[1], MostCommon)

        # Judge
        assert isinstance(pipeline.operators[1], JudgeSynthesis)


# =============================================================================
# End-to-End Functionality Tests
# =============================================================================


class TestEndToEndFunctionality:
    """Tests for end-to-end functionality of the compact notation."""

    def test_example1_basic_ensemble_judge(self):
        """Test example 1 from the example file."""
        # Using compact notation
        compact_pipeline = build_graph(
            [
                "3:E:gpt-4o:0.7",  # Ensemble with 3 GPT-4o instances at temp=0.7
                "1:J:claude-3-5-sonnet:0.0",  # Judge using Claude with temp=0
            ]
        )

        # Equivalent pipeline using standard API
        standard_pipeline = Sequential(
            operators=[
                UniformEnsemble(num_units=3, model_name="gpt-4o", temperature=0.7),
                JudgeSynthesis(model_name="claude-3-5-sonnet", temperature=0.0),
            ]
        )

        # Verify structure equality
        assert len(compact_pipeline.operators) == len(standard_pipeline.operators)
        assert type(compact_pipeline.operators[0]) == type(
            standard_pipeline.operators[0]
        )
        assert type(compact_pipeline.operators[1]) == type(
            standard_pipeline.operators[1]
        )

        ensemble1 = cast(UniformEnsemble, compact_pipeline.operators[0])
        ensemble2 = cast(UniformEnsemble, standard_pipeline.operators[0])

        assert ensemble1.num_units == ensemble2.num_units
        assert ensemble1.model_name == ensemble2.model_name
        assert ensemble1.temperature == ensemble2.temperature

        judge1 = cast(JudgeSynthesis, compact_pipeline.operators[1])
        judge2 = cast(JudgeSynthesis, standard_pipeline.operators[1])

        assert judge1.model_name == judge2.model_name
        assert judge1.temperature == judge2.temperature

    def test_example6_nested_network_equivalent(self):
        """Test example 6 from the example file - nested network equivalent."""
        # Define the SubNetwork component (ensemble → verifier pipeline)
        subnetwork = [
            "2:E:gpt-4o:0.0",  # Ensemble with 2 identical models at temp=0
            "1:V:gpt-4o:0.0",  # Verify first response from ensemble
        ]

        # Create components map with the SubNetwork component
        component_map = {
            "sub": subnetwork,  # Reusable SubNetwork definition
        }

        # Build the NestedNetwork
        nested_network = build_graph(
            [
                "$sub",  # First branch: SubNetwork instance
                "$sub",  # Second branch: SubNetwork instance
                "1:J:gpt-4o:0.0",  # Final Judge synthesizes results
            ],
            components=component_map,
        )

        # Verify the structure
        assert isinstance(nested_network, Sequential)
        assert len(nested_network.operators) == 3

        # Both branches should be identical SubNetwork instances
        for i in range(2):
            branch = nested_network.operators[i]
            assert isinstance(branch, Sequential)
            assert len(branch.operators) == 2
            assert isinstance(branch.operators[0], UniformEnsemble)
            assert branch.operators[0].num_units == 2
            assert branch.operators[0].model_name == "gpt-4o"
            assert branch.operators[0].temperature == 0.0
            assert isinstance(branch.operators[1], Verifier)
            assert branch.operators[1].model_name == "gpt-4o"

        # Final judge
        assert isinstance(nested_network.operators[2], JudgeSynthesis)
        assert nested_network.operators[2].model_name == "gpt-4o"
        assert nested_network.operators[2].temperature == 0.0


# =============================================================================
# Property-based Testing
# =============================================================================


class TestProperties:
    """Property-based tests for the compact notation."""

    @pytest.mark.parametrize(
        "count,type_code,model,temp",
        [
            (1, "E", "gpt-4o", 0.0),
            (3, "E", "gpt-4o", 0.7),
            (5, "E", "claude-3-5-sonnet", 1.0),
            (1, "J", "gpt-4o", 0.5),
            (1, "V", "claude-3-5-haiku", 0.2),
        ],
    )
    def test_property_parse_spec_valid(self, count, type_code, model, temp):
        """Test property: parse_spec correctly parses valid specifications."""
        spec = f"{count}:{type_code}:{model}:{temp}"
        op = parse_spec(spec)

        if type_code == "E":
            assert isinstance(op, UniformEnsemble)
            assert op.num_units == count
            assert op.model_name == model
            assert op.temperature == temp
        elif type_code == "J":
            assert isinstance(op, JudgeSynthesis)
            assert op.model_name == model
            assert op.temperature == temp
        elif type_code == "V":
            assert isinstance(op, Verifier)
            assert op.model_name == model
            assert op.temperature == temp

    @pytest.mark.parametrize(
        "elements",
        [
            [["3:E:gpt-4o:0.7"], ["1:J:claude-3-5-sonnet:0.0"]],
            [["3:E:gpt-4o:0.7", "1:V:gpt-4o:0.0"], "1:J:claude-3-5-sonnet:0.0"],
            ["3:E:gpt-4o:0.7", ["1:V:gpt-4o:0.0", "1:J:claude-3-5-sonnet:0.0"]],
        ],
    )
    def test_property_nested_structures(self, elements):
        """Test property: build_graph correctly builds nested structures."""
        graph = build_graph(elements)
        assert isinstance(graph, Sequential)

        # Flatten the expected structure for validation
        flat_elements = []
        for el in elements:
            if isinstance(el, list):
                if all(isinstance(x, str) for x in el):
                    # Count all string specs in nested lists
                    flat_elements.extend([x for x in el if not x.startswith("$")])
                else:
                    # Count elements in lists of lists
                    for sub_el in el:
                        if isinstance(sub_el, list):
                            flat_elements.extend(
                                [x for x in sub_el if not x.startswith("$")]
                            )
                        elif isinstance(sub_el, str) and not sub_el.startswith("$"):
                            flat_elements.append(sub_el)
            elif isinstance(el, str) and not el.startswith("$"):
                flat_elements.append(el)

        # Count spec strings to determine expected operator count
        spec_count = len(flat_elements)

        # Count the actual operators in the flattened graph
        def count_operators(op):
            if isinstance(op, Sequential):
                return sum(count_operators(sub_op) for sub_op in op.operators)
            return 1

        assert count_operators(graph) >= spec_count


if __name__ == "__main__":
    pytest.main()
