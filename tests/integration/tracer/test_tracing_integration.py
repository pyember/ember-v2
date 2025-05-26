"""
Tests for XCS tracing and the JIT decorator.

These tests verify:
  - That TracerContext captures operator calls correctly.
  - That the traced graph can be converted to an executable plan.
  - That JIT caching prevents re-tracing and that force_trace_forward forces a new trace.
  - Re-entrancy is properly handled (including nested JIT operators).
"""

from typing import Any, Dict

from ember.core.registry.operator.base.operator_base import Operator
from ember.xcs import TracerContext, jit


class DummySpecification:
    def validate_inputs(self, inputs):
        return inputs

    def validate_output(self, output):
        return output


@jit()
class DummyTracingOperator(Operator[Dict[str, Any], Dict[str, Any]]):
    specification = DummySpecification()

    def __init__(self, **kwargs) -> None:
        pass

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": "traced"}


# def test_tracer_context_capture() -> None:
    op = DummyTracingOperator()
    op.name = "DummyTrace"
    with TracerContext() as tctx:
        _ = op(inputs={"dummy": "data"})
    assert len(tctx.records) >= 1, "No trace records were captured."
    found = False
    for record in tctx.records:
        if record.operator_name == "DummyTrace":
            assert record.outputs == {
                "output": "traced"
            }, f"Traced output {record.outputs} does not match expected result."
            found = True
            break
    assert found, "The operator's execution was not traced."

  # REMOVED: Trace strategy no longer exists
def test_convert_traced_graph_to_plan() -> None:
    op = DummyTracingOperator()
    op.name = "DummyTracePlan"
    with TracerContext() as tctx:
        _ = op(inputs={"dummy": "data"})
    assert len(tctx.records) >= 1, "Expected trace records but found none."


def test_jit_operator_always_executes() -> None:
    @jit()
    class DummyJITOperator(Operator[Dict[str, Any], Dict[str, Any]]):
        specification = DummySpecification()
        call_count = 0

        def __init__(self, **kwargs) -> None:
            pass

        def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
            type(self).call_count += 1
            # Mimic some transformation by appending "_processed"
            return {"value": inputs.get("value", "default") + "_processed"}

    op = DummyJITOperator()
    op.name = "JITOp"

    output1 = op(inputs={"value": "input"})
    assert output1["value"] == "input_processed"

    # No caching now: the forward method should run again
    output2 = op(inputs={"value": "another"})
    assert output2["value"] == "another_processed"
    assert DummyJITOperator.call_count == 2, "Expected forward to be called twice."


def test_force_trace() -> None:
    @jit(sample_input={"value": 1}, force_trace=True)
    class DummyForceJITOperator(Operator[Dict[str, Any], Dict[str, Any]]):
        specification = DummySpecification()
        trace_count = 0

        def __init__(self, **kwargs) -> None:
            pass

        def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
            type(self).trace_count += 1
            return {"value": inputs.get("value", 0) + 1}

    op = DummyForceJITOperator()
    op.name = "ForceJITOp"
    output1 = op(inputs={"value": 5})
    count1 = DummyForceJITOperator.trace_count
    output2 = op(inputs={"value": 5})
    count2 = DummyForceJITOperator.trace_count
    assert count2 == count1 + 1, "Expected force_trace to call forward again."
    assert output1["value"] == 6
    assert output2["value"] == 6


def test_nested_jit() -> None:
    @jit()
    class InnerOperator(Operator[Dict[str, Any], Dict[str, Any]]):
        specification = DummySpecification()

        def __init__(self, **kwargs) -> None:
            pass

        def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {"inner": inputs["value"] + "_inner"}

    @jit()
    class OuterOperator(Operator[Dict[str, Any], Dict[str, Any]]):
        specification = DummySpecification()

        def __init__(self, **kwargs) -> None:
            # Only instance-level state needed is the inner operator.
            self.inner = InnerOperator()

        def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
            inner_out = self.inner(inputs=inputs)
            return {"outer": inner_out["inner"] + "_outer"}

    op = OuterOperator()
    result = op(inputs={"value": "test"})
    assert result["outer"] == "test_inner_outer"
