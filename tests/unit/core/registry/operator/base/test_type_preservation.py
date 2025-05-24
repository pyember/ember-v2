"""Test that operators preserve types correctly in all execution contexts."""

import pytest
from typing import ClassVar, Dict, Any

from ember.core.registry.operator.base.operator_base import Operator
from ember.core.registry.specification.specification import Specification
from ember.core.types.ember_model import EmberModel


class TestInput(EmberModel):
    """Test input model."""
    value: str


class TestOutput(EmberModel):
    """Test output model."""
    result: str
    processed: bool = True


class DictReturningOperator(Operator[TestInput, TestOutput]):
    """Operator that returns a dict instead of model (simulating JIT behavior)."""
    
    specification: ClassVar[Specification] = Specification(
        input_model=TestInput,
        structured_output=TestOutput
    )
    
    def forward(self, *, inputs: TestInput) -> TestOutput:
        # Simulate JIT by returning a dict instead of the model
        return {"result": f"Processed: {inputs.value}", "processed": True}


class ModelReturningOperator(Operator[TestInput, TestOutput]):
    """Operator that properly returns a model."""
    
    specification: ClassVar[Specification] = Specification(
        input_model=TestInput,
        structured_output=TestOutput
    )
    
    def forward(self, *, inputs: TestInput) -> TestOutput:
        return TestOutput(result=f"Processed: {inputs.value}")


class TestTypePreservation:
    """Test that type preservation works correctly."""
    
    def test_dict_returning_operator_coerces_to_model(self):
        """Test that operators returning dicts are coerced to proper models."""
        op = DictReturningOperator()
        
        # Call with model input
        result = op(inputs=TestInput(value="test"))
        
        # Should get back a proper TestOutput model, not a dict
        assert isinstance(result, TestOutput)
        assert result.result == "Processed: test"
        assert result.processed is True
        
    def test_dict_input_dict_returning_operator(self):
        """Test dict input with dict-returning operator."""
        op = DictReturningOperator()
        
        # Call with dict input
        result = op(inputs={"value": "test"})
        
        # Should still get back a proper TestOutput model
        assert isinstance(result, TestOutput)
        assert result.result == "Processed: test"
        
    def test_model_returning_operator_unchanged(self):
        """Test that operators already returning models work unchanged."""
        op = ModelReturningOperator()
        
        result = op(inputs=TestInput(value="test"))
        
        assert isinstance(result, TestOutput)
        assert result.result == "Processed: test"
        
    def test_kwargs_input_style(self):
        """Test kwargs input style with type preservation."""
        op = DictReturningOperator()
        
        # Call with kwargs
        result = op(value="test")
        
        assert isinstance(result, TestOutput)
        assert result.result == "Processed: test"
        
    def test_nested_dict_structures(self):
        """Test that nested dict structures are properly converted."""
        
        class NestedOutput(EmberModel):
            data: Dict[str, Any]
            metadata: Dict[str, str]
            
        class NestedOperator(Operator[TestInput, NestedOutput]):
            specification: ClassVar[Specification] = Specification(
                input_model=TestInput,
                structured_output=NestedOutput
            )
            
            def forward(self, *, inputs: TestInput) -> NestedOutput:
                # Return dict that should be converted to NestedOutput
                return {
                    "data": {"value": inputs.value, "count": 1},
                    "metadata": {"source": "test"}
                }
        
        op = NestedOperator()
        result = op(inputs=TestInput(value="nested"))
        
        assert isinstance(result, NestedOutput)
        assert result.data["value"] == "nested"
        assert result.metadata["source"] == "test"