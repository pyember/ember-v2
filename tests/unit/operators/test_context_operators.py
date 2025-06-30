"""Test context operators (ContextAgnostic, ContextAware, ContextualData, InitialContext).

Following CLAUDE.md principles:
- Test actual implementation behavior
- No assumptions about API
- Clear test cases
"""

import pytest
from typing import Any, Callable, Dict

from ember.api.operators import (
    Operator, ContextAgnostic, ContextAware, ContextualData, 
    InitialContext
)


# Global trackers for testing (needed for Equinox compatibility)
_call_tracker = {}
_last_input_tracker = {}

# Helper operators for testing
class MockOperator(Operator):
    """Mock operator that returns predefined result."""
    result: Any
    
    def __init__(self, result):
        self.result = result
        # Use id as unique identifier
        _call_tracker[id(self)] = 0
        _last_input_tracker[id(self)] = None
        
    def forward(self, input):
        _call_tracker[id(self)] += 1
        _last_input_tracker[id(self)] = input
        if callable(self.result):
            return self.result(input)
        return self.result
    
    @property 
    def call_count(self):
        return _call_tracker.get(id(self), 0)
    
    @property
    def last_input(self):
        return _last_input_tracker.get(id(self))


class MockContextAwareOperator(Operator):
    """Mock operator that expects ContextualData and returns ContextualData."""
    data_transform: Any
    context_transform: Any
    
    def __init__(self, data_transform=None, context_transform=None):
        self.data_transform = data_transform or (lambda x: f"processed: {x}")
        self.context_transform = context_transform or (lambda ctx: ctx)
        # Use id as unique identifier  
        _call_tracker[id(self)] = 0
        _last_input_tracker[id(self)] = None
        
    def forward(self, input: ContextualData) -> ContextualData:
        _call_tracker[id(self)] += 1
        _last_input_tracker[id(self)] = input
        
        new_data = self.data_transform(input.data)
        new_context = self.context_transform(input.context.copy())
        
        return ContextualData(context=new_context, data=new_data)
    
    @property 
    def call_count(self):
        return _call_tracker.get(id(self), 0)
    
    @property
    def last_input(self):
        return _last_input_tracker.get(id(self))


class TestContextualData:
    """Test the ContextualData model."""
    
    def test_contextual_data_creation(self):
        """Test creating ContextualData with context and data."""
        context = {"user_id": "123", "session": "abc"}
        data = "Hello world"
        
        contextual = ContextualData(context=context, data=data)
        
        assert contextual.context == context
        assert contextual.data == data
    
    def test_contextual_data_empty_context(self):
        """Test ContextualData with empty context."""
        contextual = ContextualData(context={}, data="test")
        
        assert contextual.context == {}
        assert contextual.data == "test"
    
    def test_contextual_data_complex_data(self):
        """Test ContextualData with complex data types."""
        context = {"metadata": "test"}
        data = {"numbers": [1, 2, 3], "text": "example"}
        
        contextual = ContextualData(context=context, data=data)
        
        assert contextual.context == context
        assert contextual.data == data


class TestContextAgnostic:
    """Test the ContextAgnostic wrapper operator."""
    
    def test_basic_context_preservation(self):
        """Test basic context preservation around context-unaware operator."""
        # Context-unaware operator that just uppercases strings
        unaware_op = MockOperator(lambda x: x.upper())
        context_op = ContextAgnostic(unaware_op)
        
        input_data = ContextualData(
            context={"user_id": "123", "original_query": "test"},
            data="hello world"
        )
        
        result = context_op(input_data)
        
        # Should preserve context
        assert result.context == {"user_id": "123", "original_query": "test"}
        # Should transform data
        assert result.data == "HELLO WORLD"
        # Wrapped operator should have been called with just the data
        assert unaware_op.last_input == "hello world"
        assert unaware_op.call_count == 1
    
    def test_context_transform_function(self):
        """Test ContextAgnostic with context transformation."""
        def add_metadata(context, input_data, output_data):
            return {
                **context,
                "processing_time": 0.5,
                "input_length": len(str(input_data)),
                "output_length": len(str(output_data))
            }
        
        unaware_op = MockOperator(lambda x: x * 2)  # Double the input
        context_op = ContextAgnostic(unaware_op, context_transform=add_metadata)
        
        input_data = ContextualData(
            context={"session": "abc"},
            data="test"
        )
        
        result = context_op(input_data)
        
        # Should have original context plus metadata
        expected_context = {
            "session": "abc",
            "processing_time": 0.5,
            "input_length": 4,  # len("test")
            "output_length": 8   # len("testtest")
        }
        assert result.context == expected_context
        assert result.data == "testtest"
    
    def test_numeric_data_processing(self):
        """Test ContextAgnostic with numeric data."""
        add_ten = MockOperator(lambda x: x + 10)
        context_op = ContextAgnostic(add_ten)
        
        input_data = ContextualData(
            context={"operation": "add_ten"},
            data=5
        )
        
        result = context_op(input_data)
        
        assert result.context == {"operation": "add_ten"}
        assert result.data == 15
    
    def test_list_data_processing(self):
        """Test ContextAgnostic with list data."""
        reverse_op = MockOperator(lambda x: x[::-1])
        context_op = ContextAgnostic(reverse_op)
        
        input_data = ContextualData(
            context={"task": "reverse"},
            data=[1, 2, 3, 4]
        )
        
        result = context_op(input_data)
        
        assert result.context == {"task": "reverse"}
        assert result.data == [4, 3, 2, 1]


class TestContextAware:
    """Test the ContextAware wrapper operator."""
    
    def test_full_context_passing(self):
        """Test ContextAware passes full ContextualData to wrapped operator."""
        # Context-aware operator that uses context information
        def context_processor(input_data: ContextualData) -> ContextualData:
            prefix = input_data.context.get("prefix", "")
            new_data = f"{prefix}{input_data.data}"
            return ContextualData(context=input_data.context, data=new_data)
        
        aware_op = MockOperator(context_processor)
        context_op = ContextAware(aware_op)
        
        input_data = ContextualData(
            context={"prefix": "Hello, "},
            data="world!"
        )
        
        result = context_op(input_data)
        
        assert result.context == {"prefix": "Hello, "}
        assert result.data == "Hello, world!"
        # Wrapped operator should receive full ContextualData
        assert isinstance(aware_op.last_input, ContextualData)
        assert aware_op.last_input.context == {"prefix": "Hello, "}
        assert aware_op.last_input.data == "world!"
    
    def test_context_transform_on_output(self):
        """Test ContextAware with output context transformation."""
        def add_processing_metadata(context, input_data, output_data):
            return {
                **context,
                "processed": True,
                "output_type": type(output_data).__name__
            }
        
        # Simple processor that converts to uppercase
        processor = MockContextAwareOperator(
            data_transform=lambda x: x.upper()
        )
        
        context_op = ContextAware(processor, context_transform=add_processing_metadata)
        
        input_data = ContextualData(
            context={"user": "alice"},
            data="hello"
        )
        
        result = context_op(input_data)
        
        expected_context = {
            "user": "alice",
            "processed": True,
            "output_type": "str"
        }
        assert result.context == expected_context
        assert result.data == "HELLO"
    
    def test_context_aware_with_complex_operator(self):
        """Test ContextAware with operator that modifies both context and data."""
        def complex_transform(ctx):
            return {**ctx, "step_count": ctx.get("step_count", 0) + 1}
        
        processor = MockContextAwareOperator(
            data_transform=lambda x: x * 2,
            context_transform=complex_transform
        )
        
        context_op = ContextAware(processor)
        
        input_data = ContextualData(
            context={"step_count": 2},
            data="test"
        )
        
        result = context_op(input_data)
        
        assert result.context == {"step_count": 3}
        assert result.data == "testtest"


class TestInitialContext:
    """Test the InitialContext operator."""
    
    def test_basic_context_initialization(self):
        """Test InitialContext creates ContextualData from raw input."""
        context_init = InitialContext()
        
        result = context_init("What is AI?")
        
        assert isinstance(result, ContextualData)
        assert result.context == {"original_query": "What is AI?"}
        assert result.data == "What is AI?"
    
    def test_context_with_defaults(self):
        """Test InitialContext with default context values."""
        context_init = InitialContext(
            user_id="123", 
            session="abc", 
            priority="high"
        )
        
        result = context_init("Hello world")
        
        expected_context = {
            "user_id": "123",
            "session": "abc", 
            "priority": "high",
            "original_query": "Hello world"
        }
        assert result.context == expected_context
        assert result.data == "Hello world"
    
    def test_complex_input(self):
        """Test InitialContext with complex input types."""
        context_init = InitialContext(source="api")
        
        input_data = {"question": "test", "options": ["A", "B", "C"]}
        result = context_init(input_data)
        
        assert result.context == {"source": "api", "original_query": input_data}
        assert result.data == input_data


class TestContextOperatorIntegration:
    """Test integration of context operators together."""
    
    def test_agnostic_to_aware_chain(self):
        """Test chaining ContextAgnostic and ContextAware operators."""
        # Step 1: Context-agnostic processing
        uppercase_op = MockOperator(lambda x: x.upper())
        agnostic_step = ContextAgnostic(uppercase_op)
        
        # Step 2: Context-aware processing that uses original query
        def aware_processor(input_data: ContextualData) -> ContextualData:
            original = input_data.context.get("original_query", "")
            processed = input_data.data
            final_data = f"Original: '{original}' -> Processed: '{processed}'"
            return ContextualData(context=input_data.context, data=final_data)
        
        aware_op = MockOperator(aware_processor)
        aware_step = ContextAware(aware_op)
        
        # Create chain manually (simplified)
        input_data = ContextualData(
            context={"original_query": "hello world"},
            data="hello world"
        )
        
        # Step 1: ContextAgnostic
        intermediate = agnostic_step(input_data)
        assert intermediate.data == "HELLO WORLD"
        assert intermediate.context == {"original_query": "hello world"}
        
        # Step 2: ContextAware
        final_result = aware_step(intermediate)
        expected = "Original: 'hello world' -> Processed: 'HELLO WORLD'"
        assert final_result.data == expected
        assert final_result.context == {"original_query": "hello world"}
    
    def test_initial_context_chain(self):
        """Test InitialContext followed by context operators."""
        # Initialize context
        init_op = InitialContext(user_id="123", priority="high")
        
        # Process with context-agnostic operator
        double_op = MockOperator(lambda x: x * 2)
        agnostic_op = ContextAgnostic(double_op)
        
        # Chain: raw input -> InitialContext -> ContextAgnostic
        initial_result = init_op("test")
        final_result = agnostic_op(initial_result)
        
        expected_context = {
            "user_id": "123", 
            "priority": "high", 
            "original_query": "test"
        }
        assert final_result.context == expected_context
        assert final_result.data == "testtest"
    
    def test_context_data_type_consistency(self):
        """Test that all context operators work consistently with ContextualData."""
        # Create initial context
        init_op = InitialContext(source="test")
        contextual_data = init_op("input")
        
        # Test ContextAgnostic accepts and returns ContextualData
        agnostic_op = ContextAgnostic(MockOperator(lambda x: x.upper()))
        agnostic_result = agnostic_op(contextual_data)
        assert isinstance(agnostic_result, ContextualData)
        
        # Test ContextAware accepts and returns ContextualData
        aware_op = ContextAware(MockContextAwareOperator())
        aware_result = aware_op(agnostic_result)
        assert isinstance(aware_result, ContextualData)
        
        # Verify context flows through the chain
        assert "source" in aware_result.context
        assert "original_query" in aware_result.context 