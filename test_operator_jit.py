"""Test operator with JIT to debug the dict return issue."""

from typing import ClassVar
from ember.api.operators import Operator, Specification, EmberModel
from ember.api.xcs import jit

class TestInput(EmberModel):
    """Test input model."""
    message: str

class TestOutput(EmberModel):
    """Test output model."""
    result: str

class TestSpecification(Specification):
    """Test specification."""
    input_model = TestInput
    structured_output = TestOutput

# Test without JIT first
class SimpleOperator(Operator[TestInput, TestOutput]):
    """Simple operator without JIT."""
    specification: ClassVar[Specification] = TestSpecification()
    
    def forward(self, *, inputs: TestInput) -> TestOutput:
        return TestOutput(result=f"Processed: {inputs.message}")

# Test with JIT
@jit()
class JITOperator(Operator[TestInput, TestOutput]):
    """Operator with JIT decorator."""
    specification: ClassVar[Specification] = TestSpecification()
    
    def forward(self, *, inputs: TestInput) -> TestOutput:
        return TestOutput(result=f"JIT Processed: {inputs.message}")

# Test both operators
print("Testing SimpleOperator:")
simple_op = SimpleOperator()
result1 = simple_op(inputs=TestInput(message="Hello"))
print(f"Result type: {type(result1)}")
print(f"Result: {result1}")
print(f"Has 'result' attr: {hasattr(result1, 'result')}")
if hasattr(result1, 'result'):
    print(f"Result value: {result1.result}")

print("\nTesting JITOperator:")
jit_op = JITOperator()
result2 = jit_op(inputs=TestInput(message="Hello"))
print(f"Result type: {type(result2)}")
print(f"Result: {result2}")
print(f"Has 'result' attr: {hasattr(result2, 'result')}")
if hasattr(result2, 'result'):
    print(f"Result value: {result2.result}")
elif isinstance(result2, dict):
    print(f"Result as dict: {result2}")
    print(f"Dict keys: {result2.keys()}")