# Migration Guide: Specification to Signature

This guide shows how to migrate from the old Specification system to the new signature-based validation approach.

## Quick Comparison

### Old Way (Specification)
```python
from ember.core.registry.specification import Specification
from ember.core.types import EmberModel

class InputModel(EmberModel):
    question: str
    context: Optional[str] = None

class OutputModel(EmberModel):
    answer: str
    confidence: float

spec = Specification(
    input_model=InputModel,
    structured_output=OutputModel,
    prompt_template="Question: {question}\nContext: {context}",
    check_all_placeholders=True
)

# Usage
validated_input = spec.validate_inputs(inputs={"question": "What?"})
prompt = spec.render_prompt(inputs=validated_input)
validated_output = spec.validate_output(output={"answer": "42", "confidence": 0.9})
```

### New Way (Signature)
```python
from ember.core.operators.signature import Signature, InputField, OutputField, signature

class QASignature(Signature):
    question: str = InputField(desc="The question")
    context: Optional[str] = InputField(default=None, desc="Context")
    answer: str = OutputField(desc="The answer")
    confidence: float = OutputField(desc="Confidence score")

@signature(QASignature)
def qa_operator(inputs: QASignature) -> Dict[str, Any]:
    # inputs already validated
    prompt = inputs.render_prompt()
    return {"answer": "42", "confidence": 0.9}
```

## Migration Patterns

### 1. Simple Input/Output Models

**Before:**
```python
class SimpleInput(EmberModel):
    text: str

class SimpleOutput(EmberModel):
    result: str

spec = Specification(
    input_model=SimpleInput,
    structured_output=SimpleOutput
)

def process(inputs):
    validated = spec.validate_inputs(inputs=inputs)
    result = validated.text.upper()
    return spec.validate_output(output={"result": result})
```

**After:**
```python
class ProcessSignature(Signature):
    text: str = InputField()
    result: str = OutputField()

@signature(ProcessSignature)
def process(inputs: ProcessSignature):
    return {"result": inputs.text.upper()}
```

### 2. Complex Types and Defaults

**Before:**
```python
class ComplexInput(EmberModel):
    items: List[str]
    options: Dict[str, Any]
    threshold: float = 0.5

class ComplexOutput(EmberModel):
    filtered: List[str]
    metadata: Dict[str, int]

spec = Specification(
    input_model=ComplexInput,
    structured_output=ComplexOutput
)
```

**After:**
```python
class ComplexSignature(Signature):
    items: List[str] = InputField(desc="Items to process")
    options: Dict[str, Any] = InputField(desc="Processing options")
    threshold: float = InputField(default=0.5, desc="Filter threshold")
    
    filtered: List[str] = OutputField(desc="Filtered items")
    metadata: Dict[str, int] = OutputField(desc="Processing metadata")
```

### 3. Prompt Templates

**Before:**
```python
spec = Specification(
    input_model=InputModel,
    prompt_template="Context: {context}\nQuestion: {question}\nAnswer:",
    check_all_placeholders=True
)

prompt = spec.render_prompt(inputs={"context": "...", "question": "..."})
```

**After:**
```python
@signature(
    MySignature,
    template="Context: {context}\nQuestion: {question}\nAnswer:"
)
def my_operator(inputs: MySignature):
    prompt = inputs.render_prompt()  # Uses template
    # Or auto-generate:
    # prompt = inputs.render_prompt()  # Auto-formats from fields
```

### 4. Validation Without Decorators

**Before:**
```python
spec = Specification(input_model=InputModel)

# Manual validation
try:
    validated = spec.validate_inputs(inputs=raw_data)
except SpecificationValidationError as e:
    handle_error(e)
```

**After:**
```python
# Direct instantiation with validation
try:
    validated = MySignature(**raw_data)
except ValidationError as e:
    handle_error(e)
```

### 5. Optional Fields and Validation

**Before:**
```python
class InputWithOptional(EmberModel):
    required_field: str
    optional_field: Optional[int] = None

spec = Specification(
    input_model=InputWithOptional,
    check_all_placeholders=False  # Don't require optional fields
)
```

**After:**
```python
class SignatureWithOptional(Signature):
    required_field: str = InputField(desc="Required")
    optional_field: Optional[int] = InputField(default=None, desc="Optional")
```

## Feature Mapping

| Specification Feature | Signature Equivalent |
|----------------------|---------------------|
| `input_model` | Input fields with `InputField()` |
| `structured_output` | Output fields with `OutputField()` |
| `prompt_template` | `template` parameter or `render_prompt()` |
| `check_all_placeholders` | Automatic with Pydantic validation |
| `validate_inputs()` | Automatic with `@signature` |
| `validate_output()` | Automatic with `@signature` |
| `render_prompt()` | `inputs.render_prompt()` |
| `model_json_schema()` | `Signature.model_json_schema()` |

## Best Practices for Migration

1. **Start Simple**: For basic type checking, consider using `@validate` instead
2. **Group Related Fields**: Signatures work best with cohesive input/output groups
3. **Use Descriptive Names**: Field descriptions become documentation
4. **Leverage Defaults**: Use `InputField(default=...)` for optional parameters
5. **Type Everything**: Signatures enforce types, so be explicit

## When NOT to Migrate

Keep using simple `@validate` for:
- Single input/output validation
- Simple type checks (str, int, float)
- Performance-critical code
- Temporary validation needs

## Advanced Migration: Custom Validators

**Before (with Specification subclass):**
```python
class CustomSpec(Specification):
    def validate_inputs(self, inputs):
        validated = super().validate_inputs(inputs=inputs)
        # Custom validation
        if validated.value < 0:
            raise ValueError("Value must be positive")
        return validated
```

**After (with Signature validator):**
```python
from pydantic import validator

class CustomSignature(Signature):
    value: int = InputField(desc="Must be positive")
    
    @validator('value')
    def value_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Value must be positive")
        return v
```

## Summary

The new signature-based approach provides:
- Cleaner, more intuitive API
- Better integration with modern Python typing
- Automatic validation through decorators
- Progressive enhancement from simple to complex

Most migrations are straightforward field-mapping exercises. The main benefits are less boilerplate and better ergonomics while maintaining the same validation power.