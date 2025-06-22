# Operators: Complete Progressive Disclosure

## Yes, You Can Still Use EmberModel and Specifications!

The new operator system supports **full progressive disclosure**:

### Level 1: Simple Functions (90% of use cases)
```python
def summarize(text):
    return models("gpt-4", f"Summarize: {text}")
```

### Level 2: Validated Functions (9% of use cases)
```python
@validate(input=str, output=dict)
def analyze(text: str) -> dict:
    return {"sentiment": "positive", "score": 0.9}
```

### Level 3: Full EmberModel + Specification (1% of use cases)
```python
# Define models
class AnalysisInput(EmberModel):
    text: str
    options: Dict[str, Any]

class AnalysisOutput(EmberModel):
    results: List[Dict]
    metadata: Dict[str, Any]

# Create specification
spec = Specification(
    input_model=AnalysisInput,
    structured_output=AnalysisOutput
)

# Use with operator
@with_specification(spec)
def complex_analysis(inputs: AnalysisInput) -> AnalysisOutput:
    # Full type safety and validation
    return AnalysisOutput(...)
```

## Key Points

1. **Nothing is removed** - EmberModel and Specifications still work
2. **Progressive enhancement** - Start simple, add complexity only when needed
3. **Full backward compatibility** - Old EmberModel-based operators still work
4. **Clean composition** - All levels work together seamlessly

## When to Use Each Level

- **Level 1 (Functions)**: Most operators - transformations, API calls, simple logic
- **Level 2 (@validate)**: When you need basic runtime type checking
- **Level 3 (EmberModel)**: Complex multi-field inputs/outputs, strict validation, generated clients

The beauty is that you can start at Level 1 and progressively add features as needed, without rewriting your code. This is what Dean, Ghemawat, Jobs, and others would appreciate - power through simplicity, complexity when necessary.