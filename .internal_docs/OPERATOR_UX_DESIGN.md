# Operator UX Design Philosophy

## The Tension

There's a fundamental tension in API design between:
1. **Simplicity**: One obvious way to do things
2. **Flexibility**: Supporting diverse use cases
3. **Type Safety**: Catching errors early
4. **Pythonic Feel**: Natural to Python developers

## Current State

Operators can be called in multiple ways:
```python
# 1. Kwargs (clean, pythonic)
result = op(name="World", temperature=0.7)

# 2. Dict with inputs parameter
result = op(inputs={"name": "World", "temperature": 0.7})

# 3. Typed models
result = op(inputs=MyInput(name="World", temperature=0.7))

# 4. Positional dict (should we support this?)
result = op({"name": "World", "temperature": 0.7})
```

## Design Principles

### 1. Progressive Disclosure
Start simple, reveal complexity as needed:
- **Beginner**: Just use kwargs `op(name="World")`
- **Intermediate**: Add type safety with models when beneficial
- **Advanced**: Use dicts for dynamic scenarios

### 2. Context-Appropriate Defaults
Different contexts naturally favor different patterns:
- **Scripts/Notebooks**: Kwargs for exploration
- **Production Code**: Typed models for reliability
- **Data Pipelines**: Dicts for dynamic data

### 3. No Artificial Restrictions
Don't block valid use cases in pursuit of purity. If someone needs to pass a dict, let them.

## Recommended Approach for Examples

### Getting Started (01-02)
```python
# Simplest possible - just kwargs
class HelloOperator(Operator):
    specification = HelloSpec()
    
    def forward(self, *, inputs):
        name = inputs.get("name", "World")
        return {"greeting": f"Hello, {name}!"}

# Usage is clean
op = HelloOperator()
result = op(name="Alice")
```

### Core Concepts (03-04)
```python
# Introduce models for structure
class QueryInput(EmberModel):
    question: str
    context: Optional[str] = None

class AnswerOutput(EmberModel):
    answer: str
    confidence: float

# But still callable with kwargs
qa = QAOperator()
result = qa(question="What is Python?", context="Programming language")
```

### Data Processing (05)
```python
# Show dict usage for dynamic data
for record in dataset:
    # Records might have varying fields
    result = op(**record)  # Unpack dict as kwargs
    # or
    result = op(inputs=record)  # When field names might conflict
```

### Advanced (07+)
```python
# Show the full flexibility
class FlexibleOperator(Operator):
    def forward(self, *, inputs):
        # Handle both simple and complex cases
        if isinstance(inputs, dict):
            # Dynamic processing
            pass
        else:
            # Typed model processing
            pass
```

## Key Insights

1. **The simple case should be simple**: Most users just want `op(x="value")`

2. **Type safety is opt-in**: Start with dicts, graduate to models when you need validation

3. **Framework boundaries**: When integrating with other systems, flexibility matters

4. **Explicit > Implicit**: If we support multiple patterns, be clear about when to use each

## Implementation Guidelines

1. **All operators should support kwargs** for the common case

2. **The `inputs` parameter is the escape hatch** for when kwargs aren't enough

3. **Models are for contracts**, not requirements - they add value through validation and documentation

4. **Examples should show the 80% case first**, then explain the 20% case

## What This Means for Examples

- **01_getting_started**: Pure kwargs, no complexity
- **02_core_concepts**: Introduce models as "here's how to add type safety"  
- **03_operators**: Show composition - how operators flow data between each other
- **05_data_processing**: Show real-world patterns with dynamic data
- **07_advanced**: Show the full power and flexibility

The goal: Someone should be able to use Ember productively after just the first example, but have a clear path to more sophisticated usage as their needs grow.