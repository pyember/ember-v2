# Example Migration Guide

This guide helps contributors migrate existing examples to use the new conditional execution pattern.

## Overview

The migration ensures all examples:
1. Work without API keys (educational value)
2. Provide realistic simulated outputs
3. Can be tested automatically
4. Maintain consistent user experience

## Migration Steps

### 1. Add Imports

```python
import sys
from pathlib import Path

# Add the shared utilities to path
sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header
from _shared.conditional_execution import conditional_llm, SimulatedResponse
```

### 2. Decorate Main Function

Replace the manual API key checking:

```python
# Before
def main():
    if not ensure_api_key("openai"):
        print("No API key found...")
        demo_mode()
        return
    
    # Real implementation
```

With the decorator:

```python
# After
@conditional_llm(providers=["openai"])
def main(_simulated_mode=False):
    print_section_header("Example Name")
    
    if _simulated_mode:
        return run_simulated_example()
    
    # Real implementation
```

### 3. Create Simulated Function

Add a function that provides realistic outputs without API calls:

```python
def run_simulated_example():
    """Run example with simulated responses."""
    
    # Simulate API responses
    response = SimulatedResponse(
        text="This is a simulated response that demonstrates the expected output.",
        model_id="gpt-3.5-turbo"
    )
    
    # Use same output formatting as real example
    print_example_output("Response", response.text)
    print_example_output("Tokens", response.usage["total_tokens"])
    
    return 0
```

### 4. Update Entry Point

```python
if __name__ == "__main__":
    sys.exit(main())
```

## Migration Patterns

### Simple Model Call

```python
# Real mode
response = models("gpt-3.5-turbo", prompt)
print(response.text)

# Simulated mode
response = SimulatedResponse(
    text="Simulated response showing expected format",
    model_id="gpt-3.5-turbo"
)
print(response.text)
```

### Multiple Model Calls

```python
def run_simulated_example():
    # Simulate each model call
    responses = {
        "summary": SimulatedResponse(text="This is a summary."),
        "analysis": SimulatedResponse(text="Detailed analysis here."),
        "recommendation": SimulatedResponse(text="Recommendations: 1, 2, 3")
    }
    
    # Display results matching real output format
    for task, response in responses.items():
        print(f"\n{task.title()}:")
        print(response.text)
```

### Streaming Responses

```python
def run_simulated_example():
    # Simulate streaming by printing incrementally
    import time
    
    full_text = "This simulates a streaming response..."
    words = full_text.split()
    
    for word in words:
        print(word, end=" ", flush=True)
        time.sleep(0.1)  # Simulate streaming delay
    print()
```

### Batch Processing

```python
def run_simulated_example():
    # Simulate batch results
    items = ["item1", "item2", "item3"]
    results = []
    
    for item in items:
        results.append({
            "input": item,
            "output": f"Processed {item}",
            "score": 0.85
        })
    
    # Display as table or list
    for result in results:
        print(f"{result['input']} -> {result['output']} (score: {result['score']})")
```

## Best Practices

### 1. Realistic Outputs

Make simulated outputs educational and realistic:

```python
# Bad - Too generic
response = SimulatedResponse(text="Response")

# Good - Shows expected format and content
response = SimulatedResponse(
    text="The capital of France is Paris. It has been the capital since 987 AD and is known for landmarks like the Eiffel Tower."
)
```

### 2. Preserve Educational Value

Simulated mode should teach the same concepts:

```python
def run_simulated_example():
    # Show the concept being demonstrated
    print("Demonstrating prompt engineering principles:")
    
    # Vague prompt
    print("\nVague prompt: 'Tell me about AI'")
    print("Response: [Long, unfocused response about many AI topics...]")
    
    # Specific prompt  
    print("\nSpecific prompt: 'List 3 applications of AI in healthcare'")
    print("Response:")
    print("1. Diagnostic imaging analysis")
    print("2. Drug discovery acceleration") 
    print("3. Personalized treatment planning")
```

### 3. Match Output Format

Ensure simulated output matches real output structure:

```python
# If real mode prints JSON
if not _simulated_mode:
    result = model_call()
    print(json.dumps(result, indent=2))
else:
    # Simulated should also print JSON
    result = {
        "status": "success",
        "data": {"key": "value"},
        "metadata": {"tokens": 150}
    }
    print(json.dumps(result, indent=2))
```

### 4. Timing Simulation

For performance examples:

```python
import time

def run_simulated_example():
    start = time.time()
    
    # Simulate some processing time
    time.sleep(0.5)
    
    # Show performance metrics
    duration = time.time() - start
    print(f"Processing time: {duration:.2f}s")
    print("(Simulated timing for demonstration)")
```

## Testing Your Migration

1. **Test without API keys:**
   ```bash
   unset OPENAI_API_KEY
   python examples/your_example.py
   ```

2. **Test with API keys:**
   ```bash
   export OPENAI_API_KEY="your-key"
   python examples/your_example.py
   ```

3. **Run automated tests:**
   ```bash
   pytest tests/examples/test_your_category.py -v
   ```

4. **Generate golden output:**
   ```bash
   python tests/examples/update_golden.py --example "your_category/your_example.py"
   ```

## Common Issues

### Import Errors

If you get import errors for `_shared`:
- Ensure the sys.path modification is before the import
- Check you're in the correct directory structure

### Decorator Not Working

If the decorator doesn't detect API keys:
- Check environment variable names match
- Use correct provider names: "openai", "anthropic", "google"

### Output Mismatch

If tests fail due to output differences:
- Ensure simulated output follows same structure
- Use consistent formatting (JSON, tables, etc.)
- Include all expected sections

## Examples of Migrated Files

See these files for reference:
- `examples/01_getting_started/first_model_call.py`
- `examples/01_getting_started/basic_prompt_engineering.py`
- `examples/03_simplified_apis/natural_api_showcase.py`
- `examples/03_simplified_apis/simplified_workflows.py`