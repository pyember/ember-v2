# Ember Examples Testing Framework

This directory contains a comprehensive testing framework for all Ember examples, ensuring they remain functional, educational, and maintainable.

## Overview

The testing framework provides:
- **Golden testing**: Captures and validates example outputs
- **Dual execution modes**: Tests both with and without API keys
- **Performance validation**: Ensures examples complete within time bounds
- **CI integration**: Automated testing on every change
- **Easy maintenance**: Scripts to update golden outputs

## Structure

```
tests/examples/
├── __init__.py                 # Package initialization
├── test_base.py               # Base test class with utilities
├── conftest.py                # Pytest configuration
├── update_golden.py           # Update golden outputs
├── validate_golden.py         # Validate golden consistency
├── golden_outputs/            # Golden output files
│   └── {example}_simulated.json
├── test_*.py                  # Test files for each example directory
└── README.md                  # This file
```

## Running Tests

### Basic Usage

```bash
# Run all example tests (simulated mode only)
pytest tests/examples/ --no-api-keys

# Run with real API calls (requires API keys)
pytest tests/examples/

# Run tests for specific example
pytest tests/examples/ --example "01_getting_started/hello_world.py"

# Update golden outputs
python tests/examples/update_golden.py

# Validate golden outputs
python tests/examples/validate_golden.py
```

### Command Line Options

- `--no-api-keys`: Run tests without requiring API keys (simulated mode only)
- `--update-golden`: Update golden outputs from current execution
- `--example PATH`: Run tests for a specific example only

## Golden Output Format

Golden outputs are stored as JSON files with the following structure:

```json
{
  "version": "1.0",
  "example": "01_getting_started/first_model_call.py",
  "execution_mode": "simulated",
  "sections": [
    {
      "header": "Section Header",
      "output": "Section output content..."
    }
  ],
  "total_time": 2.1,
  "api_keys_required": ["OPENAI_API_KEY"],
  "metrics": {
    "lines_of_code": 45,
    "api_calls": 1
  }
}
```

## Writing Tests

### Basic Test Pattern

```python
from .test_base import ExampleGoldenTest

class TestMyExamples(ExampleGoldenTest):
    
    def test_my_example(self):
        """Test description."""
        self.run_example_test(
            "directory/example.py",
            requires_api_keys=["OPENAI_API_KEY"],
            max_execution_time=30.0,
            validate_sections=["Expected Section"]
        )
```

### Test Parameters

- `example_path`: Path relative to examples directory
- `requires_api_keys`: List of required API keys
- `max_execution_time`: Maximum allowed execution time in seconds
- `validate_sections`: Specific sections that must be present in output
- `skip_real_mode`: Skip testing with real API keys

## CI Integration

The GitHub Actions workflow runs on:
- Every push/PR affecting examples or tests
- Daily schedule to catch regressions
- Manual dispatch with option to test with real API keys

### Workflow Features

- Tests across Python 3.8-3.11
- Parallel test execution
- Result artifacts upload
- PR commenting with results
- Automatic golden output updates (manual trigger)

## Maintenance

### Updating Golden Outputs

When examples change, update golden outputs:

```bash
# Update all golden outputs
python tests/examples/update_golden.py

# Update specific example
python tests/examples/update_golden.py --example "01_getting_started/hello_world.py"

# Update with real API calls
python tests/examples/update_golden.py --mode real
```

### Adding New Examples

1. Create the example in `examples/`
2. Add conditional execution with `@conditional_llm` if it uses APIs
3. Create test in appropriate `test_*.py` file
4. Run `update_golden.py` to generate initial golden output
5. Verify the golden output is correct

### Debugging Failed Tests

1. Check the error message for specific failures
2. Compare actual output with golden output
3. Verify API keys are set correctly (for real mode)
4. Check performance bounds aren't too restrictive
5. Update golden outputs if changes are intentional

## Best Practices

1. **Always use conditional execution**: Ensures examples work without API keys
2. **Provide realistic simulated responses**: Maintains educational value
3. **Keep performance bounds reasonable**: Allow for variability
4. **Update golden outputs promptly**: When examples change intentionally
5. **Review golden output changes**: Ensure they're expected and correct

## Troubleshooting

### Common Issues

**Tests fail with "Golden output not found"**
- Run `update_golden.py` to generate initial golden outputs

**Performance test failures**
- Increase `max_execution_time` if reasonable
- Check for unexpected delays in example code

**API key errors in CI**
- Ensure secrets are configured in GitHub repository
- Use `--no-api-keys` for CI runs without secrets

**Golden validation failures**
- Run `validate_golden.py` to check consistency
- Ensure golden files are valid JSON
- Check for version mismatches