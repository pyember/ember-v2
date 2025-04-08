# Context System Integration Tests

These tests verify that the EmberContext system works correctly with real components.

## Test Files

- `test_basic_context.py`: Basic context creation, component registration and retrieval
- `test_thread_isolation.py`: Thread-local isolation of contexts
- `test_mock_models.py`: Using the context with mock model implementations

## Running Tests

```bash
# Run all context integration tests
uv run pytest tests/integration/core/context/integration

# Run specific test
uv run pytest tests/integration/core/context/integration/test_basic_context.py
```
