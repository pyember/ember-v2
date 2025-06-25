# Testing Principles

Following CLAUDE.md and Google Python Style Guide principles:

## Core Principles

1. **Minimal**: Each test should test exactly one thing
2. **Focused**: No unnecessary setup or assertions  
3. **No Magic**: Explicit is better than implicit
4. **Fast**: Tests should run in milliseconds
5. **Hermetic**: Tests should be completely isolated
6. **Deterministic**: Same result every time

## Good Patterns

### 1. Single Assertion Per Test (When Possible)
```python
def test_config_get_returns_none_for_missing_key(self, tmp_ctx):
    """Missing keys return None."""
    assert tmp_ctx.get_config("missing") is None
```

### 2. Parametrized Tests for Similar Cases
```python
@pytest.mark.parametrize("key,expected", [
    ("", None),
    (None, None),
    (123, None),
])
def test_invalid_keys_return_none(self, tmp_ctx, key, expected):
    """Invalid keys return None."""
    assert tmp_ctx.get_config(key) == expected
```

### 3. Explicit Fixtures
```python
@pytest.fixture
def isolated_context(tmp_path):
    """Create isolated context with temporary directory."""
    ctx = EmberContext(isolated=True)
    ctx._config_file = tmp_path / "config.yaml"
    return ctx
```

### 4. Direct Testing (No Subprocess)
```python
def test_cli_command(self, mock_cli_args, capsys):
    """Test CLI directly without subprocess."""
    mock_cli_args("configure", "get", "test.key")
    ret = main()
    assert ret == 0
```

## Bad Patterns to Avoid

### 1. Testing Multiple Things
```python
# BAD
def test_config_operations(self):
    ctx.set_config("key", "value")
    assert ctx.get_config("key") == "value"
    ctx.set_config("nested.key", "value")
    assert ctx.get_config("nested.key") == "value"
    # Testing too many things in one test
```

### 2. Implicit Dependencies
```python
# BAD
def test_something(self):
    # Assumes previous test set up state
    assert ctx.get_config("from_previous_test") == "value"
```

### 3. Magic Mock Behavior
```python
# BAD
mock = MagicMock()
# Relies on MagicMock's auto-creation of attributes

# GOOD
mock = Mock()
mock.specific_method = Mock(return_value="explicit")
```

### 4. Overly Complex Setup
```python
# BAD
def test_with_complex_setup(self):
    # 20 lines of setup
    # 1 line assertion
    
# GOOD - Extract to fixture or helper
@pytest.fixture
def complex_scenario(self):
    return create_scenario()
```

## Test Organization

1. **Group by functionality**: `TestContextCore`, `TestCredentials`, etc.
2. **Clear test names**: `test_<what>_<condition>_<expected>`
3. **Docstrings**: One line explaining the test
4. **No comments**: Test name and code should be self-documenting

## Performance Tests

- Use `time.perf_counter()` for timing
- Set reasonable thresholds (e.g., < 10ms for simple operations)
- Test both individual operations and bulk operations

## Error Cases

- Test one error condition per test
- Use `pytest.raises` with specific exception and match pattern
- Verify error doesn't corrupt state

## Cleanup

- Use fixtures for setup/teardown
- Ensure tests don't leak state
- Use `tmp_path` for file operations