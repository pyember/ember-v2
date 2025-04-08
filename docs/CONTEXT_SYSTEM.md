# Ember Context System

## Overview

The Ember Context System provides a lightweight, zero-overhead approach to component discovery and dependency management. This redesigned system eliminates circular dependencies, improves thread safety, and enhances performance while significantly reducing complexity.

## Key Concepts

### Registry

The `Registry` is the only global abstraction in the system. It provides a thread-local dictionary that components use to discover each other:

```python
from ember.core.context import Registry

# Get current thread's registry
registry = Registry.current()

# Register a component
registry.register("my_component", component_instance)

# Get a component
component = registry.get("my_component")
```

### Component

The `Component` base class provides common functionality for all components:

```python
from ember.core.context import Component

class MyComponent(Component):
    def _register(self):
        """Register in registry."""
        self._registry.register("my_component", self)
    
    def _initialize(self):
        """Initialize lazily."""
        # Initialization logic
```

### Core Components

The system includes these core components:

- **ConfigComponent**: Configuration management
- **ModelComponent**: Model discovery and creation
- **DataComponent**: Dataset management
- **MetricsComponent**: Performance metrics collection

## Migration Guide

### Old API vs New API

Old API (EmberContext):

```python
from ember.core.app_context import get_app_context

# Get context
context = get_app_context()

# Get model and dataset
model = context.get_model("my_model")
dataset = context.get_dataset("my_dataset")
```

New API (Component-based):

```python
from ember.core.context.model import ModelComponent
from ember.core.context.data import DataComponent

# Get components
model_component = ModelComponent()
data_component = DataComponent()

# Get model and dataset
model = model_component.get_model("my_model")
dataset = data_component.get_dataset("my_dataset")
```

### Step 1: Update Imports

Change your imports from:

```python
from ember.core.app_context import get_app_context, EmberContext
```

To:

```python
from ember.core.context.compatibility import current_context, EmberContext
```

This provides compatibility with the old API while using the new implementation.

### Step 2: Migrate to Component-Based API

For each usage of `get_app_context()`, consider migrating to the direct component API:

Before:

```python
context = get_app_context()
model = context.get_model("gpt-4")
```

After:

```python
from ember.core.context.model import ModelComponent
model_component = ModelComponent()
model = model_component.get_model("gpt-4")
```

### Step 3: Update Mocks in Tests

Update test mocks to mock the component directly rather than the context:

Before:

```python
# Mock EmberContext
context_mock = MagicMock()
context_mock.get_model.return_value = model_mock
monkeypatch.setattr("ember.core.app_context.get_app_context", lambda: context_mock)
```

After:

```python
# Mock ModelComponent
model_component_mock = MagicMock()
model_component_mock.get_model.return_value = model_mock
monkeypatch.setattr("ember.core.context.model.ModelComponent.get", 
                    lambda cls: model_component_mock)
```

Or use the scoped registry for cleaner tests:

```python
from ember.core.context import scoped_registry
from ember.core.context.model import ModelComponent

# Create isolated registry for test
with scoped_registry() as registry:
    # Create component with explicit registry
    model = ModelComponent(registry)
    
    # Register mock model
    model.register_model("test_model", mock_model)
    
    # Test code that uses model
    result = test_function()  # Will use mock_model
```

## Best Practices

1. **Direct Component Usage**: Access components directly instead of through EmberContext
2. **Explicit Dependencies**: Pass components as arguments instead of implicitly accessing them
3. **Scoped Testing**: Use `scoped_registry()` for isolated tests
4. **Lazy Loading**: Let components initialize themselves when first accessed
5. **Thread Safety**: Design components to be thread-safe using the double-checked locking pattern

## Performance Considerations

The new context system is designed for optimal performance:

- Thread-local storage eliminates most contention
- Lazy initialization reduces startup costs
- Component caching accelerates repeated lookups
- Direct component interaction reduces indirection

## Thread Safety

Thread safety is achieved through:

1. **Thread-Local Storage**: Each thread has its own isolated registry
2. **Double-Checked Locking**: Efficient lazy initialization pattern
3. **Fine-Grained Locking**: Component-specific locks for minimal contention

With this approach, most operations require no locking, and only registration and initialization need synchronization.

## Advanced Usage

### Custom Components

Creating custom components is straightforward:

```python
from ember.core.context import Component, Registry

class MyCustomComponent(Component):
    def __init__(self, registry=None):
        super().__init__(registry)
        self._data = {}
    
    def _register(self):
        self._registry.register("my_custom", self)
    
    def _initialize(self):
        # Lazy initialization logic
        config = self._registry.get("config")
        if config:
            self._data = config.get_config("my_custom") or {}
    
    def get_item(self, key):
        self._ensure_initialized()
        return self._data.get(key)
```

### Registry Scoping

For isolated execution contexts:

```python
from ember.core.context import scoped_registry
from ember.core.context.config import ConfigComponent

def isolated_function():
    with scoped_registry() as registry:
        # Create components with isolated registry
        config = ConfigComponent(registry, config_data={"custom": {"key": "value"}})
        
        # Function logic using isolated components
        result = process_with_config(config)
        
        return result
```

This ensures that components created within the function don't affect the rest of the application.

## Architecture

The new architecture simplifies the design to a single abstraction with direct component interaction:

```
┌─────────────────────────────────────────────────────────────┐
│ User Code                                                   │
│                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│  │ ModelComponent│    │ DataComponent │    │ConfigComponent│  │
│  │ get_model()   │    │ get_dataset() │    │ get_config()  │  │
│  └───────────────┘    └───────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Thread-Local Registry                                        │
│                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│  │ Thread 1      │    │ Thread 2      │    │ Thread 3      │  │
│  │ Registry      │    │ Registry      │    │ Registry      │  │
│  └───────────────┘    └───────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Component Registration                                       │
│                                                             │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│  │ "model" →     │    │ "data" →      │    │ "config" →    │  │
│  │ ModelComponent│    │ DataComponent │    │ConfigComponent│  │
│  └───────────────┘    └───────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

This design:
- Eliminates circular dependencies
- Reduces indirection
- Improves component discoverability
- Enhances thread safety
- Optimizes performance

## Implementation Status

- ✅ Core thread-local registry - Fully implemented with `Registry` class
- ✅ Component base class - Common functionality for all components
- ✅ Core components - Config, Model, Data, and Metrics implementations
- ✅ Management utilities - Scoped registry and temporary components
- ✅ Compatibility layer - Integration with existing code
- ✅ Documentation - Migration guide and best practices