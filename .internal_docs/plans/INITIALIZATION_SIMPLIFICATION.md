# Ember Initialization Simplification

## Goal
Simplify the initialization and configuration system while **preserving all core functionality** (XCS, operators, type system, etc.)

## Current Problems

### 1. Multiple Initialization Paths (5+)
```python
# Method 1: Full initialization
ember.initialize_ember(config_path="...", auto_discover=True, ...)

# Method 2: Simple init
ember.init(config={...})

# Method 3: Direct registry
initialize_registry(...)

# Method 4: Auto-init via context
EmberContext.current()  # Creates default

# Method 5: Direct context creation
EmberContext(config_manager=...)
```

### 2. Dual Context Systems
- `EmberContext` - General framework context
- `ModelContext` - Model-specific context
- Both doing similar things, causing confusion

### 3. Complex Configuration
- YAML files with nested schemas
- Environment variable magic
- Auto-discovery creating 200+ entries
- Multiple config sources

## Proposed Solution

### 1. Single Initialization Method
```python
# Zero config - just works with env vars
import ember
# Ready to use - auto-initializes on first use

# With explicit config (optional)
import ember
ember.configure(
    auto_discover=False,  # Don't scan for all models
    cache_enabled=True,   # Enable response caching
    # Any other settings
)
```

### 2. Unified Context
Merge `EmberContext` and `ModelContext` into one:
```python
class Context:
    """Single context for all Ember components."""
    
    def __init__(self):
        self._models = None      # Lazy init
        self._operators = None   # Lazy init
        self._datasets = None    # Lazy init
        self._xcs_engine = None  # Lazy init
    
    @property
    def models(self):
        if self._models is None:
            self._models = self._init_models()
        return self._models
```

### 3. Environment-First Config
```python
# Primary configuration via environment
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=ant-...
EMBER_CACHE_ENABLED=true
EMBER_AUTO_DISCOVER=false

# Optional Python config for advanced cases
ember.configure(cache_enabled=False)  # Overrides env
```

### 4. Lazy Everything
- No initialization on import
- Components created on first use
- Fast startup, no delays

## What Stays The Same

- **XCS System** - All optimization and graph execution
- **Operators** - Full operator framework and patterns
- **Type System** - All type checking and specifications
- **Models API** - Current simple interface
- **Data System** - All dataset functionality
- **Evaluation** - All evaluation pipelines

## Implementation Plan

### Phase 1: Context Unification
1. Create unified `Context` class
2. Migrate EmberContext functionality
3. Migrate ModelContext functionality
4. Update all components to use unified context

### Phase 2: Simplify Configuration
1. Make environment variables primary
2. Remove YAML requirement (keep as option)
3. Add simple `ember.configure()` for overrides
4. Remove complex schema validation

### Phase 3: Remove Extra Init Methods
1. Keep only auto-init and `configure()`
2. Add deprecation warnings to old methods
3. Update documentation and examples

### Phase 4: Optimize Startup
1. Ensure all components lazy-load
2. Remove auto-discovery by default
3. Profile and optimize import time

## Example Usage After Changes

### Basic Usage (99% of cases)
```python
import ember
from ember.api import models, operators

# Just works - reads API keys from env
response = models("gpt-4", "Hello!")

# Operators work as before
ensemble = operators.Ensemble(
    models=["gpt-4", "claude-3"],
    aggregation="majority"
)
result = ensemble("What is 2+2?")

# XCS works as before  
from ember.xcs import jit

@jit
def pipeline(text):
    summary = models("gpt-4", f"Summarize: {text}")
    return summary
```

### Advanced Usage
```python
import ember

# Optional configuration
ember.configure(
    auto_discover=True,      # Discover all available models
    cache_dir="/tmp/ember",  # Custom cache location
    retry_count=5,           # Custom retry logic
)

# Everything else works the same
```

## Benefits

1. **Simpler Mental Model** - One context, one config method
2. **Faster Startup** - No auto-discovery, lazy loading
3. **Cleaner API** - Just `import ember` and go
4. **Preserves Power** - All advanced features intact
5. **Better Errors** - Clear messages when API keys missing

## Migration Path

```python
# Old code continues to work with warnings
ember.initialize_ember(...)  # Warning: Use ember.configure()

# New code is simpler
ember.configure(...)  # Optional
```

## Success Metrics

1. Time to first model call: < 100ms
2. Lines of init code: Reduce by 70%
3. Number of init methods: From 5+ to 2 (auto + configure)
4. User understanding: One obvious way to start