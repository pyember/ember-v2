# Ember Dependency Analysis

## Root Dependency Issues

### 1. Equinox (CRITICAL)
**Location**: `src/ember/core/module.py`
**Usage**: Base Module class for all Ember components
**Problem**: 
- Equinox is a heavy neural network library built on JAX
- We only use it for its Module class to get:
  - Immutability via frozen dataclasses
  - JAX pytree registration for automatic differentiation
- This single import brings in the entire neural network framework

**Solution**:
Replace with a simple custom Module class that provides:
- Immutability through `__setattr__` override
- Optional JAX pytree registration only when JAX is available
- Same API surface but without the heavy dependencies

### 2. PSUtil (MODERATE)
**Location**: `src/ember/core/utils/eval/code_execution.py`
**Usage**: Monitor memory usage of code execution processes
**Problem**:
- System-level dependency that may not be available on all platforms
- Only used in one specific evaluator for competitive programming
- Makes the core untestable in restricted environments

**Solution**:
Make it an optional import with graceful degradation:
```python
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
```

### 3. JAX (CRITICAL)
**Locations**: 
- Used with equinox for pytree operations
- XCS module for parallelism analysis
**Problem**:
- Heavy ML framework dependency
- Only needed for advanced XCS optimization features
- Makes basic Ember usage require GPU-capable JAX installation

**Solution**:
Make JAX optional - only import when XCS features are used

### 4. Pandas (MODERATE)
**Usage**: Data processing in data API
**Problem**:
- Heavy dependency with many transitive dependencies
- Not needed for core operator/model functionality

**Solution**:
Move to optional dependency for data processing features

### 5. Datasets (MODERATE)  
**Usage**: Hugging Face datasets integration
**Problem**:
- Large library with many dependencies
- Only needed for specific dataset loading

**Solution**:
Make it part of the `[data]` extra, not core

## Principled Refactoring Plan

### Phase 1: Remove Equinox Dependency
1. Create `src/ember/core/base_module.py` with a simple Module implementation
2. Update `src/ember/core/module.py` to use the new base
3. Ensure backward compatibility with same API

### Phase 2: Make Heavy Dependencies Optional
1. Move pandas, datasets to optional imports
2. Add proper error messages when features requiring them are used
3. Update pyproject.toml to move them to extras

### Phase 3: Fix PSUtil Usage
1. Make psutil optional in code_execution.py
2. Return 0.0 memory usage when not available
3. Add warning in logs when memory monitoring is disabled

### Phase 4: Create Minimal Core
Update pyproject.toml dependencies to have truly minimal core:
```toml
dependencies = [
    # Absolute minimum
    "pydantic>=2.7.4",
    "pydantic-settings>=2.3.0", 
    "PyYAML>=6.0.1",
    "typing_extensions>=4.12.2",
    
    # For API calls
    "httpx>=0.27.0",
    "tenacity>=9.0.0",
    
    # Basic utilities
    "tqdm>=4.67.1",
    "cachetools>=5.4.0",
]
```

## Benefits
1. **Faster installation**: Core goes from ~1GB to ~50MB of dependencies
2. **Better testability**: Can run tests without GPU or system dependencies  
3. **Cleaner architecture**: Dependencies match actual usage
4. **Progressive enhancement**: Users only install what they need

## Next Steps
1. Implement custom Module class to replace equinox
2. Make psutil optional with graceful degradation
3. Move heavy dependencies to extras in pyproject.toml
4. Update imports to handle optional dependencies gracefully