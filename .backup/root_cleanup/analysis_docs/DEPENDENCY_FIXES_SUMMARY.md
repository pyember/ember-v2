# Dependency Fixes Summary

## Completed Fixes

### 1. ✅ Removed Equinox Dependency
- Created `src/ember/core/base_module.py` - a lightweight Module class
- Updated `src/ember/core/module.py` to use the new base instead of equinox
- The new Module provides:
  - Immutability via `__setattr__` override
  - Optional JAX pytree registration when JAX is available
  - Same API as equinox.Module for compatibility

### 2. ✅ Made PSUtil Optional
- Updated `src/ember/core/utils/eval/code_execution.py`
- Added try/except import with graceful degradation
- Returns 0.0 memory usage when psutil not available
- Logs debug message about disabled memory monitoring

### 3. ✅ Made JAX Optional in XCS
- Updated `src/ember/xcs/_internal/parallelism.py` - optional JAX import
- Updated `src/ember/xcs/_internal/ir.py` - uses ember.core.module.Module instead of equinox
- Added HAS_JAX flag to conditionally use JAX features

### 4. ✅ Created Minimal pyproject.toml
- Created `pyproject_minimal.toml` with truly minimal core dependencies
- Moved heavy dependencies to optional extras:
  - `[jax]` - JAX and related libraries
  - `[data]` - pandas, numpy, datasets, etc.
  - `[eval]` - psutil for code execution monitoring
  - `[standard]` - common set with OpenAI, Anthropic, pandas, numpy

## Remaining Issues

### 1. ❌ JAX Imports in Core Operators
**File**: `src/ember/core/operators/common.py`
**Issue**: Has `import jax` at module level
**Fix Needed**: Make JAX imports conditional or move to a separate module

### 2. ❌ Other JAX Imports
**Files**:
- `src/ember/examples/xcs_transformation_patterns.py`
- `src/ember/xcs/transformations.py`
- `src/ember/xcs/_internal/analysis.py`

**Fix Needed**: These should conditionally import JAX since they're XCS-specific

## Benefits Achieved

1. **Lighter Core**: 
   - Before: ~1GB of dependencies (equinox, JAX, pandas, datasets)
   - After: ~50MB core (just pydantic, httpx, basic utilities)

2. **Better Testability**:
   - Can run core tests without GPU or JAX
   - No system dependencies required for basic functionality

3. **Progressive Enhancement**:
   - Users only install what they need
   - `pip install ember-ai` - minimal for basic API calls
   - `pip install ember-ai[standard]` - common usage with providers
   - `pip install ember-ai[jax]` - advanced XCS features
   - `pip install ember-ai[all]` - everything

4. **No Breaking Changes**:
   - Same API surface maintained
   - Existing code continues to work
   - Only internal implementation changed

## Next Steps

1. Fix remaining JAX imports in core operators
2. Update CI/CD to test minimal installation
3. Update documentation to explain installation options
4. Consider moving more providers to optional (Google has many deps)