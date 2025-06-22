# Merge Plan: xcs-radical-simplification + feat/simple-operators-v2 → core

## Overview
This document outlines the plan to merge two significant feature branches into a unified `core` branch that combines the best of both improvements while maintaining a clean, consistent API.

## Branch Analysis

### xcs-radical-simplification
**Philosophy**: "No false promises" - minimal API surface, zero configuration
- Simplified XCS from ~60 exports to 4 essential ones
- Removed internal implementation details from public API
- Made @jit zero-configuration
- Removed pmap (not useful for I/O-bound LLM operations)
- Clean, principled architecture

### feat/simple-operators-v2
**Philosophy**: Progressive complexity - simple for beginners, powerful for experts
- Dual syntax operators (simple dict-based + typed)
- 80-95% reduction in operator boilerplate
- Fixed type consistency issues across execution contexts
- Simplified model and data APIs
- Maintained backward compatibility

## Merge Strategy

### 1. Core Principles for Unified Branch
- **Progressive Simplicity**: Start simple, add complexity only when needed
- **Zero Configuration**: Everything should "just work" by default
- **Clean APIs**: One obvious way to do each task
- **Type Safety**: Optional but encouraged for production code
- **No False Promises**: Don't expose internals that users shouldn't touch

### 2. File-by-File Resolution Strategy

#### A. XCS Module (`src/ember/xcs/`)

**`src/ember/xcs/__init__.py`**
- **Take from**: xcs-radical-simplification
- **Reason**: Cleaner minimal API (4 exports vs 60+)
- **Exports to keep**: `jit`, `trace`, `vmap`, `get_jit_stats`
- **Not merging**: No content from feat/simple-operators-v2 needed here

**`src/ember/xcs/jit/core.py`**
- **Take from**: xcs-radical-simplification
- **Reason**: Removed unnecessary features (sample_input, TRACE mode)
- **Keep the `force_strategy` alias**: Good UX improvement
- **Not merging**: Discard feat/simple-operators-v2's extra parameters

**`src/ember/xcs/jit/__init__.py`**
- **Take from**: xcs-radical-simplification
- **Reason**: Doesn't expose JITMode enum (internal detail)
- **Not merging**: No changes needed from feat/simple-operators-v2

**`src/ember/xcs/transforms/`** (if conflicts exist)
- **Take from**: xcs-radical-simplification
- **Reason**: Removed pmap (not useful for I/O-bound operations)
- **Keep only**: vmap implementation

#### B. API Module (`src/ember/api/`)

**`src/ember/api/operators.py`**
- **Take from**: Either branch (identical content)
- **Reason**: No conflicts - same file in both branches
- **Already includes**: Dual syntax support, simplified imports

**`src/ember/api/models.py`**
- **Take from**: feat/simple-operators-v2
- **Reason**: Has the direct invocation pattern (`models("gpt-4", "prompt")`)
- **Not merging**: xcs-radical-simplification has older version

**`src/ember/api/data.py`**
- **Take from**: feat/simple-operators-v2
- **Reason**: Has the fluent builder pattern improvements
- **Not merging**: xcs-radical-simplification has older version

**`src/ember/api/xcs.py`**
- **Take from**: xcs-radical-simplification
- **Reason**: Exports only the minimal XCS API
- **Exports**: `jit`, `trace`, `vmap`, `get_jit_stats` only
- **Remove**: pmap, ExecutionOptions, Node, Graph exports

**`src/ember/api/xcs_old.py`**
- **Action**: Delete this file
- **Reason**: Deprecated, not needed with new simplified API

#### C. Core Implementation Files

**`src/ember/core/registry/operator/` (all operator files)**
- **Take from**: feat/simple-operators-v2
- **Reason**: Has the type consistency fixes
- **Critical fix**: Operators return consistent types in all contexts

**`src/ember/xcs/graph/` (if present)**
- **Take from**: xcs-radical-simplification
- **Reason**: Simplified Graph implementation
- **Hide**: Node class from public API

#### D. Examples (`src/ember/examples/`)

**All example files**
- **Take from**: feat/simple-operators-v2 
- **Reason**: Already updated to use simplified APIs
- **Additional changes**: Update any XCS imports to use minimal API

#### E. Documentation Files

**`README.md`**
- **MERGE BOTH**: Combine improvements
- **From xcs-radical-simplification**: Zero-config @jit description
- **From feat/simple-operators-v2**: Dual syntax operator examples
- **How**: Take structure from feat/simple-operators-v2, update XCS section with radical simplification philosophy

**`docs/xcs/README.md`**
- **Take from**: xcs-radical-simplification 
- **Reason**: Explains the simplified XCS philosophy
- **Add**: Note about removed features (pmap, ExecutionOptions)

**`docs/xcs/TRANSFORMS.md`**
- **Take from**: xcs-radical-simplification
- **Reason**: Documents only vmap (pmap removed)
- **Not merging**: No pmap documentation

**`api_sketch_slides.md`** (if including)
- **Take from**: Current version (already merged improvements)
- **Reason**: Already combines both branches' improvements

#### F. Test Files

**`tests/unit/xcs/`**
- **Take from**: xcs-radical-simplification
- **Reason**: Tests aligned with simplified API
- **Remove**: Tests for removed features (pmap, execution options)

**`tests/integration/`**
- **Take from**: feat/simple-operators-v2
- **Reason**: Tests for operator type consistency
- **Update**: Remove XCS strategy selection tests

#### G. Configuration Files

**`pyproject.toml`, `setup.py`, etc.**
- **Take from**: Either branch (check for version differences)
- **Update**: Version number to reflect merged improvements

### 3. Special Merge Cases

**Files requiring careful merging (not just taking one version):**

1. **`README.md`**
   - Take overall structure from feat/simple-operators-v2
   - Replace XCS section with content from xcs-radical-simplification
   - Combine examples showing both operator dual syntax AND zero-config JIT

2. **Stashed changes** (from stash@{0})
   - Cherry-pick only the docstring improvements
   - Skip any changes that reintroduce complexity
   - Apply formatting improvements

3. **Any file with import statements**
   - Update imports to use new minimal APIs
   - Remove imports of deleted features (pmap, ExecutionOptions, etc.)

### 3. Conflict Resolution Rules

1. **When in doubt, choose simplicity**
   - If a feature can be internal, make it internal
   - If an API can be simpler, simplify it

2. **Preserve user-facing improvements**
   - Keep dual syntax for operators
   - Keep direct model invocation
   - Keep fluent data builder

3. **Remove complexity**
   - No manual JIT strategy selection
   - No execution options in public API
   - No pmap (not useful for LLM operations)

## Testing Strategy

### 1. Pre-Merge Testing
- Run all tests on both branches independently
- Document current test pass rates
- Identify tests that will need updates

### 2. Post-Merge Testing
- Run full test suite
- Update tests for new APIs
- Add tests for merged functionality
- Target: >95% test pass rate

### 3. Integration Testing
- Test all examples work correctly
- Test backward compatibility shims
- Test performance characteristics

## Documentation Review Plan

### 1. Google Python Style Guide Compliance
- Docstrings: One-line summary, Args, Returns, Raises, Examples
- Comments: Explain why, not what
- Type hints: Use throughout
- Line length: 80 characters for docstrings

### 2. Core Functions Documentation
Each core function needs:
- Clear one-line summary
- Detailed description if needed
- Usage examples
- Common patterns
- Performance notes

### 3. File Categories for Review
1. **Core API files** (highest priority)
   - `src/ember/api/*.py`
   - `src/ember/xcs/__init__.py`
   - `src/ember/xcs/jit/core.py`

2. **Operator implementations**
   - `src/ember/core/registry/operator/*.py`
   - Focus on base classes and core operators

3. **Examples** (ensure they're educational)
   - All files in `src/ember/examples/`
   - Clear, progressive complexity

## Execution Steps

1. **Create core branch from main**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b core
   ```

2. **Merge xcs-radical-simplification**
   ```bash
   git merge xcs-radical-simplification
   # Resolve conflicts favoring simplification
   ```

3. **Merge feat/simple-operators-v2**
   ```bash
   git merge feat/simple-operators-v2
   # Resolve conflicts per strategy above
   ```

4. **Apply stashed changes selectively**
   ```bash
   git stash show -p stash@{0}
   # Cherry-pick valuable changes
   ```

5. **Run tests and fix issues**
   ```bash
   pytest tests/
   # Fix any failures
   ```

6. **Documentation review**
   - Review EVERY Python file
   - Update docstrings to Google style
   - Add usage examples
   - Clean up comments

7. **Final testing**
   ```bash
   pytest tests/
   mypy src/ember/
   ```

8. **Create PR**
   - Comprehensive description
   - Breaking changes documented
   - Migration guide if needed

## Success Criteria

1. **Unified API** that combines best of both branches
2. **>95% test pass rate**
3. **All docstrings** follow Google Python Style Guide
4. **Core functions** have usage examples
5. **Clean architecture** with no leaked abstractions
6. **Progressive complexity** preserved
7. **Zero configuration** for common cases

## Risk Mitigation

1. **Breaking Changes**
   - Document all breaking changes
   - Provide migration guide
   - Consider compatibility shims for critical APIs

2. **Test Failures**
   - Fix tests incrementally
   - Don't merge until tests pass
   - Add new tests for merged features

3. **Documentation Gaps**
   - Review systematically
   - Use tooling to check docstring format
   - Get examples from actual usage

## Timeline Estimate

1. Branch setup and initial merge: 1 hour
2. Conflict resolution: 2-3 hours  
3. Test fixing: 2-3 hours
4. Documentation review: 4-6 hours
5. Final testing and cleanup: 1-2 hours

**Total: 10-15 hours of focused work**