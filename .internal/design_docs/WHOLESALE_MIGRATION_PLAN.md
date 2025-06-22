# Wholesale Migration Plan - Models Registry

## Overview
Complete transition from old complex registry to new simplified registry with proper naming.

## Step 1: Backup Old Files
```bash
mkdir -p backup/old_model_registry
cp -r src/ember/core/registry/model/base/registry/* backup/old_model_registry/
cp -r src/ember/core/registry/model/base/services/* backup/old_model_registry/
cp -r src/ember/core/registry/model/base/context/* backup/old_model_registry/
```

## Step 2: Rename New Files (Remove "Simple" Prefix)
- `simple_model_registry.py` → `model_registry.py`
- `SimpleModelRegistry` class → `ModelRegistry` class

## Step 3: Update Core Imports

### Files that need updating:
1. **src/ember/api/models.py**
   - Change: `from ...simple_model_registry import SimpleModelRegistry`
   - To: `from ...model_registry import ModelRegistry`

2. **src/ember/core/registry/model/base/services/model_service.py**
   - Change: `SimpleModelRegistry` → `ModelRegistry`

3. **src/ember/core/context/ember_context.py**
   - Already imports ModelRegistry, just needs to use new implementation

4. **src/ember/core/registry/model/initialization.py**
   - Update to use new ModelRegistry

5. **src/ember/core/registry/model/config/settings.py**
   - Update to use new ModelRegistry

## Step 4: Update Provider Registry
- Keep `_registry.py` as is (already simplified)
- Keep `_costs.py` as is (already simplified)

## Step 5: Remove Old Code
- Delete old model_registry.py (compatibility shim)
- Delete old discovery.py, factory.py if they still exist
- Delete old context files if unused

## Step 6: Update Tests
- Move any remaining old tests to backup
- Ensure new tests use correct imports
- Run all tests to verify

## Files to Update (Priority Order):

1. **Critical Path**:
   - `simple_model_registry.py` → `model_registry.py` (rename)
   - `models.py` (update import)
   - `model_service.py` (update import)

2. **Secondary**:
   - `ember_context.py` (should work with new registry)
   - `initialization.py`
   - `config/settings.py`

3. **Examples/Legacy**:
   - Can be updated later or marked as deprecated

## Expected Outcome
- Clean architecture with standard naming
- No "Simple" prefix anywhere
- No compatibility shims
- All tests passing
- ~40% less code overall