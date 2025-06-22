# Models API Migration Plan

## Overview

This document outlines the migration from the old models.py to the new models_v2.py implementation. The migration is designed to minimize disruption while providing a clean break from the old complexity.

## Current Usage Analysis

### 1. Direct API Usage Locations
- **Core modules**: 
  - `src/ember/core/non.py` - Uses models and ModelBinding
  - `src/ember/core/registry/operator/base/operator_base.py` - Uses models and ModelBinding
  - `src/ember/core/registry/operator/core/ensemble.py` - Uses models
  
- **Provider self-tests**:
  - `src/ember/core/registry/model/providers/anthropic/anthropic_provider.py`
  - `src/ember/core/registry/model/providers/deepmind/deepmind_provider.py`
  - `src/ember/core/registry/model/providers/openai/openai_provider.py`

- **CLI commands**:
  - `src/ember/cli/commands/model.py`
  - `src/ember/cli/commands/invoke.py`
  - `src/ember/cli/commands/project.py`

- **Examples**: 30+ example files in src/ember/examples/

- **Tests**: Multiple integration and unit tests

### 2. Import Patterns Found
```python
# Pattern 1: Direct models import
from ember.api import models

# Pattern 2: From models module
from ember.api.models import models

# Pattern 3: Additional imports
from ember.api import models, ModelBinding
from ember.api.models import ModelService, UsageService
```

## Migration Strategy

### Phase 1: Fix Plugin System Imports (Blocking Issue)

The providers currently import from `ember.plugin_system` which should be `ember.core.plugin_system`:

```python
# Current (broken)
from ember.plugin_system import PluginSystem

# Fixed
from ember.core.plugin_system import PluginSystem
```

**Files to fix**:
- `src/ember/core/registry/model/providers/anthropic/anthropic_provider.py`
- `src/ember/core/registry/model/providers/deepmind/deepmind_provider.py`
- `src/ember/core/registry/model/providers/openai/openai_provider.py`

### Phase 2: Update API Exports

Modify `src/ember/api/__init__.py` to use models_v2:

```python
# Change from:
from ember.api.models import models

# To:
from ember.api.models_v2 import models
```

### Phase 3: Update Core Module Imports

The good news: Most code uses `from ember.api import models`, which means we only need to change the export in `api/__init__.py`. The import statements in consuming code remain the same!

### Phase 4: Handle Breaking Changes

#### 4.1 ModelBinding Import
The new API exports ModelBinding directly. Update imports:

```python
# Old
from ember.api import models, ModelBinding

# New (no change needed - ModelBinding is exported from models_v2)
from ember.api import models, ModelBinding
```

#### 4.2 Service/Registry Access
Some advanced code imports internals:

```python
# Old
from ember.api.models import ModelService, UsageService

# New path
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.registry.model.base.services.usage_service import UsageService
```

### Phase 5: Update Tests

Tests that rely on internal implementation details need updates:
- Mock objects may need adjustment
- Registry access patterns change
- Provider discovery tests need rewriting

### Phase 6: Update Examples

Most examples should work unchanged due to API compatibility. Only advanced examples accessing internals need updates.

### Phase 7: Remove Old Implementation

Once all migrations are complete:
1. Delete `src/ember/api/models.py`
2. Rename `models_v2.py` to `models.py`
3. Remove old registry implementation files
4. Clean up deprecated provider discovery code

## Implementation Order

1. **Fix plugin_system imports** (Critical blocker)
2. **Update api/__init__.py** (Single change affects everything)
3. **Run tests** to identify specific breakages
4. **Fix tests** one by one
5. **Update examples** that fail
6. **Remove old code** (clean break)

## Risk Assessment

- **Low Risk**: Most code uses the public API which remains compatible
- **Medium Risk**: Tests that mock internals
- **High Risk**: Provider implementations (but we control these)

## Validation Steps

1. All unit tests pass
2. All integration tests pass
3. All examples run successfully
4. CLI commands work
5. Provider self-tests pass

## Timeline Estimate

- Phase 1 (Plugin fixes): 30 minutes
- Phase 2-3 (API export): 15 minutes
- Phase 4 (Breaking changes): 1 hour
- Phase 5 (Tests): 2-3 hours
- Phase 6 (Examples): 1 hour
- Phase 7 (Cleanup): 30 minutes

**Total: 5-6 hours of focused work**

## Success Criteria

1. All existing code works with minimal changes
2. No public API breaking changes (only internal refactoring)
3. Performance improves due to simpler architecture
4. Code is ~40% smaller and easier to understand