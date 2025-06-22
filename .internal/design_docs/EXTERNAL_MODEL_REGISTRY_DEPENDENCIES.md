# External Dependencies on ember.core.registry.model

This document lists all files outside of the `ember/core/registry/model` directory that import from `ember.core.registry.model`.

## Core Package Files

### src/ember/__init__.py
- Line 138: `from ember.core.registry.model.initialization import initialize_registry`

## Test Files

### tests/conftest.py
- Line 60: `from ember.core.registry.model.base.schemas.chat_schemas import ...`
- Line 65: `from ember.core.registry.model.providers.base_provider import BaseProviderModel`

### tests/unit/api/models/test_costs.py
- Line 11: `from ember.core.registry.model._costs import ...`

### tests/unit/api/models/test_model_registry.py
- Line 11: `from ember.core.registry.model.base.registry.model_registry import ModelRegistry`
- Line 12: `from ember.core.registry.model.base.utils.model_registry_exceptions import ModelNotFoundError`

### tests/unit/api/models/test_provider_registry.py
- Line 10: `from ember.core.registry.model.providers._registry import ...`
- Line 20: `from ember.core.registry.model.providers.base_provider import BaseProviderModel`

### tests/integration/core/test_integration_core.py
- Line 15: `from ember.core.registry.model.base.registry.model_registry import ModelRegistry`
- Line 16: `from ember.core.registry.model.base.schemas.cost import ModelCost, RateLimit`
- Line 17: `from ember.core.registry.model.base.schemas.model_info import ModelInfo`
- Line 18: `from ember.core.registry.model.base.schemas.provider_info import ProviderInfo`
- Line 19: `from ember.core.registry.model.base.services.model_service import ModelService`
- Line 20: `from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig`
- Line 242: `from ember.core.registry.model.base.schemas.chat_schemas import ...`
- Line 245: `from ember.core.registry.model.providers.base_provider import BaseProviderModel`

### tests/integration/core/test_minimal_config.py
- Line 21: `from ember.core.registry.model.base.registry.model_registry import ModelRegistry`

### tests/integration/core/registry/test_provider_discovery.py
- Line 12: `from ember.core.registry.model.base.registry.model_registry import ModelRegistry`
- Line 13: `from ember.core.registry.model.providers.anthropic.anthropic_discovery import ...`
- Line 15: `from ember.core.registry.model.providers.deepmind.deepmind_discovery import ...`
- Line 17: `from ember.core.registry.model.providers.openai.openai_discovery import OpenAIDiscovery`

### tests/golden/test_models_examples.py
- Line 164: Contains check for "from ember.core.registry.model" in content

### tests/golden/run_golden_tests.py
- Line 70: Contains pattern check for "from ember.core.registry.model"

## Example Files

### src/ember/examples/legacy/models/function_style_api.py
- Line 94: `from ember.core.registry.model.base.errors import ...`
- Line 197: Contains string reference "from ember.core.registry.model import lm"

### src/ember/examples/legacy/operators/custom_prompt_example_caravan.py
- Line 107: `from ember.core.registry.model.base.schemas.model_info import ModelInfo, ModelCost, RateLimit`

### src/ember/examples/legacy/advanced/diagnose_model_discovery.py
- Line 32: `from ember.core.registry.model.base.registry.discovery import ModelDiscoveryService`
- Line 33: `from ember.core.registry.model.base.registry.model_registry import ModelRegistry`
- Line 34: `from ember.core.registry.model.providers.base_discovery import BaseDiscoveryProvider`
- Line 35: `from ember.core.registry.model.providers.registry import PROVIDER_REGISTRY`
- Line 102: `from ember.core.registry.model.providers import ...`
- Line 285: `from ember.core.registry.model.base.registry.model_registry import ModelRegistry`

## Maintenance Scripts

### .internal_docs/maintenance_scripts/validate_migration.py
- Contains imports from ember.core.registry.model

## Backup Files

### .backup/models_v1/* (multiple files)
- These are backup files that contain old model registry imports

## Summary

The main external dependencies are:
1. **Core initialization**: `src/ember/__init__.py` uses the initialization module
2. **Test infrastructure**: `conftest.py` and various test files import schemas and providers
3. **Example files**: Legacy examples still reference the model registry
4. **Golden tests**: Check for model registry imports as part of their validation

These files will need to be updated if the model registry is moved or restructured.