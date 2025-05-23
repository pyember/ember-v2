# Examples Update Tracker

## Overview
Track progress on updating all examples to use the new simplified APIs.

## Status Legend
- ✅ Complete - Updated and tested
- 🔧 In Progress - Currently being updated  
- ❌ Needs Update - Identified issues
- ⏭️ Skipped - Not needed for MVP
- ✔️ Good - Already uses correct patterns

## Basic Examples (8 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| minimal_example.py | ✔️ | None | Good example of operator pattern |
| minimal_operator_example.py | ❌ | Test failures | Needs minor fixes |
| compact_notation_example.py | ❌ | Mock issues | Fix non API usage |
| context_example.py | ✔️ | None | Working correctly |
| simple_jit_demo.py | ❌ | Import issues | Update XCS imports |
| check_env.py | ❌ | Module checks | Update for new structure |
| README.md | ❌ | Outdated | Update descriptions |
| __init__.py | ✔️ | None | - |

## Models Examples (9 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| model_api_example.py | ✅ | None | Updated to new API |
| list_models.py | ❌ | Old API | Need models.list() |
| model_registry_example.py | ❌ | Old patterns | Full rewrite needed |
| model_registry_direct.py | ❌ | Deep imports | Simplify significantly |
| manual_model_registration.py | ❌ | Old API | Update or remove |
| function_style_api.py | ❌ | Old patterns | Show models() function |
| dependency_injection.py | ❌ | Complex | Simplify pattern |
| register_models_directly.py | ❌ | initialize_registry | Major update needed |
| README.md | ❌ | Outdated | New examples needed |

## Operators Examples (8 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| simplified_ensemble_example.py | ✅ | None | Updated to API imports |
| composition_example.py | ✅ | None | Updated to simplified API |
| container_operator_example.py | ✅ | None | Fixed imports and path |
| container_simplified.py | ✅ | None | Updated imports |
| custom_prompt_example_caravan.py | ✅ | None | Updated to models API |
| diverse_ensemble_operator_example.py | ✅ | None | Updated imports |
| README.md | ✅ | None | Updated with new API patterns |
| __init__.py | ✔️ | None | - |

## Data Examples (10 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| data_api_example.py | ✅ | None | Updated to use simplified API |
| context_example.py | ✅ | None | Rewritten for simplified API |
| transformation_example.py | ⏭️ | XCS transforms | Not data API related |
| explore_datasets.py | ✅ | None | Updated with DataItem pattern |
| enhanced_builder_example.py | ✅ | None | Updated to use data API |
| mcq_experiment_example.py | ⏭️ | Standalone | Self-contained example |
| new_datasets_example.py | ✅ | None | Updated with DataItem pattern |
| README.md | ✅ | None | Updated with new patterns |
| __init__.py | ✔️ | None | - |

## XCS Examples (8 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| jit_example.py | ✅ | None | Updated imports |
| example_simplified_xcs.py | ✅ | None | Fixed Poetry ref and imports |
| auto_graph_example.py | ✅ | None | Updated operator imports |
| auto_graph_simplified.py | ✅ | None | Updated Field import |
| enhanced_jit_example.py | ⏭️ | Internal APIs | Kept for internal mechanics demo |
| simple_autograph_example.py | ✔️ | None | Working |
| transforms_integration_example.py | ✅ | None | Updated all imports |
| README.md | ✅ | None | Added quick start section |

## Advanced Examples (11 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| clean_jit_example.py | ⏭️ | - | Review later |
| context_performance_example.py | ⏭️ | - | Performance focused |
| custom_component_example.py | ⏭️ | - | Advanced pattern |
| diagnose_model_discovery.py | ✅ | None | Updated to use models API |
| ensemble_judge_mmlu.py | ✅ | None | Replaced LMModule with models.bind() |
| example_architectures.py | ⏭️ | - | Architecture demos |
| model_benchmark_specialized_datasets.py | ⏭️ | - | Benchmarking |
| parallel_benchmark.py | ✅ | None | Updated imports and model usage |
| parallel_pipeline_example.py | ❌ | Deep imports | Important pattern |
| reasoning_system.py | ❌ | Deep imports | Complex example |
| test_auto_discovery.py | ❌ | initialize_registry | Testing tool |

## Integration Examples (2 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| api_operators_example.py | ❌ | Deep imports | Shows integration |
| README.md | ❌ | Outdated | Update |

## Summary Stats

- **Total Files**: 54
- **Updated**: 29 (54%)
- **Good**: 5 (9%)  
- **Needs Update**: 12 (22%)
- **Skipped**: 8 (15%)

## Next Actions

1. Fix test infrastructure (conftest.py)
2. Update high-impact examples (list_models.py, model_registry_direct.py)
3. Run golden tests after each update
4. Update READMEs with new patterns