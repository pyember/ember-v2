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
| data_api_example.py | ❌ | Deep imports | Consider API usage |
| custom_dataset_example.py | ❌ | Test failures | Fix mocks |
| context_example.py | ❌ | Different from basic | Rename? |
| transformation_example.py | ❌ | Test failures | Update patterns |
| explore_datasets.py | ❌ | Registry issues | Fix dataset registry |
| enhanced_builder_example.py | ⏭️ | - | Check if needed |
| mcq_experiment_example.py | ⏭️ | - | Advanced example |
| new_datasets_example.py | ❌ | Unknown | Review needed |
| README.md | ❌ | Outdated | Update |
| __init__.py | ✔️ | None | - |

## XCS Examples (8 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| jit_example.py | ❌ | Test failures | Update imports |
| example_simplified_xcs.py | ❌ | Poetry ref | Good patterns otherwise |
| auto_graph_example.py | ❌ | Test failures | Review graph API |
| auto_graph_simplified.py | ❌ | Test failures | Simpler version |
| enhanced_jit_example.py | ❌ | Complex | Performance focused |
| simple_autograph_example.py | ✔️ | None | Working |
| transforms_integration_example.py | ❌ | Test failures | Update transforms |
| README.md | ❌ | Outdated | Update |

## Advanced Examples (11 files)

| File | Status | Issues | Notes |
|------|--------|---------|--------|
| clean_jit_example.py | ⏭️ | - | Review later |
| context_performance_example.py | ⏭️ | - | Performance focused |
| custom_component_example.py | ⏭️ | - | Advanced pattern |
| diagnose_model_discovery.py | ❌ | Old API | Debugging tool |
| ensemble_judge_mmlu.py | ❌ | Deep imports | Real use case |
| example_architectures.py | ⏭️ | - | Architecture demos |
| model_benchmark_specialized_datasets.py | ⏭️ | - | Benchmarking |
| parallel_benchmark.py | ❌ | Deep imports | Performance test |
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
- **Updated**: 1 (2%)
- **Good**: 5 (9%)  
- **Needs Update**: 40 (74%)
- **Skipped**: 8 (15%)

## Next Actions

1. Fix test infrastructure (conftest.py)
2. Update high-impact examples (list_models.py, model_registry_direct.py)
3. Run golden tests after each update
4. Update READMEs with new patterns