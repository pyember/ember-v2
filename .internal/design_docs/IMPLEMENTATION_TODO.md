# Implementation TODO List

*Explicit, trackable tasks for the internal simplification project*

## Day 0: Foundation Fixes (MUST DO FIRST)

### Morning: Remove Public API Leaks (4 hours)
- [ ] Change ModelsAPI.get_registry() to _get_registry() or remove (1h)
  - Exit: No public registry access in dir(ember.api.models)
- [ ] Add deprecation warning to operators.py (1h)
  - Exit: Warning shown on import
- [ ] Re-export operators_v2 content from operators.py (0.5h)
  - Exit: All operators_v2 exports available
- [ ] Fix or remove async methods that raise NotImplementedError (0.5h)
  - Exit: All public methods work or don't exist

### Afternoon: Scaffold Missing Pieces (4 hours)
- [ ] Create src/ember/ir package structure (0.5h)
  - Exit: __init__.py, ops.py, graph.py exist
- [ ] Implement OpType enum and Operation dataclass (1.5h)
  - Exit: Can create LLM_CALL operation
- [ ] Implement basic Graph class with optimize() stub (1h)
  - Exit: Graph([op]).optimize() runs (even if no-op)
- [ ] Export validate decorator from operators_v2 (1h)
  - Exit: @validate available in public API

## Day 0.5: CI/CD Infrastructure

### Morning: Automated Quality Gates (4 hours)
- [ ] Create scripts/check_module_size.py (1h)
  - Exit: Script fails if any module > 1000 LOC
- [ ] Add pre-commit hook for module size (0.5h)
  - Exit: Git commit warns about large modules
- [ ] Add CI job for module size check (0.5h)
  - Exit: PR fails if modules too large
- [ ] Create contract tests for public API (2h)
  - Exit: Tests fail if API grows unexpectedly

### Afternoon: Tech Debt Tracking (4 hours)
- [ ] Implement minimal DatasetMetadata (2h)
  - Exit: 8-field dataclass as designed
- [ ] Update DataContext to return new metadata (1h)
  - Exit: DataContext uses DatasetMetadata
- [ ] Create DEPRECATION.md with timeline (1h)
  - Exit: Clear dates for removal

## Project Setup (After Day 0/0.5)
- [ ] Create new branch: `internal-simplification`
- [ ] Set up testing framework for continuous validation
- [ ] Configure strict mypy settings
- [ ] Set up benchmark suite for performance tracking

## Week 1: Models Module Simplification

### Day 1: Research & Analysis (8 hours)
- [ ] **Morning: Current State Analysis**
  - [ ] Verify registry is actually hidden after Day 0 fixes (0.5h)
    - Exit: get_registry() not in public API
  - [ ] Map out current models.py dependencies (1.5h)
  - [ ] Document provider registry architecture (1h)
  - [ ] Identify all calling patterns in use (1h)

- [ ] **Afternoon: External Research**
  - [ ] Study LiteLLM implementation patterns (1h)
  - [ ] Study OpenRouter API design (1h)
  - [ ] Study Anthropic SDK patterns (1h)
  - [ ] Document best practices found (1h)

### Day 2: Design & Documentation (8 hours)
- [ ] **Morning: Questioning & Design**
  - [ ] Document why each registry component exists (2h)
  - [ ] Design new public API surface (1h)
  - [ ] Create API comparison chart (old vs new) (1h)

- [ ] **Afternoon: Detailed Design**
  - [ ] Design internal provider routing (2h)
  - [ ] Design cost tracking integration (1h)
  - [ ] Create implementation plan (1h)

### Day 3: Core Implementation (8 hours)
- [ ] **Morning: Base Implementation**
  - [ ] Implement new Models class (2h)
  - [ ] Implement __call__ method (1h)
  - [ ] Implement instance() method (1h)

- [ ] **Afternoon: Provider Integration**
  - [ ] Implement provider registry (hidden) (2h)
  - [ ] Implement provider routing logic (1h)
  - [ ] Add error handling (1h)

### Day 4: Advanced Features (8 hours)
- [ ] **Morning: Async Support**
  - [ ] Implement working async_call method (2h)
    - Exit: Async calls actually work
  - [ ] Add streaming support (2h)
    - Exit: Can iterate over streaming responses

- [ ] **Afternoon: Cost Tracking**
  - [ ] Implement CostTracker class (2h)
  - [ ] Integrate with Models class (1h)
  - [ ] Add cost reporting (1h)

### Day 5: Testing & Documentation (8 hours)
- [ ] **Morning: Testing**
  - [ ] Write unit tests for Models class (2h)
  - [ ] Write integration tests with providers (2h)

- [ ] **Afternoon: Documentation**
  - [ ] Write migration guide (2h)
  - [ ] Update examples (1h)
  - [ ] Create performance benchmarks (1h)

## Week 2: Data Module Simplification

### Day 1: Research & Analysis (8 hours)
- [ ] **Morning: Current State**
  - [ ] Map current data module structure (2h)
  - [ ] Document metadata system (1h)
  - [ ] Analyze dataset registry (1h)

- [ ] **Afternoon: External Research**
  - [ ] Study PyTorch DataLoader patterns (2h)
  - [ ] Study HuggingFace datasets design (2h)

### Day 2: Metadata Design (8 hours)
- [ ] **Morning: Masters' Convergence**
  - [ ] Remove legacy DatasetInfo imports (0.5h)
    - Exit: No DatasetInfo in new code
  - [ ] Verify DatasetMetadata from Day 0 (0.5h)
  - [ ] Document each field's purpose (1h)
  - [ ] Create metadata examples (1h)
  - [ ] Remove DataItem normalizer complexity (1h)
    - Exit: < 50 LOC for DataItem

- [ ] **Afternoon: API Design**
  - [ ] Design simplified Data class (2h)
  - [ ] Design streaming interface (1h)
  - [ ] Plan registry integration (1h)

### Day 3: Core Implementation (8 hours)
- [ ] **Morning: Base Implementation**
  - [ ] Implement Data class (2h)
  - [ ] Implement __call__ method (1h)
  - [ ] Implement list() method (1h)

- [ ] **Afternoon: Dataset Loading**
  - [ ] Implement dataset loading (2h)
  - [ ] Add transformation support (1h)
  - [ ] Add streaming support (1h)

### Day 4: Metadata & Registry (8 hours)
- [ ] **Morning: Metadata Implementation**
  - [ ] Implement DatasetMetadata class (2h)
  - [ ] Integrate with loading (2h)

- [ ] **Afternoon: Registry Integration**
  - [ ] Update registry to use new metadata (2h)
  - [ ] Ensure backward compatibility of datasets (2h)

### Day 5: Testing & Documentation (8 hours)
- [ ] **Morning: Testing**
  - [ ] Unit tests for Data class (2h)
  - [ ] Test streaming functionality (1h)
  - [ ] Test transformations (1h)

- [ ] **Afternoon: Documentation**
  - [ ] Write usage examples (2h)
  - [ ] Document metadata schema (1h)
  - [ ] Create migration guide (1h)

## Week 3: Operators Module Simplification

### Day 1: Research & Analysis (8 hours)
- [ ] **Morning: Current State**
  - [ ] Analyze specification system (2h)
  - [ ] Document validation patterns (1h)
  - [ ] Map operator hierarchies (1h)

- [ ] **Afternoon: External Research**
  - [ ] Study JAX Equinox patterns (1.5h)
  - [ ] Study DSPy modules (1.5h)
  - [ ] Document composition patterns (1h)

### Day 2: Progressive Disclosure Design (8 hours)
- [ ] **Morning: Design Levels**
  - [ ] Design Level 1: Just functions (2h)
  - [ ] Design Level 2: With validation (1h)
  - [ ] Design Level 3: Full specifications (1h)

- [ ] **Afternoon: Integration Design**
  - [ ] Design validation decorator (2h)
  - [ ] Plan progressive enhancement (1h)
  - [ ] Create usage examples (1h)

### Day 3: Protocol Implementation (8 hours)
- [ ] **Morning: Core Protocol**
  - [ ] Delete legacy operators.py after deprecation (0.5h)
    - Exit: Only operators_v2 remains
  - [ ] Implement Operator protocol (1h)
  - [ ] Implement composition utilities (1.5h)
  - [ ] Add type annotations (1h)

- [ ] **Afternoon: Validation System**
  - [ ] Verify validate decorator exported (0.5h)
  - [ ] Enhance validate implementation (1.5h)
  - [ ] Add example validation (1h)
  - [ ] Add error handling (1h)

### Day 4: Composition Utilities (8 hours)
- [ ] **Morning: Basic Composition**
  - [ ] Implement chain() (1h)
  - [ ] Implement parallel() (1h)
  - [ ] Implement ensemble() (1h)
  - [ ] Implement conditional() (1h)

- [ ] **Afternoon: Advanced Patterns**
  - [ ] Implement map_operator() (1h)
  - [ ] Add retry patterns (1h)
  - [ ] Add timeout patterns (1h)
  - [ ] Create complex examples (1h)

### Day 5: Testing & Documentation (8 hours)
- [ ] **Morning: Testing All Levels**
  - [ ] Test simple functions (1h)
  - [ ] Test validated functions (1h)
  - [ ] Test specifications (1h)
  - [ ] Test composition (1h)

- [ ] **Afternoon: Documentation**
  - [ ] Document progressive disclosure (2h)
  - [ ] Create examples for each level (1h)
  - [ ] Write migration guide (1h)

## Week 4: XCS Module Simplification

### Day 1: IR Research (8 hours)
- [ ] **Morning: Current IR Analysis**
  - [ ] Deep dive into current IR system (2h)
  - [ ] Document use cases (1h)
  - [ ] Identify optimization opportunities (1h)

- [ ] **Afternoon: External IR Research**
  - [ ] Study MLIR design principles (1.5h)
  - [ ] Study LLVM IR patterns (1h)
  - [ ] Study JAX HLO design (1h)
  - [ ] Document applicable patterns (0.5h)

### Day 2: IR Design (8 hours)
- [ ] **Morning: Core IR Design**
  - [ ] Design OpType enum for LLM operations (1h)
  - [ ] Design Operation dataclass (1h)
  - [ ] Design Graph structure (1h)
  - [ ] Design value semantics (1h)

- [ ] **Afternoon: Optimization Design**
  - [ ] Design optimization passes (2h)
  - [ ] Design cloud export format (1h)
  - [ ] Plan future extensibility (1h)

### Day 3: IR Implementation (8 hours)
- [ ] **Morning: Core Structures**
  - [ ] Verify IR package from Day 0 (0.5h)
    - Exit: Can import from ember.ir
  - [ ] Enhance Operation class with cost/latency (1h)
  - [ ] Implement Graph.to_dict() (1h)
  - [ ] Add basic validation (1.5h)

- [ ] **Afternoon: Graph Building**
  - [ ] Implement GraphBuilder class (2h)
  - [ ] Add operation factories (1h)
  - [ ] Add graph validation (1h)

### Day 4: JIT & Optimization (8 hours)
- [ ] **Morning: JIT Implementation**
  - [ ] Implement simplified jit decorator (2h)
  - [ ] Add IR generation from functions (2h)

- [ ] **Afternoon: Optimization Passes**
  - [ ] Implement operation fusion (1h)
  - [ ] Implement batching optimization (1h)
  - [ ] Implement dead code elimination (1h)
  - [ ] Add cloud export (1h)

### Day 5: Testing & Documentation (8 hours)
- [ ] **Morning: Testing**
  - [ ] Test IR construction (1h)
  - [ ] Test optimization passes (1h)
  - [ ] Test JIT compilation (1h)
  - [ ] Benchmark performance (1h)

- [ ] **Afternoon: Documentation**
  - [ ] Document IR design (1h)
  - [ ] Create optimization examples (1h)
  - [ ] Document cloud scheduler interface (1h)
  - [ ] Write migration guide (1h)

## Week 5: Integration & Polish

### Day 1: Cross-Module Integration (8 hours)
- [ ] Test models + operators integration (2h)
- [ ] Test data + operators integration (2h)
- [ ] Test operators + XCS integration (2h)
- [ ] Fix integration issues (2h)

### Day 2: Performance Validation (8 hours)
- [ ] Benchmark model calls (2h)
- [ ] Benchmark data loading (2h)
- [ ] Benchmark operator composition (2h)
- [ ] Benchmark JIT compilation (2h)

### Day 3: Documentation Sprint (8 hours)
- [ ] Update main README (2h)
- [ ] Create getting started guide (2h)
- [ ] Update all docstrings (2h)
- [ ] Create architecture diagram (2h)

### Day 4: Example Updates (8 hours)
- [ ] Update basic examples (2h)
- [ ] Create advanced examples (2h)
- [ ] Add cookbook recipes (2h)
- [ ] Create troubleshooting guide (2h)

### Day 5: Final Review (8 hours)
- [ ] Code review with fresh eyes (2h)
- [ ] Run full test suite (1h)
- [ ] Check mypy strict mode (1h)
- [ ] Performance regression check (2h)
- [ ] Create release notes (2h)

## Continuous Tasks (Throughout All Weeks)

### Daily
- [ ] Ask "What would the masters do?" before each decision
- [ ] Document why things exist before deleting
- [ ] Run tests after each change
- [ ] Check for UX confusion

### Weekly
- [ ] Review progress against simplicity metrics
- [ ] Benchmark performance
- [ ] Update documentation
- [ ] Reflect on architectural decisions

## Definition of Done

### For Each Module
- [ ] Public API fits on one page
- [ ] All tests passing
- [ ] Zero mypy errors (strict mode)
- [ ] Documented with examples
- [ ] Migration guide complete
- [ ] Performance benchmarked
- [ ] **Contract tests verify no API leakage**
- [ ] **Module < 1000 LOC (CI enforced)**

### For Overall Project
- [ ] Total codebase under 10,000 lines
- [ ] Each module under 1,000 lines
- [ ] 80% of use cases use simplest API
- [ ] No performance regressions
- [ ] Clean break from old API (no compatibility shims)
- [ ] **No public registry access**
- [ ] **All async methods work**
- [ ] **IR system operational**

### Weekly Exit Criteria
- [ ] Tech debt triage completed
- [ ] Dead code removed
- [ ] CI remains green
- [ ] Performance benchmarks run