# Comprehensive Logging Improvement Plan for Ember

## Executive Summary
This plan addresses user complaints about excessive and ugly logging in Ember examples by implementing a multi-phase approach to improve logging UX, reduce noise, and provide better user feedback mechanisms.

## Phase 1: Immediate Fixes (1-2 days) ✅ COMPLETED
**Goal**: Quick wins to reduce log noise without breaking changes

### Tactical Slice 1.1: Adjust Log Levels ✅
**Checklist:**
- [x] Change model discovery INFO logs to DEBUG in `discovery.py`
  - [x] Line: `logger.info(f"Discovered {len(models)} models: {models}")` → DEBUG
  - [x] Line: `logger.info(f"Provider {pname} completed in {duration:.2f}s")` → DEBUG
  - [x] Line: `logger.info("Returning cached discovery results.")` → DEBUG
- [x] Change model initialization INFO logs to DEBUG in `initialization.py`
  - [x] Line: `logger.info(f"Registered model from config: {model_id}")` → DEBUG
  - [x] Line: `logger.info("Execute model discovery...")` → Simplified to "Discovering available models..."
- [x] Change XCS engine execution logs to DEBUG in `xcs_engine.py`
  - [x] All wave execution logs → DEBUG
  - [x] Node-by-node processing logs → DEBUG
  - [x] Changed logger.exception to logger.error with exc_info=True

### Tactical Slice 1.2: Suppress HTTP Library Logs ✅
**Checklist:**
- [x] Update `configure_logging()` to set HTTP libraries to ERROR level by default
- [x] Add early initialization hook in `__init__.py` to suppress logs on import
- [x] Create environment variable `EMBER_HTTP_LOG_LEVEL` for override
- [x] Extended HTTP library list to include more libraries
- [x] Added NullHandler to prevent propagation

### Tactical Slice 1.3: One-Time Migration Warnings ✅
**Checklist:**
- [x] Implement `_shown_warnings` set in `lm.py`
- [x] Modify LMModule warning to show only once per session
- [x] Changed migration log from INFO to DEBUG
- [x] Also applied to `get_default_model_service()` function

## Phase 2: Logging Infrastructure (2-3 days) ✅ COMPLETED
**Goal**: Build better logging utilities and context managers

### Tactical Slice 2.1: Create Progress Reporter ✅
**Checklist:**
- [x] Create `ember/core/utils/progress.py` with:
  - [x] `ProgressReporter` class with methods for common operations
  - [x] `discovery_start()`, `discovery_complete(count)`
  - [x] `loading_start()`, `loading_complete()`
  - [x] `execution_start()`, `execution_complete()`
- [x] Add `--quiet` mode support
- [x] Add emoji/icon support with fallback for non-Unicode terminals
- [x] Added timing support with human-readable duration formatting
- [x] Added context manager for timed operations

### Tactical Slice 2.2: Implement Log Context Manager ✅
**Checklist:**
- [x] Create `suppress_logs()` context manager
- [x] Create `log_level()` context manager for temporary level changes
- [x] Add module-specific suppression patterns
- [x] Added `verbose_mode()` context manager for temporary verbosity

### Tactical Slice 2.3: Structured Logging Enhancement ✅
**Checklist:**
- [x] Extend `structured_logging.py` with user-facing formatters
- [x] Add JSON output mode for machine parsing
- [x] Add simplified format for interactive use
- [x] Create log filtering by component groups
- [x] Added color support with automatic detection
- [x] Added `SimpleFormatter`, `DetailedFormatter`, and `JSONFormatter`
- [x] Added `configure_xcs_logging()` for easy setup

## Phase 3: User Experience (3-4 days) ✅ COMPLETED
**Goal**: Implement CLI-friendly progress indicators and clean output

### Tactical Slice 3.1: CLI Progress Integration ✅
**Checklist:**
- [x] Implemented text-based progress without external dependencies
- [x] Created progress indicators in `progress.py`:
  - [x] Model discovery progress
  - [x] Dataset loading progress
  - [x] Execution progress
  - [x] General operation timing
- [x] Added emoji support with automatic fallback
- [x] Added environment variable support (EMBER_QUIET)

### Tactical Slice 3.2: Clean Output Formatting ✅
**Checklist:**
- [x] Create `ember/core/utils/output.py` with:
  - [x] `print_header(title)` - formatted section headers
  - [x] `print_summary(results)` - clean result tables
  - [x] `print_models(models)` - grouped model listings
  - [x] `print_metrics(metrics)` - formatted performance metrics
- [x] Add color support with NO_COLOR environment variable respect
- [x] Added additional utilities:
  - [x] `print_table()` - clean table formatting
  - [x] `print_progress()` - progress bar
  - [x] `print_error()`, `print_warning()`, `print_success()`, `print_info()`

### Tactical Slice 3.3: Verbosity Control ✅
**Checklist:**
- [x] Create `ember/core/utils/verbosity.py` with full verbosity management
- [x] Implement verbosity levels: quiet, normal, verbose, debug
- [x] Create `VerbosityLevel` enum
- [x] Added `add_verbosity_args()` for easy argparse integration
- [x] Added `VerbosityManager` class for structured verbosity control
- [x] Environment variable support (EMBER_VERBOSITY)
- [x] Context managers for temporary verbosity changes

## Phase 4: Examples Update (2-3 days) ✅ COMPLETED
**Goal**: Refactor all examples to use new logging approach

### Tactical Slice 4.1: Basic Examples ✅
**Checklist:**
- [x] Update `minimal_example.py`
  - [x] Added verbosity argument parsing
  - [x] Use clean output formatting utilities
  - [x] Added --verbose/--quiet flags
- [x] Update `check_env.py`
  - [x] Added clean headers and status indicators
  - [x] Suppressed model discovery logs
  - [x] Added verbosity controls

### Tactical Slice 4.2: Model Examples ✅
**Checklist:**
- [x] Update `list_models.py`
  - [x] Replace logging with ProgressReporter
  - [x] Use clean output formatting (print_models, print_table)
  - [x] Add progress indicators for discovery
  - [x] Suppress internal logs with context managers
  - [x] Add verbosity controls

### Tactical Slice 4.3: Advanced Examples ✅
**Checklist:**
- [x] Update `ensemble_judge_mmlu.py` with new logging approach
  - [x] Added verbosity parsing
  - [x] Replaced verbose console output with clean formatters
  - [x] Added log suppression for cleaner output

## Phase 5: Documentation (1-2 days)
**Goal**: Document new logging behavior and migration path

### Tactical Slice 5.1: User Documentation
**Checklist:**
- [ ] Create `docs/logging.md` with:
  - [ ] Logging philosophy and defaults
  - [ ] Verbosity control guide
  - [ ] Environment variables reference
  - [ ] Troubleshooting guide
- [ ] Update README with logging section

### Tactical Slice 5.2: Migration Guide
**Checklist:**
- [ ] Document changes for library users
- [ ] Provide before/after examples
- [ ] List breaking changes (if any)
- [ ] Add to CHANGELOG.md

## Implementation Priority Matrix

| Phase | Impact | Effort | Priority | Timeline |
|-------|--------|--------|----------|----------|
| 1.1 Log Levels | High | Low | Critical | Day 1 |
| 1.2 HTTP Suppression | High | Low | Critical | Day 1 |
| 1.3 Migration Warnings | Medium | Low | High | Day 2 |
| 2.1 Progress Reporter | High | Medium | High | Days 3-4 |
| 2.2 Context Managers | Medium | Medium | Medium | Day 5 |
| 3.1 CLI Progress | High | High | Medium | Days 6-7 |
| 3.2 Clean Output | High | Medium | High | Days 8-9 |
| 4.1-4.3 Examples | High | Medium | High | Days 10-12 |
| 5.1-5.2 Documentation | Low | Low | Low | Days 13-14 |

## Success Metrics
1. **Noise Reduction**: 80% fewer log lines in typical example runs
2. **User Satisfaction**: Clean, professional output in all examples
3. **Developer Experience**: Easy verbosity control for debugging
4. **Performance**: No measurable overhead from logging changes

## Risk Mitigation
1. **Breaking Changes**: All changes maintain backward compatibility
2. **Dependency Issues**: Optional dependencies with fallbacks
3. **Testing**: Comprehensive tests for each phase before moving forward
4. **Rollback Plan**: Git tags at each phase completion

## Next Steps
1. Begin with Phase 1.1 - adjusting log levels in discovery and initialization modules
2. Create feature branch `improve-logging-ux`
3. Set up tracking issue with checkboxes for all tasks
4. Daily progress updates with completed checkboxes

## Current Logging Issues (Reference)

### 1. **Excessive Verbosity During Model Discovery**
- Every API key check logs whether found or not
- Each provider logs all discovered models
- Timing information for each provider
- Aggregated results listing all model IDs
- Cache hit/miss messages

### 2. **HTTP Library Noise**
- httpx, httpcore, urllib3, and openai libraries produce verbose HTTP request/response logs
- Debug-level stack traces for expected errors (e.g., 401 for missing API keys)

### 3. **XCS Engine Execution Details**
- Wave-by-wave execution logs
- Node-by-node processing details
- Parallelization decisions
- Error stack traces for each failed node

### 4. **Migration Nudges**
- LMModule deprecation warnings on every instantiation
- Suggestions to migrate to newer APIs

### 5. **Poor Log Format**
- Long timestamps and module names clutter output
- No distinction between framework logs and user-relevant information
- Debug/info logs mixed with actual results

## Example Log Output (Before)
```
2025-05-23 13:20:41,851 [DEBUG] openai._base_client: Sending HTTP Request: GET https://api.openai.com/v1/models
2025-05-23 13:20:41,988 [INFO] httpx: HTTP Request: GET https://api.openai.com/v1/models "HTTP/1.1 401 Unauthorized"
2025-05-23 13:20:41,994 [ERROR] ember.core.registry.model.providers.openai.openai_discovery: OpenAI API error: Error code: 401
2025-05-23 13:20:41,995 [WARNING] ember.core.registry.model.providers.openai.openai_discovery: No fallback models provided - API discovery required
```

## Expected Output (After)
```
🔍 Discovering available models...
✅ Found 25 models

OpenAI: 15 models
Anthropic: 10 models

Use --verbose to see detailed model list
```