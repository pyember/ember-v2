# Convergence Plan: Building the Perfect JIT System

## Current State
1. **We have a working parallel execution system** (ParallelScheduler + WaveOrderingStrategy)
2. **We have a clean Graph implementation** (our new graph.py with wave-based execution)
3. **JIT strategies don't leverage parallelism** (they create sequential graphs)
4. **Sleep-based tests prove parallelism works** (>5x speedup on I/O operations)

## Goal State
A JIT system that:
1. **Detects I/O-bound operations** (sleep, API calls, network I/O)
2. **Identifies parallelization opportunities** (independent operations)
3. **Builds graphs with proper dependencies** (not just sequential)
4. **Leverages existing parallel execution** (ThreadPoolExecutor for I/O)

## Implementation Steps

### Step 1: Create I/O-Aware structural strategy
**Status: Already drafted in io_aware_trace.py**
- Detects I/O patterns in code
- Identifies independent operations
- Builds graphs with parallel structure

### Step 2: Enhance Graph Building from Traces
Instead of:
```python
# Current: Everything sequential
node1 → node2 → node3 → node4
```

Build:
```python
# Smart: Parallel where possible
     ┌→ node1 ┐
start┼→ node2 ┼→ aggregate
     └→ node3 ┘
```

### Step 3: Integration with Existing Schedulers
- Use our Graph's built-in wave detection
- OR integrate with existing ParallelScheduler
- Ensure I/O operations use ThreadPoolExecutor

### Step 4: Smart Strategy Selection
Update strategy selection heuristics:
```python
if has_io_operations and has_loops:
    return IOAwareTraceStrategy  # New
elif has_ensemble_pattern:
    return EnhancedStrategy      # Existing
elif is_operator:
    return StructuralStrategy    # Existing
else:
    return TraceStrategy         # Fallback
```

### Step 5: Proper Benchmarks
Create benchmarks that reflect real Ember usage:
- LLM ensemble calls
- API aggregation patterns
- Mixed I/O and computation

## Key Design Decisions

### 1. Graph Structure
Our new Graph class already implements wave-based parallelism. We should:
- Keep the clean API
- Ensure it integrates with existing XCS infrastructure
- Add compatibility layer if needed

### 2. I/O Detection
Two approaches:
- **Static analysis**: Look for sleep, requests, etc. in source
- **Dynamic analysis**: Measure execution time during tracing

We'll use both - static for hints, dynamic for confirmation.

### 3. Dependency Analysis
Current trace assumes sequential dependencies. We need:
- Detect when operations are actually independent
- Look for data flow dependencies, not just execution order
- Conservative default: assume sequential unless proven parallel

### 4. Performance Guarantees
- **I/O-bound**: Expect near-linear speedup with parallelism
- **CPU-bound**: No speedup (GIL limitation)
- **Mixed**: Speedup proportional to I/O percentage

## Success Metrics
1. **Sleep-based ensemble**: >3x speedup with 3 parallel operations
2. **Real LLM ensemble**: Measurable speedup on actual API calls
3. **No regression**: CPU-bound operations not slower
4. **Clean API**: Users don't need to change code

## Next Immediate Steps
1. ✅ Rename simple.py → graph.py
2. Fix structural strategy to build parallel graphs
3. Test with sleep-based ensemble
4. Integrate with existing test suite
5. Document the architecture