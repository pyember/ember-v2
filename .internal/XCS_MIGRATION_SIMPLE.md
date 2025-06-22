# XCS Migration: From ThreadPoolExecutor to UnifiedDispatcher

## The Change

Replace every `ThreadPoolExecutor` with `UnifiedDispatcher`. That's it.

## Examples

### Basic Pattern
```python
# Before
with ThreadPoolExecutor(max_workers=n) as executor:
    futures = [executor.submit(fn, x) for x in items]
    results = [f.result() for f in futures]

# After  
dispatcher = UnifiedDispatcher(max_workers=n)
results = dispatcher.map(fn, items)
```

### With Error Handling
```python
# Before
with ThreadPoolExecutor() as executor:
    futures = {}
    for item in items:
        futures[executor.submit(process, item)] = item
    
    results = {}
    for future in as_completed(futures):
        item = futures[future]
        try:
            results[item] = future.result()
        except Exception as e:
            results[item] = {"error": str(e)}

# After
dispatcher = UnifiedDispatcher()
results = dispatcher.map_with_ids(
    lambda d: process(d["item"]), 
    [(str(i), {"item": item}) for i, item in enumerate(items)]
)
```

### With Context (Optional)
```python
# Give hints for better optimization
context = ExecutionContext(
    component="vmap",
    is_io_heavy=False  # CPU-bound computation
)

with context:
    dispatcher = UnifiedDispatcher()
    results = dispatcher.map(compute, inputs)
```

## Component-Specific Examples

### XCS Engine
```python
# In xcs_engine.py TopologicalSchedulerWithParallelDispatch
def run_plan(self, plan, graph, inputs):
    # Before: 40+ lines of ThreadPoolExecutor code
    
    # After:
    dispatcher = UnifiedDispatcher(max_workers=self.max_workers)
    
    for wave in self.schedule(graph):
        items = [(node_id, {
            "op": plan.tasks[node_id].operator,
            "inputs": prepare_inputs(node_id)
        }) for node_id in wave]
        
        results = dispatcher.map_with_ids(
            lambda d: d["op"](inputs=d["inputs"]),
            items
        )
```

### Transforms (vmap, pmap, mesh)
```python
# In vmap.py
def _process_parallel(self, fn, inputs, batch_size):
    # Before: Custom Dispatcher configuration
    
    # After:
    dispatcher = UnifiedDispatcher(max_workers=self.options.max_workers)
    return dispatcher.map(fn, inputs)
```

### Schedulers
```python
# In base_scheduler_impl.py  
def execute_wave(self, wave, graph):
    # Before: Complex futures management
    
    # After:
    context = ExecutionContext(component="scheduler")
    with context:
        dispatcher = UnifiedDispatcher()
        return dispatcher.map(execute_node, wave)
```

## Migration Checklist

1. **Find ThreadPoolExecutor**
   ```bash
   grep -r "ThreadPoolExecutor" src/
   ```

2. **Replace with UnifiedDispatcher**
   - Import: `from ember.xcs.utils._executor_next_v2 import UnifiedDispatcher`
   - Create: `dispatcher = UnifiedDispatcher(max_workers=...)`
   - Execute: `results = dispatcher.map(fn, inputs)`

3. **Add Context (Optional but Recommended)**
   ```python
   context = ExecutionContext(
       component="your_component",
       is_io_heavy=True  # If doing API calls or I/O
   )
   ```

4. **Test**
   - Run existing tests - they should pass
   - Performance should be same or better
   - Code should be cleaner

## FAQ

**Q: What about timeout and error handling?**
A: UnifiedDispatcher handles both by default. Errors become None in results.

**Q: How does it know to use threads vs async?**
A: It learns. First run uses heuristics, then remembers what was fast.

**Q: Do I need to change my functions?**
A: No. They work exactly as before.

**Q: What about resource cleanup?**
A: Call `dispatcher.close()` when done, or use `parallel_map()` helper.

## The Payoff

- **Less Code**: ~70% reduction
- **Smarter Execution**: Automatically optimizes
- **Cleaner**: No more futures management
- **Consistent**: Same pattern everywhere

Start with one component, see how clean it becomes, then migrate the rest.