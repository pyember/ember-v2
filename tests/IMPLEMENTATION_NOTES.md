# Implementation Notes for Tests

## Operators

### Base Operator
- Inherits from `Module` (equinox) - immutable
- Requires implementing `forward()` method
- Has optional `input_spec` and `output_spec` for validation
- `__call__` handles validation, `forward` does the work

### Common Operators
1. **Ensemble**
   - Constructor: `Ensemble(operators, aggregator=None)`
   - If no aggregator, returns list of results
   - Aggregator function takes list of results and returns single result

2. **Chain**
   - Constructor: `Chain(operators)`
   - Passes output of each operator as input to next

3. **Router**
   - Constructor: `Router(routes, router_fn, default_route=None)`
   - `routes`: Dict[str, Operator]
   - `router_fn`: Function that returns route name
   - Raises KeyError if no route found and no default

4. **LearnableRouter**
   - Constructor: `LearnableRouter(operators, embedding_dim, hidden_dim, key)`
   - Uses MLP to learn routing
   - Has `weights`, `bias`, etc. as JAX arrays

5. **Retry**
   - Constructor: `Retry(operator, max_attempts=3, should_retry=None)`
   - Wraps single operator with retry logic

6. **Cache**
   - Constructor: `Cache(operator, max_size=100, key_fn=None)`
   - LRU cache around single operator

### Convenience Functions
- `ensemble(*operators, **kwargs)` -> Ensemble
- `chain(*operators)` -> Chain
- `router(routes, **kwargs)` -> Router

## XCS

### @jit decorator
- Located in `ember.xcs._simple`
- Function: `jit(func=None, *, _config=None)`
- Handles both `@jit` and `@jit()` syntax
- Returns wrapped function with `_xcs_optimized` attribute
- Uses IRBuilder, ParallelismAnalyzer, ExecutionEngine internally
- Caches optimization decision

### Other transformations
- `vmap`, `pmap`, `scan`, `grad` from `ember.xcs.transformations`
- These are XCSTransformation instances

## Data API

### stream() function
- Returns `StreamIterator` instance
- Parameters: source, subset, split, filter, transform, batch_size, max_items, normalize
- Supports method chaining (.filter(), .transform(), .limit())

### load() function
- Similar to stream but returns materialized list
- Uses stream internally with `.to_list()`

### DataSource protocol
- Must implement `read_batches(batch_size)`
- Built-in sources: HuggingFaceSource, FileSource

## Models API (Already tested)
- `models(model, prompt, **params)` - function-based API
- `models.instance(model, **params)` - creates ModelBinding
- Response object with .text, .usage, .model_id