# Pure IR Design for XCS

## Core Principles

### 1. **No Hardcoded Assumptions**
- Works with any Python code
- No assumptions about attribute names (like `operators`, `models`)
- No monkey-patching or modifying user objects
- Handles any callable: functions, methods, lambdas, objects

### 2. **Pure Functional Design**
- Immutable IR (operations and values are frozen)
- No side effects during graph construction
- Deterministic execution
- Cacheable results for pure operations

### 3. **Clean Separation of Concerns**

```
Source Code → [Builder] → IR → [Optimizer] → Optimized IR → [Executor] → Result
```

Each component has a single responsibility and clean interface.

## The IR Design

```python
# Pure, general operations
class OpType(Enum):
    CALL = "call"              # Any callable invocation
    LOAD = "load"              # Load a value
    STORE = "store"            # Store a value
    CONSTANT = "constant"      # Literal values
    BINOP = "binop"           # +, -, *, /, etc.
    COMPARE = "compare"        # ==, <, >, etc.
    GETATTR = "getattr"       # x.y
    GETITEM = "getitem"       # x[y]
    BRANCH = "branch"         # if/else
    LOOP = "loop"             # for/while
    RETURN = "return"         # return statement

# Immutable value references (SSA form)
@dataclass(frozen=True)
class Value:
    id: str  # Unique identifier

# Immutable operations
@dataclass(frozen=True)
class Operation:
    op_type: OpType
    inputs: Tuple[Value, ...]
    output: Optional[Value]
    attributes: Dict[str, Any]  # Additional metadata

# Basic blocks for control flow
@dataclass
class Block:
    operations: List[Operation]
    terminator: Optional[Operation]  # Branch/return

# Complete computation graph
@dataclass
class Graph:
    blocks: Dict[str, Block]
    values: Dict[str, Any]  # Constants
```

## Building IR - Multiple Approaches

### 1. **Tracing** (Dynamic)
```python
# Observe actual execution
tracer = Tracer()
graph, result = tracer.trace(func, example_input)
```

### 2. **AST Analysis** (Static)
```python
# Analyze code structure
analyzer = ASTAnalyzer()
graph = analyzer.analyze(func)
```

### 3. **Bytecode Analysis** (Static)
```python
# Analyze compiled bytecode
analyzer = BytecodeAnalyzer()
graph = analyzer.analyze(func.__code__)
```

### 4. **Hybrid** (Best of both)
```python
# Combine approaches for completeness
builder = HybridBuilder()
graph = builder.build(func)
```

## Optimization Pipeline

```python
# All optimizations work on the same IR
optimizers = [
    DeadCodeElimination(),
    CommonSubexpressionElimination(),
    ParallelizationAnalysis(),
    OperationFusion(),
    MemoryOptimization(),
]

for optimizer in optimizers:
    graph = optimizer.optimize(graph)
```

## Execution

```python
# Pure executor with automatic parallelization
executor = PureExecutor(max_workers=8)
result = executor.execute(graph, inputs)

# Cached executor for repeated calls
executor = CachedExecutor()
result = executor.execute(graph, inputs)
```

## Key Benefits

1. **Universal** - Works with any Python code
2. **Pure** - No side effects, predictable behavior
3. **Composable** - Mix and match builders, optimizers
4. **Efficient** - Automatic parallelization, caching
5. **Maintainable** - Clean interfaces, single responsibilities

## Integration with XCS

All JIT strategies become thin wrappers:

```python
class StructuralStrategy(IRStrategy):
    def build_ir(self, func):
        return ASTAnalyzer().analyze(func)

class EnhancedStrategy(IRStrategy):
    def build_ir(self, func):
        return HybridBuilder().build(func)

class TracingStrategy(IRStrategy):
    def build_ir(self, func):
        return Tracer().trace(func, examples)
```

They all:
- Build the same IR
- Use the same optimizers
- Execute through the same backend

## Example: Automatic Parallelization

```python
# User writes natural code
def process_items(processors, data):
    results = []
    for proc in processors:
        result = proc(data)
        results.append(result)
    return results

# System automatically:
# 1. Traces execution → builds IR
# 2. Analyzes dependencies → finds independent calls
# 3. Executes in parallel → achieves speedup

# No hardcoded assumptions about 'processors' or loop structure!
```

## Conclusion

This pure IR design achieves our goals:
- **General** - No brittle assumptions
- **Pure** - Functional, immutable, predictable
- **Elegant** - Clean abstractions
- **Efficient** - Automatic optimization
- **Concise** - Minimal, focused code

It provides a solid foundation for XCS that can grow and adapt without accumulating technical debt.