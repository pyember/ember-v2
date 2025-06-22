# XCS Implementation Roadmap: From Design to Reality

## Overview

This roadmap details the specific implementation tasks to transform XCS from its current disconnected state into a fully integrated, robust system for automatic parallelism discovery.

## Phase 1: Core Pipeline Connection (Days 1-5)

### Day 1: Runtime Tracer Foundation

**Morning: Proxy System**
```python
# ember/xcs/_internal/proxy.py
class XCSProxy:
    """Proxy object that records operations during tracing."""
    def __init__(self, var_name: str, shape: Optional[tuple] = None, dtype: Optional[type] = None):
        self.var_name = var_name
        self.shape = shape
        self.dtype = dtype
        self._trace_context = None
    
    def __call__(self, *args, **kwargs):
        """Record function calls on this proxy."""
        if self._trace_context:
            return self._trace_context.record_call(self, args, kwargs)
        raise RuntimeError("Proxy used outside tracing context")
    
    def __getattr__(self, name):
        """Record attribute access."""
        if self._trace_context:
            return self._trace_context.record_attr(self, name)
        raise RuntimeError("Proxy used outside tracing context")
```

**Afternoon: Tracing Context**
```python
# ember/xcs/_internal/tracer.py
class TracingContext:
    """Manages tracing state and builds IR."""
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.proxy_cache = {}
        
    def record_call(self, target: XCSProxy, args: tuple, kwargs: dict) -> XCSProxy:
        """Record a function call and return result proxy."""
        # Create node in IR
        output_proxy = XCSProxy(self.builder.next_var())
        
        # Extract operator from target
        if hasattr(target, '_operator'):
            operator = target._operator
        else:
            operator = target  # Direct function call
            
        # Add to graph
        self.builder.add_operation(
            operator=operator,
            inputs=[self._unwrap_proxy(arg) for arg in args],
            outputs=[output_proxy.var_name]
        )
        
        return output_proxy
```

### Day 2: Connect Tracer to IR Builder

**Morning: Enhanced IR Builder**
```python
# ember/xcs/_internal/ir_builder.py
class IRBuilder:
    """Builds computation graphs from traced operations."""
    
    def add_operation(self, operator: Any, inputs: List[str], outputs: List[str]) -> IRNode:
        """Add operation with full metadata."""
        node_id = self.next_node_id()
        
        # Analyze operation characteristics
        signature = self._analyze_operator(operator)
        
        # Create node
        node = IRNode(
            id=node_id,
            operator=operator,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            metadata={
                'signature': signature,
                'can_parallelize': signature.is_pure,
                'can_batch': signature.can_batch,
                'is_tensor_op': signature.is_tensor_op,
                'is_orchestration_op': signature.is_orchestration_op,
            }
        )
        
        # Update graph
        self.nodes[node_id] = node
        self._update_edges(node_id, inputs)
        
        return node
    
    def _analyze_operator(self, operator: Any) -> OperationSignature:
        """Analyze operator to determine its characteristics."""
        # Type-based analysis
        if isinstance(operator, ModelBinding):
            return OperationSignature(
                is_orchestration_op=True,
                can_batch=True,
                is_pure=True  # Models are pure functions
            )
        elif hasattr(operator, '__module__') and 'jax' in operator.__module__:
            return OperationSignature(
                is_tensor_op=True,
                can_batch=True,
                is_pure=True
            )
        # ... more analysis
```

**Afternoon: Integration with JIT**
```python
# ember/xcs/_simple.py
def jit(func: Optional[Callable] = None, *, _config: Optional['Config'] = None) -> Callable:
    """Enhanced JIT that actually optimizes."""
    if func is None:
        return functools.partial(jit, _config=_config)
    
    # Check if already jitted
    if hasattr(func, '_xcs_optimized'):
        return func
    
    # Create optimized version
    compiled_cache = {}
    
    @functools.wraps(func)
    def optimized_func(*args, **kwargs):
        # Get cache key
        cache_key = _make_cache_key(args, kwargs)
        
        # Check cache
        if cache_key in compiled_cache:
            graph, analysis, executor = compiled_cache[cache_key]
        else:
            # Trace to build graph
            tracer = XCSTracer()
            graph = tracer.trace(func, args, kwargs)
            
            # Analyze for parallelism
            analyzer = ParallelismAnalyzer()
            analysis = analyzer.analyze_graph(graph)
            
            # Create executor
            engine = _get_engine()
            executor = engine.compile(graph, analysis)
            
            # Cache
            compiled_cache[cache_key] = (graph, analysis, executor)
        
        # Execute optimized version
        return executor.execute(args, kwargs)
    
    # Mark and return
    optimized_func._xcs_optimized = True
    optimized_func._xcs_original = func
    optimized_func._xcs_cache = compiled_cache
    
    return optimized_func
```

### Day 3: Parallelism Analyzer Integration

**Morning: Enhanced Analysis**
```python
# ember/xcs/_internal/parallelism.py
class ParallelismAnalyzer:
    """Discovers parallelism from graph structure and signatures."""
    
    def analyze_graph(self, graph: IRGraph) -> GraphParallelismAnalysis:
        """Complete analysis with execution strategy."""
        # Analyze nodes
        node_info = self._analyze_nodes(graph)
        
        # Find opportunities
        parallel_groups = self._find_parallel_groups(graph, node_info)
        vectorizable_chains = self._find_vectorizable_chains(graph, node_info)
        fusion_opportunities = self._find_fusion_opportunities(graph, node_info)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(graph, node_info)
        
        # Choose strategy
        strategy = self._choose_execution_strategy(
            graph, parallel_groups, vectorizable_chains, fusion_opportunities
        )
        
        # Estimate speedup
        speedup = self._estimate_speedup(strategy, parallel_groups, vectorizable_chains)
        
        return GraphParallelismAnalysis(
            node_info=node_info,
            parallel_groups=parallel_groups,
            vectorizable_chains=vectorizable_chains,
            fusion_opportunities=fusion_opportunities,
            bottlenecks=bottlenecks,
            execution_strategy=strategy,
            estimated_speedup=speedup
        )
    
    def _choose_execution_strategy(self, graph, parallel_groups, vectorizable_chains, fusion_ops):
        """Choose best execution strategy."""
        # Count different operation types
        tensor_count = sum(1 for n in graph.nodes.values() 
                          if n.metadata.get('is_tensor_op'))
        orch_count = sum(1 for n in graph.nodes.values() 
                        if n.metadata.get('is_orchestration_op'))
        
        # Decision logic
        if tensor_count > 0 and orch_count > 0:
            return 'hybrid'
        elif len(parallel_groups) > 0:
            return 'parallel'
        elif len(vectorizable_chains) > 0:
            return 'vectorized'
        elif len(fusion_ops) > 0:
            return 'fused'
        else:
            return 'sequential'
```

**Afternoon: Execution Strategy Mapping**
```python
# ember/xcs/_internal/strategies.py
@dataclass
class ExecutionStrategy:
    """Execution strategy with specific optimizations."""
    name: str
    parallel_groups: List[Set[str]]
    vectorizable_chains: List[List[str]]
    fusion_groups: List[Set[str]]
    execution_order: List[str]
    
    def apply(self, graph: IRGraph, context: ExecutionContext) -> Any:
        """Apply this strategy to execute the graph."""
        raise NotImplementedError

class ParallelStrategy(ExecutionStrategy):
    """Execute independent operations in parallel."""
    
    def apply(self, graph: IRGraph, context: ExecutionContext) -> Any:
        """Execute with thread pool for parallel groups."""
        with ThreadPoolExecutor() as executor:
            # Execute each parallel group
            for group in self.parallel_groups:
                futures = []
                for node_id in group:
                    node = graph.nodes[node_id]
                    future = executor.submit(
                        self._execute_node, node, context
                    )
                    futures.append((node_id, future))
                
                # Wait for group completion
                for node_id, future in futures:
                    result = future.result()
                    self._store_result(context, node_id, result)
        
        return self._get_final_result(context)
```

### Day 4: Execution Engine Enhancement

**Morning: Compiled Executors**
```python
# ember/xcs/_internal/engine.py
class ExecutionEngine:
    """Compiles and executes computation graphs."""
    
    def compile(self, graph: IRGraph, analysis: GraphParallelismAnalysis) -> CompiledExecutor:
        """Compile graph with chosen strategy."""
        strategy_name = analysis.execution_strategy
        
        # Create appropriate executor
        if strategy_name == 'parallel':
            return ParallelExecutor(graph, analysis)
        elif strategy_name == 'vectorized':
            return VectorizedExecutor(graph, analysis)
        elif strategy_name == 'hybrid':
            return HybridExecutor(graph, analysis)
        elif strategy_name == 'fused':
            return FusedExecutor(graph, analysis)
        else:
            return SequentialExecutor(graph, analysis)

class CompiledExecutor:
    """Base class for compiled executors."""
    
    def __init__(self, graph: IRGraph, analysis: GraphParallelismAnalysis):
        self.graph = graph
        self.analysis = analysis
        self._prepare_execution_plan()
    
    def _prepare_execution_plan(self):
        """Pre-compute execution order and dependencies."""
        raise NotImplementedError
    
    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Execute with pre-compiled plan."""
        raise NotImplementedError

class ParallelExecutor(CompiledExecutor):
    """Executes graphs with parallel operations."""
    
    def _prepare_execution_plan(self):
        """Pre-compute waves and dependencies."""
        self.execution_waves = self._compute_waves()
        self.wave_dependencies = self._compute_wave_deps()
    
    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Execute waves in order, parallelize within waves."""
        context = self._create_context(args, kwargs)
        
        with ThreadPoolExecutor() as executor:
            for wave in self.execution_waves:
                if len(wave) == 1:
                    # Single node, execute directly
                    self._execute_node(wave[0], context)
                else:
                    # Multiple nodes, parallelize
                    self._execute_wave_parallel(wave, context, executor)
        
        return self._extract_result(context)
```

**Afternoon: Error Handling**
```python
# ember/xcs/_internal/errors.py
class XCSExecutionError(Exception):
    """Rich error with execution context."""
    
    def __init__(self, message: str, node: Optional[IRNode] = None, 
                 original_error: Optional[Exception] = None):
        self.node = node
        self.original_error = original_error
        
        # Build helpful message
        parts = [message]
        if node:
            parts.append(f"\nAt node: {node.id}")
            parts.append(f"Operator: {node.operator}")
            parts.append(f"Inputs: {node.inputs}")
        if original_error:
            parts.append(f"\nOriginal error: {type(original_error).__name__}: {original_error}")
        
        super().__init__(''.join(parts))

# In executors
def _execute_node_safe(self, node: IRNode, context: ExecutionContext) -> Any:
    """Execute node with proper error handling."""
    try:
        # Gather inputs
        inputs = [context.get(var) for var in node.inputs]
        
        # Execute
        result = node.operator(*inputs)
        
        # Store result
        if node.outputs:
            context.set(node.outputs[0], result)
        
        return result
        
    except Exception as e:
        # Fail fast with context
        raise XCSExecutionError(
            f"Failed to execute {node.operator.__name__ if hasattr(node.operator, '__name__') else node.operator}",
            node=node,
            original_error=e
        ) from e
```

### Day 5: Integration Testing

**Morning: Test Infrastructure**
```python
# tests/integration/xcs/test_pipeline_integration.py
import pytest
from ember.xcs import jit, get_jit_stats
from ember.xcs._internal.errors import XCSExecutionError

class TestXCSPipeline:
    """Test the complete XCS pipeline."""
    
    def test_simple_parallelism_discovery(self):
        """Test that independent operations run in parallel."""
        call_order = []
        
        def track_call(name):
            def wrapper(x):
                call_order.append((name, time.time()))
                time.sleep(0.1)  # Simulate work
                return x * 2
            return wrapper
        
        @jit
        def parallel_ops(x):
            # These should run in parallel
            a = track_call('op1')(x)
            b = track_call('op2')(x)
            c = track_call('op3')(x)
            return a + b + c
        
        # Execute
        result = parallel_ops(10)
        assert result == 60  # (10*2) + (10*2) + (10*2)
        
        # Check parallelism happened
        times = [t for _, t in call_order]
        max_time_diff = max(times) - min(times)
        assert max_time_diff < 0.05  # Should start nearly simultaneously
        
        # Check stats
        stats = get_jit_stats(parallel_ops)
        assert stats['parallel_groups'] == 1
        assert stats['group_sizes'] == [3]
    
    def test_vectorization_discovery(self):
        """Test that map patterns are vectorized."""
        @jit
        def map_operation(items):
            return [model(x) for x in items]
        
        result = map_operation([1, 2, 3, 4])
        
        stats = get_jit_stats(map_operation)
        assert stats['vectorizable_chains'] > 0
        assert stats['execution_strategy'] == 'vectorized'
    
    def test_hybrid_execution(self):
        """Test mixed tensor/orchestration handling."""
        @jit
        def hybrid_pipeline(x):
            # Tensor op
            embedding = encode_tensor(x)
            
            # Orchestration op
            analysis = llm_analyze(embedding)
            
            # Tensor op
            return decode_tensor(analysis)
        
        result = hybrid_pipeline(data)
        
        stats = get_jit_stats(hybrid_pipeline)
        assert stats['execution_strategy'] == 'hybrid'
        assert stats['tensor_operations'] == 2
        assert stats['orchestration_operations'] == 1
```

**Afternoon: Error Handling Tests**
```python
def test_error_propagation():
    """Test that errors are properly reported."""
    
    def failing_op(x):
        raise ValueError("Intentional failure")
    
    @jit
    def pipeline_with_error(x):
        y = good_op(x)
        z = failing_op(y)  # This will fail
        return final_op(z)
    
    with pytest.raises(XCSExecutionError) as exc_info:
        pipeline_with_error(10)
    
    error = exc_info.value
    assert "failing_op" in str(error)
    assert "Intentional failure" in str(error)
    assert error.node is not None
    assert error.original_error is not None

def test_graceful_fallback():
    """Test fallback for untraceable functions."""
    
    @jit
    def untraceable_function(x):
        # Dynamic behavior that's hard to trace
        if random.random() > 0.5:
            return option_a(x)
        else:
            return option_b(x)
    
    # Should still work, just without optimization
    result = untraceable_function(10)
    assert result is not None
    
    stats = get_jit_stats(untraceable_function)
    assert stats['execution_strategy'] == 'fallback'
```

## Phase 2: Robustness Improvements (Days 6-10)

### Day 6: Type-Based Operation Detection

**Morning: Operation Signatures**
```python
# ember/xcs/_internal/signatures.py
class OperationDetector:
    """Detect operation types from runtime information."""
    
    def detect(self, operator: Any) -> OperationSignature:
        """Detect operation characteristics."""
        # Check against known types
        for detector in self._detectors:
            if detector.matches(operator):
                return detector.signature_for(operator)
        
        # Unknown operation - analyze dynamically
        return self._analyze_unknown(operator)
    
    def _analyze_unknown(self, operator: Any) -> OperationSignature:
        """Analyze unknown operator."""
        signature = OperationSignature()
        
        # Check for tensor indicators
        if hasattr(operator, '__array_ufunc__'):
            signature.is_tensor_op = True
        
        # Check for model indicators
        if hasattr(operator, 'generate') or hasattr(operator, 'complete'):
            signature.is_orchestration_op = True
        
        # Check purity
        if hasattr(operator, '__code__'):
            # Analyze bytecode for side effects
            signature.is_pure = not self._has_side_effects(operator.__code__)
        
        return signature
```

**Afternoon: Detector Registry**
```python
# Register known operators
@dataclass
class OperatorDetector:
    """Detects specific operator types."""
    
    def matches(self, operator: Any) -> bool:
        raise NotImplementedError
    
    def signature_for(self, operator: Any) -> OperationSignature:
        raise NotImplementedError

class ModelBindingDetector(OperatorDetector):
    """Detects Ember model bindings."""
    
    def matches(self, operator: Any) -> bool:
        return isinstance(operator, ModelBinding)
    
    def signature_for(self, operator: Any) -> OperationSignature:
        return OperationSignature(
            is_orchestration_op=True,
            is_pure=True,
            can_batch=True,
            estimated_cost='high'
        )

class JAXOperatorDetector(OperatorDetector):
    """Detects JAX operations."""
    
    def matches(self, operator: Any) -> bool:
        return (hasattr(operator, '__module__') and 
                operator.__module__.startswith('jax'))
    
    def signature_for(self, operator: Any) -> OperationSignature:
        return OperationSignature(
            is_tensor_op=True,
            is_pure=True,
            can_batch=True,
            can_vmap=True,
            estimated_cost='low'
        )
```

### Day 7: Thread Safety and Immutability

**Morning: Immutable Execution Context**
```python
# ember/xcs/_internal/context.py
@dataclass(frozen=True)
class ImmutableExecutionContext:
    """Thread-safe immutable execution context."""
    variables: FrozenDict[str, Any]
    metadata: FrozenDict[str, Any]
    
    def get(self, var_name: str) -> Any:
        """Get variable value."""
        return self.variables.get(var_name)
    
    def with_update(self, var_name: str, value: Any) -> 'ImmutableExecutionContext':
        """Create new context with updated value."""
        new_vars = dict(self.variables)
        new_vars[var_name] = value
        return ImmutableExecutionContext(
            variables=FrozenDict(new_vars),
            metadata=self.metadata
        )
    
    def with_updates(self, updates: Dict[str, Any]) -> 'ImmutableExecutionContext':
        """Create new context with multiple updates."""
        new_vars = dict(self.variables)
        new_vars.update(updates)
        return ImmutableExecutionContext(
            variables=FrozenDict(new_vars),
            metadata=self.metadata
        )

# Frozen dictionary implementation
class FrozenDict(dict):
    """Immutable dictionary."""
    
    def __setitem__(self, key, value):
        raise TypeError("FrozenDict is immutable")
    
    def __delitem__(self, key):
        raise TypeError("FrozenDict is immutable")
    
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
```

**Afternoon: Safe Parallel Execution**
```python
# ember/xcs/_internal/parallel_executor.py
class SafeParallelExecutor:
    """Thread-safe parallel execution."""
    
    def execute_parallel(self, 
                        nodes: List[IRNode], 
                        context: ImmutableExecutionContext) -> Dict[str, Any]:
        """Execute nodes in parallel with immutable context."""
        with ThreadPoolExecutor() as executor:
            # Submit all tasks with immutable context
            futures = {}
            for node in nodes:
                future = executor.submit(
                    self._execute_node_isolated,
                    node,
                    context  # Safe to share - it's immutable
                )
                futures[node.id] = future
            
            # Collect results
            results = {}
            for node_id, future in futures.items():
                try:
                    result = future.result()
                    results[node_id] = result
                except Exception as e:
                    # Handle per-node failures
                    node = next(n for n in nodes if n.id == node_id)
                    raise XCSExecutionError(
                        f"Parallel execution failed for {node_id}",
                        node=node,
                        original_error=e
                    ) from e
            
            return results
    
    def _execute_node_isolated(self, 
                              node: IRNode, 
                              context: ImmutableExecutionContext) -> Any:
        """Execute single node in isolation."""
        # Gather inputs
        inputs = [context.get(var) for var in node.inputs]
        
        # Execute operator
        result = node.operator(*inputs)
        
        # Return result (don't mutate context)
        return result
```

### Day 8: Improved Fallback Strategies

**Morning: Graceful Degradation**
```python
# ember/xcs/_internal/fallback.py
class FallbackStrategy:
    """Handles cases where optimization fails."""
    
    def __init__(self):
        self.fallback_reasons = []
    
    def try_trace(self, func: Callable, args: tuple, kwargs: dict) -> Optional[IRGraph]:
        """Try to trace, with fallback on failure."""
        try:
            tracer = XCSTracer()
            return tracer.trace(func, args, kwargs)
        except Exception as e:
            self.fallback_reasons.append(f"Tracing failed: {e}")
            return None
    
    def try_optimize(self, func: Callable, args: tuple, kwargs: dict) -> Callable:
        """Try optimization with graceful fallback."""
        # Try full optimization
        graph = self.try_trace(func, args, kwargs)
        if graph:
            try:
                analyzer = ParallelismAnalyzer()
                analysis = analyzer.analyze_graph(graph)
                
                if analysis.estimated_speedup > 1.1:
                    # Worth optimizing
                    engine = ExecutionEngine()
                    executor = engine.compile(graph, analysis)
                    return executor.execute
                else:
                    self.fallback_reasons.append("No speedup opportunity detected")
            except Exception as e:
                self.fallback_reasons.append(f"Analysis failed: {e}")
        
        # Try simple optimizations
        if self._can_batch(func):
            return self._create_batched_version(func)
        
        if self._can_cache(func):
            return self._create_cached_version(func)
        
        # Ultimate fallback - just wrap for monitoring
        return self._create_monitored_version(func)
```

**Afternoon: Partial Optimization**
```python
# ember/xcs/_internal/partial_optimization.py
class PartialOptimizer:
    """Optimize what we can, leave the rest."""
    
    def optimize_partially(self, graph: IRGraph) -> IRGraph:
        """Optimize portions of the graph we understand."""
        optimized_nodes = {}
        
        for node_id, node in graph.nodes.items():
            # Try to optimize this node
            if self._can_optimize_node(node):
                optimized_nodes[node_id] = self._optimize_node(node)
            else:
                # Keep original
                optimized_nodes[node_id] = node
        
        # Rebuild graph with optimized nodes
        return IRGraph(nodes=optimized_nodes, edges=graph.edges)
    
    def _can_optimize_node(self, node: IRNode) -> bool:
        """Check if we can optimize this node."""
        sig = node.metadata.get('signature')
        if not sig:
            return False
        
        # Can optimize pure operations
        if sig.is_pure and (sig.is_tensor_op or sig.can_batch):
            return True
        
        # Can optimize known patterns
        if self._matches_known_pattern(node):
            return True
        
        return False
    
    def _optimize_node(self, node: IRNode) -> IRNode:
        """Create optimized version of node."""
        sig = node.metadata.get('signature')
        
        if sig.is_tensor_op:
            # Wrap with JAX transformations
            optimized_op = self._wrap_tensor_op(node.operator)
        elif sig.can_batch:
            # Wrap with batching logic
            optimized_op = self._wrap_batchable_op(node.operator)
        else:
            optimized_op = node.operator
        
        # Return new node with optimized operator
        return IRNode(
            id=node.id,
            operator=optimized_op,
            inputs=node.inputs,
            outputs=node.outputs,
            metadata={**node.metadata, 'optimized': True}
        )
```

### Day 9: Advanced Profiling

**Morning: Detailed Performance Tracking**
```python
# ember/xcs/_internal/profiler.py
@dataclass
class ExecutionProfile:
    """Detailed execution profile."""
    total_time_ms: float
    node_times_ms: Dict[str, float]
    parallel_efficiency: float
    cache_hit_rate: float
    optimization_overhead_ms: float
    
    def to_stats(self) -> Dict[str, Any]:
        """Convert to user-friendly stats."""
        return {
            'total_time_ms': self.total_time_ms,
            'optimization_overhead_ms': self.optimization_overhead_ms,
            'net_speedup': self._calculate_net_speedup(),
            'parallel_efficiency': self.parallel_efficiency,
            'cache_hit_rate': self.cache_hit_rate,
            'slowest_operations': self._get_slowest_ops(),
            'optimization_opportunities': self._suggest_optimizations()
        }

class Profiler:
    """Advanced profiling with actionable insights."""
    
    def __init__(self):
        self.profiles = defaultdict(list)
        self.optimization_time = defaultdict(float)
    
    def profile_execution(self, 
                         func_name: str,
                         graph: IRGraph,
                         analysis: GraphParallelismAnalysis,
                         execution_time: float,
                         node_times: Dict[str, float]):
        """Record execution profile."""
        profile = ExecutionProfile(
            total_time_ms=execution_time * 1000,
            node_times_ms={k: v * 1000 for k, v in node_times.items()},
            parallel_efficiency=self._calculate_efficiency(analysis, node_times),
            cache_hit_rate=self._get_cache_stats(),
            optimization_overhead_ms=self.optimization_time[func_name] * 1000
        )
        
        self.profiles[func_name].append(profile)
    
    def get_function_stats(self, func_name: str) -> Dict[str, Any]:
        """Get aggregated stats for function."""
        profiles = self.profiles.get(func_name, [])
        if not profiles:
            return {'status': 'no_data'}
        
        # Aggregate profiles
        avg_profile = self._aggregate_profiles(profiles)
        stats = avg_profile.to_stats()
        
        # Add trends
        if len(profiles) > 1:
            stats['performance_trend'] = self._calculate_trend(profiles)
        
        return stats
```

**Afternoon: Optimization Suggestions**
```python
# ember/xcs/_internal/optimization_advisor.py
class OptimizationAdvisor:
    """Provides actionable optimization suggestions."""
    
    def analyze_profile(self, profile: ExecutionProfile, graph: IRGraph) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Check parallel efficiency
        if profile.parallel_efficiency < 0.7:
            suggestions.append(
                "Low parallel efficiency detected. Consider:\n"
                "- Reducing shared state between parallel operations\n"
                "- Batching smaller operations together\n"
                "- Using process pool for CPU-bound operations"
            )
        
        # Check for bottlenecks
        bottlenecks = self._find_bottlenecks(profile.node_times_ms)
        if bottlenecks:
            suggestions.append(
                f"Bottleneck operations: {', '.join(bottlenecks)}\n"
                "Consider:\n"
                "- Caching results if operations are deterministic\n"
                "- Replacing with more efficient implementations\n"
                "- Parallelizing within these operations"
            )
        
        # Check cache opportunities
        if profile.cache_hit_rate < 0.3:
            repeated_ops = self._find_repeated_operations(graph)
            if repeated_ops:
                suggestions.append(
                    "Low cache utilization with repeated operations.\n"
                    "Consider:\n"
                    "- Enabling caching for deterministic operations\n"
                    "- Restructuring to avoid redundant computation"
                )
        
        return suggestions
```

### Day 10: Integration Testing Suite

**Full Pipeline Tests**
```python
# tests/integration/xcs/test_robust_pipeline.py
class TestRobustXCSPipeline:
    """Comprehensive tests for robustness."""
    
    def test_complex_graph_optimization(self):
        """Test optimization of complex computation graphs."""
        @jit
        def complex_pipeline(data):
            # Parallel preprocessing
            cleaned = [preprocess(x) for x in data]
            
            # Parallel model calls
            results = []
            for item in cleaned:
                r1 = model1(item)
                r2 = model2(item)
                r3 = model3(item)
                results.append(aggregate([r1, r2, r3]))
            
            # Reduction
            return summarize(results)
        
        # Execute
        result = complex_pipeline(test_data)
        
        # Verify optimization
        stats = get_jit_stats(complex_pipeline)
        assert stats['parallel_groups'] >= 2  # At least 2 parallel sections
        assert stats['vectorizable_chains'] >= 1  # List comprehension
        assert stats['net_speedup'] > 1.5  # Significant speedup
    
    def test_error_recovery(self):
        """Test graceful handling of errors during optimization."""
        @jit
        def partially_failing_pipeline(x):
            a = good_op1(x)
            b = flaky_op(a)  # Sometimes fails
            c = good_op2(b)
            return c
        
        # Should work even if optimization fails
        result = partially_failing_pipeline(10)
        assert result is not None
        
        # Check fallback was used appropriately
        stats = get_jit_stats(partially_failing_pipeline)
        assert 'fallback_reason' in stats or stats['partial_optimization']
    
    def test_thread_safety(self):
        """Test concurrent execution safety."""
        shared_state = {'count': 0}
        
        @jit
        def concurrent_pipeline(x):
            # Multiple operations that might race
            a = increment_op(x, shared_state)
            b = increment_op(x, shared_state)
            c = increment_op(x, shared_state)
            return a + b + c
        
        # Run concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(concurrent_pipeline, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # Verify no race conditions
        assert all(r is not None for r in results)
        # State mutations should be atomic or prevented
```

## Phase 3: Advanced Features (Days 11-15)

### Day 11: Hybrid Operation Splitting

```python
# ember/xcs/_internal/hybrid_executor.py
class HybridExecutor(CompiledExecutor):
    """Executes graphs with mixed tensor/orchestration operations."""
    
    def _prepare_execution_plan(self):
        """Split graph into tensor and orchestration subgraphs."""
        self.tensor_subgraph, self.orch_subgraph = self._split_graph()
        self.boundary_nodes = self._identify_boundaries()
    
    def _split_graph(self) -> Tuple[IRGraph, IRGraph]:
        """Split into tensor and orchestration subgraphs."""
        tensor_nodes = {}
        orch_nodes = {}
        
        for node_id, node in self.graph.nodes.items():
            sig = node.metadata.get('signature', OperationSignature())
            if sig.is_tensor_op:
                tensor_nodes[node_id] = node
            else:
                orch_nodes[node_id] = node
        
        # Build subgraphs preserving dependencies
        tensor_graph = self._build_subgraph(tensor_nodes)
        orch_graph = self._build_subgraph(orch_nodes)
        
        return tensor_graph, orch_graph
    
    def execute(self, args: tuple, kwargs: dict) -> Any:
        """Execute with appropriate strategy for each subgraph."""
        context = self._create_context(args, kwargs)
        
        # Execute in topological order respecting boundaries
        for phase in self._execution_phases():
            if phase.is_tensor:
                self._execute_tensor_phase(phase, context)
            else:
                self._execute_orchestration_phase(phase, context)
        
        return self._extract_result(context)
```

### Day 12: Complete vmap/pmap for Orchestration

```python
# ember/xcs/transformations.py additions
def _parallel_orchestration_vmap(func, args, kwargs, in_axes):
    """Enhanced vmap for orchestration operations."""
    # Detect batch structure
    batch_info = _detect_batch_structure(args, in_axes)
    
    # Create execution plan
    plan = ParallelOrchestrationPlan(
        func=func,
        batch_size=batch_info.size,
        input_slicing=batch_info.slicing_strategy
    )
    
    # Execute with optimal parallelism
    with OrchestrationExecutor(max_workers=_optimal_workers(batch_info)) as executor:
        # Submit all tasks
        futures = []
        for i in range(batch_info.size):
            item_args = plan.slice_inputs(args, i)
            item_kwargs = plan.slice_kwargs(kwargs, i)
            
            future = executor.submit(func, *item_args, **item_kwargs)
            futures.append(future)
        
        # Collect with proper error handling
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                raise XCSExecutionError(
                    f"vmap failed at index {i}",
                    original_error=e
                ) from e
    
    # Reconstruct output structure
    return plan.reconstruct_output(results)
```

### Day 13-15: Testing and Polish

Complete test coverage, documentation, and performance benchmarks.

## Success Criteria

1. **All components connected**: Tracer → IR → Analyzer → Engine working together
2. **Robust operation detection**: >95% accuracy in identifying operation types
3. **Error handling**: All errors provide actionable context
4. **Performance**: Measurable speedup on parallel workloads
5. **Reliability**: <0.1% failure rate in production use

This roadmap transforms XCS from a promising design into a production-ready system that delivers on the promise of automatic parallelism discovery.