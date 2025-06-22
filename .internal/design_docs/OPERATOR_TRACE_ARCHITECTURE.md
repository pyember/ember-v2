# Operator Trace Architecture

## Design Philosophy

Drawing inspiration from the engineering principles of Jeff Dean/Sanjay Ghemawat (simple, powerful abstractions), Robert C. Martin (clean architecture), and Steve Jobs (obsessive simplicity), this design enables sophisticated operator analysis through elegant primitives.

## Core Principles

1. **Structured Events, Not Complex Traces**: Like structured logging revolutionized debugging, structured operator events enable powerful analysis
2. **Automatic Context Propagation**: Hierarchy and relationships are captured automatically
3. **Zero-Cost When Unused**: No performance penalty unless actively analyzing
4. **Composable Analysis**: Simple queries compose into complex analyses

## Architecture Overview

### 1. Event Model

```python
@dataclasses.dataclass
class OperatorEvent:
    """Single event from operator execution."""
    operator_id: str
    operator_type: str
    stage: str  # e.g., "ensemble_1", "judge_1"
    event_type: str  # "call", "result", "metric"
    timestamp: float
    context: Dict[str, Any]  # Parent operators, tags, etc.
    data: Any  # Event-specific data
```

### 2. Automatic Context Tracking

```python
class ExecutionContext:
    """Thread-local context for tracking operator hierarchy."""
    _context: ContextVar[List[Dict]] = ContextVar('execution_context', default=[])
    
    @contextmanager
    def operator(self, op_id: str, op_type: str, stage: str):
        """Context manager for operator execution."""
        ctx = self._context.get()
        new_ctx = ctx + [{"id": op_id, "type": op_type, "stage": stage}]
        token = self._context.set(new_ctx)
        try:
            yield
        finally:
            self._context.reset(token)
```

### 3. Enhanced Operator Base

```python
class TracedOperator(EmberModule):
    """Operator that automatically emits structured events."""
    
    func: Callable
    metadata: OperatorMetadata
    
    def __call__(self, *args, **kwargs):
        # Automatic context tracking
        with ExecutionContext().operator(
            self.metadata.id,
            self.metadata.type_hint,
            self.metadata.tags.get("stage", "default")
        ):
            # Emit call event
            event = OperatorEvent(
                operator_id=self.metadata.id,
                operator_type=self.metadata.type_hint,
                stage=self.metadata.tags.get("stage", "default"),
                event_type="call",
                timestamp=time.time(),
                context=ExecutionContext.current(),
                data={"args": args, "kwargs": kwargs}
            )
            EventCollector.emit(event)
            
            # Execute
            result = self.func(*args, **kwargs)
            
            # Emit result event
            EventCollector.emit(OperatorEvent(
                ...
                event_type="result",
                data={"result": result}
            ))
            
            return result
```

### 4. Analysis API

```python
class OperatorAnalysis:
    """Fluent API for analyzing operator execution."""
    
    def __init__(self, events: List[OperatorEvent]):
        self.events = events
    
    def stage(self, stage_name: str) -> 'OperatorAnalysis':
        """Filter to specific stage."""
        return OperatorAnalysis([
            e for e in self.events 
            if e.stage == stage_name
        ])
    
    def ensemble_disagreement(self) -> float:
        """Calculate disagreement rate among ensemble members."""
        # Group by parent ensemble
        ensembles = self._group_by_parent("ensemble")
        
        disagreements = []
        for ensemble_id, events in ensembles.items():
            results = [e.data["result"] for e in events if e.event_type == "result"]
            if len(results) > 1:
                # Calculate disagreement (% of non-unanimous decisions)
                unique_results = len(set(str(r) for r in results))
                disagreement = (unique_results - 1) / (len(results) - 1)
                disagreements.append(disagreement)
        
        return sum(disagreements) / len(disagreements) if disagreements else 0.0
    
    def accuracy_distribution(self) -> Dict[str, float]:
        """Get accuracy distribution by operator type."""
        accuracies = {}
        
        for event in self.events:
            if event.event_type == "metric" and "accuracy" in event.data:
                key = f"{event.operator_type}:{event.stage}"
                if key not in accuracies:
                    accuracies[key] = []
                accuracies[key].append(event.data["accuracy"])
        
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "samples": len(v)}
            for k, v in accuracies.items()
        }
    
    def judge_accuracy_by_agreement(self) -> Dict[int, float]:
        """Analyze judge accuracy based on ensemble agreement levels."""
        results = {}
        
        # Find judge decisions and their corresponding ensemble inputs
        for judge_event in self.events:
            if judge_event.operator_type == "judge" and judge_event.event_type == "result":
                # Find ensemble that fed this judge
                ensemble_events = self._find_parent_ensemble_results(judge_event)
                
                if ensemble_events:
                    agreement_count = self._count_majority(ensemble_events)
                    total = len(ensemble_events)
                    
                    key = f"{agreement_count}/{total}"
                    if key not in results:
                        results[key] = []
                    
                    # Compare judge decision with ground truth
                    results[key].append(judge_event.data.get("correct", False))
        
        # Calculate accuracy for each agreement level
        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in results.items()
        }
```

### 5. Usage Examples

```python
# Simple usage - automatic tracing
@trace()
class MyEnsemble(Ensemble):
    operators: Tuple[Operator, ...]
    
    def __call__(self, input):
        # Automatically traced
        return super().__call__(input)

# Analysis
with collect_events() as events:
    # Run your pipeline
    result = pipeline(data)

# Analyze
analysis = OperatorAnalysis(events)

# How much disagreement in ensemble stage 1?
disagreement = analysis.stage("ensemble_1").ensemble_disagreement()

# What's the accuracy distribution?
accuracies = analysis.accuracy_distribution()

# How do judges perform based on ensemble agreement?
judge_perf = analysis.judge_accuracy_by_agreement()
# Output: {"3/3": 1.0, "2/3": 0.85, "1/3": 0.45}
```

## Implementation Details

### Event Collection

```python
class EventCollector:
    """Global event collector with minimal overhead."""
    _collectors: List[Callable] = []
    _enabled: bool = False
    
    @classmethod
    def emit(cls, event: OperatorEvent):
        """Emit event to all collectors."""
        if cls._enabled:
            for collector in cls._collectors:
                collector(event)
    
    @classmethod
    @contextmanager
    def collect(cls) -> List[OperatorEvent]:
        """Context manager for collecting events."""
        events = []
        cls._collectors.append(events.append)
        cls._enabled = True
        try:
            yield events
        finally:
            cls._collectors.remove(events.append)
            if not cls._collectors:
                cls._enabled = False
```

### Automatic Operator Enhancement

```python
def trace(stage: Optional[str] = None):
    """Decorator to add tracing to operators."""
    def decorator(cls):
        # Wrap __call__ method
        original_call = cls.__call__
        
        def traced_call(self, *args, **kwargs):
            # Add stage tag if provided
            if stage and hasattr(self, 'metadata'):
                self.metadata.tags.add(f"stage:{stage}")
            
            # Emit events and track context
            with ExecutionContext().operator(...):
                # ... event emission logic ...
                return original_call(self, *args, **kwargs)
        
        cls.__call__ = traced_call
        return cls
    
    return decorator
```

### Performance Optimizations

1. **Zero-cost when disabled**: Events only collected when explicitly enabled
2. **Structured data**: Events use simple dictionaries, not complex objects
3. **Lazy analysis**: Analysis computed on-demand, not during execution
4. **Context propagation**: Uses Python's ContextVar for efficient thread-local storage

## Advanced Analysis Patterns

### Custom Metrics

```python
class AccuracyMetric:
    """Automatically emit accuracy metrics."""
    def __init__(self, ground_truth_key: str):
        self.ground_truth_key = ground_truth_key
    
    def __call__(self, operator_result, input_data):
        accuracy = self.calculate_accuracy(
            operator_result,
            input_data[self.ground_truth_key]
        )
        
        EventCollector.emit(OperatorEvent(
            event_type="metric",
            data={"accuracy": accuracy}
        ))
```

### Visualization

```python
def visualize_operator_flow(events: List[OperatorEvent]):
    """Generate Graphviz visualization of operator execution flow."""
    # Group events by operator relationships
    # Generate DOT notation
    # Show disagreement rates, accuracies on edges/nodes
```

### Real-time Monitoring

```python
class OperatorMonitor:
    """Real-time monitoring of operator performance."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.events = deque(maxlen=window_size)
        EventCollector.register(self.events.append)
    
    def get_rolling_stats(self) -> Dict:
        """Get rolling statistics."""
        analysis = OperatorAnalysis(list(self.events))
        return {
            "disagreement_rate": analysis.ensemble_disagreement(),
            "accuracy_by_stage": analysis.accuracy_distribution()
        }
```

## Design Rationale

### Why Structured Events?

1. **Simplicity**: Events are just data, easy to understand and process
2. **Flexibility**: Any analysis can be built on top of the event stream
3. **Performance**: Minimal overhead, can be sampled or filtered
4. **Debugging**: Events can be logged, stored, replayed

### Why Automatic Context?

1. **Zero Configuration**: Operators don't need to know about tracing
2. **Accurate Hierarchy**: Parent-child relationships captured automatically
3. **Thread Safety**: ContextVar handles thread-local state correctly

### Why Fluent Analysis API?

1. **Discoverable**: Methods chain naturally, IDE autocomplete helps
2. **Composable**: Complex analyses built from simple operations
3. **Testable**: Each method has single responsibility

## Conclusion

This architecture achieves the goal of making simple analyses trivial while enabling sophisticated investigations. By focusing on structured events and automatic context propagation, we avoid the complexity of traditional tracing systems while providing more power for the specific use case of analyzing operator behavior in ensemble/judge configurations.

The design embodies:
- **Jeff Dean/Sanjay's simplicity**: Just events and context, no complex machinery
- **Robert C. Martin's principles**: Single responsibility, open for extension
- **Steve Jobs' elegance**: The API is so simple it's obvious how to use it