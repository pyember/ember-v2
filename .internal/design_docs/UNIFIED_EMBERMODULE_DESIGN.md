# Unified EmberModule and Operator Design

## Executive Summary

This document unifies the insights from PyTree analysis, module system design, and operator trace architecture into a cohesive vision for EmberModule and Operator. The design prioritizes simplicity, power, and analyzability while maintaining the functional programming principles that make Ember elegant.

## Core Design Philosophy

Channeling the engineering principles of our inspirations:
- **Jeff Dean/Sanjay Ghemawat**: Simple, powerful abstractions that compose naturally
- **Robert C. Martin**: Clean architecture with single responsibilities
- **Steve Jobs**: Obsessive simplicity - remove everything unnecessary
- **John Carmack**: Performance through simplicity, not cleverness

## Architecture Overview

### 1. EmberModule: The Foundation

```python
"""EmberModule: Immutable, transformable, traceable computation units."""

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from contextvars import ContextVar
from contextlib import contextmanager
import time

T = TypeVar('T')

# Global execution context for tracing
_execution_context: ContextVar[List[Dict]] = ContextVar('execution_context', default=[])

class ModuleMeta(type):
    """Metaclass that converts classes to frozen, traceable dataclasses."""
    
    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Skip base class and already processed classes
        if cls.__name__ == 'EmberModule' or dataclasses.is_dataclass(cls):
            return cls
        
        # Convert to frozen dataclass
        cls = dataclasses.dataclass(frozen=True, eq=False)(cls)
        
        # Wrap __call__ for automatic tracing
        if hasattr(cls, '__call__'):
            original_call = cls.__call__
            
            def traced_call(self, *args, **kwargs):
                # Only trace if we have metadata (operators have it)
                if hasattr(self, '_ember_metadata'):
                    with trace_context(self._ember_metadata):
                        start_time = time.perf_counter()
                        try:
                            result = original_call(self, *args, **kwargs)
                            emit_event(OperatorEvent(
                                operator_id=self._ember_metadata.get('id'),
                                event_type='complete',
                                duration=time.perf_counter() - start_time,
                                data={'result_type': type(result).__name__}
                            ))
                            return result
                        except Exception as e:
                            emit_event(OperatorEvent(
                                operator_id=self._ember_metadata.get('id'),
                                event_type='error',
                                duration=time.perf_counter() - start_time,
                                data={'error': str(e)}
                            ))
                            raise
                else:
                    return original_call(self, *args, **kwargs)
            
            cls.__call__ = traced_call
        
        # Register for tree transformations
        TreeRegistry.register(cls, module_flatten, module_unflatten)
        
        return cls

class EmberModule(metaclass=ModuleMeta):
    """Base class for all Ember computational units.
    
    Automatically provides:
    - Immutability through frozen dataclasses
    - Tree transformation support for functional operations
    - Automatic tracing when metadata is present
    - Clean replace() method for updates
    """
    
    def replace(self, **updates) -> 'EmberModule':
        """Create new instance with updated fields."""
        return dataclasses.replace(self, **updates)
    
    def with_metadata(self, **metadata) -> 'EmberModule':
        """Attach metadata for tracing and analysis."""
        # Create a new instance with metadata attached
        new_instance = self.replace()
        object.__setattr__(new_instance, '_ember_metadata', metadata)
        return new_instance
```

### 2. Simplified Tree Operations

```python
class TreeRegistry:
    """Minimal tree transformation registry."""
    _registry: Dict[type, Tuple[Callable, Callable]] = {}
    
    @classmethod
    def register(cls, node_type: type, flatten_fn: Callable, unflatten_fn: Callable):
        """Register tree transformation functions."""
        cls._registry[node_type] = (flatten_fn, unflatten_fn)

def module_flatten(module: EmberModule) -> Tuple[List[Any], Dict]:
    """Flatten module to leaves and structure."""
    # Get all fields
    fields = dataclasses.fields(module)
    
    # Separate static and dynamic
    dynamic_values = []
    dynamic_keys = []
    static_data = {}
    
    for field in fields:
        value = getattr(module, field.name)
        if field.metadata.get('static', False):
            static_data[field.name] = value
        else:
            dynamic_values.append(value)
            dynamic_keys.append(field.name)
    
    # Include metadata if present
    if hasattr(module, '_ember_metadata'):
        static_data['_ember_metadata'] = module._ember_metadata
    
    return dynamic_values, {
        'type': type(module),
        'dynamic_keys': dynamic_keys,
        'static_data': static_data
    }

def module_unflatten(aux: Dict, leaves: List[Any]) -> EmberModule:
    """Reconstruct module from leaves and structure."""
    cls = aux['type']
    kwargs = dict(zip(aux['dynamic_keys'], leaves))
    kwargs.update(aux['static_data'])
    
    # Remove metadata from kwargs, apply separately
    metadata = kwargs.pop('_ember_metadata', None)
    
    instance = cls(**kwargs)
    if metadata:
        object.__setattr__(instance, '_ember_metadata', metadata)
    
    return instance
```

### 3. Event System for Analysis

```python
@dataclasses.dataclass
class OperatorEvent:
    """Structured event from operator execution."""
    operator_id: Optional[str]
    operator_type: Optional[str] = None
    stage: Optional[str] = None
    event_type: str = "trace"  # trace, complete, error, metric
    timestamp: float = dataclasses.field(default_factory=time.time)
    context: List[Dict] = dataclasses.field(default_factory=list)
    duration: Optional[float] = None
    data: Dict[str, Any] = dataclasses.field(default_factory=dict)

class EventCollector:
    """Lightweight event collection system."""
    _collectors: List[Callable[[OperatorEvent], None]] = []
    _enabled: bool = False
    
    @classmethod
    def emit(cls, event: OperatorEvent):
        """Emit event to registered collectors."""
        if cls._enabled and cls._collectors:
            # Capture current context
            event.context = _execution_context.get()
            for collector in cls._collectors:
                collector(event)
    
    @classmethod
    @contextmanager
    def collect(cls):
        """Context manager for collecting events."""
        events = []
        cls._collectors.append(events.append)
        cls._enabled = True
        try:
            yield events
        finally:
            cls._collectors.pop()
            if not cls._collectors:
                cls._enabled = False

@contextmanager
def trace_context(metadata: Dict[str, Any]):
    """Add metadata to execution context."""
    ctx = _execution_context.get()
    new_ctx = ctx + [metadata]
    token = _execution_context.set(new_ctx)
    try:
        yield
    finally:
        _execution_context.reset(token)

# Convenience function
emit_event = EventCollector.emit
```

### 4. Enhanced Operator Base

```python
from typing import ClassVar, Generic
from ember.core.types import InputT, OutputT

class Operator(EmberModule, Generic[InputT, OutputT]):
    """Base operator with automatic tracing and metadata.
    
    Combines the power of EmberModule with operator-specific functionality:
    - Type-safe input/output specifications
    - Automatic tracing and performance monitoring
    - Metadata for analysis and debugging
    """
    
    # Required: specification defines I/O contract
    specification: ClassVar[Specification[InputT, OutputT]]
    
    # Operator metadata (set via fluent interface)
    _ember_metadata: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {'id': None, 'tags': set(), 'type': 'operator'},
        init=False,
        repr=False
    )
    
    def __post_init__(self):
        """Initialize operator metadata."""
        # Generate ID if not set
        if not self._ember_metadata.get('id'):
            self._ember_metadata['id'] = f"{self.__class__.__name__}_{id(self)}"
        
        # Set operator type from class name
        self._ember_metadata['type'] = self.__class__.__name__
    
    @abc.abstractmethod
    def forward(self, *, inputs: InputT) -> OutputT:
        """Implement operator logic."""
        raise NotImplementedError
    
    def __call__(self, *, inputs: Optional[InputT] = None, **kwargs) -> OutputT:
        """Execute with validation and tracing."""
        # Validation logic (from existing implementation)
        # ... existing validation code ...
        
        # The traced_call wrapper from ModuleMeta handles tracing
        return self.forward(inputs=validated_inputs)
    
    # Fluent interface for metadata
    def tag(self, *tags: str) -> 'Operator':
        """Add tags for analysis."""
        new_metadata = self._ember_metadata.copy()
        new_metadata['tags'] = new_metadata.get('tags', set()) | set(tags)
        return self.with_metadata(**new_metadata)
    
    def stage(self, stage_name: str) -> 'Operator':
        """Set execution stage for analysis."""
        new_metadata = self._ember_metadata.copy()
        new_metadata['stage'] = stage_name
        return self.with_metadata(**new_metadata)
    
    def id(self, operator_id: str) -> 'Operator':
        """Set operator ID."""
        new_metadata = self._ember_metadata.copy()
        new_metadata['id'] = operator_id
        return self.with_metadata(**new_metadata)
```

### 5. Analysis API

```python
class OperatorAnalysis:
    """Fluent API for analyzing operator executions."""
    
    def __init__(self, events: List[OperatorEvent]):
        self.events = events
    
    def stage(self, stage_name: str) -> 'OperatorAnalysis':
        """Filter events by stage."""
        filtered = [e for e in self.events if e.context and 
                   any(c.get('stage') == stage_name for c in e.context)]
        return OperatorAnalysis(filtered)
    
    def operator_type(self, type_name: str) -> 'OperatorAnalysis':
        """Filter by operator type."""
        filtered = [e for e in self.events if e.operator_type == type_name]
        return OperatorAnalysis(filtered)
    
    def ensemble_disagreement(self) -> Dict[str, float]:
        """Calculate disagreement rates for ensemble operators."""
        # Group events by ensemble parent
        ensembles = self._group_by_parent_type('EnsembleOperator')
        
        results = {}
        for ensemble_id, events in ensembles.items():
            # Get results from child operators
            child_results = [
                e.data.get('result') for e in events 
                if e.event_type == 'complete' and e.context
            ]
            
            if len(child_results) > 1:
                # Calculate disagreement
                unique_results = len(set(str(r) for r in child_results))
                disagreement = (unique_results - 1) / (len(child_results) - 1)
                results[ensemble_id] = disagreement
        
        return results
    
    def accuracy_by_stage(self) -> Dict[str, Dict[str, float]]:
        """Get accuracy metrics grouped by stage."""
        accuracies = {}
        
        for event in self.events:
            if event.event_type == 'metric' and 'accuracy' in event.data:
                stage = next((c.get('stage') for c in event.context if 'stage' in c), 'unknown')
                
                if stage not in accuracies:
                    accuracies[stage] = []
                accuracies[stage].append(event.data['accuracy'])
        
        # Calculate statistics
        return {
            stage: {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
            for stage, values in accuracies.items()
        }
    
    def judge_effectiveness(self) -> Dict[str, float]:
        """Analyze judge performance based on ensemble agreement."""
        # Implementation similar to the trace architecture document
        # but integrated with the event system
        pass
    
    def performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics."""
        durations = [e.duration for e in self.events if e.duration]
        
        if not durations:
            return {}
        
        return {
            'total_time': sum(durations),
            'mean_time': sum(durations) / len(durations),
            'operator_times': self._operator_performance(),
            'stage_times': self._stage_performance()
        }
```

## Usage Examples

### Basic Operator Definition

```python
class ChainOfThought(Operator[Question, Answer]):
    """Example operator with automatic tracing."""
    
    specification = ChainOfThoughtSpec()
    model: str = "gpt-4"
    temperature: float = 0.7
    
    def forward(self, *, inputs: Question) -> Answer:
        # Reasoning step
        reasoning = self.llm(
            f"Think step by step: {inputs.text}",
            temperature=self.temperature
        )
        
        # Answer step
        answer = self.llm(
            f"Given: {reasoning}\nAnswer: {inputs.text}",
            temperature=0.1
        )
        
        return Answer(text=answer)

# Usage with tracing
cot = ChainOfThought()
    .tag("reasoning", "gpt4")
    .stage("reasoning_stage")
    .id("cot_primary")
```

### Ensemble Analysis

```python
# Create ensemble with metadata
ensemble = EnsembleOperator(
    operators=[
        DirectAnswer().tag("direct").stage("ensemble_1"),
        ChainOfThought().tag("cot").stage("ensemble_1"),
        StepByStep().tag("steps").stage("ensemble_1")
    ]
).stage("ensemble_1")

# Create judge
judge = SelectorJudge().stage("judge_1")

# Run with event collection
with EventCollector.collect() as events:
    # Run pipeline
    ensemble_results = ensemble(question)
    final_answer = judge(ensemble_results)
    
    # Analyze
    analysis = OperatorAnalysis(events)
    
    # How much disagreement?
    disagreement = analysis.stage("ensemble_1").ensemble_disagreement()
    print(f"Ensemble disagreement: {disagreement}")
    
    # Performance breakdown
    perf = analysis.performance_summary()
    print(f"Total time: {perf['total_time']:.2f}s")
```

### Transformation Support

```python
# EmberModules support functional transformations
from ember.xcs import vmap

# Batch processing
batch_cot = vmap(cot)  # Works because EmberModule supports tree operations
batch_results = batch_cot(batch_of_questions)

# The metadata is preserved through transformations
assert batch_cot._ember_metadata['tags'] == {'reasoning', 'gpt4'}
```

## Design Rationale

### Why Combine Module and Tracing?

1. **Zero Configuration**: Tracing happens automatically when metadata is present
2. **No Performance Penalty**: Without metadata, no tracing overhead
3. **Transformation Friendly**: Metadata travels with transformations
4. **Clean Separation**: Base functionality (EmberModule) vs domain-specific (Operator)

### Why Simple Events?

1. **Structured Logging Paradigm**: Events are just structured data
2. **Flexible Analysis**: Any analysis can be built on the event stream
3. **Easy Testing**: Events can be collected and asserted in tests
4. **Production Ready**: Can pipe to real monitoring systems

### Why Metadata Pattern?

1. **Opt-in Complexity**: Only add metadata when you need analysis
2. **Fluent Interface**: Natural way to configure operators
3. **Composable**: Metadata merges naturally in composed operators
4. **Debugging**: Rich context for understanding execution

## Implementation Strategy

### Phase 1: Core Infrastructure
- Implement new EmberModule with metaclass
- Add tree registry and basic transformations
- Create event system

### Phase 2: Operator Integration
- Update Operator to use new EmberModule
- Add metadata fluent interface
- Implement automatic tracing

### Phase 3: Analysis Tools
- Build OperatorAnalysis API
- Create visualization tools
- Add real-time monitoring

### Phase 4: Ecosystem
- Update all existing operators
- Create migration guide
- Build example notebooks

## Conclusion

This unified design achieves our goals:

1. **Simple**: EmberModule is just a frozen dataclass with benefits
2. **Powerful**: Automatic tracing, tree transformations, metadata
3. **Analyzable**: Rich event stream enables any analysis
4. **Performant**: Zero cost when features aren't used
5. **Pythonic**: Leverages standard Python features

The design embodies the principles of our inspirations while solving the real problems of analyzing complex operator hierarchies. It's simple enough that anyone can understand it, yet powerful enough to handle sophisticated use cases.