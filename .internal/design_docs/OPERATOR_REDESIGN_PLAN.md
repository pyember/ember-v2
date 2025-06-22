# Operator System Redesign Plan

## Core Principles
1. **Progressive Disclosure**: Simple things simple, complex things possible
2. **No Leaky Abstractions**: Rich internals, clean interfaces
3. **One Obvious Way**: Opinionated design that eliminates choice paralysis
4. **Explicit Over Magic**: Clear method names, predictable types
5. **10x Improvements**: Platform thinking, not feature thinking

## Architecture Overview

### Layer 1: Simple Function Operators (90% Use Case)
```python
@op
def classify_sentiment(text: str) -> str:
    """Simple operators are just functions with types."""
    return model.generate(f"Classify sentiment: {text}")

@op
def summarize(text: str, max_words: int = 100) -> str:
    """Type hints automatically become specifications."""
    return model.generate(f"Summarize in {max_words} words: {text}")
```

### Layer 2: Structured Operators (Advanced Use)
```python
@op
class EnsembleClassifier(EmberModule):
    """For when you need state and configuration."""
    models: List[ModelBinding]
    voting_strategy: str = "majority"
    
    def __call__(self, text: str) -> Classification:
        results = [m.generate(text) for m in self.models]
        return self.vote(results)
```

### Layer 3: Custom Operators (Power Users)
```python
class CustomOperator(Operator[InputT, OutputT]):
    """Full control when needed."""
    
    def forward(self, input: InputT) -> OutputT:
        # Custom implementation
        pass
    
    def validate_input(self, input: InputT) -> InputT:
        # Optional validation
        pass
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. Design new Operator base class hierarchy
2. Implement @op decorator with type introspection
3. Create specification inference system
4. Build validation mixin system

### Phase 2: EmberModule Enhancement
1. Restore EmberModule with cleaner implementation
2. Remove metaclass complexity
3. Implement efficient tree transformations
4. Add JAX-style pytree support

### Phase 3: Concrete Operators
1. Restore key operators (Ensemble, Verifier, etc.)
2. Implement using new patterns
3. Add model integration support
4. Create composition utilities

### Phase 4: Advanced Features
1. Parallel execution support
2. Streaming capabilities
3. Cost tracking and optimization
4. Performance monitoring

## Key Design Decisions

### 1. Operator Base Class
- Minimal abstract base with `forward()` method
- Optional validation through mixins
- Clean separation of concerns
- No forced inheritance of complex behavior

### 2. Specification System
- Automatic inference from type hints
- Optional explicit specifications
- Runtime validation only when requested
- Clean error messages

### 3. Module System
- Immutable by default
- Efficient tree transformations
- Clear static vs dynamic fields
- No hidden state or magic

### 4. Model Integration
- Clean abstraction over different providers
- Lazy model loading
- Automatic batching support
- Cost tracking built-in

## Review Checkpoints

1. **After Phase 1**: Does the @op decorator feel natural?
2. **After Phase 2**: Is EmberModule cleaner than before?
3. **After Phase 3**: Do operators compose naturally?
4. **After Phase 4**: Is the system 10x better?

## Success Criteria
- Simple operators require < 5 lines of code
- Complex operators are possible without framework fighting
- No surprising behavior or hidden complexity
- Clean stack traces and error messages
- Performance equal or better than original
- Test coverage > 95%