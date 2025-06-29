# XCS JIT Issue Analysis

## Failing Functions and Root Cause

### 1. optimization_techniques.py - `analyze_sentiment_detailed`
```python
def analyze_sentiment_detailed(text: str) -> dict:
    sentiment = models("gpt-3.5-turbo", f"Classify sentiment: {text}").text.strip()
    emotion = models("gpt-3.5-turbo", f"Primary emotion: {text}").text.strip()
    intensity = models("gpt-3.5-turbo", f"Sentiment intensity: {text}").text
    return {"sentiment": sentiment, "emotion": emotion, "intensity": intensity}

fast_analyze = jit(analyze_sentiment_detailed)  # Works
result = fast_analyze(text)  # Fails with UUID error
```

### 2. natural_api_showcase.py - `classify_email`
```python
def classify_email(email: str) -> dict:
    urgency = models("gpt-3.5-turbo", f"Rate urgency: {email}").text
    intent = models("gpt-3.5-turbo", f"What's the intent: {email}").text
    actions = models("gpt-3.5-turbo", f"List action items: {email}").text
    return {"urgency": urgency, "intent": intent, "actions": actions}

fast_classify = jit(classify_email)  # Works
result = fast_classify("When can we schedule?")  # Fails with UUID error
```

### 3. advanced_techniques.py - Stateful conversation
```python
# Similar pattern - JIT on functions with model calls

## Root Cause

The issue occurs in `src/ember/xcs/_internal/engine.py` at line 225:
```python
result = node.operator(*node.metadata['args'])
```

The XCS system is storing the actual function arguments in the IR metadata during tracing, then trying to replay them during execution. When those arguments involve model operations that generate UUIDs for request tracking, the replay fails.

## Why This Is a Bug

The Ember XCS JIT is designed to handle hybrid tensor/orchestration operations. According to the documentation:

> XCS intelligently separates:
> - Pure computations (fully compilable)
> - Model calls (cached and batched)
> - Tool/API/MCP calls (deferred execution)

However, the current implementation is not properly handling model calls. It's trying to store and replay them as if they were pure operations, leading to the UUID generation error.

## Expected Behavior

The XCS JIT should:
1. Detect that `models()` calls are orchestration operations
2. Store references to these operations, not their actual arguments
3. Execute them properly during graph execution
4. Provide caching and batching optimizations

## Fix Required

The IR builder needs to distinguish between:
- Pure tensor operations (can store/replay args)
- Orchestration operations (need special handling)

The metadata should store operation types and handle them appropriately during execution.