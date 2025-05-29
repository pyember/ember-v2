# Smart Model Discovery - Elegant Blending of Static and Dynamic

## The Vision

What would happen if Jeff Dean, Sanjay Ghemawat, Robert C. Martin, and Steve Jobs designed model discovery together?

### Their Principles

**Jeff Dean & Sanjay Ghemawat**: 
- "Make the common case fast, the rare case correct"
- Hardcoded models for instant startup
- Lazy discovery only when needed
- Smart caching with TTLs

**Robert C. Martin**:
- "Single Responsibility - discovery discovers, registry stores"
- Clean separation between static knowledge and dynamic discovery
- No mixing of concerns

**Steve Jobs**:
- "It just works - no configuration needed"
- Users see what's available without thinking about it
- Graceful degradation without API keys

## The Design

### 1. Three-Layer Architecture

```
┌─────────────────────────────────────┐
│         User Experience             │
│   "ember model list" just works     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Smart Model Registry           │
│  • Hardcoded models (instant)       │
│  • Discovery cache (1hr TTL)        │  
│  • Lazy API calls                   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│       Provider APIs                 │
│  • OpenAI /models endpoint          │
│  • Anthropic API                    │
│  • Google AI API                    │
└─────────────────────────────────────┘
```

### 2. Key Behaviors

**Fast Path (99% of requests)**:
```python
model = get_model("gpt-4")  # Instant - hardcoded
model = get_model("claude-3-opus")  # Instant - hardcoded
```

**Discovery Path (when needed)**:
```python
models = list_models()  # First call: discovers + caches
models = list_models()  # Subsequent calls: uses cache
```

**Smart Model ID Resolution**:
```python
# All of these work
get_model("gpt-4")           # Assumes OpenAI
get_model("openai:gpt-4")    # Explicit provider
get_model("gpt-4-turbo")     # Hardcoded knowledge
get_model("gpt-4-new-model") # Triggers discovery
```

### 3. Implementation Details

**Hardcoded Knowledge**:
- Common models with metadata (context length, capabilities)
- Updated with each Ember release
- Zero startup cost

**Discovery Mechanism**:
- Only runs when:
  - User lists models with API key present
  - User requests unknown model
  - Cache is stale (>1 hour)
- Gracefully handles failures
- Non-blocking for hardcoded models

**Caching Strategy**:
- Per-provider TTL (1 hour default)
- Manual refresh available (`ember model refresh`)
- Thread-safe discovery
- Memory efficient

## User Experience

### Scenario 1: New User
```bash
$ ember model list
OpenAI (✗ No API key)
  → Set OPENAI_API_KEY to use these models

Anthropic (✗ No API key)
  → Set ANTHROPIC_API_KEY to use these models
```

### Scenario 2: Configured User
```bash
$ ember model list
OpenAI (✓ Configured)
  ● gpt-4 - 8,192 tokens [chat, code]
  ● gpt-4-turbo - 128,000 tokens [chat, code, vision]
  ● gpt-3.5-turbo - 16,384 tokens [chat, code]
  ○ gpt-4-1106-preview - 128,000 tokens (discovered 5m ago)

Legend: ● = Always available, ○ = Discovered via API
```

### Scenario 3: Using Models
```bash
# Works immediately - no discovery needed
$ ember invoke gpt-4 "Hello"

# Unknown model - triggers discovery
$ ember invoke gpt-4-brand-new "Hello"
# If exists: works after quick discovery
# If not: helpful error with suggestions
```

## Benefits

1. **Fast Startup**: Zero delay for common operations
2. **Always Current**: Discovery finds new models automatically
3. **Offline Friendly**: Works without internet for known models
4. **Transparent**: Users understand what's happening
5. **Efficient**: Minimal API calls, smart caching

## Code Simplicity

**Before** (complex discovery):
```python
# 500+ lines of discovery code
# Complex initialization
# Slow startup
# Fragile API dependencies
```

**After** (smart discovery):
```python
# ~200 lines total
# Instant startup
# Graceful degradation
# Clear separation of concerns
```

## Why This Design Wins

1. **Performance**: Instant for 99% of use cases
2. **Reliability**: Works offline, handles API failures
3. **Simplicity**: Clear mental model, predictable behavior
4. **Extensibility**: Easy to add new providers
5. **User Experience**: It just works

This is what happens when you combine:
- Jeff & Sanjay's performance obsession
- Uncle Bob's clean architecture
- Steve Jobs' user experience focus

The result: A system that's fast, clean, and delightful to use.