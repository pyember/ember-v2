# Ember Radical Simplification: The 99% Solution

## Executive Summary

The Ember framework's `models()` API proves that simplicity wins. This redesign takes that success to its logical conclusion: **eliminate everything except what 99% of users actually need**. No contexts, no configuration, no initialization - just functions that work. Like `requests.get()` or `numpy.array()`, Ember should be obvious from the first import.

## Design Principles

1. **Just Functions** - No classes, no objects, no state
2. **Environment Only** - API keys from environment, nothing else
3. **Direct Dispatch** - No registries, no discovery, no indirection
4. **99% Coverage** - Optimize for what nearly everyone needs
5. **Like NumPy** - Obvious, immediate, no documentation needed

## The Problem

Even with simplifications, Ember still has:
- **2 context systems** (EmberContext, ModelContext)
- **3 initialization methods** (initialize_ember, init, auto-init)
- **4 configuration sources** (YAML, env, code, auto-discovery)
- **50+ classes** for operators, registries, builders
- **1000s of lines** of initialization code

**Users just want to:**
```python
answer = ask_ai("gpt-4", "What is 2+2?")
```

Everything else is complexity for the 1%.

## The Solution: Just Functions

### Complete Public API
```python
import ember

# Models - call any LLM
answer = ember.models("gpt-4", "What is 2+2?")

# Data - load any dataset  
dataset = ember.data("mmlu")

# Eval - evaluate anything
score = ember.eval("gpt-4", "mmlu", "accuracy")

# Fast - optimize any function
@ember.fast
def pipeline(text):
    return ember.models("gpt-4", f"Analyze: {text}")
```

**That's it. The entire API.**

### Implementation (ember/__init__.py)
```python
import os
from functools import lru_cache

# Direct provider dispatch - no registries
def models(model: str, prompt: str, **kwargs):
    """Call any model. Just works."""
    if "gpt" in model:
        return _call_openai(model, prompt, **kwargs)
    elif "claude" in model:
        return _call_anthropic(model, prompt, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")

# Direct dataset loading - no builders
def data(name: str, **kwargs):
    """Load any dataset. Returns iterator."""
    loaders = {
        "mmlu": _load_mmlu,
        "humaneval": _load_humaneval,
        "truthfulqa": _load_truthfulqa,
    }
    if name in loaders:
        return loaders[name](**kwargs)
    elif "path" in kwargs:  # Custom dataset
        return _load_file(kwargs["path"])
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Direct evaluation - no pipelines
def eval(model: str, data_name: str, metric: str = "accuracy"):
    """Evaluate model on dataset."""
    dataset = data(data_name) if isinstance(data_name, str) else data_name
    correct = 0
    total = 0
    
    for item in dataset:
        response = models(model, item["prompt"])
        if metric == "accuracy":
            correct += response.strip() == item["answer"].strip()
        total += 1
    
    return correct / total if total > 0 else 0

# Simple optimization - no XCS
fast = lru_cache(maxsize=1000)

# Private implementation details
def _call_openai(model, prompt, **kwargs):
    # Direct API call, no wrappers
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )
    return response.choices[0].message.content

# Similar for other providers...
```

### What Gets Eliminated

| Component | Lines of Code | Replacement |
|-----------|--------------|-------------|
| EmberContext | 600+ | None needed |
| ModelContext | 400+ | None needed |
| ConfigManager | 300+ | os.environ |
| Registry system | 1000+ | Direct dispatch |
| Operator classes | 2000+ | Just functions |
| XCS system | 3000+ | @lru_cache |
| Type system | 500+ | Duck typing |
| **Total** | **~8000 lines** | **~200 lines** |

### Architecture? What Architecture?

```
User Code
    ↓
ember.models() / ember.data() / ember.eval()
    ↓
Direct API calls

That's it. No layers. No indirection.
```

## Implementation Plan

### Phase 1: Create Core Functions (Week 1)
1. Implement `ember.models()` with direct provider calls
2. Implement `ember.data()` with direct loaders
3. Implement `ember.eval()` with simple loop
4. Implement `ember.fast` as lru_cache alias

### Phase 2: Remove Everything Else (Week 2)
1. Delete EmberContext and ModelContext
2. Delete ConfigManager and all YAML handling
3. Delete Registry systems
4. Delete Operator classes (keep logic as functions)
5. Delete XCS (keep @lru_cache for caching)

### Phase 3: Migration (Week 3)
1. Create ember.legacy module with old APIs
2. Update examples to use new functions
3. Add deprecation warnings
4. Update documentation

### Phase 4: Ship It (Week 4)
1. Test the 200-line implementation
2. Ensure backward compatibility via legacy module
3. Update README with new examples
4. Release

## Real-World Examples

### Example 1: Basic Q&A
```python
import ember

# Just ask
answer = ember.models("gpt-4", "What is the capital of France?")
print(answer)  # "Paris"
```

### Example 2: Evaluate a Model
```python
import ember

# One line evaluation
accuracy = ember.eval("gpt-3.5-turbo", "mmlu", "accuracy")
print(f"MMLU Score: {accuracy:.2%}")  # "MMLU Score: 67.23%"
```

### Example 3: Build a RAG System
```python
import ember

def rag(question, documents):
    # No operators needed - just use Python
    context = "\n".join(documents)
    prompt = f"Context: {context}\n\nQuestion: {question}"
    return ember.models("gpt-4", prompt)

# Use it
docs = ["Paris is the capital of France.", "London is the capital of UK."]
answer = rag("What is the capital of France?", docs)
```

### Example 4: Optimize Performance
```python
import ember

# Make any function faster with caching
@ember.fast
def expensive_analysis(text):
    summary = ember.models("gpt-4", f"Summarize: {text}")
    keywords = ember.models("gpt-4", f"Extract keywords: {summary}")
    return {"summary": summary, "keywords": keywords}

# Cached after first call
result1 = expensive_analysis("Long text...")  # Slow
result2 = expensive_analysis("Long text...")  # Instant
```

### Example 5: Custom Dataset
```python
import ember

# Load built-in dataset
for item in ember.data("mmlu"):
    question = item["prompt"]
    answer = ember.models("gpt-4", question)
    print(f"Q: {question}\nA: {answer}\n")

# Load custom dataset
my_data = ember.data("custom", path="./my_questions.jsonl")
for item in my_data:
    print(item)
```

## Migration Examples

### Complex Operator → Simple Function
```python
# Before: 50 lines of operator code
from ember.core.registry.operator import Ensemble, MajorityVote
from ember.core.registry.operator.base import OperatorBase

class MyComplexOperator(OperatorBase):
    def __init__(self, models, temperature=0.7):
        super().__init__()
        self.ensemble = Ensemble(models)
        self.voter = MajorityVote()
        self.temperature = temperature
    
    def forward(self, prompt):
        results = self.ensemble(prompt, temperature=self.temperature)
        return self.voter(results)

operator = MyComplexOperator(["gpt-4", "claude-3"])
result = operator("What is 2+2?")

# After: 3 lines
import ember

def my_operator(prompt):
    gpt = ember.models("gpt-4", prompt, temperature=0.7)
    claude = ember.models("claude-3", prompt, temperature=0.7)
    return gpt if gpt == claude else f"Disagree: {gpt} vs {claude}"

result = my_operator("What is 2+2?")
```

### Configuration Hell → Environment Variables
```python
# Before: config.yaml + initialization + context
config = {
    "registry": {
        "providers": {
            "openai": {
                "api_keys": {"default": {"key": "${OPENAI_API_KEY}"}},
                "models": [...] 
            }
        }
    }
}
ember.initialize_ember(config_path="config.yaml")
ctx = EmberContext.current()
model = ctx.get_model("gpt-4")

# After: Just use it
import ember
ember.models("gpt-4", "Hello")  # Reads OPENAI_API_KEY automatically
```

### XCS Pipeline → Regular Python
```python
# Before: Complex XCS graph
@xcs.jit(mode="structural")
@xcs.trace
def complex_pipeline(text):
    with xcs.graph() as g:
        summary = g.node(models, "gpt-4", f"Summarize: {text}")
        keywords = g.node(models, "gpt-4", f"Keywords: {summary}")
        return g.output({"summary": summary, "keywords": keywords})

# After: Just Python with caching
@ember.fast
def simple_pipeline(text):
    summary = ember.models("gpt-4", f"Summarize: {text}")
    keywords = ember.models("gpt-4", f"Keywords: {summary}")
    return {"summary": summary, "keywords": keywords}
```

## For the 1% Who Need More

```python
# Advanced users can still access internals
from ember.legacy import (
    EmberContext, 
    Operator, 
    XCS,
    ConfigManager
)

# But we don't advertise this
```

## Why This Works

### It's Obvious
```python
# NumPy
array = np.array([1, 2, 3])

# Requests  
response = requests.get("https://api.com")

# Ember
answer = ember.models("gpt-4", "Hello")
```

### It's Fast
- No initialization overhead
- No context lookups  
- No registry scanning
- Direct function calls

### It's Simple
- 200 lines vs 8000 lines
- 4 functions vs 50+ classes
- No configuration vs YAML hell
- Just works vs initialization dance

## The Philosophy

**Jeff Dean**: "Build simple things that work at scale."  
**Sanjay Ghemawat**: "Optimize for the common case."  
**Robert Martin**: "The best code is no code."  
**Steve Jobs**: "Simplicity is the ultimate sophistication."

This design embodies all four principles.

## Conclusion

By eliminating 97% of the codebase and focusing on what users actually do, we create an API that needs no documentation. Like `requests.get()` or `numpy.array()`, `ember.models()` is self-explanatory. 

The complex use cases that need operators, XCS, and configuration represent less than 1% of usage. For them, we keep the legacy module. For everyone else, we give them what they actually want: **functions that just work**.