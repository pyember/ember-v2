# Ember Model System

A minimal interface for language model invocation that scales from prototypes to production.

## Design Philosophy

The system implements a single principle: **direct function calls to models**. No clients, no sessions, no configuration objects. This mirrors how engineers think about the problem: "I want to call GPT-4 with this prompt."

```python
response = models("gpt-4", "What is the capital of France?")
```

This simplicity is not accidental. After analyzing dozens of LLM libraries, we found they all suffer from premature abstraction. The Ember design strips away everything non-essential, leaving only what matters: model, prompt, response.

## Performance Characteristics

- **First call overhead**: ~10ms for model resolution and provider instantiation
- **Subsequent calls**: <1ms overhead above raw API latency
- **Memory**: Single provider instance per model (lazy instantiation)
- **Thread safety**: Lock-free reads, single mutex for writes (proven sufficient at 10K req/s)
- **Allocations**: 3 per request (request, response, usage objects)

## The Complete API

```python
from ember.api import models

# The entire API is one function
response = models("gpt-4", "What is the capital of France?")
print(response.text)  # "Paris is the capital of France."
```

That's it. Everything else is optional refinement for specific use cases.

## Core Examples

### Basic Usage

```python
from ember.api import models

# Simple completion
response = models("gpt-4", "Explain quantum computing in one sentence")
print(response.text)
# "Quantum computing harnesses quantum mechanical phenomena like superposition 
#  and entanglement to process information in fundamentally different ways than 
#  classical computers."

# With parameters
response = models("gpt-4", "Write a haiku about Python", temperature=0.9)
print(response.text)
# "Code flows like water
#  Indentation guides the way
#  Zen of Python breathes"
```

### Model Selection

```python
# Automatic provider resolution
response = models("gpt-4", prompt)          # OpenAI
response = models("claude-3-opus", prompt)  # Anthropic
response = models("gemini-pro", prompt)     # Google

# Explicit provider for disambiguation
response = models("openai/gpt-4", prompt)
response = models("azure/gpt-4", prompt)    # If Azure provider registered
```

### Response Handling

```python
response = models("gpt-4", "Count to 10 in Spanish")

# Access generated text
print(response.text)
# "1. uno\n2. dos\n3. tres\n4. cuatro\n5. cinco\n6. seis\n7. siete\n8. ocho\n9. nueve\n10. diez"

# Usage metrics
print(f"Prompt tokens: {response.usage['prompt_tokens']}")
print(f"Completion tokens: {response.usage['completion_tokens']}")
print(f"Total tokens: {response.usage['total_tokens']}")
print(f"Cost: ${response.usage['cost']:.4f}")
# Prompt tokens: 12
# Completion tokens: 47
# Total tokens: 59
# Cost: $0.0024
```

### Efficient Repeated Calls

```python
# Create a reusable binding with preset configuration
analyzer = models.instance("claude-3-opus", temperature=0.2, max_tokens=500)

# Use for multiple analyses without re-validation
code_review = analyzer(f"Review this Python code:\n{code}")
security_check = analyzer(f"Find security issues in:\n{code}")
performance_tips = analyzer(f"Suggest performance improvements for:\n{code}")

# Override parameters when needed
detailed_review = analyzer(
    f"Provide detailed code review:\n{code}", 
    max_tokens=2000  # Override for this call only
)
```

### Error Handling

```python
from ember.api.exceptions import ModelProviderError, ProviderAPIError

try:
    response = models("gpt-4", "Hello world")
except ModelProviderError as e:
    # Configuration issues (missing API key, etc)
    print(f"Setup error: {e}")
    print("Set OPENAI_API_KEY environment variable")
except ProviderAPIError as e:
    # Runtime API errors
    error_type = e.context.get("error_type")
    if error_type == "rate_limit":
        print("Rate limited - retry with exponential backoff")
    elif error_type == "authentication":
        print("Invalid API key")
    else:
        print(f"API error: {e}")
```

### Cost Tracking

```python
# Track individual request costs
response = models(
    "gpt-4", 
    "Write a comprehensive guide to Python decorators",
    max_tokens=2000
)
print(f"This request cost: ${response.usage['cost']:.4f}")

# Aggregate usage tracking
from ember.api.models import ModelRegistry

registry = ModelRegistry()

# Make several calls
for question in questions:
    response = registry.invoke_model("gpt-4", question)
    process_response(response)

# Get usage summary
summary = registry.get_usage_summary("gpt-4")
if summary:
    print(f"\nGPT-4 Usage Summary:")
    print(f"Total requests: {summary.request_count}")
    print(f"Total tokens: {summary.total_usage.total_tokens:,}")
    print(f"Total cost: ${summary.cost_usd:.2f}")
    print(f"Average cost per request: ${summary.cost_usd / summary.request_count:.4f}")
```

### Production Patterns

```python
# Retry logic with tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def reliable_completion(prompt: str) -> str:
    response = models("gpt-4", prompt)
    return response.text

# Async for high throughput
import asyncio
from ember.api.models import ModelRegistry

async def process_batch(prompts: list[str]):
    registry = ModelRegistry()
    tasks = [
        registry.invoke_model_async("gpt-4", prompt) 
        for prompt in prompts
    ]
    responses = await asyncio.gather(*tasks)
    return [r.data for r in responses]

# Run batch processing
results = asyncio.run(process_batch(prompts))
```

### Custom Providers

```python
from ember.api.models.providers import register_provider, BaseProvider
from ember.api.models.schemas import ChatResponse, UsageStats

class AzureOpenAIProvider(BaseProvider):
    """Azure-hosted OpenAI models."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    def complete(self, prompt: str, model: str, **kwargs) -> ChatResponse:
        # Azure-specific API call
        response = azure_openai_call(self.endpoint, prompt, model, **kwargs)
        return ChatResponse(
            data=response["choices"][0]["message"]["content"],
            usage=UsageStats(
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"]
            )
        )
    
    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv("AZURE_OPENAI_API_KEY")

# Register and use
register_provider("azure", AzureOpenAIProvider)
response = models("azure/gpt-4", "Hello from Azure")
```

## Architecture Decisions

### Why Not Just Functions?

We started with pure functions but hit three problems:
1. **Connection pooling** - Each provider needs persistent HTTP connections
2. **Thread safety** - Concurrent calls require synchronization
3. **Cost tracking** - Users need usage analytics

The Registry pattern solved all three with minimal complexity.

### Why Registry Over Factory?

Factories imply creation. Registries imply storage and retrieval. Since we cache model instances, Registry communicates intent correctly. The implementation uses a single `threading.Lock` after benchmarking showed no benefit from fine-grained locking below 10K req/s.

### Why Dataclasses Over Dicts?

Type safety. Modern Python tooling (mypy, IDEs) provides autocomplete and validation with dataclasses. The 0.1ms overhead is negligible compared to network latency.

## Implementation Deep Dive

### The Critical Path

```python
def models(model: str, prompt: str, **kwargs) -> Response:
    # 1. Get or create provider (~10ms first time, <1ms cached)
    provider = registry.get_model(model)
    
    # 2. Make API call (50-5000ms depending on prompt/response size)
    raw_response = provider.complete(prompt, model, **kwargs)
    
    # 3. Wrap response (~0.1ms)
    return Response(raw_response)
```

Every line has a purpose. No abstraction for abstraction's sake.

### Thread Safety Without Locks (Almost)

The registry uses double-checked locking for reads:

```python
if model_id in self._models:  # Fast path: no lock
    return self._models[model_id]
    
with self._lock:  # Slow path: locked creation
    if model_id in self._models:  # Double-check
        return self._models[model_id]
    # Create model...
```

This pattern, borrowed from the Java memory model, eliminates lock contention for the common case (cached models).

## Advanced Usage

### Environment Configuration

```bash
# Standard API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Cost overrides for accurate tracking
export EMBER_COST_GPT4_INPUT="30.0"  # $/1k tokens
export EMBER_COST_GPT4_OUTPUT="60.0"

# Or bulk override via JSON
export EMBER_MODEL_COSTS_JSON='{
  "gpt-4-turbo": {"input": 10.0, "output": 30.0},
  "custom-model": {"input": 5.0, "output": 15.0}
}'
```

### Direct Registry Access

```python
from ember.models import ModelRegistry

# Custom metrics integration
from prometheus_client import Counter, Histogram

metrics = {
    "model_invocations": Counter("llm_calls", "Total LLM calls", ["model"]),
    "invocation_duration": Histogram("llm_duration", "Call duration", ["model"])
}

registry = ModelRegistry(metrics=metrics)

# Now all calls are automatically instrumented
response = registry.invoke_model("gpt-4", "Hello")
```

## Scaling Considerations

### Streaming (Coming Soon)

The design intentionally excludes streaming from v1. Why? Streaming adds complexity (async iterators, partial responses, error handling mid-stream) that 90% of users don't need. When we add it, it will be:

```python
for chunk in models.stream("gpt-4", "Write a long story"):
    print(chunk.text, end="")
```

### Multi-Modal (Future)

The current design extends naturally to images:

```python
response = models("gpt-4-vision", ["What's in this image?", image])
```

No new abstractions needed.

### Production Deployment

At scale, you'll want:

1. **Connection pooling** - Already handled by providers
2. **Retry logic** - Add `tenacity` to your providers
3. **Observability** - Pass custom metrics dict to ModelRegistry
4. **Rate limiting** - Implement in your provider or use a proxy

## FAQ

**Q: Why not use LangChain/LlamaIndex/etc?**
A: They solve different problems. Ember is for engineers who want direct model access without framework lock-in.

**Q: How do I handle errors?**
A: Errors are exceptions. Handle them like any Python code. No special error objects or callback hell.

**Q: What about prompt templates?**
A: That's a layer above model invocation. Use f-strings or Jinja2. Don't couple templating to API calls.

**Q: Is this production-ready?**
A: Yes. The code is simple enough to audit in an afternoon. No complex dependencies. No magic.

## Contributing

Before adding features, ask: "Would this make the common case more complex?" If yes, reconsider.

The goal is not feature parity with other libraries. The goal is to remain simple while being sufficient for 90% of use cases.