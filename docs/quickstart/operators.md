# Ember Operators

Operators are the building blocks of Ember applications. This guide shows how to create and compose operators using Ember's simplified API.

## Core Concepts

Ember embraces progressive disclosure - start simple and add complexity only when needed:

1. **Simple Functions** (90% of use cases) - Just decorate a function with `@ember.op`
2. **Validated Functions** (9% of use cases) - Add type validation with `@validate`
3. **Full Operators** (1% of use cases) - Use the `Operator` class for complex needs

## Simple Function Operators

Most operators can be simple functions:

```python
from ember.api import ember
from typing import List
from pydantic import BaseModel

# Simple text classification
@ember.op
async def classify_text(text: str, categories: List[str]) -> str:
    """Classify text into one of the given categories."""
    prompt = f"""Classify this text into one of these categories: {', '.join(categories)}
    
    Text: {text}
    
    Return only the category name."""
    
    return await ember.llm(prompt)

# With structured output
class Classification(BaseModel):
    category: str
    confidence: float
    reasoning: str

@ember.op
async def classify_with_confidence(text: str, categories: List[str]) -> Classification:
    """Classify text with confidence score and reasoning."""
    prompt = f"""Classify this text into one of these categories: {', '.join(categories)}
    
    Text: {text}"""
    
    return await ember.llm(prompt, output_type=Classification)
```

## Using Operators

```python
# Simple usage
category = await classify_text(
    "The new iPhone has amazing battery life",
    ["technology", "politics", "sports", "entertainment"]
)
print(category)  # "technology"

# Structured output
result = await classify_with_confidence(
    "The new iPhone has amazing battery life",
    ["technology", "politics", "sports", "entertainment"]
)
print(f"{result.category} (confidence: {result.confidence})")
```

## Validated Operators

Add validation when you need strict input/output guarantees:

```python
from ember.api import ember, validate
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    max_results: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = False

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)

@ember.op
@validate
async def search_web(query: SearchQuery) -> List[SearchResult]:
    """Search the web with validated inputs."""
    # Validation happens automatically
    # Invalid inputs will raise clear error messages
    results = await ember.web_search(query.query, limit=query.max_results)
    return [SearchResult(**r) for r in results]
```

## Composition Patterns

### Sequential Processing

```python
@ember.op
async def analyze_article(url: str) -> dict:
    """Complete article analysis pipeline."""
    # Fetch article
    content = await fetch_article(url)
    
    # Process in sequence
    summary = await summarize(content)
    sentiment = await analyze_sentiment(content)
    keywords = await extract_keywords(content)
    
    return {
        "summary": summary,
        "sentiment": sentiment,
        "keywords": keywords
    }
```

### Parallel Processing

```python
from ember.api import ember

@ember.op
async def analyze_multiple_articles(urls: List[str]) -> List[dict]:
    """Analyze multiple articles in parallel."""
    return await ember.parallel([
        analyze_article(url) for url in urls
    ])

# Or with more control
@ember.op
async def analyze_with_progress(urls: List[str]) -> List[dict]:
    """Analyze articles with progress tracking."""
    results = []
    async for result in ember.stream(urls, analyze_article):
        print(f"Completed: {len(results) + 1}/{len(urls)}")
        results.append(result)
    return results
```

### Conditional Logic

```python
@ember.op
async def smart_responder(message: str, context: dict) -> str:
    """Respond intelligently based on message type."""
    # Classify the message
    msg_type = await classify_message(message)
    
    # Route to appropriate handler
    if msg_type == "question":
        return await answer_question(message, context)
    elif msg_type == "request":
        return await handle_request(message, context)
    else:
        return await general_response(message)
```

## Advanced Patterns

### Ensemble Operators

```python
from ember.api import ember, ensemble

@ember.op
async def robust_classifier(text: str) -> str:
    """Classify using multiple models for robustness."""
    # Run multiple models in parallel
    results = await ensemble(
        text,
        models=["gpt-4", "claude-3", "gemini-pro"],
        task="classify"
    )
    
    # Return majority vote
    return max(set(results), key=results.count)
```

### Retry and Error Handling

```python
from ember.api import ember, retry

@ember.op
@retry(max_attempts=3, backoff=True)
async def reliable_api_call(query: str) -> dict:
    """Make API call with automatic retry."""
    return await external_api.search(query)

# Custom error handling
@ember.op
async def safe_processor(data: dict) -> dict:
    """Process data with graceful error handling."""
    try:
        result = await complex_processing(data)
        return {"success": True, "result": result}
    except Exception as e:
        # Log error and return safe default
        await ember.log_error(e)
        return {"success": False, "error": str(e)}
```

### Stateful Operators (Advanced)

For the rare cases where you need state:

```python
from ember.api import Operator

class ConversationOperator(Operator):
    """Maintains conversation history."""
    
    def __init__(self, model_name: str = "gpt-4"):
        super().__init__()
        self.model_name = model_name
        self.history = []
    
    async def forward(self, message: str) -> str:
        # Add to history
        self.history.append({"role": "user", "content": message})
        
        # Generate response with history
        response = await ember.llm(
            self.history,
            model=self.model_name
        )
        
        # Update history
        self.history.append({"role": "assistant", "content": response})
        
        return response
```

## Performance Optimization

Ember automatically optimizes your operators:

```python
from ember.api import ember, jit

# Enable JIT compilation for better performance
@ember.op
@jit
async def fast_processor(items: List[str]) -> List[str]:
    """Process items with JIT optimization."""
    return await ember.parallel([
        process_item(item) for item in items
    ])

# Batch processing
@ember.op
async def batch_analyzer(documents: List[str], batch_size: int = 10) -> List[dict]:
    """Process documents in optimized batches."""
    results = []
    for batch in ember.batch(documents, size=batch_size):
        batch_results = await ember.parallel([
            analyze_document(doc) for doc in batch
        ])
        results.extend(batch_results)
    return results
```

## Testing Operators

```python
import pytest
from ember.api import ember

@pytest.mark.asyncio
async def test_classifier():
    """Test the classifier operator."""
    result = await classify_text(
        "Python is a programming language",
        ["technology", "cooking", "sports"]
    )
    assert result == "technology"

# Test with mocked LLM
@pytest.mark.asyncio
async def test_with_mock():
    with ember.mock_llm(returns="mocked response"):
        result = await my_operator("test input")
        assert result == "mocked response"
```

## Progressive Disclosure: From Simple to Advanced

Ember supports a full spectrum of operator complexity, revealing features only when needed:

### Level 1: Simple Functions (90% of use cases)
```python
@ember.op
async def classify(text: str) -> str:
    return await ember.llm(f"Classify: {text}")
```

### Level 2: Basic Operators (8% of use cases)
```python
from ember.api import Operator

class SentimentAnalyzer(Operator):
    def __init__(self, model_name: str = "gpt-4"):
        super().__init__()
        self.model = ember.models.instance(model_name)
    
    async def forward(self, text: str) -> dict:
        sentiment = await self.model(f"Analyze sentiment: {text}")
        return {"text": text, "sentiment": sentiment}
```

### Level 3: Validated Operators (1.5% of use cases)
```python
from ember.api import Operator, Specification
from pydantic import BaseModel

class AnalysisInput(BaseModel):
    text: str
    language: str = "en"

class AnalysisOutput(BaseModel):
    sentiment: str
    confidence: float
    keywords: List[str]

class ValidatedAnalyzer(Operator):
    def __init__(self):
        super().__init__(
            spec=Specification(
                input_schema=AnalysisInput,
                output_schema=AnalysisOutput
            )
        )
    
    async def forward(self, input: AnalysisInput) -> AnalysisOutput:
        # Input/output automatically validated
        result = await ember.llm(
            f"Analyze {input.text}",
            output_type=AnalysisOutput
        )
        return result
```

### Level 4: Learnable Parameters (0.4% of use cases)

Ember automatically detects JAX arrays as learnable parameters:

```python
import jax
import jax.numpy as jnp
from ember.api import Operator

class LearnableRouter(Operator):
    """Routes requests to different models based on learned embeddings."""
    
    def __init__(self, routes: dict[str, str], embedding_dim: int = 64):
        super().__init__()
        
        # Learnable parameters (JAX arrays) - automatically detected
        key = jax.random.PRNGKey(0)
        self.routing_weights = jax.random.normal(key, (embedding_dim, len(routes)))
        self.temperature = jnp.array(1.0)
        
        # Static parameters - automatically detected as non-learnable
        self.routes = routes  # {"fast": "gpt-3.5", "accurate": "gpt-4"}
        self.models = {
            name: ember.models.instance(model) 
            for name, model in routes.items()
        }
    
    def compute_route_scores(self, text: str) -> jnp.ndarray:
        # Get text embedding (simplified for example)
        embedding = self.get_embedding(text)
        
        # Compute routing scores (differentiable)
        scores = jnp.dot(embedding, self.routing_weights)
        return jax.nn.softmax(scores / self.temperature)
    
    async def forward(self, text: str) -> dict:
        # Compute routing probabilities
        scores = self.compute_route_scores(text)
        
        # Select route (non-differentiable decision)
        route_idx = jnp.argmax(scores)
        route_name = list(self.routes.keys())[route_idx]
        
        # Call selected model
        response = await self.models[route_name](text)
        
        return {
            "route": route_name,
            "confidence": float(scores[route_idx]),
            "response": response
        }
```

### Level 5: Complex Systems (0.1% of use cases)

Build sophisticated ML systems with Ember's full power:

```python
class AdaptiveEnsemble(Operator):
    """Ensemble that learns to weight different models based on context."""
    
    def __init__(self, models: List[str]):
        super().__init__()
        
        # Learnable ensemble weights
        self.ensemble_weights = jnp.ones(len(models)) / len(models)
        self.context_encoder = self.build_encoder()
        
        # Static model instances
        self.models = [ember.models.instance(m) for m in models]
        self.aggregator = AggregationStrategy()
    
    def build_encoder(self):
        # Build a learnable context encoder
        return jax.nn.Sequential([
            jax.nn.Dense(128),
            jax.nn.relu,
            jax.nn.Dense(64)
        ])
    
    async def forward(self, query: str, context: dict) -> dict:
        # Encode context to adjust weights
        context_features = self.context_encoder(context)
        adjusted_weights = self.ensemble_weights * jax.nn.sigmoid(context_features)
        
        # Run all models in parallel
        responses = await ember.parallel([
            model(query) for model in self.models
        ])
        
        # Weighted aggregation
        final_response = self.aggregator(responses, adjusted_weights)
        
        return {
            "response": final_response,
            "model_contributions": dict(zip(self.models, adjusted_weights))
        }
```

## Key Advanced Features

### Automatic Parameter Detection

Ember automatically distinguishes between:
- **Learnable (dynamic)**: JAX arrays (`jnp.ndarray`)
- **Static**: Everything else (models, strings, configs)

No decorators or annotations needed!

### Gradient Flow

```python
# Compute gradients only through JAX operations
loss_fn = lambda params: operator.compute_loss(data)
grads = jax.grad(loss_fn)(operator.get_learnable_params())

# Update learnable parameters
operator.update_params(grads)
```

### Built-in Advanced Operators

```python
from ember.api.operators import Ensemble, Chain, Router, Cache, Retry

# Ensemble with voting
ensemble = Ensemble(
    operators=[op1, op2, op3],
    aggregation="majority_vote"
)

# Sequential pipeline
pipeline = Chain([
    preprocessor,
    analyzer,
    postprocessor
])

# Conditional routing
router = Router({
    "simple": simple_op,
    "complex": complex_op
}, router_fn=lambda x: "simple" if len(x) < 100 else "complex")

# Caching expensive operations
cached_op = Cache(expensive_op, maxsize=1000)

# Automatic retry
reliable_op = Retry(flaky_op, max_attempts=3)
```

### XCS Integration

The JIT compiler intelligently handles mixed static/dynamic operations:

```python
@ember.op
@jit
class OptimizedRouter(LearnableRouter):
    # XCS will:
    # - Keep string operations static
    # - Make JAX operations dynamic
    # - Cache based on static inputs
    # - Optimize execution graph
    pass
```

## Best Practices

1. **Start Simple**: Use function operators for most tasks
2. **Type Everything**: Use type hints and Pydantic models
3. **Small and Focused**: Create operators that do one thing well
4. **Compose Over Inherit**: Build complex behavior through composition
5. **Handle Errors Gracefully**: Use try/except or @retry decorator
6. **Test Thoroughly**: Write tests for your operators
7. **Document Clearly**: Use docstrings to explain what operators do
8. **Let Ember Detect Parameters**: Don't manually mark learnable vs static

## Next Steps

- [Models Guide](./models.md) - Working with different LLM providers
- [Data Processing](./data.md) - Loading and processing data
- [XCS Advanced](../xcs/README.md) - Advanced execution strategies
- [Performance Guide](./performance.md) - Optimization techniques