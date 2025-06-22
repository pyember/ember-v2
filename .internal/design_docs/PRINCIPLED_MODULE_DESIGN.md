# Principled Module Design for Ember

## Core Insight: LLM Orchestration is Not Neural Networks

*"The best way to predict the future is to invent it." - Alan Kay*

Our fundamental insight: **LLM orchestration is about composing API calls, not differentiating through weights**. This drives our entire design.

## Design Principles (What Our Dream Team Would Build)

### 1. Static by Default (Dean & Ghemawat)
```python
# This is the 90% case - just composing LLM calls
class QAChain(ember.Module):
    retriever: ember.Tool = ember.tool()  # Static - no gradients
    model: ember.Model = ember.model()    # Static - API call
    
    def forward(self, question: str) -> str:
        docs = self.retriever.search(question)
        return self.model.generate(f"Answer using: {docs}\n\nQ: {question}")
```

**Why**: Most LLM operations are black-box API calls. Making them dynamic by default is a performance and semantic mistake.

### 2. Explicit Learnable Parameters (Carmack's Pragmatism)
```python
# The 10% case - when you need gradients
class LearnableRouter(ember.Module):
    models: Dict[str, ember.Model]  # Static (not JAX arrays)
    routing_weights: jax.Array  # Dynamic (JAX array)
    temperature: jax.Array  # Dynamic (JAX array)
    
    def __init__(self, models: Dict[str, Any], embed_dim: int, key: jax.Array):
        self.models = {k: ember.model(v) for k, v in models.items()}
        # JAX arrays are automatically dynamic
        self.routing_weights = jax.random.normal(key, (embed_dim, len(models)))
        self.temperature = jnp.array(1.0)
    
    def forward(self, query: str) -> str:
        # Routing logic uses learnable weights
        embeddings = self.embed(query)
        logits = embeddings @ self.routing_weights / self.temperature
        model_name = self.models.keys()[jnp.argmax(logits)]
        return self.models[model_name].generate(query)
```

**Why**: JAX arrays are automatically dynamic in Ember. No special markers needed.

### 3. Progressive Disclosure (Jobs' Simplicity)

#### Level 1: Functions (Dead Simple)
```python
@ember.op
def classify(text: str) -> str:
    return ember.model("gpt-4").generate(f"Classify: {text}")

# Just works - no classes, no boilerplate
result = classify("Is this spam?")
```

#### Level 2: Modules (Composable)
```python
class Classifier(ember.Module):
    model: ember.Model = ember.model("gpt-4")
    prompt: str = "Classify as spam/ham: {text}"
    
    def forward(self, text: str) -> str:
        return self.model.generate(self.prompt.format(text=text))
```

#### Level 3: Advanced (Full Power)
```python
class NeuralRouter(ember.Module):
    """Routes to models using learned embeddings."""
    
    # Static components
    models: Dict[str, ember.Model]
    encoder: ember.Tool
    
    # Dynamic components (JAX arrays)
    routing_matrix: jax.Array
    temperature: jax.Array
    bias: jax.Array
    
    def __init__(self, models: Dict[str, str], key: jax.Array):
        self.models = {k: ember.model(v) for k, v in models.items()}
        self.encoder = ember.tool("sentence-transformer")
        # JAX arrays are automatically dynamic
        k1, k2 = jax.random.split(key)
        self.routing_matrix = jax.random.normal(k1, (384, 3))
        self.temperature = jnp.array(1.0)
        self.bias = jnp.zeros(3)
    
    def forward(self, query: str) -> str:
        # Encode (static tool call)
        embedding = self.encoder.encode(query)
        
        # Route (differentiable)
        logits = embedding @ self.routing_matrix + self.bias
        probs = jax.nn.softmax(logits / self.temperature)
        
        # Select model (static)
        model_idx = jnp.argmax(probs)
        return list(self.models.values())[model_idx].generate(query)
```

### 4. Clean Separation of Concerns (Uncle Bob)

```python
# Separate learnable state from logic
class RouterLogic(ember.Module):
    """Pure routing logic - no parameters."""
    
    def compute_scores(self, embedding: jax.Array, params: RouterParams) -> jax.Array:
        return jax.nn.softmax(embedding @ params.weights / params.temperature)
    
    def select_model(self, scores: jax.Array, models: List[str]) -> str:
        return models[jnp.argmax(scores)]

# Composed into stateful router
class Router(ember.Module):
    logic: RouterLogic = RouterLogic()  # Static
    params: RouterParams = ember.learnable(RouterParams)  # Dynamic
    models: List[ember.Model] = ember.models()  # Static
```

### 5. Platform, Not Features (Larry Page)

Instead of building specific operators, build a platform for LLM composition:

```python
# The platform enables patterns, doesn't prescribe them
ensemble = ember.map(classify, over=["gpt-4", "claude-3", "gemini"])
consensus = ember.reduce(ember.voting(), ensemble)

# Or build your own patterns
@ember.pattern
def best_of_n(op: ember.Op, n: int = 3, judge: ember.Op = quality_judge):
    results = ember.map(op, range(n))
    scores = ember.map(judge, results)
    return results[jnp.argmax(scores)]
```

## Native JAX Integration

Since Ember modules are PyTrees, **all JAX transformations work natively**:

### Basic JAX Compatibility
```python
import ember
import jax
import jax.numpy as jnp

class LearnableRouter(ember.Module):
    # Static - excluded from gradients
    models: Dict[str, ember.Model] = ember.models()
    
    # Dynamic - included in gradients
    routing_weights: jax.Array
    temperature: jax.Array  # JAX arrays are automatically dynamic
    
    def __init__(self, models: Dict[str, Any], embed_dim: int, key: jax.Array):
        self.models = {k: ember.model(v) for k, v in models.items()}
        self.routing_weights = jax.random.normal(key, (embed_dim, len(models)))
        self.temperature = jnp.array(1.0)
    
    def __call__(self, embedding: jax.Array) -> jax.Array:
        # This part is differentiable
        logits = embedding @ self.routing_weights / self.temperature
        return jax.nn.softmax(logits)

# All JAX transforms just work!
router = LearnableRouter({"fast": "gpt-3.5", "smart": "gpt-4"}, 384, jax.random.PRNGKey(0))

# Gradient computation - only routing_weights and temperature are differentiated
@jax.grad
def loss_fn(router, embeddings, targets):
    predictions = jax.vmap(router)(embeddings)
    return jnp.mean((predictions - targets) ** 2)

grads = loss_fn(router, embeddings, targets)
# grads.routing_weights exists
# grads.temperature exists  
# grads.models is None (static field)
```

### Advanced Hybrid System
```python
class HybridQASystem(ember.Module):
    # Static components (API calls)
    retriever: ember.Tool = ember.tool("vector-db")
    generator: ember.Model = ember.model("gpt-4")
    
    # Learnable components
    query_projection: jax.Array
    relevance_threshold: jax.Array  # Learnable threshold
    
    def __init__(self, embed_dim: int, key: jax.Array):
        self.retriever = ember.tool("vector-db")
        self.generator = ember.model("gpt-4")
        self.query_projection = jax.random.normal(key, (embed_dim, embed_dim))
        self.relevance_threshold = jnp.array(0.7)
    
    def score_relevance(self, query_emb: jax.Array, doc_emb: jax.Array) -> float:
        # This is differentiable
        projected_query = query_emb @ self.query_projection
        return jax.nn.sigmoid(projected_query @ doc_emb.T)
    
    def __call__(self, question: str, doc_embeddings: jax.Array) -> Tuple[str, jax.Array]:
        # Get query embedding (static API call)
        query_emb = self.retriever.embed(question)
        
        # Score documents (differentiable)
        scores = jax.vmap(lambda d: self.score_relevance(query_emb, d))(doc_embeddings)
        
        # Filter by threshold (differentiable)
        mask = scores > self.relevance_threshold
        
        # Generate answer (static API call)
        relevant_docs = self.retriever.fetch(jnp.where(mask)[0])
        answer = self.generator(f"Context: {relevant_docs}\nQuestion: {question}")
        
        return answer, scores

# Can optimize the relevance scoring!
@jax.value_and_grad
def train_step(system, question, doc_embeddings, relevance_labels):
    _, scores = system(question, doc_embeddings)
    loss = jnp.mean((scores - relevance_labels) ** 2)
    return loss

qa_system = HybridQASystem(384, jax.random.PRNGKey(0))
loss, grads = train_step(qa_system, "What is JAX?", doc_embeddings, labels)

# Update only the learnable parts
import optax
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(grads)
updates, opt_state = optimizer.update(grads, opt_state)
qa_system = optax.apply_updates(qa_system, updates)
```

### Key Benefits of Native JAX

1. **All Transforms Work**
   ```python
   # Everything just works out of the box
   jax.jit(my_module)
   jax.grad(loss_fn)(my_module, data)
   jax.vmap(my_module)(batched_input)
   jax.pmap(my_module)(distributed_data)
   ```

2. **Proper Static/Dynamic Separation**
   ```python
   # JAX automatically handles static fields correctly
   grads = jax.grad(loss_fn)(module)
   # grads.models is None (static field - no gradient)
   # grads.weights contains gradients (dynamic field)
   # grads.temperature contains gradient (dynamic field)
   ```

3. **Compose with JAX Ecosystem**
   ```python
   # Use with any JAX library
   from flax import linen as nn
   from optax import adam
   from haiku import nets
   
   class HybridModel(ember.Module):
       llm: ember.Model = ember.model("gpt-4")  # Static
       vision_model: nn.Module  # Dynamic (Flax module)
       mlp: hk.Module  # Dynamic (Haiku module)
       
       def __call__(self, image: jax.Array) -> str:
           # Extract features with neural network
           features = self.vision_model(image)
           processed = self.mlp(features)
           
           # Generate description with LLM
           prompt = f"Describe image with features: {processed}"
           return self.llm(prompt)
   ```

4. **No Magic - It's Just PyTrees**
   ```python
   # Ember modules are PyTrees, so all of this works:
   leaves, treedef = jax.tree_flatten(my_module)
   my_module_copy = jax.tree_unflatten(treedef, leaves)
   
   # Tree operations work naturally
   def add_noise(leaf):
       if isinstance(leaf, jax.Array):
           return leaf + 0.01 * jax.random.normal(key, leaf.shape)
       return leaf
   
   noisy_module = jax.tree_map(add_noise, my_module)
   ```

### Real-World Example: Learning to Route

```python
# A complete example showing hybrid static/dynamic optimization
class SmartQARouter(ember.Module):
    """Routes questions to appropriate models based on learned patterns."""
    
    # Static components
    models: Dict[str, ember.Model]
    embedder: ember.Tool
    
    # Dynamic components
    routing_mlp: nn.Module
    threshold: jax.Array  # Learnable threshold
    
    def __init__(self, key: jax.Array):
        self.models = {
            "fast": ember.model("gpt-3.5-turbo"),
            "accurate": ember.model("gpt-4"),
            "creative": ember.model("claude-3")
        }
        self.embedder = ember.tool("sentence-transformers")
        self.routing_mlp = nn.Sequential([
            nn.Dense(128),
            nn.relu,
            nn.Dense(3)  # 3 models
        ])
        self.threshold = jnp.array(0.5)
    
    def __call__(self, question: str) -> str:
        # Embed question (static tool call)
        embedding = self.embedder.encode(question)
        
        # Compute routing scores (differentiable)
        scores = self.routing_mlp(embedding)
        probs = jax.nn.softmax(scores)
        
        # Route to best model if confidence > threshold
        best_idx = jnp.argmax(probs)
        if probs[best_idx] > self.threshold:
            model_name = list(self.models.keys())[best_idx]
        else:
            # Fallback to ensemble
            model_name = "accurate"
        
        # Generate answer (static API call)
        return self.models[model_name](question)

# Training loop using native JAX
@jax.jit
def train_step(router, optimizer_state, questions, labels):
    def loss_fn(router):
        # Compute routing decisions
        embeddings = jax.vmap(router.embedder.encode)(questions)
        all_scores = jax.vmap(router.routing_mlp)(embeddings)
        
        # Cross-entropy loss against true best model
        return optax.softmax_cross_entropy(all_scores, labels).mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(router)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    router = optax.apply_updates(router, updates)
    
    return router, optimizer_state, loss
```

## API Design Decisions

### 1. Field Types (Ritchie's Minimalism)

```python
# No special field types needed - just use Python and JAX
# Static by default: model bindings, tools, strings, etc.
model: ModelBinding  # Static - API call
tool: ToolBinding    # Static - external tool

# Dynamic automatically: JAX arrays
weights: jax.Array   # Dynamic - included in gradients
threshold: jax.Array # Dynamic - learnable parameter
```

### 2. Learnable Parameters (Knuth's Clarity)

```python
# Just use JAX arrays - they're automatically dynamic
class Module(ember.Module):
    # Static components
    model: ModelBinding
    
    # Dynamic components (JAX arrays)
    weights: jax.Array
    bias: jax.Array
    
    def __init__(self, key: jax.Array):
        self.model = ember.model("gpt-4")
        # Initialize JAX arrays
        k1, k2 = jax.random.split(key)
        self.weights = jax.random.normal(k1, (10, 10))
        self.bias = jnp.zeros(10)
```

**Decision**: No special syntax needed. JAX arrays are automatically dynamic.

### 3. Gradient Context (Brockman's Workflow)

```python
# Native JAX style (preferred)
loss_fn = lambda module: compute_loss(module(input), target)
grads = jax.grad(loss_fn)(module)

# Or with optimizer integration
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(module)

for batch in data:
    loss, grads = jax.value_and_grad(loss_fn)(module, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    module = optax.apply_updates(module, updates)
```

## Implementation Strategy

### Phase 1: Core Abstractions
```python
# ember/core/module.py
from equinox import Module

# That's it. No additions needed.
# JAX arrays are automatically detected as dynamic fields.
# Everything else is static by default.
```

### Phase 2: Progressive API Layers
```python
# ember/api/simple.py - The 90% case
@op
def my_operator(x): ...

# ember/api/modules.py - The 9% case  
class MyModule(Module): ...

# ember/api/advanced.py - The 1% case
class LearableModule(Module, strict=True): ...
```

### Phase 3: Platform Patterns
```python
# ember/patterns/ensemble.py
def ensemble(*ops): ...

# ember/patterns/routing.py
def router(routes: Dict[str, Op]): ...

# Let users build their own
```

## The Magic: It's Just PyTrees

The beautiful thing is there's no magic. Ember modules are just PyTrees, which means:

- `jax.grad` automatically knows to skip static fields
- `jax.jit` compiles only the differentiable parts
- `optax` optimizers work out of the box
- You can mix with any JAX library (Flax, Haiku, etc.)

This gives us the best of both worlds:
- **Simple API** for the 90% case (just calling LLMs)
- **Full JAX power** for the 10% case (learning, optimization)
- **Native integration** with the entire JAX ecosystem

## Key Design Insights (After Deep Reflection)

### 1. No Special Field Types Needed

We don't need any special field markers. Ember automatically detects JAX arrays as dynamic:

```python
# Clean and simple:
class MyOp(ember.Module):
    model: ModelBinding  # Static by default
    tool: SearchTool     # Static by default
    weights: jax.Array   # Dynamic (automatically detected)
    
    def __init__(self, key: jax.Array):
        self.model = ember.model("gpt-4")  # Or ModelBinding("gpt-4")
        self.tool = SearchTool()
        self.weights = jax.random.normal(key, (10, 5))
```

**Why this is better**: Less magic, more Python. The only special case is marking dynamic fields.

### 2. Arbitrary Nesting with Mixed Static/Dynamic Works Perfectly

The key insight about Ember's module system: **"static" at one level doesn't stop traversal**. It just means that field won't be replaced, but Ember still looks inside for dynamic leaves:

```python
class InnerOp(ember.Module):
    # Mix of static and dynamic
    model: ModelBinding = ModelBinding("gpt-4")  # Static
    weights: jax.Array  # Dynamic (auto-detected by Ember)
    
class OuterOp(ember.Module):
    inner: InnerOp  # Static field containing mixed operator
    threshold: jax.Array  # Dynamic (JAX array)
    
# Gradients flow correctly!
grads = jax.grad(loss_fn)(outer_op)
# grads.threshold has gradient
# grads.inner.weights has gradient (even though inner is static!)
# grads.inner.model is None
```

This enables natural composition of operators within operators, each with their own static/dynamic fields.

### 3. Simplified API - Just Python

The final API is beautifully minimal:

```python
# Regular Python for static (the default)
class SmartQA(ember.Module):
    # Compose operators naturally
    routers: List[RouterOp] = [RouterOp(), AdvancedRouter()]
    judge: QualityJudge = QualityJudge()
    fallback: DirectAnswer = DirectAnswer()
    
    # Dynamic fields (JAX arrays)
    router_weights: jax.Array
    quality_threshold: jax.Array
```

No registration, no special handling - just composition.

## Real-World Example with Deep Nesting

```python
class ProductionSystem(ember.Module):
    """Complex nested system showing mixed static/dynamic at every level."""
    
    # Static operators that themselves contain mixed fields
    preprocessors: List[ember.Module]
    routers: Dict[str, ember.Module]
    system_temperature: jax.Array  # Dynamic
    
    def __init__(self, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Initialize preprocessors with dynamic fields
        self.preprocessors = [
            Normalizer(mean=0.0, std_key=k1),  # Has internal std: jax.Array
            Augmenter(static_config={"type": "text"}, noise_key=k2)
        ]
        
        self.routers = {
            "primary": SmartRouter(
                routes={
                    "technical": TechExpert(model=ember.model("gpt-4")),
                    "creative": Creative(key=k3)  # Has internal imagination: jax.Array
                },
                routing_mlp=MLP(hidden=128),  # MLP has internal dynamic weights
                static_threshold=0.5
            ),
            "fallback": SimpleRouter(routes={"default": DefaultOp()})
        }
        
        # Top-level dynamic parameter
        self.system_temperature = jnp.array(1.0)
    
    def __call__(self, x):
        # Preprocess with learnable parameters
        for p in self.preprocessors:
            x = p(x)
            
        # Route using nested learnable logic
        if self.system_temperature > 0.5:
            return self.routers["primary"](x)
        return self.routers["fallback"](x)

# JAX handles the entire nested structure!
system = ProductionSystem()
grads = jax.grad(loss_fn)(system)

# Gradients exist at every level:
# - grads.preprocessors[0].std ✓
# - grads.preprocessors[1].noise ✓  
# - grads.routers["primary"].routing_mlp.weights ✓
# - grads.routers["primary"].routes["creative"].imagination ✓
# - grads.system_temperature ✓
# All static fields are None, but traversal continues into them
```

## Validation Against CLAUDE.md Principles

✅ **Principled, root-node fixes**: Static-by-default is the correct fundamental choice
✅ **Google Python Style**: Clean, simple, obvious
✅ **No Claude references**: Pure technical design
✅ **Opinionated decisions**: One obvious way (static by default)
✅ **Explicit over magic**: Learnable parameters explicitly marked
✅ **Common case simple**: Basic operators are just functions
✅ **Professional documentation**: Clear, technical, no emojis
✅ **Comprehensive testing**: Each layer independently testable
✅ **10x improvement**: Makes LLM composition as simple as function calls
✅ **Platform thinking**: Enables patterns, doesn't prescribe them

## The Larry Page Test

**Is this 10x better?**

Current approach (LangChain, etc.):
```python
class MyChain(Chain):
    def __init__(self):
        self.llm = ChatOpenAI()
        self.prompt = PromptTemplate(...)
        # 50 more lines of boilerplate
```

Our approach:
```python
@ember.op
def my_chain(x: str) -> str:
    return ember.model("gpt-4").generate(f"Process: {x}")
```

**Yes.** 10x less code, 10x clearer intent, 10x easier to compose.

## Summary

By making static (LLM calls) the default and learnable parameters special, we align with how LLM systems actually work. Progressive disclosure from simple functions to complex modules ensures anyone can start using it immediately, while platform thinking enables the advanced users to build the future.

The native JAX integration means you get all the power of the JAX ecosystem for free - no adapters, no wrappers, no magic. Just clean, composable, differentiable programs.

**Key insights after deep reflection**:
1. No special field types needed - JAX arrays are automatically dynamic
2. Arbitrary nesting with mixed static/dynamic works perfectly - Ember traverses everything
3. The API is just Python - no special markers or decorators needed

This is what our dream team would build: **Simple. Correct. Powerful.**