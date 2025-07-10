"""Common operators built on the simplified base class.

This module provides commonly used operators that demonstrate the power of
the simplified design. These operators showcase composition, ensemble patterns,
and integration with language models.

Following Google Python Style Guide:
    https://google.github.io/styleguide/pyguide.html
"""

from typing import List, Dict, Any, Callable, Optional, Type
import jax
import jax.numpy as jnp

from ember.operators.base import Operator
from ember.api.models import models, ModelBinding, Response
from ember.operators.ember_data import EmberData, Context


class ModelCall(Operator):
    """Operator that calls a language model with EmberData and returns EmberData.
    
    This operator wraps a model binding and provides a consistent interface
    for calling language models with EmberData. It automatically accumulates
    usage metrics and returns EmberData with the response text and updated context.
    
    Attributes:
        model: The bound model instance to call.
    
    Examples:
        Basic model calling with EmberData:
        
        >>> # Create EmberData (usually done by EmberEmbedding)
        >>> ember_data = EmberData("What is the capital of France?")
        >>> model_op = ModelCall("gpt-4")
        >>> result = model_op(ember_data)
        >>> print(result.data)  # "Paris is the capital of France."
        >>> print(result.total_usage.total_tokens)  # 25
        
        >>> # Usage is automatically accumulated
        >>> result2 = ModelCall("claude-3-sonnet")(result)
        >>> print(result2.total_usage.total_cost)  # Combined cost
        
        >>> # Chain naturally with other operators
        >>> pipeline = Chain([
        ...     ModelCall("gpt-4"),
        ...     ModelCall("claude-3-sonnet")  # Context flows through
        ... ])
        >>> final_result = pipeline(ember_data)
    """
    
    model: ModelBinding

    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        """Initialize model call operator.
        
        Args:
            model_name: Name of the model to use (e.g., "gpt-4", "claude-3").
            **kwargs: Additional arguments passed to model initialization.
        """
        self.model = models.instance(model_name, **kwargs)
    
    def forward(self, input: EmberData) -> EmberData:
        """Call the model with EmberData and return EmberData with updated context.
        
        Args:
            input: EmberData containing the input to send to the model.
            
        Returns:
            EmberData with the model response and accumulated usage metrics.
        """
        # Extract actual data from EmberData
        result = self.model(input.data)
        
        # Accumulate usage metrics if available
        if hasattr(result, 'usage') and result.usage:
            updated_ember_data = input.accumulate_usage(result.usage)
        else:
            updated_ember_data = input
        
        # Return new EmberData with the result text and updated context
        return updated_ember_data.with_data(result.text)

class Ensemble(Operator):
    """Ensemble operator that combines multiple operators.
    
    This operator runs multiple sub-operators in parallel and aggregates their
    results. It's useful for improving reliability through redundancy or
    gathering diverse perspectives.
    
    Attributes:
        operators: List of operators to run in ensemble.
        aggregator: Optional function to combine results. If None, returns list.
    
    Examples:
        Simple ensemble returning all results:
        
        >>> ensemble = Ensemble([
        ...     Classifier("gpt-4"),
        ...     Classifier("claude-3"),
        ...     Classifier("gemini")
        ... ])
        >>> results = ensemble("Is this text positive?")
        >>> # results = ["positive", "positive", "neutral"]
        
        Ensemble with custom aggregation:
        
        >>> def majority_vote(results: List[str]) -> str:
        ...     from collections import Counter
        ...     return Counter(results).most_common(1)[0][0]
        >>> 
        >>> ensemble = Ensemble(
        ...     operators=[Op1(), Op2(), Op3()],
        ...     aggregator=majority_vote
        ... )
        >>> result = ensemble(input_data)  # Returns single majority result
    """
    
    operators: List[Operator]
    aggregator: Optional[Callable[[List[Any]], Any]]
    
    def __init__(self, 
                 operators: List[Operator],
                 aggregator: Optional[Callable[[List[Any]], Any]] = None):
        """Initialize ensemble with operators and optional aggregator.
        
        Args:
            operators: List of operators to run in ensemble.
            aggregator: Optional function to aggregate results. Should accept
                a list of results and return aggregated result.
        """
        self.operators = operators
        self.aggregator = aggregator
    
    def forward(self, input: Any) -> Any:
        """Run all operators and aggregate results.
        
        Args:
            input: Input to pass to all operators.
            
        Returns:
            List of results if no aggregator, otherwise aggregated result.
        """
        results = [op(input) for op in self.operators]
        
        if self.aggregator:
            return self.aggregator(results)
        return results


class Chain(Operator):
    """Chain operator that runs operators sequentially.
    
    This operator passes the output of each operator as input to the next,
    creating a pipeline of transformations. Useful for multi-step processing.
    
    Examples:
        >>> chain = Chain([
        ...     Preprocessor(),      # Clean and normalize text
        ...     Classifier(),        # Classify cleaned text
        ...     Postprocessor()      # Format classification result
        ... ])
        >>> result = chain(raw_input)
    """
    
    operators: List[Operator]
    
    def __init__(self, operators: List[Operator]):
        """Initialize chain with list of operators.
        
        Args:
            operators: List of operators to run in sequence.
        """
        self.operators = operators
    
    def forward(self, input: Any) -> Any:
        """Pass input through all operators sequentially.
        
        Args:
            input: Initial input to the chain.
            
        Returns:
            Output from the final operator.
        """
        result = input
        for op in self.operators:
            result = op(result)
        return result


class Router(Operator):
    """Router that conditionally directs inputs to different operators.
    
    This operator uses a routing function to decide which operator should
    handle each input. Useful for specialization and conditional logic.
    
    Examples:
        Text type router:
        
        >>> def route_by_length(text: str) -> str:
        ...     return "long" if len(text) > 100 else "short"
        >>> 
        >>> router = Router(
        ...     routes={
        ...         "short": ShortTextProcessor(),
        ...         "long": LongTextProcessor()
        ...     },
        ...     router_fn=route_by_length
        ... )
        
        Domain-specific router:
        
        >>> def route_by_domain(question: str) -> str:
        ...     if "math" in question.lower():
        ...         return "math"
        ...     elif "code" in question.lower():
        ...         return "code"
        ...     else:
        ...         return "general"
        >>> 
        >>> router = Router(
        ...     routes={
        ...         "math": MathExpert(),
        ...         "code": CodeExpert(),
        ...         "general": GeneralAssistant()
        ...     },
        ...     router_fn=route_by_domain
        ... )
    """
    
    routes: Dict[str, Operator]
    router_fn: Callable[[Any], str]
    default_route: Optional[str]
    
    def __init__(self,
                 routes: Dict[str, Operator],
                 router_fn: Callable[[Any], str],
                 default_route: Optional[str] = None):
        """Initialize router with routes and routing function.
        
        Args:
            routes: Dictionary mapping route names to operators.
            router_fn: Function that takes input and returns route name.
            default_route: Optional default route if router_fn returns unknown.
        """
        self.routes = routes
        self.router_fn = router_fn
        self.default_route = default_route
    
    def forward(self, input: Any) -> Any:
        """Route input to appropriate operator.
        
        Args:
            input: Input to route.
            
        Returns:
            Result from the selected operator.
            
        Raises:
            KeyError: If route not found and no default route.
        """
        route = self.router_fn(input)
        
        if route not in self.routes:
            if self.default_route and self.default_route in self.routes:
                route = self.default_route
            else:
                raise KeyError(f"No operator for route '{route}'")
        
        return self.routes[route](input)


class LearnableRouter(Operator):
    """Router with learnable routing logic using JAX.
    
    This advanced router learns to route inputs based on embeddings and
    learnable weights. The routing decision is differentiable, enabling
    end-to-end optimization.
    
    The router can work in two modes:
    1. With an embedding function (common case) - computes embeddings from input
    2. With external embeddings - accepts pre-computed embeddings
    
    Examples:
        >>> # Mode 1: With embedding function
        >>> def text_embedder(text: str) -> jax.Array:
        ...     # Simple length-based embedding for demo
        ...     return jnp.ones(10) * len(text) / 100
        >>> 
        >>> router = LearnableRouter(
        ...     routes={
        ...         "fast": FastModel(),
        ...         "accurate": AccurateModel(),
        ...     },
        ...     embedding_fn=text_embedder,
        ...     embed_dim=10,
        ...     key=jax.random.PRNGKey(0)
        ... )
        >>> result = router("Hello world")  # Embedding computed internally
        >>> 
        >>> # Mode 2: External embeddings (set embedding_fn=None)
        >>> router = LearnableRouter(
        ...     routes={"fast": FastModel(), "accurate": AccurateModel()},
        ...     embedding_fn=None,  # Expects RoutingInput
        ...     embed_dim=384,
        ...     key=jax.random.PRNGKey(0)
        ... )
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class RoutingInput:
        ...     data: Any
        ...     embedding: jax.Array
        >>> 
        >>> result = router(RoutingInput("Hello", precomputed_embedding))
    """
    
    routes: Dict[str, Operator]
    route_names: List[str]
    embedding_fn: Optional[Callable[[Any], jax.Array]]
    routing_weights: jax.Array
    temperature: jax.Array
    
    def __init__(self,
                 routes: Dict[str, Operator],
                 embed_dim: int,
                 key: jax.Array,
                 embedding_fn: Optional[Callable[[Any], jax.Array]] = None,
                 temperature: float = 1.0):
        """Initialize learnable router.
        
        Args:
            routes: Dictionary mapping route names to operators.
            embed_dim: Dimension of input embeddings.
            key: JAX random key for initialization.
            embedding_fn: Optional function to compute embeddings from input.
                If None, expects input to have 'data' and 'embedding' attributes.
            temperature: Temperature for softmax routing (lower = more decisive).
        """
        self.routes = routes
        self.route_names = list(routes.keys())
        self.embedding_fn = embedding_fn
        
        # Learnable parameters (JAX arrays are automatically dynamic)
        self.routing_weights = jax.random.normal(key, (embed_dim, len(routes)))
        self.temperature = jnp.array(temperature)
    
    def compute_route_probabilities(self, embedding: jax.Array) -> jax.Array:
        """Compute routing probabilities from embedding.
        
        This method is differentiable and can be used in loss functions.
        
        Args:
            embedding: Input embedding of shape (embed_dim,).
            
        Returns:
            Probabilities for each route of shape (num_routes,).
        """
        logits = embedding @ self.routing_weights
        return jax.nn.softmax(logits / self.temperature)
    
    def forward(self, input: Any) -> Any:
        """Route input based on learned weights.
        
        Args:
            input: If embedding_fn is provided, this is the raw input to process.
                If embedding_fn is None, expects input with 'data' and 'embedding' attributes.
            
        Returns:
            Result from the selected operator.
        """
        # Mode 1: Compute embedding from input
        if self.embedding_fn is not None:
            data = input
            embedding = self.embedding_fn(input)
        # Mode 2: Extract pre-computed embedding
        else:
            if hasattr(input, 'data') and hasattr(input, 'embedding'):
                data = input.data
                embedding = input.embedding
            else:
                raise ValueError(
                    "When embedding_fn is None, input must have 'data' and 'embedding' attributes. "
                    "Consider using a dataclass or named tuple."
                )
        
        # Compute routing probabilities
        probs = self.compute_route_probabilities(embedding)
        
        # Select route with highest probability
        route_idx = jnp.argmax(probs)
        route_name = self.route_names[route_idx]
        
        # Route the data
        return self.routes[route_name](data)



class Retry(Operator):
    """Operator that retries on failure with configurable strategy.
    
    This operator wraps another operator and retries it on failure,
    useful for handling transient errors in API calls.
    
    Examples:
        >>> # Retry up to 3 times on any exception
        >>> reliable_op = Retry(
        ...     operator=UnreliableAPIOperator(),
        ...     max_attempts=3
        ... )
        >>> 
        >>> # Custom retry logic
        >>> def should_retry(e: Exception, attempt: int) -> bool:
        ...     return isinstance(e, RateLimitError) and attempt < 5
        >>> 
        >>> careful_op = Retry(
        ...     operator=RateLimitedOperator(),
        ...     should_retry=should_retry
        ... )
    """
    
    operator: Operator
    max_attempts: int
    should_retry: Callable[[Exception, int], bool]
    
    def __init__(self,
                 operator: Operator,
                 max_attempts: int = 3,
                 should_retry: Optional[Callable[[Exception, int], bool]] = None):
        """Initialize retry operator.
        
        Args:
            operator: The operator to wrap with retry logic.
            max_attempts: Maximum number of attempts.
            should_retry: Optional function to determine if should retry.
                Takes exception and attempt number, returns bool.
        """
        self.operator = operator
        self.max_attempts = max_attempts
        self.should_retry = should_retry or (lambda e, n: n < max_attempts)
    
    def forward(self, input: Any) -> Any:
        """Execute operator with retry logic.
        
        Args:
            input: Input to pass to wrapped operator.
            
        Returns:
            Result from successful execution.
            
        Raises:
            Exception: The last exception if all retries fail.
        """
        last_error = None
        
        for attempt in range(self.max_attempts):
            try:
                return self.operator(input)
            except Exception as e:
                last_error = e
                if not self.should_retry(e, attempt + 1):
                    raise
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Retry exhausted without error")


class Cache(Operator):
    """Operator that caches results for repeated inputs.
    
    This operator wraps another operator and caches its results,
    useful for expensive operations like LLM calls.
    
    Examples:
        >>> # Simple in-memory cache
        >>> cached_classifier = Cache(
        ...     operator=ExpensiveClassifier(),
        ...     max_size=1000
        ... )
        >>> 
        >>> # First call computes result
        >>> result1 = cached_classifier("test input")
        >>> # Second call returns cached result
        >>> result2 = cached_classifier("test input")
    """
    
    operator: Operator
    max_size: int
    key_fn: Callable[[Any], str]
    cache: Dict[str, Any]
    access_order: List[str]
    
    def __init__(self,
                 operator: Operator,
                 max_size: int = 100,
                 key_fn: Optional[Callable[[Any], str]] = None):
        """Initialize cache operator.
        
        Args:
            operator: The operator to wrap with caching.
            max_size: Maximum number of entries to cache.
            key_fn: Optional function to compute cache key from input.
                Defaults to str() conversion.
        """
        self.operator = operator
        self.max_size = max_size
        self.key_fn = key_fn or str
        self.cache = {}
        self.access_order = []
    
    def forward(self, input: Any) -> Any:
        """Execute operator with caching.
        
        Args:
            input: Input to pass to wrapped operator.
            
        Returns:
            Cached result if available, otherwise computed result.
        """
        key = self.key_fn(input)
        
        # Check cache
        if key in self.cache:
            # Move to end (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # Compute result
        result = self.operator(input)
        
        # Add to cache
        self.cache[key] = result
        self.access_order.append(key)
        
        # Evict if necessary
        if len(self.cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        return result

# Convenience operators for common model calling tasks

class ExtractText(Operator):
    """Operator that extracts text from a model response.
    
    This operator takes a Response object and returns just the text content;
    useful for chaining after model calls when you only need the text output.
    """
    
    def forward(self, response: Response) -> str:
        """Return the text from the response object."""
        return response.text

class ModelText(Operator):
    """Operator that calls a model and returns the text from the response."""
    model_text: Operator

    def __init__(self, model_name: str, **kwargs):
        """Initialize the model text operator."""
        self.model_text = Chain([ModelCall(model_name, **kwargs), ExtractText()])

    def forward(self, input: Any) -> Any:
        """Call the model and return the text from the response."""
        return self.model_text(input)


class EmberEmbedding(Operator):
    """Operator that creates EmberData from raw input (like PyTorch embedding layer).
    
    This operator converts raw input into EmberData, which carries both the data
    and metadata (usage metrics, original prompt) through the
    computation graph. This is similar to PyTorch's embedding layers that
    convert tokens into rich tensor representations.
    
    Examples:
        >>> # Create embedding layer for a model
        >>> model = EmberEmbedding(ThinkingModel())
        >>> result = model("What is 2+2?")
        >>> # result is EmberData with Context
        
        >>> # Use with complex operators
        >>> router = EmberEmbedding(RouterOperator([...]))
        >>> result = router("Complex question")
        >>> # result is EmberData with Context
    """
    
    inner: Operator
    
    def __init__(self, inner_operator: Operator):
        """Initialize with the operator to wrap.
        
        Args:
            inner_operator: The operator to wrap with EmberData conversion.
        """
        self.inner = inner_operator
    
    def forward(self, input: Any) -> EmberData:
        """Convert raw input to EmberData and forward to inner operator.
        
        Args:
            input: The raw input to convert to EmberData.
            
        Returns:
            EmberData result from the inner operator.
        """
        # Create EmberData with Context
        context = Context(original_input=input)
        ember_data = EmberData(input, context)
        
        # Forward to inner operator
        return self.inner(ember_data)


# Convenience functions for creating common patterns

def ensemble(*operators: Operator, **kwargs) -> Ensemble:
    """Create an ensemble from operators.
    
    Args:
        *operators: Operators to include in ensemble.
        **kwargs: Additional arguments for Ensemble constructor.
        
    Returns:
        Ensemble operator.
    """
    return Ensemble(list(operators), **kwargs)


def chain(*operators: Operator) -> Chain:
    """Create a chain from operators.
    
    Args:
        *operators: Operators to chain in sequence.
        
    Returns:
        Chain operator.
    """
    return Chain(list(operators))


def router(routes: Dict[str, Operator], **kwargs) -> Router:
    """Create a router from route dictionary.
    
    Args:
        routes: Dictionary mapping route names to operators.
        **kwargs: Additional arguments for Router constructor.
        
    Returns:
        Router operator.
    """
    return Router(routes, **kwargs)


__all__ = [
    "ModelCall", "Ensemble", "Chain", "Router", "LearnableRouter", 
    "Retry", "Cache", "ExtractText", "ModelText", "EmberEmbedding",
    "ensemble", "chain", "router"
]