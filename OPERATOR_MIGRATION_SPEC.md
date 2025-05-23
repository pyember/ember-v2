# Operator Migration Technical Specification

## Overview

This document provides detailed technical specifications for migrating each operator from LMModule to ModelBinding.

## Base Operator Enhancement

### File: `src/ember/core/registry/operator/base/operator_base.py`

```python
from typing import Union, List, Optional
from ember.api import models, ModelBinding, Response

class BaseOperator:
    """Enhanced base operator with model binding support."""
    
    def _normalize_models(
        self, 
        model_specs: Union[str, ModelBinding, List[Union[str, ModelBinding]]],
        default_params: Optional[dict] = None
    ) -> Union[ModelBinding, List[ModelBinding]]:
        """Convert model specifications to ModelBinding instances.
        
        Args:
            model_specs: Model ID string, ModelBinding, or list of either
            default_params: Default parameters for string specifications
            
        Returns:
            ModelBinding or list of ModelBinding instances
        """
        default_params = default_params or {}
        
        if isinstance(model_specs, list):
            return [self._normalize_single_model(m, default_params) for m in model_specs]
        return self._normalize_single_model(model_specs, default_params)
    
    def _normalize_single_model(
        self, 
        model_spec: Union[str, ModelBinding],
        default_params: dict
    ) -> ModelBinding:
        """Convert a single model specification to ModelBinding."""
        if isinstance(model_spec, str):
            return models.bind(model_spec, **default_params)
        elif isinstance(model_spec, ModelBinding):
            return model_spec
        else:
            raise TypeError(
                f"Model must be string or ModelBinding, got {type(model_spec)}"
            )
```

## Individual Operator Migrations

### 1. EnsembleOperator

**Current Implementation Analysis**:
- Takes list of LMModule instances
- Calls each module with prompt
- Aggregates string responses

**Migration Specification**:

```python
class EnsembleOperator(BaseOperator):
    def __init__(
        self,
        models: List[Union[str, ModelBinding]],
        aggregation_method: str = "most_common",
        **kwargs
    ):
        """Initialize ensemble with multiple models.
        
        Args:
            models: List of model IDs or ModelBinding instances
            aggregation_method: How to aggregate responses
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Extract any model-specific parameters
        model_params = {
            k: v for k, v in kwargs.items() 
            if k in ['temperature', 'max_tokens', 'top_p']
        }
        
        # Normalize all models to bindings
        self.models = self._normalize_models(models, model_params)
        self.aggregation_method = aggregation_method
    
    def forward(self, prompt: str, **kwargs) -> str:
        """Execute ensemble and aggregate responses."""
        responses = []
        
        for model in self.models:
            try:
                response = model(prompt, **kwargs)
                responses.append(response.text)
            except Exception as e:
                # Log error but continue with other models
                logger.warning(f"Model {model.model_id} failed: {e}")
                continue
        
        if not responses:
            raise RuntimeError("All models in ensemble failed")
        
        return self._aggregate_responses(responses)
```

**Test Migration Example**:

```python
def test_ensemble_operator():
    # Before
    lm_modules = [
        LMModule(config=LMModuleConfig(id="gpt-4")),
        LMModule(config=LMModuleConfig(id="claude-3"))
    ]
    operator = EnsembleOperator(lm_modules=lm_modules)
    
    # After
    operator = EnsembleOperator(
        models=["gpt-4", "claude-3"],
        temperature=0.7
    )
    
    # Or with explicit bindings
    operator = EnsembleOperator(
        models=[
            models.bind("gpt-4", temperature=0.5),
            models.bind("claude-3", temperature=0.9)
        ]
    )
```

### 2. VerifierOperator

**Current Implementation Analysis**:
- Single LMModule for verification
- Binary output (verified/not verified)
- Simple prompt template

**Migration Specification**:

```python
class VerifierOperator(BaseOperator):
    def __init__(
        self,
        model: Union[str, ModelBinding],
        verification_prompt_template: str = DEFAULT_VERIFY_TEMPLATE,
        **kwargs
    ):
        """Initialize verifier with a single model.
        
        Args:
            model: Model ID or ModelBinding instance
            verification_prompt_template: Template for verification
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Extract model parameters
        model_params = {
            k: v for k, v in kwargs.items() 
            if k in ['temperature', 'max_tokens']
        }
        
        # Default to low temperature for consistency
        if 'temperature' not in model_params:
            model_params['temperature'] = 0.3
            
        self.model = self._normalize_models(model, model_params)
        self.prompt_template = verification_prompt_template
    
    def forward(self, statement: str, context: str = "", **kwargs) -> bool:
        """Verify a statement's correctness."""
        prompt = self.prompt_template.format(
            statement=statement,
            context=context
        )
        
        response = self.model(prompt, **kwargs)
        return self._parse_verification(response.text)
```

### 3. MostCommonOperator

**Current Implementation Analysis**:
- Wraps EnsembleOperator
- Counts response frequencies
- Returns most common response

**Migration Specification**:

```python
class MostCommonOperator(BaseOperator):
    def __init__(
        self,
        models: List[Union[str, ModelBinding]],
        threshold: float = 0.5,
        **kwargs
    ):
        """Initialize with models for consensus.
        
        Args:
            models: List of models to query
            threshold: Minimum frequency for consensus
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Use EnsembleOperator internally
        self.ensemble = EnsembleOperator(
            models=models,
            aggregation_method="all",  # Get all responses
            **kwargs
        )
        self.threshold = threshold
    
    def forward(self, prompt: str, **kwargs) -> str:
        """Get most common response from models."""
        # Get all responses
        all_responses = self.ensemble.forward_all(prompt, **kwargs)
        
        # Count frequencies
        from collections import Counter
        counts = Counter(all_responses)
        
        # Check threshold
        most_common, count = counts.most_common(1)[0]
        frequency = count / len(all_responses)
        
        if frequency < self.threshold:
            raise ValueError(
                f"No consensus reached (highest frequency: {frequency:.2f})"
            )
        
        return most_common
```

### 4. SynthesisJudgeOperator

**Current Implementation Analysis**:
- Uses LMModule for synthesis judgment
- Complex prompt construction
- Structured output parsing

**Migration Specification**:

```python
class SynthesisJudgeOperator(BaseOperator):
    def __init__(
        self,
        model: Union[str, ModelBinding],
        judgment_criteria: List[str],
        output_format: str = "json",
        **kwargs
    ):
        """Initialize synthesis judge.
        
        Args:
            model: Model for judgment
            judgment_criteria: Criteria to evaluate
            output_format: Expected output format
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Synthesis benefits from higher temperature
        model_params = kwargs.copy()
        if 'temperature' not in model_params:
            model_params['temperature'] = 0.7
            
        self.model = self._normalize_models(model, model_params)
        self.criteria = judgment_criteria
        self.output_format = output_format
    
    def forward(
        self, 
        responses: List[str], 
        original_prompt: str,
        **kwargs
    ) -> dict:
        """Synthesize and judge multiple responses."""
        synthesis_prompt = self._build_synthesis_prompt(
            responses, 
            original_prompt,
            self.criteria
        )
        
        response = self.model(synthesis_prompt, **kwargs)
        return self._parse_judgment(response.text)
```

## Migration Patterns

### Pattern 1: Simple Model Replacement

```python
# Before
self.lm_module = LMModule(config=LMModuleConfig(id=model_name))
response = self.lm_module(prompt=prompt)

# After
self.model = models.bind(model_name)
response = self.model(prompt).text
```

### Pattern 2: Multiple Models

```python
# Before
self.lm_modules = [LMModule(config=cfg) for cfg in configs]
responses = [lm(prompt=p) for lm in self.lm_modules]

# After
self.models = [models.bind(cfg.id, **cfg.params) for cfg in configs]
responses = [m(p).text for m in self.models]
```

### Pattern 3: Dynamic Model Creation

```python
# Before
def create_model(self, model_id: str) -> LMModule:
    return LMModule(config=LMModuleConfig(id=model_id))

# After
def create_model(self, model_id: str) -> ModelBinding:
    return models.bind(model_id, **self.default_params)
```

## Error Handling Updates

### Consistent Error Types

```python
from ember.api.models import (
    ModelError,
    ModelNotFoundError,
    RateLimitError,
    AuthenticationError
)

class OperatorError(Exception):
    """Base exception for operator errors."""
    pass

class OperatorExecutionError(OperatorError):
    """Raised when operator execution fails."""
    pass

# In operators
def forward(self, prompt: str) -> str:
    try:
        response = self.model(prompt)
        return response.text
    except ModelNotFoundError as e:
        raise OperatorError(f"Model configuration error: {e}")
    except RateLimitError:
        # Could implement retry logic here
        raise
    except ModelError as e:
        raise OperatorExecutionError(f"Model invocation failed: {e}")
```

## Testing Strategy

### Mock Fixtures

```python
@pytest.fixture
def mock_model_binding():
    """Create a mock ModelBinding for testing."""
    mock = MagicMock(spec=ModelBinding)
    mock.model_id = "test-model"
    
    # Configure response
    response = MagicMock(spec=Response)
    response.text = "test response"
    response.usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "cost": 0.001
    }
    
    mock.return_value = response
    return mock

@pytest.fixture
def mock_models_api(monkeypatch, mock_model_binding):
    """Mock the models API."""
    mock_api = MagicMock()
    mock_api.bind.return_value = mock_model_binding
    
    monkeypatch.setattr("ember.api.models", mock_api)
    return mock_api
```

### Integration Test Pattern

```python
def test_operator_with_real_models():
    """Test operator with actual model service."""
    # Use test configuration
    operator = EnsembleOperator(
        models=["gpt-3.5-turbo", "claude-instant"],
        temperature=0.5
    )
    
    result = operator.forward("What is 2+2?")
    assert "4" in result
```

## Performance Considerations

### Memory Usage

- ModelBinding objects are lighter than LMModule
- Shared model service reduces duplication
- Lazy initialization of model connections

### Latency Improvements

```python
# Benchmark comparison
def benchmark_invocation():
    # Old way: ~2.5ms overhead
    lm_module = LMModule(config=LMModuleConfig(id="gpt-4"))
    start = time.time()
    response = lm_module(prompt="test")
    old_time = time.time() - start
    
    # New way: ~0.5ms overhead
    model = models.bind("gpt-4")
    start = time.time()
    response = model("test").text
    new_time = time.time() - start
    
    print(f"Improvement: {(old_time - new_time) / old_time * 100:.1f}%")
```

## Rollout Phases

### Phase 1: Non-Breaking Changes
1. Add ModelBinding support to BaseOperator
2. Update operators to accept both LMModule and ModelBinding
3. Add deprecation warnings to LMModule usage

### Phase 2: Migration
1. Update all operator instantiations in examples
2. Migrate test files to use ModelBinding
3. Update documentation

### Phase 3: Cleanup
1. Remove LMModule support from operators
2. Delete LMModule and LMModuleConfig
3. Final performance validation

## Validation Checklist

For each operator migration:

- [ ] Unit tests pass with ModelBinding
- [ ] Integration tests pass
- [ ] Performance benchmarks show improvement
- [ ] Error handling is consistent
- [ ] Documentation is updated
- [ ] Examples work correctly
- [ ] No regressions in functionality