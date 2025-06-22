# Models Module Implementation Plan (Final)

*Based on deep analysis and master's wisdom*

## Executive Summary

After deep analysis, we found:
1. **Threading is already correct** - Single lock, no change needed
2. **Configuration needs hybrid approach** - Hardcode defaults, allow env overrides
3. **Provider mapping should be explicit** - But allow extension
4. **Service layer must stay** - It does essential work (cost calc, metrics)

## Implementation Architecture

### Keep What Works
```
API → Service → Registry → Provider
```
Each layer has a clear, essential purpose.

### Simplify How It Works

## 1. Configuration: Hybrid Approach

```python
# models/_costs.py - Hardcoded defaults
DEFAULT_MODEL_COSTS = {
    # OpenAI
    "gpt-4": {"input": 30.0, "output": 60.0, "context": 8192},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0, "context": 128000},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5, "context": 4096},
    
    # Anthropic  
    "claude-3-opus": {"input": 15.0, "output": 75.0, "context": 200000},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0, "context": 200000},
    
    # Google
    "gemini-pro": {"input": 0.5, "output": 1.5, "context": 32768},
}

def get_model_costs() -> dict:
    """Get model costs with environment overrides."""
    costs = DEFAULT_MODEL_COSTS.copy()
    
    # Override entire cost structure
    if override_json := os.getenv("EMBER_MODEL_COSTS_JSON"):
        costs.update(json.loads(override_json))
    
    # Override specific model costs
    # EMBER_COST_GPT4_INPUT=25.0
    for key, value in os.environ.items():
        if key.startswith("EMBER_COST_"):
            parts = key[11:].lower().split("_")
            if len(parts) >= 2:
                model = parts[0].replace("_", "-")
                field = parts[1]  # "input" or "output"
                if model in costs:
                    costs[model][field] = float(value)
    
    return costs
```

**Why This Works**:
- **Dean & Ghemawat**: "Defaults handle 99% of cases, env vars handle updates"
- **Jobs**: "No config files to manage"
- **Carmack**: "Can update costs without code changes"

## 2. Provider Mapping: Explicit + Extensible

```python
# providers/_registry.py
from typing import Type, Dict
from .base import BaseProviderModel
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider

# Explicit core providers
CORE_PROVIDERS: Dict[str, Type[BaseProviderModel]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider, 
    "google": GoogleProvider,
    "azure": AzureOpenAIProvider,
}

# Extension point for custom providers
_custom_providers: Dict[str, Type[BaseProviderModel]] = {}

def register_provider(name: str, provider_class: Type[BaseProviderModel]) -> None:
    """Register a custom provider (advanced use)."""
    if not issubclass(provider_class, BaseProviderModel):
        raise TypeError("Provider must inherit from BaseProviderModel")
    _custom_providers[name] = provider_class

def get_provider_class(name: str) -> Type[BaseProviderModel]:
    """Get provider class by name."""
    # Custom providers can override core
    if name in _custom_providers:
        return _custom_providers[name]
    if name in CORE_PROVIDERS:
        return CORE_PROVIDERS[name]
    
    # Helpful error
    available = list(CORE_PROVIDERS.keys()) + list(_custom_providers.keys())
    raise ValueError(
        f"Unknown provider '{name}'. Available: {', '.join(available)}"
    )
```

**Why This Works**:
- **Ritchie**: "I can see all providers at a glance"
- **Brockman**: "Power users can still extend"
- **Carmack**: "No filesystem scanning magic"

## 3. Simplified Context/Registry Creation

```python
# models.py
class ModelsAPI:
    def __init__(self, registry: Optional[ModelRegistry] = None):
        """Initialize with optional custom registry."""
        if registry is None:
            # Direct creation, no complex context
            self._registry = ModelRegistry()
            self._usage_service = UsageService()
            self._service = ModelService(
                registry=self._registry,
                usage_service=self._usage_service,
                metrics=_create_metrics() if os.getenv("EMBER_METRICS_ENABLED") else None
            )
        else:
            # Advanced: custom registry
            self._registry = registry
            self._usage_service = UsageService()
            self._service = ModelService(
                registry=self._registry,
                usage_service=self._usage_service
            )
```

## 4. Model ID Resolution

```python
def resolve_model_id(model: str) -> tuple[str, str]:
    """Resolve model string to (provider, model_name).
    
    Examples:
        "gpt-4" → ("openai", "gpt-4")
        "openai/gpt-4" → ("openai", "gpt-4")
        "claude-3" → ("anthropic", "claude-3")
    """
    # Explicit provider
    if "/" in model:
        return model.split("/", 1)
    
    # Well-known models
    if model.startswith("gpt-") or model in ["davinci-002", "babbage-002"]:
        return ("openai", model)
    elif model.startswith("claude-"):
        return ("anthropic", model)
    elif model.startswith("gemini-"):
        return ("google", model)
    elif model.startswith("llama-"):
        return ("meta", model)
    
    # Unknown - let registry handle error
    return ("unknown", model)
```

## Implementation Steps

### Day 2: Core Simplification
1. Create `models/_costs.py` with hybrid cost system
2. Create `providers/_registry.py` with explicit mapping
3. Simplify ModelRegistry (already has single lock)
4. Update ModelsAPI to use direct creation

### Day 3: Integration
1. Wire up environment-based cost overrides
2. Implement model ID resolution
3. Add provider preference support (optional param)
4. Ensure all tests pass

### Day 4: Polish
1. Add helpful error messages
2. Document environment variables
3. Create migration guide
4. Performance benchmarks

## What We're NOT Changing

1. **Response Object** - Perfect as is
2. **ModelBinding** - Unique performance optimization
3. **Service Layer** - Does essential work
4. **Single Registry Lock** - Already correct
5. **No Client Init** - Beautiful API

## Environment Variables

```bash
# API Keys (existing)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Cost Overrides (new)
EMBER_MODEL_COSTS_JSON='{"gpt-4": {"input": 25.0, "output": 50.0}}'
# OR
EMBER_COST_GPT4_INPUT=25.0
EMBER_COST_GPT4_OUTPUT=50.0

# Optional Features
EMBER_METRICS_ENABLED=true
EMBER_FALLBACK_ENABLED=true  # Future
```

## Success Metrics

1. **Same UX** - All examples work unchanged
2. **Simpler Code** - ~30% reduction in complexity
3. **Faster Startup** - No filesystem scanning
4. **Flexible Costs** - Can update without deployment
5. **Extensible** - Power users can add providers

## Why This Is Right

**Dean & Ghemawat**: "We kept the architecturally important parts (Service layer) and simplified implementation details."

**Jobs**: "Users still just call models('gpt-4', 'Hello'). Perfect."

**Carmack**: "Explicit providers, overridable costs, no magic. This is production-ready."

**Ritchie**: "Four clear layers, each doing one thing well."

**Knuth**: "The code clearly shows what providers exist and their default costs."

**Brockman**: "New users are productive immediately, power users have control."

## The Philosophy

This isn't minimalism for its own sake. It's **principled simplification**:
- Keep what provides value (cost tracking, metrics, great UX)
- Simplify how it's implemented (explicit providers, env config)
- Enable advanced use cases (custom providers, cost overrides)
- Make the common case fast and the complex case possible