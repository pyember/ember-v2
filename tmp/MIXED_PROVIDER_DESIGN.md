# Mixed Provider Design for Codex

## Core Insight

Different models excel at different tasks:
- **OpenAI**: Best for tool use (function calling is mature)
- **Anthropic**: Superior for planning and reasoning
- **Ember Ensembles**: For consensus on critical decisions

We need a minimal change that routes requests based on intent, not a full rewrite.

## Surgical Integration Points

### 1. Minimal AgentLoop Changes

```typescript
// In agent-loop.ts - add provider routing
interface ProviderRouter {
  getProvider(intent: 'tool_use' | 'planning' | 'synthesis'): string;
}

class SmartProviderRouter implements ProviderRouter {
  getProvider(intent: string): string {
    switch(intent) {
      case 'tool_use':
        return 'openai';  // Keep using OpenAI for tools
      case 'planning':
        return 'anthropic';  // Claude for planning
      case 'synthesis':
        return 'ensemble';  // Multiple models for important decisions
      default:
        return 'openai';
    }
  }
}

// Minimal change in agentLoop function
export async function agentLoop({
  model,
  provider,
  router = new SmartProviderRouter(),  // NEW: inject router
  // ... rest of params
}: AgentLoopParams) {
  
  // Existing code...
  
  // When making completion requests:
  const effectiveProvider = router.getProvider('tool_use');
  const client = createClient(effectiveProvider);  // Existing function
  
  // For planning phases:
  if (needsPlanning(conversationHistory)) {
    const planningProvider = router.getProvider('planning');
    const planningClient = createClient(planningProvider);
    // Use for high-level reasoning
  }
}
```

### 2. Ember Bridge as a Drop-in Client

```python
# codex_ember_bridge.py - Acts like OpenAI client
from ember.api import models
from typing import AsyncIterator
import json

class EmberOpenAICompatibleClient:
    """Drop-in replacement for OpenAI client using Ember."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        self.chat = self.Chat(self)
    
    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)
        
        class Completions:
            def __init__(self, parent):
                self.parent = parent
            
            async def create(self, **kwargs):
                """OpenAI-compatible completion."""
                # Use Ember models under the hood
                model_name = kwargs.get('model', self.parent.model)
                
                # Special handling for tool calls - always use OpenAI
                if kwargs.get('tools'):
                    model_name = 'openai:gpt-4o'  # Force OpenAI for tools
                
                # Convert to Ember format
                ember_model = models.instance(model_name)
                
                if kwargs.get('stream'):
                    return self._stream_wrapper(ember_model, kwargs)
                else:
                    response = await ember_model.forward_async({
                        'messages': kwargs['messages'],
                        'temperature': kwargs.get('temperature'),
                        'max_tokens': kwargs.get('max_tokens')
                    })
                    return self._convert_response(response)
            
            async def _stream_wrapper(self, model, kwargs):
                """Wrap Ember streaming to match OpenAI format."""
                async for chunk in model.stream_async(kwargs):
                    yield self._convert_chunk(chunk)

def createClient(provider: string) -> EmberOpenAICompatibleClient:
    """Factory function matching existing Codex code."""
    if provider == 'ensemble':
        # Special handling for ensembles
        return EmberEnsembleClient(['gpt-4o', 'claude-3'])
    return EmberOpenAICompatibleClient(provider)
```

### 3. Configuration-Driven Routing

```yaml
# .codex/config.yaml
providers:
  default: openai
  
  # Intent-based routing
  routing:
    tool_use: openai        # Always use OpenAI for function calling
    planning: anthropic     # Use Claude for planning/reasoning
    code_gen: anthropic     # Claude excels at code
    safety_check: ensemble  # Multiple models for safety
    
  # Model-specific overrides
  models:
    openai:
      default: gpt-4o
      tool_model: gpt-4o    # Can specify different model for tools
    anthropic:
      default: claude-3-opus
      planning_model: claude-3-opus

# Ensemble configurations
ensembles:
  safety:
    models: [gpt-4o, claude-3-opus]
    strategy: unanimous  # All must agree
  
  quality:
    models: [gpt-4o, claude-3-sonnet, gpt-4-turbo]
    strategy: majority   # 2/3 must agree
```

### 4. Minimal Changes to Tool Handling

```typescript
// In handle-exec-command.ts
export async function handleExecCommand(
  command: string[],
  provider?: string  // NEW: optional provider override
) {
  // Critical: Use OpenAI for tool execution
  const toolProvider = provider || 'openai';
  const client = createClient(toolProvider);
  
  // Rest stays the same...
}

// In apply-patch.ts
export async function applyPatch(
  patch: string,
  provider?: string  // NEW: optional provider override
) {
  // Always use OpenAI for patch application (reliable tool use)
  const client = createClient(provider || 'openai');
  
  // Rest stays the same...
}
```

## Implementation Strategy

### Phase 1: Drop-in Compatibility Layer (2 days)
1. Create `EmberOpenAICompatibleClient` that matches OpenAI client interface
2. Implement `createClient` factory to return appropriate client
3. No changes to existing code paths initially

### Phase 2: Intent-Based Routing (1 day)
1. Add `ProviderRouter` interface
2. Inject router into `agentLoop`
3. Route based on operation type (tool use vs planning)

### Phase 3: Configuration System (1 day)
1. Add provider routing configuration
2. Load from `.codex/config.yaml`
3. Environment variable overrides

### Phase 4: Testing & Optimization (2 days)
1. A/B test different providers for different tasks
2. Measure latency and quality
3. Fine-tune routing rules

## Key Design Decisions

### 1. **Preserve OpenAI for Tools**
- OpenAI's function calling is mature and reliable
- Keep using it for all tool operations
- No risk to existing functionality

### 2. **Inject Providers Strategically**
- Planning/reasoning: Use Claude
- Code generation: Use Claude
- Safety checks: Use ensembles
- Tool execution: Stay with OpenAI

### 3. **Minimal API Changes**
- `createClient()` returns compatible interface
- Existing code continues to work
- New capabilities are opt-in

### 4. **Configuration Over Code**
- Route providers via config, not hardcoded
- Easy to experiment and tune
- No code changes for different strategies

## Example Usage

```bash
# Use default routing (OpenAI for tools, Claude for planning)
codex "refactor this codebase"

# Force single provider
codex --provider openai "fix this bug"

# Use ensemble for critical operations
codex --provider ensemble "delete all unused files"

# Custom routing via config
codex --config ./safety-first.yaml "update dependencies"
```

## Benefits

1. **Minimal Changes**: ~50 lines of code in Codex
2. **Best Tool for Job**: Use each model's strengths
3. **Backwards Compatible**: Existing behavior preserved
4. **Future Proof**: Easy to add new providers
5. **Cost Optimized**: Use cheaper models where appropriate

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Different response formats | Normalize in EmberOpenAICompatibleClient |
| Tool calling incompatibilities | Always use OpenAI for tools |
| Latency from multiple providers | Cache and parallelize where possible |
| Cost increases | Monitor usage, default to efficient routing |

## Next Steps

1. Implement `EmberOpenAICompatibleClient` class
2. Test with existing Codex functionality
3. Add configuration loading
4. Implement intent detection
5. Roll out gradually with feature flags