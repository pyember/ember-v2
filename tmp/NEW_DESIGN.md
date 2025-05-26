# Codex-Ember Integration Design

## Core Insight

Instead of building a new provider abstraction, we leverage Ember's existing model registry and operator system to power Codex. This gives us:

1. **Existing provider infrastructure** - OpenAI, Anthropic, etc. already implemented
2. **Operator patterns** - Ensembles, voting, verification built-in
3. **Model registry** - Thread-safe, lazy loading, cost tracking
4. **Unified API** - Clean models() and operators() APIs

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Codex CLI │────▶│ CodexEmberBridge │────▶│ Ember Framework │
└─────────────┘     └──────────────────┘     └─────────────────┘
                            │                          │
                            ▼                          ▼
                    ┌──────────────┐          ┌─────────────────┐
                    │ AgentAdapter │          │   Model Registry │
                    └──────────────┘          ├─────────────────┤
                            │                 │     Operators    │
                            ▼                 ├─────────────────┤
                    ┌──────────────┐          │    Providers    │
                    │ Tool Handler │          └─────────────────┘
                    └──────────────┘
```

## Implementation Strategy

### 1. CodexEmberBridge - The Integration Layer

```python
from ember.api import models, operators
from ember.core.registry.model.base.schemas.chat_schemas import ChatRequest, ChatResponse, Message
from typing import AsyncIterator, List, Optional
import json

class CodexEmberBridge:
    """Bridge between Codex's agent loop and Ember's model system."""
    
    def __init__(self, default_model: str = "gpt-4o"):
        self.default_model = default_model
        # Can configure ensemble operators here
        self.ensemble = None
        
    def setup_ensemble(self, model_names: List[str]):
        """Configure an ensemble of models for robust responses."""
        from ember.api.operators import EnsembleOperator, MostCommonAnswerSelector
        
        # Create operators for each model
        model_operators = [models.instance(name) for name in model_names]
        
        # Create ensemble with voting
        self.ensemble = MostCommonAnswerSelector(
            operator=EnsembleOperator(operators=model_operators)
        )
    
    async def complete_with_tools(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        model: Optional[str] = None,
        stream: bool = True
    ) -> AsyncIterator[dict]:
        """
        Convert Codex tool-calling format to Ember format and stream responses.
        """
        # Convert to Ember's ChatRequest format
        ember_messages = [
            Message(
                role=msg["role"],
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id")
            )
            for msg in messages
        ]
        
        request = ChatRequest(
            messages=ember_messages,
            tools=tools,
            stream=stream
        )
        
        # Use configured model or ensemble
        if self.ensemble and not model:
            # Use ensemble for important decisions
            response = await self.ensemble.forward_async(request)
            yield self._convert_to_codex_format(response)
        else:
            # Use single model
            model_instance = models.instance(model or self.default_model)
            
            if stream:
                async for chunk in model_instance.stream_async(request):
                    yield self._convert_chunk_to_codex_format(chunk)
            else:
                response = await model_instance.forward_async(request)
                yield self._convert_to_codex_format(response)
    
    def _convert_to_codex_format(self, response: ChatResponse) -> dict:
        """Convert Ember ChatResponse to Codex response format."""
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls
                },
                "finish_reason": response.finish_reason
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
```

### 2. Enhanced AgentLoop Integration

```python
# In agent-loop.ts - minimal changes needed
class EnhancedAgentLoop:
    def __init__(self, config):
        # Initialize Ember bridge
        self.ember_bridge = CodexEmberBridge(
            default_model=config.model
        )
        
        # Configure ensemble for critical operations
        if config.use_ensemble:
            self.ember_bridge.setup_ensemble([
                "gpt-4o",
                "claude-3-sonnet",
                "gpt-4-turbo"
            ])
    
    async def execute_with_tools(self, messages, tools):
        """Execute using Ember's infrastructure."""
        # Tool definitions stay the same
        # Message format stays the same
        # Just swap the backend
        
        async for response in self.ember_bridge.complete_with_tools(
            messages=messages,
            tools=tools,
            model=self.model,
            stream=True
        ):
            # Process response exactly as before
            yield response
```

### 3. Leveraging Ember Operators for Advanced Features

```python
from ember.api.operators import Operator, EmberModel, Field
from ember.core.types.ember_model import EmberComputed

class CodexCommand(EmberModel):
    """Model for shell commands in Codex."""
    command: List[str] = Field(..., description="Command to execute")
    workdir: Optional[str] = Field(None, description="Working directory")
    
class CommandSafetyChecker(Operator[CodexCommand, bool]):
    """Use LLM to check if command is safe."""
    
    system_prompt = """
    You are a security expert. Analyze if this command is safe to execute.
    Consider: file system damage, network access, system changes.
    Return true only if completely safe.
    """
    
    def forward(self, cmd: CodexCommand) -> bool:
        response = self.model(
            f"Is this command safe? {' '.join(cmd.command)}",
            context=self.system_prompt
        )
        return "true" in response.lower()

class CommandExplainer(Operator[CodexCommand, str]):
    """Explain what a command does."""
    
    def forward(self, cmd: CodexCommand) -> str:
        return self.model(
            f"Explain this command: {' '.join(cmd.command)}"
        )

# Use in Codex
safety_checker = CommandSafetyChecker(model="gpt-4o")
explainer = CommandExplainer(model="claude-3")

# Before executing any command
if not safety_checker(command):
    explanation = explainer(command)
    # Ask user for confirmation with explanation
```

### 4. Configuration Integration

```yaml
# codex.yaml - extends Ember's configuration
ember:
  # Use existing Ember model configuration
  models:
    openai:
      api_key: ${OPENAI_API_KEY}
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
  
  # Operator configurations
  operators:
    safety_ensemble:
      type: ensemble
      models: ["gpt-4o", "claude-3"]
      voting: majority
    
    code_specialist:
      type: routing
      rules:
        - pattern: ".*\\.py$"
          model: "claude-3"  # Claude for Python
        - pattern: ".*\\.ts$"
          model: "gpt-4o"    # GPT-4 for TypeScript

codex:
  # Codex-specific settings
  default_model: "gpt-4o"
  use_ensemble_for_commands: true
  safety_check_enabled: true
```

## Key Benefits of Ember Integration

### 1. **Immediate Multi-Provider Support**
- OpenAI, Anthropic already implemented
- Standardized error handling
- Retry logic and rate limiting built-in

### 2. **Advanced Operator Patterns**
```python
# Example: Multi-stage code generation with verification
from ember.api.operators import Pipeline, VerifierOperator

code_pipeline = Pipeline([
    CodeGenerator(model="gpt-4o"),
    SyntaxChecker(),
    VerifierOperator(
        candidate_operator=CodeGenerator(model="claude-3"),
        verifier_model="gpt-4o"
    )
])

# Use in Codex for high-quality code generation
result = code_pipeline(request)
```

### 3. **Cost Tracking and Optimization**
```python
# Ember tracks costs automatically
response = models("gpt-4o", prompt)
print(f"Cost: ${response.usage.cost_usd:.4f}")

# Can optimize by routing to cheaper models
if is_simple_query(prompt):
    response = models("gpt-4o-mini", prompt)  # 10x cheaper
```

### 4. **Unified Context Management**
```python
# Ember's context system for configuration
from ember.core.context import EmberContext

with EmberContext(temperature=0.2, max_tokens=2000):
    # All model calls use these settings
    response = models("gpt-4o", prompt)
```

## Migration Path

### Phase 1: Minimal Integration (Week 1)
1. Create CodexEmberBridge
2. Replace OpenAI client with Ember models API
3. Maintain exact same behavior
4. Test with existing functionality

### Phase 2: Enhanced Features (Week 2)
1. Add ensemble support for critical operations
2. Implement safety checking operators
3. Add cost tracking to CLI output
4. Enable provider switching via flags

### Phase 3: Advanced Operators (Week 3)
1. Create Codex-specific operators (command validation, code analysis)
2. Implement graph-based workflows for complex tasks
3. Add specialized routing for different file types
4. Integrate with Ember's operator composition

### Phase 4: Full Integration (Week 4)
1. Unified configuration system
2. Seamless model/operator switching
3. Advanced ensemble strategies
4. Performance optimization with caching

## Comparison with Original Design

| Original Design | Ember Integration |
|----------------|-------------------|
| Build provider abstraction from scratch | Use existing BaseProviderModel |
| Create new registry | Leverage ModelRegistry |
| Simple message format | Rich ChatRequest/ChatResponse |
| Manual error handling | Standardized error hierarchy |
| No cost tracking | Built-in usage calculation |
| Single model calls | Ensemble operators available |
| Basic streaming | Advanced streaming with transforms |

## Next Steps

1. Study Ember's provider implementations in detail
2. Create minimal CodexEmberBridge prototype
3. Test with existing Codex functionality
4. Gradually add operator-based enhancements
5. Benchmark performance vs direct OpenAI calls