# Ember Framework Integrations Design Document

## Executive Summary

This document outlines the design and implementation strategy for integrating Ember with three major AI frameworks: DSPy, OpenAI Swarm, and Anthropic's Model Context Protocol (MCP). These integrations will enable users to leverage Ember's unified model interface, cost tracking, and performance optimizations within their existing AI workflows.

## Design Principles

1. **Minimal Overhead**: Integrations should add minimal complexity to user code
2. **Feature Preservation**: Maintain all Ember capabilities (streaming, batching, metrics)
3. **Framework Idioms**: Respect each framework's design patterns and conventions
4. **Progressive Disclosure**: Simple by default, powerful when needed
5. **Type Safety**: Leverage Python's type system for robust integrations

## Integration Architecture

### 1. DSPy Integration

#### Overview
DSPy is a framework for declarative self-improving language model programs. Ember will integrate as a custom language model backend, providing access to its entire model registry while maintaining DSPy's optimization capabilities.

#### Architecture
```
DSPy Program
    ↓
DSPy Modules (Predict, ChainOfThought, etc.)
    ↓
EmberLM (Custom LM Implementation)
    ↓
Ember Model Registry
    ↓
Multiple Model Providers (OpenAI, Anthropic, etc.)
```

#### Key Components

**EmberLM Class**
```python
from typing import Any, Dict, List, Optional
import dspy
from ember import models, metrics

class EmberLM(dspy.BaseLM):
    """Ember language model backend for DSPy."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        **kwargs
    ):
        super().__init__(
            model=model,
            model_type='chat',
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs
        )
        self.ember_model = models.get_model(model)
        self.metrics_collector = metrics.MetricsCollector()
    
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Execute model call through Ember."""
        # Convert DSPy format to Ember format
        if messages:
            ember_messages = self._convert_messages(messages)
        else:
            ember_messages = [{"role": "user", "content": prompt}]
        
        # Merge kwargs with instance defaults
        call_kwargs = {**self.kwargs, **kwargs}
        
        # Call Ember model
        with self.metrics_collector.track():
            response = self.ember_model.complete(
                messages=ember_messages,
                temperature=call_kwargs.get('temperature', self.temperature),
                max_tokens=call_kwargs.get('max_tokens', self.max_tokens),
                **{k: v for k, v in call_kwargs.items() 
                   if k not in ['temperature', 'max_tokens']}
            )
        
        # Track in DSPy history
        self.history.append({
            'prompt': prompt or str(messages),
            'response': response.content,
            'kwargs': call_kwargs,
            'ember_metrics': self.metrics_collector.get_last_metrics()
        })
        
        return response.content
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert DSPy message format to Ember format."""
        return [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in messages
        ]
```

**Usage Example**
```python
import dspy
from ember.integrations.dspy import EmberLM

# Initialize Ember backend
ember_lm = EmberLM(model="claude-3-opus-20240229", temperature=0.7)
dspy.configure(lm=ember_lm)

# Use with DSPy modules
classify = dspy.Predict("text -> sentiment")
result = classify(text="I love this framework!")

# Advanced usage with optimization
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=my_accuracy_metric)
optimized_classifier = optimizer.compile(classify, trainset=train_data)
```

### 2. OpenAI Swarm Integration

#### Overview
OpenAI Swarm is a lightweight multi-agent orchestration framework. Ember will integrate by providing a drop-in replacement for the OpenAI client, enabling Swarm to use any model in Ember's registry.

#### Architecture
```
Swarm Agent System
    ↓
Swarm Core (run, get_chat_completion)
    ↓
EmberSwarmClient (OpenAI-compatible interface)
    ↓
Ember Model Registry
    ↓
Multiple Model Providers
```

#### Key Components

**EmberSwarmClient Class**
```python
from typing import Any, Dict, Iterator, List, Optional, Union
from dataclasses import dataclass
from ember import models, streaming

@dataclass
class ChatCompletionMessage:
    role: str
    content: str
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

@dataclass
class ChatCompletion:
    choices: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

class EmberSwarmClient:
    """OpenAI-compatible client for Swarm using Ember backend."""
    
    def __init__(self, default_model: str = "gpt-4"):
        self.default_model = default_model
        self.chat = self.Chat(self)
    
    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(client)
        
        class Completions:
            def __init__(self, client):
                self.client = client
            
            def create(
                self,
                model: str,
                messages: List[Dict[str, Any]],
                temperature: float = 0.0,
                max_tokens: Optional[int] = None,
                functions: Optional[List[Dict[str, Any]]] = None,
                function_call: Optional[Union[str, Dict[str, str]]] = None,
                stream: bool = False,
                **kwargs
            ) -> Union[ChatCompletion, Iterator[Dict[str, Any]]]:
                """Create chat completion using Ember."""
                # Get Ember model
                ember_model = models.get_model(model)
                
                # Convert functions to Ember tool format if present
                tools = None
                if functions:
                    tools = [self._function_to_tool(f) for f in functions]
                
                # Prepare call parameters
                call_params = {
                    "messages": messages,
                    "temperature": temperature,
                    "tools": tools,
                    "tool_choice": function_call,
                    **kwargs
                }
                
                if max_tokens:
                    call_params["max_tokens"] = max_tokens
                
                # Handle streaming
                if stream:
                    return self._stream_response(ember_model, call_params)
                
                # Non-streaming response
                response = ember_model.complete(**call_params)
                
                # Convert to OpenAI format
                return self._convert_response(response, model)
            
            def _function_to_tool(self, function: Dict[str, Any]) -> Dict[str, Any]:
                """Convert OpenAI function format to Ember tool format."""
                return {
                    "type": "function",
                    "function": function
                }
            
            def _convert_response(self, ember_response: Any, model: str) -> ChatCompletion:
                """Convert Ember response to OpenAI format."""
                # Extract tool calls if present
                tool_calls = None
                if hasattr(ember_response, 'tool_calls') and ember_response.tool_calls:
                    tool_calls = [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for i, tc in enumerate(ember_response.tool_calls)
                    ]
                
                choice = {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ember_response.content,
                    },
                    "finish_reason": "stop"
                }
                
                if tool_calls:
                    choice["message"]["tool_calls"] = tool_calls
                
                return ChatCompletion(
                    choices=[choice],
                    model=model,
                    usage={
                        "prompt_tokens": ember_response.usage.input_tokens,
                        "completion_tokens": ember_response.usage.output_tokens,
                        "total_tokens": ember_response.usage.total_tokens
                    }
                )
            
            def _stream_response(self, ember_model: Any, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
                """Stream responses in OpenAI format."""
                for chunk in ember_model.stream(**params):
                    yield {
                        "choices": [{
                            "delta": {
                                "content": chunk.content if hasattr(chunk, 'content') else chunk
                            },
                            "index": 0
                        }]
                    }
```

**Usage Example**
```python
from swarm import Swarm, Agent
from ember.integrations.swarm import EmberSwarmClient

# Create Ember-backed Swarm client
ember_client = EmberSwarmClient(default_model="claude-3-opus-20240229")
swarm = Swarm(client=ember_client)

# Define agents using any Ember-supported model
sales_agent = Agent(
    name="Sales Assistant",
    model="gpt-4-turbo-preview",  # Can use any model in Ember's registry
    instructions="You are a helpful sales assistant.",
    functions=[transfer_to_support, check_inventory]
)

support_agent = Agent(
    name="Support Assistant", 
    model="claude-3-sonnet-20240229",  # Mix models across agents
    instructions="You are a technical support specialist."
)

# Run conversation
response = swarm.run(
    agent=sales_agent,
    messages=[{"role": "user", "content": "I need help with my order"}]
)
```

### 3. Anthropic MCP Integration

#### Overview
Model Context Protocol (MCP) is a universal protocol for connecting AI systems with data sources. Ember will implement an MCP server that exposes its model orchestration capabilities as tools, resources, and prompts.

#### Architecture
```
MCP Client (Claude Desktop, IDE, etc.)
    ↓ (JSON-RPC 2.0)
Ember MCP Server
    ├── Tools (model inference, ensemble, etc.)
    ├── Resources (model registry, metrics)
    └── Prompts (templates, workflows)
    ↓
Ember Core System
```

#### Key Components

**Ember MCP Server**
```python
from typing import Any, Dict, List, Optional
from mcp.server import Server, Tool, Resource, Prompt
from mcp.types import TextContent, ImageContent
from ember import models, operators, data
import json

class EmberMCPServer:
    """MCP server exposing Ember capabilities."""
    
    def __init__(self):
        self.server = Server("ember-mcp-server")
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        """Register Ember capabilities as MCP tools."""
        
        @self.server.tool(
            name="ember_generate",
            description="Generate text using any model in Ember's registry",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Input prompt"},
                    "model": {"type": "string", "description": "Model name (e.g., claude-3-opus)"},
                    "temperature": {"type": "number", "description": "Sampling temperature", "default": 0.7},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"}
                },
                "required": ["prompt", "model"]
            }
        )
        async def ember_generate(prompt: str, model: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
            """Generate text using Ember."""
            ember_model = models.get_model(model)
            response = await ember_model.agenerate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content
        
        @self.server.tool(
            name="ember_ensemble",
            description="Run ensemble voting across multiple models",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Input prompt"},
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model names"
                    },
                    "strategy": {
                        "type": "string", 
                        "enum": ["majority_vote", "weighted", "confidence"],
                        "description": "Voting strategy"
                    }
                },
                "required": ["prompt", "models"]
            }
        )
        async def ember_ensemble(prompt: str, models: List[str], strategy: str = "majority_vote") -> Dict[str, Any]:
            """Run ensemble across multiple models."""
            ensemble = operators.EnsembleOperator(
                operators=[operators.Operator(model=m) for m in models],
                strategy=strategy
            )
            result = await ensemble.arun(prompt)
            return {
                "consensus": result.content,
                "votes": result.metadata.get("votes", {}),
                "confidence": result.metadata.get("confidence", 0.0)
            }
        
        @self.server.tool(
            name="ember_verify",
            description="Verify and improve model output",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Original prompt"},
                    "output": {"type": "string", "description": "Output to verify"},
                    "criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Verification criteria"
                    },
                    "model": {"type": "string", "description": "Model for verification"}
                },
                "required": ["prompt", "output", "model"]
            }
        )
        async def ember_verify(prompt: str, output: str, model: str, criteria: List[str] = None) -> Dict[str, Any]:
            """Verify and potentially improve output."""
            verifier = operators.VerifierOperator(model=model, criteria=criteria)
            result = await verifier.averify(prompt, output)
            return {
                "is_valid": result.is_valid,
                "issues": result.issues,
                "improved_output": result.improved_output if not result.is_valid else output
            }
    
    def _register_resources(self):
        """Register Ember data as MCP resources."""
        
        @self.server.resource(
            uri="ember://models/registry",
            name="Model Registry",
            description="Available models with capabilities and pricing"
        )
        async def get_model_registry() -> List[TextContent]:
            """Return model registry information."""
            registry = models.get_registry()
            content = json.dumps({
                model_id: {
                    "provider": info.provider,
                    "capabilities": info.capabilities,
                    "context_length": info.context_length,
                    "cost_per_1k_tokens": {
                        "input": info.input_cost,
                        "output": info.output_cost
                    }
                }
                for model_id, info in registry.items()
            }, indent=2)
            
            return [TextContent(
                type="text",
                text=content,
                mimeType="application/json"
            )]
        
        @self.server.resource(
            uri="ember://metrics/usage",
            name="Usage Metrics",
            description="Current session usage statistics"
        )
        async def get_usage_metrics() -> List[TextContent]:
            """Return usage metrics."""
            metrics = models.get_usage_metrics()
            return [TextContent(
                type="text",
                text=json.dumps(metrics, indent=2),
                mimeType="application/json"
            )]
    
    def _register_prompts(self):
        """Register Ember prompt templates."""
        
        @self.server.prompt(
            name="code_review",
            description="Comprehensive code review with multiple perspectives"
        )
        async def code_review_prompt(code: str, language: str = "python") -> List[Dict[str, str]]:
            """Generate code review prompt."""
            return [
                {
                    "role": "system",
                    "content": f"You are an expert {language} code reviewer. Review the following code for correctness, performance, security, and style."
                },
                {
                    "role": "user", 
                    "content": f"Please review this {language} code:\n\n```{language}\n{code}\n```"
                }
            ]
        
        @self.server.prompt(
            name="chain_of_thought",
            description="Step-by-step reasoning prompt"
        )
        async def chain_of_thought_prompt(question: str) -> List[Dict[str, str]]:
            """Generate chain of thought prompt."""
            return [
                {
                    "role": "system",
                    "content": "You are a logical reasoning expert. Break down problems step by step."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nPlease solve this step by step, showing your reasoning at each stage."
                }
            ]
    
    async def run(self, transport: str = "stdio"):
        """Run the MCP server."""
        if transport == "stdio":
            await self.server.run_stdio()
        elif transport == "http":
            await self.server.run_http(host="localhost", port=3000)

# FastMCP Alternative Implementation
from fastmcp import FastMCP

mcp = FastMCP("ember-mcp-server")

@mcp.tool()
async def ember_generate(prompt: str, model: str = "claude-3-opus-20240229") -> str:
    """Generate text using Ember."""
    from ember import models
    ember_model = models.get_model(model)
    response = await ember_model.agenerate(prompt)
    return response.content

@mcp.resource("ember://models/list")
async def list_models() -> str:
    """List available models."""
    from ember import models
    return json.dumps(models.list_models(), indent=2)
```

**Usage Example**
```python
# Server setup (run as separate process)
from ember.integrations.mcp import EmberMCPServer

server = EmberMCPServer()
asyncio.run(server.run(transport="stdio"))

# Client configuration (e.g., in Claude Desktop)
{
  "mcpServers": {
    "ember": {
      "command": "python",
      "args": ["-m", "ember.integrations.mcp.server"],
      "env": {
        "OPENAI_API_KEY": "...",
        "ANTHROPIC_API_KEY": "..."
      }
    }
  }
}

# Usage in Claude Desktop
# User: "Use ember to generate a haiku using gpt-4"
# Claude would call: ember_generate(prompt="Write a haiku", model="gpt-4")
```

## Implementation Plan

### Phase 1: Core Integration Components
1. Create `src/ember/integrations/` directory structure
2. Implement base classes for each integration
3. Add comprehensive type hints and documentation
4. Create unit tests for each integration

### Phase 2: Examples and Documentation
1. Create working examples for each integration
2. Write integration guides with best practices
3. Add performance benchmarking examples
4. Create troubleshooting documentation

### Phase 3: Advanced Features
1. Add streaming support for all integrations
2. Implement batch processing optimizations
3. Add metrics and observability hooks
4. Create integration-specific utilities

## Directory Structure
```
src/ember/integrations/
├── __init__.py
├── dspy/
│   ├── __init__.py
│   ├── ember_lm.py
│   ├── utils.py
│   └── examples/
│       ├── basic_usage.py
│       ├── optimization.py
│       └── advanced_patterns.py
├── swarm/
│   ├── __init__.py
│   ├── client.py
│   ├── utils.py
│   └── examples/
│       ├── multi_agent.py
│       ├── tool_calling.py
│       └── mixed_models.py
└── mcp/
    ├── __init__.py
    ├── server.py
    ├── fastmcp_server.py
    ├── utils.py
    └── examples/
        ├── basic_server.py
        ├── claude_desktop_config.json
        └── advanced_tools.py

tests/integrations/
├── test_dspy.py
├── test_swarm.py
└── test_mcp.py

docs/integrations/
├── dspy_guide.md
├── swarm_guide.md
├── mcp_guide.md
└── troubleshooting.md
```

## Testing Strategy

### Unit Tests
- Mock external dependencies
- Test format conversions
- Verify error handling
- Check type safety

### Integration Tests
- Test against real frameworks
- Verify end-to-end workflows
- Performance benchmarks
- Multi-model scenarios

### Example Tests
```python
# tests/integrations/test_dspy.py
import pytest
from ember.integrations.dspy import EmberLM
import dspy

def test_ember_lm_initialization():
    """Test EmberLM can be initialized and configured."""
    lm = EmberLM(model="gpt-4", temperature=0.5)
    assert lm.model == "gpt-4"
    assert lm.kwargs['temperature'] == 0.5

def test_dspy_integration():
    """Test basic DSPy integration."""
    lm = EmberLM(model="gpt-3.5-turbo")
    dspy.configure(lm=lm)
    
    predict = dspy.Predict("question -> answer")
    result = predict(question="What is 2+2?")
    assert result.answer is not None
```

## Performance Considerations

1. **Caching**: Leverage Ember's built-in caching for repeated calls
2. **Batching**: Use Ember's batch processing for multiple requests
3. **Streaming**: Implement streaming for long-running operations
4. **Connection Pooling**: Reuse connections for MCP server
5. **Async Support**: Provide async variants for all integrations

## Security Considerations

1. **API Key Management**: Use Ember's secure credential storage
2. **Input Validation**: Validate all inputs before forwarding
3. **Rate Limiting**: Respect provider rate limits
4. **Error Handling**: Don't leak sensitive information in errors
5. **Transport Security**: Use TLS for remote MCP connections

## Success Metrics

1. **Adoption**: Number of users adopting integrations
2. **Performance**: Latency overhead < 10ms per call
3. **Reliability**: 99.9% uptime for MCP server
4. **Coverage**: Support for 90% of framework features
5. **Documentation**: 95% of users can integrate without support

## Future Enhancements

1. **Additional Frameworks**: LangChain, CrewAI, AutoGen
2. **Visual Tools**: Integration with Gradio/Streamlit
3. **Monitoring**: Prometheus/Grafana integration
4. **Model Router**: Intelligent model selection based on task
5. **Cost Optimization**: Automatic model selection for cost/performance

This design provides a solid foundation for integrating Ember with major AI frameworks while maintaining its core strengths of simplicity, performance, and unified model access.