# Ember Framework Integration Guide

This guide provides detailed instructions for integrating Ember with DSPy, OpenAI Swarm, and Anthropic's Model Context Protocol (MCP).

## Table of Contents
1. [Installation](#installation)
2. [DSPy Integration](#dspy-integration)
3. [OpenAI Swarm Integration](#openai-swarm-integration)
4. [MCP Integration](#mcp-integration)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Installation

First, install Ember and the frameworks you want to integrate with:

```bash
# Install Ember
pip install ember-ai

# For DSPy integration
pip install dspy-ai

# For Swarm integration
pip install git+https://github.com/openai/swarm.git

# For MCP integration
pip install mcp
# Or for simplified setup
pip install fastmcp
```

## DSPy Integration

### Overview
DSPy integration allows you to use any Ember-supported model as a language model backend for declarative self-improving programs.

### Quick Start

```python
import dspy
from ember.integrations.dspy import EmberLM

# Initialize Ember backend
lm = EmberLM(model="claude-3-opus-20240229", temperature=0.7)
dspy.configure(lm=lm)

# Use with DSPy modules
classify = dspy.Predict("text -> sentiment")
result = classify(text="I love this integration!")
print(result.sentiment)  # Output: positive
```

### Advanced Usage

#### Model Switching
```python
# Dynamically switch models based on task
for model in ["gpt-3.5-turbo", "claude-3-haiku-20240307", "gpt-4"]:
    lm = EmberLM(model=model)
    dspy.configure(lm=lm)
    
    result = classify(text="Test sentiment")
    print(f"{model}: {result.sentiment}")
```

#### Optimization with Bootstrap
```python
from dspy.teleprompt import BootstrapFewShot

# Define metric
def accuracy_metric(example, prediction, trace=None):
    return prediction.sentiment.lower() == example.sentiment.lower()

# Optimize program
optimizer = BootstrapFewShot(metric=accuracy_metric)
optimized_classify = optimizer.compile(classify, trainset=train_data)
```

#### Cost Tracking
```python
# After running DSPy programs
metrics = lm.get_usage_metrics()
print(f"Total cost: ${metrics['total_cost']:.4f}")
print(f"Tokens used: {metrics['total_tokens']}")
```

### Complete Example
See `src/ember/integrations/dspy/examples/basic_usage.py` for comprehensive examples including:
- Basic prediction
- Chain of thought reasoning
- Multi-hop QA
- Model comparison
- Custom signatures

## OpenAI Swarm Integration

### Overview
Swarm integration enables using any Ember model in multi-agent systems with seamless handoffs and tool calling.

### Quick Start

```python
from swarm import Swarm, Agent
from ember.integrations.swarm import EmberSwarmClient

# Create Ember-backed client
client = EmberSwarmClient(default_model="claude-3-opus-20240229")
swarm = Swarm(client=client)

# Define agent with any Ember model
agent = Agent(
    name="Assistant",
    model="gpt-4",  # Can use any model in Ember's registry
    instructions="You are a helpful assistant.",
    functions=[my_function]
)

# Run conversation
response = swarm.run(
    agent=agent,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Multi-Agent Systems

#### Agent Handoffs
```python
def transfer_to_expert():
    """Transfer to expert agent."""
    return expert_agent

triage_agent = Agent(
    name="Triage",
    model="gpt-3.5-turbo",  # Fast, efficient
    instructions="Route questions to appropriate experts.",
    functions=[transfer_to_expert]
)

expert_agent = Agent(
    name="Expert",
    model="claude-3-opus-20240229",  # More capable
    instructions="Provide detailed expert answers."
)
```

#### Mixed Model Agents
```python
# Use different models for different agents
sales_agent = Agent(
    name="Sales",
    model="claude-3-haiku-20240307",
    instructions="Handle sales inquiries."
)

support_agent = Agent(
    name="Support",
    model="gpt-4",
    instructions="Resolve technical issues."
)
```

### Tool Calling
```python
def check_inventory(product: str) -> str:
    """Check product availability."""
    # Your logic here
    return f"{product} is in stock"

agent = Agent(
    name="Sales Assistant",
    model="gpt-4",
    instructions="Help customers with purchases.",
    functions=[check_inventory]
)
```

### Complete Example
See `src/ember/integrations/swarm/examples/multi_agent.py` for examples including:
- Customer service journey
- Parallel agent execution
- Dynamic model selection
- Cost optimization strategies

## MCP Integration

### Overview
MCP integration exposes Ember's capabilities as tools, resources, and prompts accessible from any MCP-compatible client like Claude Desktop.

### Quick Start (FastMCP)

```python
from fastmcp import FastMCP
from ember.api import models

mcp = FastMCP("ember-mcp")

@mcp.tool()
async def generate_text(prompt: str, model: str = "claude-3-haiku-20240307") -> str:
    """Generate text using Ember."""
    ember_model = models.get_model(model)
    response = await ember_model.agenerate(prompt)
    return response.content

# Run server
if __name__ == "__main__":
    mcp.run()
```

### Claude Desktop Configuration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "ember": {
      "command": "python",
      "args": ["-m", "ember.integrations.mcp.fastmcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key",
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

### Available Tools

#### Text Generation
```python
# In Claude Desktop after configuration
# User: "Use ember to write a haiku using GPT-4"
# Claude will call: ember_generate(prompt="Write a haiku", model="gpt-4")
```

#### Model Comparison
```python
# User: "Compare how GPT-4 and Claude explain quantum computing"
# Claude will call: ember_compare(
#     prompt="Explain quantum computing",
#     models=["gpt-4", "claude-3-opus-20240229"]
# )
```

#### Ensemble Voting
```python
# User: "Get consensus from multiple models on this code review"
# Claude will call: ember_ensemble(
#     prompt="Review this code: ...",
#     models=["gpt-4", "claude-3-opus", "gemini-pro"],
#     strategy="majority_vote"
# )
```

### Resources

MCP clients can access:
- `ember://models/list` - Available models
- `ember://models/costs` - Pricing information
- `ember://usage/current` - Usage statistics

### Complete Example
See `src/ember/integrations/mcp/examples/` for:
- Basic server setup
- FastMCP implementation
- Claude Desktop configuration
- Advanced tool examples

## Best Practices

### 1. Model Selection
- Use faster, cheaper models for simple tasks (e.g., `gpt-3.5-turbo`)
- Reserve powerful models for complex reasoning (e.g., `claude-3-opus`)
- Consider task-specific model strengths

### 2. Cost Management
```python
# Track costs across integrations
from ember.api import models

# Get session costs
costs = models.get_session_costs()
print(f"Total session cost: ${costs['total']:.4f}")
```

### 3. Error Handling
```python
try:
    result = ember_model.generate(prompt)
except Exception as e:
    logger.error(f"Model call failed: {e}")
    # Fallback logic
```

### 4. Performance Optimization
- Use streaming for long responses
- Batch similar requests
- Cache frequently used outputs
- Monitor latency metrics

### 5. Security
- Store API keys in environment variables
- Validate inputs before processing
- Implement rate limiting
- Log but don't expose sensitive data

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure frameworks are installed
pip install dspy-ai swarm mcp fastmcp
```

#### 2. API Key Issues
```bash
# Set environment variables
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

#### 3. Model Not Found
```python
# List available models
from ember.api import models
print(models.list_models())
```

#### 4. MCP Connection Issues
```bash
# Test MCP server directly
python -m ember.integrations.mcp.fastmcp_server --stdio
```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. Check the examples in `src/ember/integrations/*/examples/`
2. Review the design document: `docs/integrations/EMBER_INTEGRATIONS_DESIGN.md`
3. Open an issue on GitHub with:
   - Integration type (DSPy/Swarm/MCP)
   - Error message
   - Minimal reproducible example
   - Ember version

## Next Steps

1. Explore the example scripts for each integration
2. Experiment with different models and configurations
3. Build your own multi-agent systems and tools
4. Share your integrations with the community

For more advanced usage and architectural details, see the [design document](EMBER_INTEGRATIONS_DESIGN.md).