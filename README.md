# Ember

Build AI systems with the elegance of print("Hello World").

## Installation

### From PyPI

```bash
pip install ember-ai
```

### From Source (Development)

```bash
git clone https://github.com/pyember/ember.git
cd ember
uv sync
```

## Quick Setup

Run our interactive setup wizard for the best experience:

```bash
# If installed from PyPI
ember setup

# If running from source
uv run ember setup
```

This will:
- Help you choose an AI provider (OpenAI, Anthropic, or Google)
- Configure your API keys securely  
- Test your connection
- Save configuration to ~/.ember/config.yaml

## Getting Started

### 1. Set up API Keys

Ember can store API keys in multiple ways:

**Option 1: Environment Variables**
```bash
# For OpenAI (GPT-4, GPT-3.5)
export OPENAI_API_KEY="sk-..."

# For Anthropic (Claude models)
export ANTHROPIC_API_KEY="sk-ant-..."

# For Google (Gemini models)
export GOOGLE_API_KEY="..."
```

**Option 2: Configuration File** (Recommended)
```bash
# Run setup for each provider
ember setup  # Interactive wizard (if installed from PyPI)
# OR
uv run ember setup  # If running from source

# Or configure manually
ember configure set credentials.openai_api_key "sk-..."
ember configure set credentials.anthropic_api_key "sk-ant-..."
ember configure set credentials.google_api_key "..."
# OR (if running from source)
uv run ember configure set credentials.openai_api_key "sk-..."
uv run ember configure set credentials.anthropic_api_key "sk-ant-..."
uv run ember configure set credentials.google_api_key "..."
```

**Option 3: Runtime Context**
```python
from ember import context

with context.manager(credentials={"openai_api_key": "sk-..."}):
    response = models("gpt-4", "Hello!")
```

### 2. Verify Setup

```python
from ember.api import models

# Discover available models
print(models.list())  # Shows all available models
print(models.providers())  # Shows available providers

# Get detailed model information
info = models.discover()
for model_id, details in info.items():
    print(f"{model_id}: {details['description']} (context: {details['context_window']})")

# This will work if you have OPENAI_API_KEY set
response = models("gpt-4", "Hello, world!")
print(response)
```

If API keys are missing, you'll get a clear error message:
```
ModelProviderError: No API key available for model gpt-4. 
Please set via OPENAI_API_KEY environment variable.
```

If you use an unknown model name, you'll see available options:
```
ModelNotFoundError: Cannot determine provider for model 'claude-3'. 
Available models: claude-2.1, claude-3-haiku, claude-3-opus, claude-3-sonnet, 
gemini-pro, gemini-pro-vision, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-4, 
gpt-4-turbo, gpt-4o, gpt-4o-mini, ...
```

### 3. Choose Your Style: Strings or Constants

```python
from ember.api import models, Models

# Option 1: Direct strings (simple, works everywhere)
response = models("gpt-4", "Hello, world!")
response = models("claude-3-opus", "Write a haiku")

# Option 2: Constants for IDE autocomplete and typo prevention
response = models(Models.GPT_4, "Hello, world!")
response = models(Models.CLAUDE_3_OPUS, "Write a haiku")

# Both are exactly equivalent - Models.GPT_4 == "gpt-4"
```

## Quick Start

```python
from ember import models

# Direct LLM calls - no setup required
response = models("gpt-4", "Explain quantum computing in one sentence")
print(response)
```

### Available Models

Common model identifiers:
- **OpenAI**: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, `claude-2.1`
- **Google**: `gemini-pro`, `gemini-ultra`

Models are automatically routed to the correct provider based on their name.

## Core Patterns

### Context Management

Ember uses a unified context system for configuration and state management:

```python
from ember import context

# Get the current context
ctx = context.get()

# Temporary configuration overrides
with context.manager(models={"default": "gpt-4", "temperature": 0.9}) as ctx:
    # All operations in this block use these settings
    response = models("Hello")  # Uses gpt-4 with temperature 0.9
```

The context system provides:
- Thread-safe and async-safe configuration management
- Hierarchical configuration with proper isolation
- Clean scoping for temporary overrides

### Progressive Disclosure in APIs

Ember APIs follow a pattern of progressive disclosure - simple things are simple, complex things are possible:

```python
from ember import models

# Level 1: Simple one-off calls
response = models("gpt-4", "Hello world")

# Level 2: Reusable configured instances  
assistant = models.instance("gpt-4", temperature=0.7, system="You are helpful")
response = assistant("How do I center a div?")
```

This pattern appears throughout Ember:
- `models()` for quick calls, `models.instance()` for configured instances
- `operators.op` for simple functions, full `Operator` classes for complex needs
- Direct data access for simple cases, streaming pipelines for scale

## Core Concepts

Ember provides four primitives that compose into powerful AI systems:

### 1. Models - Direct LLM Access

```python
from ember import models

# Simple invocation with string model names
response = models("claude-3-opus", "Write a haiku about programming")

# Reusable configuration
assistant = models.instance("gpt-4", temperature=0.7, system="You are helpful")
response = assistant("How do I center a div?")

# Alternative: Use constants for autocomplete
from ember import Models
response = models(Models.CLAUDE_3_OPUS, "Write a haiku")
```

### 2. Operators - Composable AI Building Blocks

```python
from ember import operators

# Transform any function into an AI operator
@operators.op
def summarize(text: str) -> str:
    return models("gpt-4", f"Summarize in one sentence: {text}")

@operators.op
def translate(text: str, language: str = "Spanish") -> str:
    return models("gpt-4", f"Translate to {language}: {text}")

# Compose operators naturally
pipeline = summarize >> translate
result = pipeline("Long technical article...")
```

### 3. Data - Streaming-First Data Pipeline

```python
from ember.api import data

# Stream data efficiently
for example in data.stream("mmlu"):
    answer = models("gpt-4", example["question"])
    print(f"Q: {example['question']}")
    print(f"A: {answer}")

# Chain transformations
results = (data.stream("gsm8k")
    .filter(lambda x: x["difficulty"] > 7)
    .transform(preprocess)
    .batch(32))
```

### 4. XCS - Zero-Config Optimization

```python
from ember import xcs

# Automatic JIT compilation
@xcs.jit
def process_batch(items):
    return [models("gpt-4", item) for item in items]

# Automatic parallelization
fast_process = xcs.vmap(process_batch)
results = fast_process(large_dataset)  # Runs in parallel
```

## Real-World Examples

### Building a Code Reviewer

```python
from ember import models, operators

@operators.op
def review_code(code: str) -> dict:
    """AI-powered code review"""
    prompt = f"""Review this code for:
    1. Bugs and errors
    2. Performance issues
    3. Best practices
    
    Code:
    {code}
    """
    
    review = models("claude-3", prompt)
    
    # Extract structured feedback
    return {
        "summary": models("gpt-4", f"Summarize in one line: {review}"),
        "issues": review,
        "severity": models("gpt-4", f"Rate severity 1-10: {review}")
    }

# Use directly
feedback = review_code("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""")
```

### Parallel Document Processing

```python
from ember import models, xcs, data

# Define processing pipeline
@xcs.jit
def analyze_document(doc: dict) -> dict:
    # Extract key information
    summary = models("gpt-4", f"Summarize: {doc['content']}")
    entities = models("gpt-4", f"Extract entities: {doc['content']}")
    sentiment = models("gpt-4", f"Analyze sentiment: {doc['content']}")
    
    return {
        "id": doc["id"],
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment
    }

# Process documents in parallel
documents = data.stream("research_papers").first(1000)
results = xcs.vmap(analyze_document)(documents)

# Results computed optimally with automatic parallelization
```

### Multi-Model Ensemble

```python
from ember import models, operators

@operators.op
def consensus_answer(question: str) -> str:
    """Get consensus from multiple models"""
    # Query different models
    gpt4_answer = models("gpt-4", question)
    claude_answer = models("claude-3-opus", question) 
    gemini_answer = models("gemini-pro", question)
    
    # Synthesize consensus
    synthesis_prompt = f"""
    Question: {question}
    
    Model answers:
    - GPT-4: {gpt4_answer}
    - Claude: {claude_answer}
    - Gemini: {gemini_answer}
    
    Synthesize the best answer combining insights from all models.
    """
    
    return models("gpt-4", synthesis_prompt)

# Use for critical decisions
answer = consensus_answer("What's the best approach to distributed systems?")
```

## Command Line Interface

Ember provides a comprehensive CLI for setup, configuration, and introspection.

**Note:** If you installed Ember from PyPI, use `ember` directly. If running from source, prefix commands with `uv run`.

### Setup and Configuration

```bash
# Interactive setup wizard (recommended for first-time setup)
ember setup   # or: uv run ember setup

# Test your API connection
ember test   # or: uv run ember test
ember test --model claude-3-opus

# Configuration management
ember configure get models.default              # Get a config value
ember configure set models.default "gpt-4"     # Set a config value
ember configure list                            # Show all configuration
ember configure show credentials               # Show specific section

# Version information
ember version   # or: uv run ember version
```

### Introspection Commands

```bash
# Context introspection
ember context view                              # View current configuration
ember context view --format json                # Output as JSON
ember context view --filter models              # Show only models config
ember context validate                          # Validate configuration

# Registry introspection  
ember registry list-models                      # List available models
ember registry list-models --provider openai    # Filter by provider
ember registry list-models --verbose            # Detailed information
ember registry list-providers                   # Show provider status
ember registry info gpt-4                       # Detailed model info
```

### Advanced Configuration

The context system supports multiple configuration sources with priority:

1. **Runtime context** (highest priority)
2. **Environment variables** 
3. **Configuration file** (~/.ember/config.yaml)
4. **Defaults** (lowest priority)

```python
from ember import context

# Use context manager for temporary overrides
with context.manager(
    models={"default": "gpt-3.5-turbo", "temperature": 0.0},
    credentials={"openai_api_key": "dev-key"}
) as dev_ctx:
    # Development operations here
    pass

with context.manager(
    models={"default": "gpt-4", "temperature": 0.7},
    credentials={"openai_api_key": "prod-key"}
) as prod_ctx:
    # Production operations here
    response = models("Production query")
```

## Advanced Features

### Type-Safe Operators

```python
from ember import Operator
from pydantic import BaseModel

class CodeInput(BaseModel):
    language: str
    code: str

class CodeOutput(BaseModel):
    is_valid: bool
    errors: list[str]
    suggestions: list[str]

class CodeValidator(Operator):
    input_spec = CodeInput
    output_spec = CodeOutput
    
    def call(self, input: CodeInput) -> CodeOutput:
        prompt = f"Validate this {input.language} code: {input.code}"
        result = models("gpt-4", prompt)
        # Automatic validation against output_spec
        return CodeOutput(...)
```

### Custom Data Loaders

```python
from ember.api import data

# Register custom dataset
@data.register("my-dataset")
def load_my_data():
    with open("data.jsonl") as f:
        for line in f:
            yield json.loads(line)

# Use like built-in datasets
for item in data.stream("my-dataset"):
    process(item)
```

### Performance Profiling

```python
from ember import xcs

# Automatic profiling
with xcs.profile() as prof:
    results = expensive_operation()

print(prof.report())
# Shows execution time, parallelism achieved, bottlenecks
```

## Design Principles

1. **Simple by Default** - Basic usage requires no configuration
2. **Progressive Disclosure** - Complexity available when needed
3. **Composition Over Configuration** - Build complex from simple
4. **Performance Without Sacrifice** - Fast by default, no manual tuning

## Architecture

Ember uses a registry-based architecture with four main components:

- **Model Registry** - Manages LLM providers and connections
- **Operator System** - Composable computation units with JAX integration
- **Data Pipeline** - Streaming-first data loading and transformation
- **XCS Engine** - Automatic optimization and parallelization

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Development

```bash
# Clone and install development dependencies
git clone https://github.com/pyember/ember.git
cd ember
uv sync --all-extras

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Benchmarks
uv run python -m benchmarks.suite
```

## Contributing

We welcome contributions that align with Ember's philosophy of simplicity and power. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Ember is inspired by the engineering excellence of:
- JAX's functional transformations
- PyTorch's intuitive API
- Langchain's comprehensive features (but simpler)
- The Unix philosophy of composable tools

Built with principles from leaders who shaped modern computing.