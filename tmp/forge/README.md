# Forge 🔨

A powerful coding assistant that orchestrates multiple AI models for optimal results.

## Why Forge?

Traditional coding assistants use a single AI model for all tasks. Forge intelligently routes requests to the best model for each specific task:

- **Tool Usage** → OpenAI (mature function calling)
- **Planning & Architecture** → Anthropic Claude (superior reasoning)  
- **Code Generation** → Anthropic Claude (higher quality code)
- **Critical Decisions** → Ensemble (multiple models vote)

## Installation

```bash
# Clone and build
git clone https://github.com/ember-ai/forge
cd forge
npm install
npm run build
npm link

# Or install globally (when published)
npm install -g @ember-ai/forge
```

## Quick Start

```bash
# Basic usage
forge "create a Python web server"

# Specific provider
forge --provider anthropic "explain this architecture"

# Ensemble for critical operations  
forge --ensemble gpt-4 claude-3-opus "refactor this legacy codebase"

# Debug mode to see routing decisions
forge --debug "implement user authentication"
```

## Configuration

Forge can be configured via `~/.forge/config.yaml`:

```yaml
providers:
  default: openai
  
  routing:
    tool_use: openai      # Best for function calling
    planning: anthropic   # Superior reasoning
    code_gen: anthropic   # Better code quality
    synthesis: ensemble   # Multiple perspectives
    
  models:
    openai:
      default: gpt-4
    anthropic:
      default: claude-3-opus

ensembles:
  default:
    models: [gpt-4, claude-3-opus]
    strategy: majority  # or: unanimous, best_of
```

Initialize configuration:
```bash
forge config --init
```

## Features

### 🧠 Intelligent Routing
Automatically selects the best model based on task type:
- Commands with file operations → OpenAI
- Strategic planning → Claude
- Code writing → Claude
- Safety-critical operations → Ensemble voting

### 🔧 Full Codex Compatibility
Drop-in replacement for OpenAI Codex:
- Same tool interface (shell, file operations)
- Compatible with existing workflows
- Enhanced with multi-model capabilities

### 💰 Cost Optimization
- Routes to cheaper models when appropriate
- Tracks token usage and costs
- Configurable limits and alerts

### 🛡️ Safety Features
- Ensemble voting for destructive operations
- Command confirmation with explanations
- Blocked command patterns

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   CLI       │────▶│ Ember Bridge │────▶│ Model Providers │
└─────────────┘     └──────────────┘     └─────────────────┘
                            │                      │
                            ▼                      ▼
                    ┌──────────────┐      ┌─────────────┐
                    │   Router     │      │   OpenAI    │
                    └──────────────┘      ├─────────────┤
                            │             │  Anthropic  │
                            ▼             ├─────────────┤
                    ┌──────────────┐      │  Ensemble   │
                    │ Ember Models │      └─────────────┘
                    └──────────────┘
```

## Advanced Usage

### Custom Routing

Create `.forge/config.yaml` in your project:

```yaml
providers:
  routing:
    # Route Python files to Claude
    "*.py": anthropic
    # Route critical files to ensemble
    "*/production/*": ensemble
    # Everything else to GPT-4
    default: openai
```

### Programmatic Usage

```typescript
import { createForgeClient } from '@ember-ai/forge';

const client = createForgeClient({
  router: new ProviderRouter({
    tool_use: 'openai',
    planning: 'anthropic'
  })
});

// Use exactly like OpenAI client
const completion = await client.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello' }]
});
```

## Environment Variables

```bash
# API Keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Feature Flags
export FORGE_AUTO_ROUTING=true
export FORGE_DEBUG=true
export FORGE_DEFAULT_PROVIDER=anthropic
```

## Comparison with Codex

| Feature | Codex | Forge |
|---------|-------|-------|
| Shell commands | ✅ | ✅ |
| File operations | ✅ | ✅ |
| Single model | ✅ | ❌ |
| Multi-model routing | ❌ | ✅ |
| Ensemble decisions | ❌ | ✅ |
| Cost tracking | ❌ | ✅ |
| Provider flexibility | ❌ | ✅ |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with ❤️ by the Ember team