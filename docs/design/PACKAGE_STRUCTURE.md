# Ember Package Structure

## Repository Layout

```
ember/                          # Repository root
├── src/
│   └── ember/                  # Core library (distributed via pip)
│       ├── _internal/          # Internal implementation
│       ├── api/                # Public API
│       ├── models/             # Model management
│       ├── onboard/            # Interactive onboarding
│       ├── operators/          # Operator system
│       ├── utils/              # Utilities
│       └── xcs/                # Optimization system
├── examples/                   # Example code (NOT distributed)
│   ├── 01_getting_started/
│   ├── 02_core_concepts/
│   └── ...
├── integrations/               # Framework integrations (NOT distributed)
│   ├── dspy/
│   ├── mcp/
│   └── swarm/
├── tests/                      # Test suite
├── docs/                       # Documentation
├── pyproject.toml              # Package configuration
└── setup.py                    # Modern setup delegating to pyproject.toml
```

## What Gets Installed

When users run `pip install ember-ai`, they get ONLY:
- `ember._internal` - Core infrastructure
- `ember.api` - Public API
- `ember.models` - Model management
- `ember.onboard` - Onboarding experience
- `ember.operators` - Operator system
- `ember.utils` - Utilities
- `ember.xcs` - Optimization

They do NOT get:
- Examples (550KB+ of tutorial code)
- Integrations (framework-specific adapters)
- Tests
- Documentation

## Why This Structure?

### 1. **Lean Installation**
- Users only download what they'll actually import
- No bloat from examples or optional integrations
- Faster installation and smaller footprint

### 2. **Clear Boundaries**
```python
# This makes sense:
from ember.operators import Operator

# This would never make sense (and isn't possible):
from ember.examples.chatbot import something
```

### 3. **Flexible Integrations**
Future installation options:
```bash
pip install ember-ai              # Core only
pip install ember-ai[dspy]        # Core + DSPy integration
pip install ember-ai[all]         # Everything
```

### 4. **Repository Organization**
- Library code in `src/ember/`
- Examples at root level for easy access
- Integrations separate for optional features
- Clear separation of concerns

## For Developers

### Running Examples
```bash
# Clone the repo
git clone https://github.com/anthropics/ember
cd ember

# Install ember
pip install -e .

# Run examples
python examples/01_getting_started/hello_world.py
```

### Using Integrations
```bash
# Currently: Import directly (requires repo clone)
from integrations.dspy import EmberLM

# Future: Install as extra
pip install ember-ai[dspy]
from ember.integrations.dspy import EmberLM
```

## Design Philosophy

This structure follows best practices from:
- **TensorFlow/JAX**: Examples in separate directories
- **FastAPI**: Minimal core, examples in docs/repo
- **Django**: Clear separation of core/contrib/examples
- **Go**: Examples never in the import path

The masters (Carmack, Uncle Bob, Pike, etc.) would approve:
- Clean separation of library vs documentation
- No mixing of concerns
- Efficient distribution
- Clear user expectations