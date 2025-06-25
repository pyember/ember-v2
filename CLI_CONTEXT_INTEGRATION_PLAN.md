# CLI-Context Integration Plan

## Current State Analysis

### What exists:
1. **Context System**: EmberContext provides centralized configuration
2. **CLI**: Basic commands (setup, version, models, test) 
3. **Setup Wizard**: npm-based interactive setup
4. **Credential Manager**: Stores API keys persistently

### What's missing:
1. CLI commands don't use EmberContext
2. No `ember configure` command for runtime config
3. Setup wizard writes directly to ~/.ember/credentials
4. No unified configuration flow
5. Each CLI invocation starts fresh (no context persistence)

## Design Principles (Jeff Dean/Sanjay Ghemawat Style)

1. **Single source of truth**: All configuration flows through EmberContext
2. **Zero duplication**: No parallel configuration systems
3. **Clean abstractions**: CLI commands are thin wrappers over context operations
4. **Persistence strategy**: Context state persists between CLI invocations
5. **Progressive disclosure**: Simple commands for common cases, advanced for power users

## Architecture

```
CLI Entry Point (main.py)
    ↓
Initialize EmberContext (shared)
    ↓
Command Handler
    ↓
Context Operations (get/set config, credentials, etc.)
    ↓
Persist Changes (if needed)
```

## Implementation Tasks

### 1. CLI Context Initialization
- Create context at CLI startup
- Load persisted state from ~/.ember/config.yaml
- Pass context to all commands

### 2. Add `ember configure` Command
```bash
ember configure set openai.api_key "sk-..."
ember configure get openai.api_key
ember configure list
ember configure show
```

### 3. Update Existing Commands
- `ember setup`: Use context for credential storage
- `ember test`: Use context for model access
- `ember models`: Use context for discovery

### 4. Context Persistence
- Save context state to ~/.ember/config.yaml
- Merge with existing credentials system
- Handle concurrent access safely

### 5. Setup Wizard Integration
- Update TypeScript code to use `ember configure` internally
- Or provide Python endpoint for configuration

## Code Changes Required

### 1. CLI Main Entry
```python
# src/ember/cli/main.py
def main():
    # Initialize context early
    ctx = EmberContext.current()
    
    # Pass to commands
    args.context = ctx
```

### 2. Configure Command
```python
# src/ember/cli/commands/configure.py
def cmd_configure(args, ctx):
    if args.action == "set":
        ctx.set_config(args.key, args.value)
        ctx.save()  # Persist
    elif args.action == "get":
        print(ctx.get_config(args.key))
```

### 3. Context Persistence
```python
# src/ember/_internal/context.py
def save(self):
    """Save context state to disk."""
    config_dir = Path.home() / ".ember"
    config_file = config_dir / "config.yaml"
    
    # Atomic write
    with tempfile.NamedTemporaryFile(...) as tmp:
        yaml.dump(self._config, tmp)
        tmp.replace(config_file)
```

## Testing Strategy

1. **Unit tests**: Each command with mocked context
2. **Integration tests**: Full CLI flow with real context
3. **Concurrent access**: Multiple CLI instances
4. **Migration tests**: Existing credentials preserved

## Migration Path

1. Detect existing ~/.ember/credentials
2. Import into context on first run
3. Maintain backward compatibility
4. Gradual deprecation of old system

## Success Criteria

1. All CLI commands use EmberContext
2. Configuration persists between invocations
3. Setup wizard integrates cleanly
4. No duplicate configuration systems
5. Clean, simple API following YAGNI

## Timeline

1. Phase 1: Add context to CLI main (30 min)
2. Phase 2: Implement configure command (1 hour)
3. Phase 3: Update existing commands (1 hour)
4. Phase 4: Add persistence (1 hour)
5. Phase 5: Setup wizard integration (2 hours)
6. Phase 6: Testing & polish (1 hour)

Total: ~6 hours of focused work