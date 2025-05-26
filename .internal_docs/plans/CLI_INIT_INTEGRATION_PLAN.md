# CLI Integration with Simplified Initialization

## Current CLI Analysis

### Good News: Already Aligned ✓
The CLI already follows many of our simplification principles:
- **Direct imports**: Uses `from ember.api import models, data, operators`
- **No explicit init**: Commands work without initialization calls
- **Environment-based**: Relies on environment variables for API keys

### Current CLI Commands
1. **`ember invoke`** - Direct model calls
2. **`ember model`** - Model management (list, info, cost)
3. **`ember eval`** - Run evaluations
4. **`ember project`** - Project scaffolding
5. **`ember config`** - Configuration management

## Required CLI Updates

### 1. Update Imports
```python
# Before
from ember.api import models, data, operators

# After (with new simplified init)
import ember
from ember import models, data, operators  # Via __getattr__
```

### 2. Simplify Discovery Commands
```python
# Current (relies on auto-discovery)
@model.command(name="list")
def list_models():
    available = models.available()  # This does discovery
    
# Updated (use simple registry)
@model.command(name="list")
def list_models():
    # Get from simplified registry without discovery
    from ember.core.context.unified_context import current_context
    registry = current_context().model_registry
    available = registry.list_models()
```

### 3. Simplify Config Command
```python
# Current (complex YAML management)
@cli.group()
def config():
    """Manage Ember configuration"""
    # Complex config file handling

# Updated (environment-focused)
@cli.group()
def config():
    """View and set Ember configuration"""
    # Simple env var management
    
@config.command()
def show():
    """Show current configuration"""
    from ember.core.config.simple_config import _config
    for key, value in _config.to_dict().items():
        click.echo(f"{key}: {value}")

@config.command()
def set(key: str, value: str):
    """Set configuration value"""
    ember.configure(**{key: value})
    click.echo(f"Set {key} = {value}")
```

### 4. Update Project Templates
```python
# Update generated project files to use simplified init
def create_project_files(name: str):
    # main.py template
    main_content = '''
import ember
from ember import models, operators
from ember.xcs import jit

# No initialization needed! Just use ember directly
response = models("gpt-4", "Hello from {name}!")
print(response)
'''.format(name=name)
```

### 5. Add Migration Command
```python
@cli.command()
def migrate():
    """Migrate old Ember configuration to new format"""
    # Check for old config.yaml
    if os.path.exists("config.yaml"):
        click.echo("Found config.yaml - migrating to environment variables...")
        # Parse YAML and suggest environment variables
        with open("config.yaml") as f:
            old_config = yaml.safe_load(f)
        
        click.echo("\nAdd these to your .env or shell:")
        if "openai" in old_config.get("providers", {}):
            click.echo("export OPENAI_API_KEY=<your-key>")
        # etc...
```

## Environment Variable Integration

### Current CLI Behavior
- Already checks for API keys in environment
- Shows warnings if keys are missing
- Good error messages

### Enhanced for Simplified Init
```python
# Add to CLI startup
def check_environment():
    """Verify environment is properly configured"""
    from ember.core.config.simple_config import get_api_key
    
    warnings = []
    if not get_api_key("openai"):
        warnings.append("OPENAI_API_KEY not set")
    if not get_api_key("anthropic"):  
        warnings.append("ANTHROPIC_API_KEY not set")
    
    if warnings:
        click.echo("⚠️  Missing API keys:", err=True)
        for w in warnings:
            click.echo(f"  - {w}", err=True)
        click.echo("\nSet them with: export OPENAI_API_KEY=sk-...", err=True)
```

## Benefits for CLI Users

1. **Faster Startup** - No auto-discovery delays
2. **Simpler Commands** - Less configuration complexity
3. **Better Errors** - Clear messages about missing API keys
4. **Easier Debugging** - No hidden initialization magic

## Implementation Priority

1. **High Priority**
   - Update imports in all commands
   - Fix model list command (no discovery)
   - Update error messages

2. **Medium Priority**
   - Simplify config command
   - Add migration helper
   - Update project templates

3. **Low Priority**
   - Remove YAML config commands
   - Update documentation
   - Add environment checker

## Testing the Integration

```bash
# Test basic commands work
ember invoke gpt-4 "test"
ember model list
ember eval run gpt-4 mmlu

# Test configuration
ember config show
ember config set retry_count 5

# Test project creation
ember project new my-app
cd my-app && python main.py
```

## Conclusion

The CLI is already well-positioned for the simplified initialization. Most commands will work with minimal changes, and the user experience will actually improve with faster startup and clearer configuration.