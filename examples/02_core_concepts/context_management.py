"""Context Management - Managing state and configuration in AI applications.

Learn how to effectively manage context, configuration, and state across
your AI application components using Ember's context system.

Example:
    >>> from ember.context import get_context
    >>> ctx = get_context()
    >>> ctx.get_model("gpt-4")
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def example_basic_context():
    """Show basic context usage."""
    print("\n=== Basic Context Management ===\n")
    
    from ember.context import get_context, create_context
    
    # Get current context (creates if needed)
    ctx = get_context()
    
    print("Getting current context:")
    print("  ctx = get_context()")
    print("  Thread-safe: Yes")
    print("  Isolated: Yes")
    print("  Auto-configures from environment\n")
    
    # Access components through context
    print("Accessing components:")
    print("  model_registry = ctx.model_registry")
    print("  data_registry = ctx.data_registry")
    print("  metrics = ctx.metrics")


def example_configuration_context():
    """Demonstrate configuration management."""
    print("\n\n=== Configuration in Context ===\n")
    
    from ember.context import get_context, create_context
    
    # Create context with custom config
    config = {
        "models": {
            "default": "gpt-3.5-turbo",
            "temperature": 0.7
        },
        "cache": {
            "enabled": True,
            "ttl": 3600
        }
    }
    
    print("Custom configuration:")
    print("  config = {")
    print("      'models': {'default': 'gpt-3.5-turbo', 'temperature': 0.7},")
    print("      'cache': {'enabled': True, 'ttl': 3600}")
    print("  }")
    
    # Create context with custom configuration
    ctx = create_context(**config)
    
    print("\nContext manages configuration:")
    print("  • Validates configuration schema")
    print("  • Provides type-safe access")
    print("  • Supports environment variables")
    print("  • Handles defaults gracefully")


def example_model_context():
    """Show model management through context."""
    print("\n\n=== Model Management in Context ===\n")
    
    print("Model access patterns:")
    print()
    
    # Mock model access
    print("1. Get default model:")
    print("   model = ctx.get_model()")
    print("   → Returns configured default model\n")
    
    print("2. Get specific model:")
    print("   model = ctx.get_model('gpt-4')")
    print("   → Returns requested model if available\n")
    
    print("3. List available models:")
    print("   models = ctx.list_models()")
    print("   → ['gpt-3.5-turbo', 'gpt-4', 'claude-3-opus', ...]\n")
    
    print("4. Model with custom settings:")
    print("   model = ctx.get_model('gpt-4', temperature=0.2)")
    print("   → Model instance with overridden parameters")


def example_data_context():
    """Demonstrate data management in context."""
    print("\n\n=== Data Management in Context ===\n")
    
    print("Data access through context:")
    print()
    
    print("1. Load dataset:")
    print("   dataset = ctx.load_dataset('mmlu')")
    print("   → Loads and caches dataset\n")
    
    print("2. List available datasets:")
    print("   datasets = ctx.list_datasets()")
    print("   → ['mmlu', 'gsm8k', 'humaneval', ...]\n")
    
    print("3. Register custom dataset:")
    print("   ctx.register_dataset('my_data', loader_func)")
    print("   → Makes dataset available in context\n")
    
    print("4. Dataset with transforms:")
    print("   dataset = ctx.load_dataset('mmlu', transform=preprocess)")
    print("   → Applies transformation pipeline")


def example_context_isolation():
    """Show context isolation and thread safety."""
    print("\n\n=== Context Isolation ===\n")
    
    from ember.context import create_context
    import threading
    
    print("Context isolation ensures:")
    print("  • Each context has its own state")
    print("  • No interference between contexts")
    print("  • Thread-safe operations")
    print("  • Clean testing environments\n")
    
    # Demonstrate isolation
    print("Example: Multiple contexts")
    print("  ctx1 = create_context()  # Production")
    print("  ctx2 = create_context()  # Testing")
    print("  ctx3 = create_context()  # Development\n")
    
    print("Each context maintains:")
    print("  • Separate model instances")
    print("  • Independent caches")
    print("  • Isolated metrics")
    print("  • Own configuration")


def example_context_sharing():
    """Show how to share context across components."""
    print("\n\n=== Sharing Context ===\n")
    
    # Define a service that uses context
    class AIService:
        def __init__(self, context):
            self.ctx = context
            self.model = None
        
        def initialize(self):
            # Get model from context
            self.model = "model from context"
            return True
        
        def process(self, text: str) -> str:
            # Use context resources
            return f"Processed: {text}"
    
    print("Sharing context across services:")
    print()
    print("class AIService:")
    print("    def __init__(self, context):")
    print("        self.ctx = context")
    print("        self.model = self.ctx.get_model()\n")
    
    print("# Create shared context")
    print("ctx = get_context()")
    print()
    print("# Initialize services with same context")
    print("classifier = TextClassifier(ctx)")
    print("summarizer = TextSummarizer(ctx)")
    print("analyzer = SentimentAnalyzer(ctx)")
    print()
    print("Benefits:")
    print("  • Shared configuration")
    print("  • Consistent model access")
    print("  • Unified metrics collection")
    print("  • Centralized resource management")


def example_context_lifecycle():
    """Demonstrate context lifecycle management."""
    print("\n\n=== Context Lifecycle ===\n")
    
    print("Context lifecycle stages:")
    print()
    
    print("1. Creation:")
    print("   ctx = get_context()")
    print("   → Initializes registries")
    print("   → Loads configuration")
    print("   → Sets up metrics\n")
    
    print("2. Usage:")
    print("   model = ctx.get_model()")
    print("   data = ctx.load_dataset('mmlu')")
    print("   → Resources loaded on demand")
    print("   → Automatic caching\n")
    
    print("3. Cleanup:")
    print("   ctx.close()  # or use context manager")
    print("   → Releases resources")
    print("   → Flushes metrics")
    print("   → Clears caches\n")
    
    print("Context manager pattern:")
    print("  with create_context() as ctx:")
    print("      model = ctx.get_model()")
    print("      # Automatic cleanup on exit")


def example_advanced_patterns():
    """Show advanced context patterns."""
    print("\n\n=== Advanced Context Patterns ===\n")
    
    print("1. Context Inheritance:")
    print("   base_ctx = create_context(**base_config)")
    print("   dev_ctx = create_context(parent=base_ctx, **dev_overrides)")
    print("   → Child inherits parent configuration\n")
    
    print("2. Context Decorators:")
    print("   @with_context")
    print("   def my_function(ctx, data):")
    print("       model = ctx.get_model()")
    print("       return model.process(data)\n")
    
    print("3. Global Context:")
    print("   from ember.context import get_context")
    print("   ctx = get_context()")
    print("   → Singleton for simple scripts\n")
    
    print("4. Testing Context:")
    print("   from ember.testing import TestContext")
    print("   ctx = TestContext(mock_models=True)")
    print("   → Isolated context for tests")


def main():
    """Run all context management examples."""
    print_section_header("Context Management")
    
    print("🎯 Context Management in Ember:\n")
    print("• Centralized configuration and state")
    print("• Thread-safe resource access")
    print("• Isolated environments")
    print("• Lifecycle management")
    print("• Dependency injection")
    
    example_basic_context()
    example_configuration_context()
    example_model_context()
    example_data_context()
    example_context_isolation()
    example_context_sharing()
    example_context_lifecycle()
    example_advanced_patterns()
    
    print("\n" + "="*50)
    print("✅ Context Best Practices")
    print("="*50)
    print("\n1. Create context early in application lifecycle")
    print("2. Share context instead of creating multiple")
    print("3. Use context managers for automatic cleanup")
    print("4. Leverage context for dependency injection")
    print("5. Keep context immutable after initialization")
    print("6. Use child contexts for testing/experimentation")
    print("7. Access all resources through context")
    
    print("\n🔧 Common Patterns:")
    print("• Application context: Single shared context")
    print("• Request context: Per-request isolation")
    print("• Test context: Isolated test environment")
    print("• Worker context: Per-worker in parallel processing")
    
    print("\nNext: Learn about error handling in 'error_handling.py'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())