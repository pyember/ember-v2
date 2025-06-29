"""Context Management - Managing state and configuration in AI applications.

Learn how to effectively manage context, configuration, and state across
your AI application components using Ember's context system.

Example:
    >>> from ember.context import context
    >>> ctx = context.get()
    >>> api_key = ctx.get_credential("openai", "OPENAI_API_KEY")
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def example_basic_context():
    """Show basic context usage."""
    print("\n=== Basic Context Management ===\n")

    from ember.context import context

    # Get current context (creates if needed)
    ctx = context.get()

    print("Getting current context:")
    print("  ctx = context.get()")
    print("  Thread-safe: Yes")
    print("  Isolated: Yes")
    print("  Auto-configures from environment\n")

    # Access components through context
    print("Accessing components:")
    print("  credentials = ctx.get_credential('provider', 'key')")
    print("  config = ctx.get_config('models.default')")
    print("  • Manages API keys and configuration")


def example_configuration_context():
    """Demonstrate configuration management."""
    print("\n\n=== Configuration in Context ===\n")

    from ember.context import context

    # Show context configuration pattern
    print("Configuration with context manager:")
    print("  with context.manager(models={'default': 'gpt-4'}) as ctx:")
    print("      # All operations use gpt-4 in this scope")
    print("      response = models('Hello')  # Uses gpt-4")
    print()

    print("Configuration access:")
    print("  # Get configuration values")
    print("  model = context.get().get_config('models.default')")
    print("  temp = context.get().get_config('models.temperature', 0.7)")
    print()

    print("Context manages configuration:")
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

    from ember.context import context
    import threading

    print("Context isolation ensures:")
    print("  • Each context has its own state")
    print("  • No interference between contexts")
    print("  • Thread-safe operations")
    print("  • Clean testing environments\n")

    # Demonstrate isolation
    print("Example: Multiple contexts with manager")
    print("  with context.manager(models={'default': 'gpt-4'}) as ctx1:")
    print("      # Production context")
    print("  with context.manager(models={'default': 'gpt-3.5'}) as ctx2:")
    print("      # Testing context\n")

    print("Each context maintains:")
    print("  • Separate configuration")
    print("  • Independent state")
    print("  • Isolated credentials")
    print("  • Own settings")


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
    print("    def __init__(self, ctx):")
    print("        self.ctx = ctx")
    print("        self.config = ctx.get_config('models.default')\n")

    print("# Get current context")
    print("ctx = context.get()")
    print()
    print("# Initialize services with same context")
    print("classifier = TextClassifier(ctx)")
    print("summarizer = TextSummarizer(ctx)")
    print("analyzer = SentimentAnalyzer(ctx)")
    print()
    print("Benefits:")
    print("  • Shared configuration")
    print("  • Consistent settings")
    print("  • Unified state management")
    print("  • Centralized resource access")


def example_context_lifecycle():
    """Demonstrate context lifecycle management."""
    print("\n\n=== Context Lifecycle ===\n")

    print("Context lifecycle stages:")
    print()

    print("1. Access:")
    print("   ctx = context.get()")
    print("   → Gets current context")
    print("   → Auto-creates if needed")
    print("   → Thread-safe access\n")

    print("2. Usage:")
    print("   config = ctx.get_config('key')")
    print("   cred = ctx.get_credential('provider', 'key')")
    print("   → Access configuration")
    print("   → Manage credentials\n")

    print("3. Scoped overrides:")
    print("   with context.manager(**overrides) as ctx:")
    print("       # Use modified context")
    print("   → Temporary changes")
    print("   → Automatic restoration\n")

    print("Context manager pattern:")
    print("  with context.manager(models={'default': 'gpt-4'}) as ctx:")
    print("      # All operations use overridden config")
    print("      # Automatic cleanup on exit")


def example_advanced_patterns():
    """Show advanced context patterns."""
    print("\n\n=== Advanced Context Patterns ===\n")

    print("1. Nested Context Managers:")
    print("   with context.manager(models={'default': 'gpt-4'}) as ctx1:")
    print("       with context.manager(models={'temperature': 0.2}) as ctx2:")
    print("           # Nested configuration overrides\n")

    print("2. Configuration Access:")
    print("   from ember.context import get_config, set_config")
    print("   value = get_config('models.default')")
    print("   set_config('models.temperature', 0.7)\n")

    print("3. Global Context:")
    print("   from ember.context import context")
    print("   ctx = context.get()")
    print("   → Thread-local context access\n")

    print("4. Testing Pattern:")
    print("   with context.manager(test_mode=True) as ctx:")
    print("       # Run tests with isolated context")
    print("   → Clean test environment")


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

    print("\n" + "=" * 50)
    print("✅ Context Best Practices")
    print("=" * 50)
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
