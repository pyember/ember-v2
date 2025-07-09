"""Context Management - Managing state and configuration in AI applications.

Learn how to effectively manage context, configuration, and state across
your AI application components using Ember's context system.

Example:
    >>> from ember.context import get_context
    >>> ctx = get_context()
    >>> ctx.get_model("gpt-4")
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def show_the_problem():
    """Demonstrate why context management matters."""
    print("\n=== The Problem: Direct Usage Doesn't Scale ===\n")
    
    print("🤔 Let's say you start simple...")
    print()
    
    # Show the naive approach
    print("# Simple approach that everyone starts with:")
    print("from ember.api import models")
    print()
    print("def analyze_text(text):")
    print("    model = models('gpt-4', temperature=0.7)")
    print("    return model(f'Analyze: {text}').text")
    print()
    print("result = analyze_text('Some text')")
    print()
    
    print("✅ This works great... until you need:")
    print("  • Different settings for dev vs production")
    print("  • Testing with mock models")
    print("  • Multiple API keys or providers")
    print("  • Sharing configuration across components")
    print("  • Different models per environment")
    print()
    
    print("❌ Problems with direct usage:")
    print("  • Hard-coded configuration scattered everywhere")
    print("  • No way to easily switch environments")
    print("  • Testing requires changing production code")
    print("  • Configuration duplication")
    print("  • No centralized resource management")


def show_basic_solution():
    """Show how context solves the basic problem."""
    print("\n\n=== The Solution: Context Management ===\n")
    
    print("🎯 Context centralizes configuration and state:")
    print()
    
    # Mock context API for demonstration
    class MockContext:
        def __init__(self, config):
            self.config = config
        
        def get_model(self, name=None, **overrides):
            model_config = self.config.get('models', {})
            if name:
                return f"Model({name}, {model_config})"
            else:
                default = model_config.get('default', 'gpt-3.5-turbo')
                return f"Model({default}, {model_config})"
    
    # Show the context approach
    print("# Context approach:")
    print("from ember.context import get_context")
    print()
    print("def analyze_text(text):")
    print("    ctx = get_context()")
    print("    model = ctx.get_model()  # Uses configured default")
    print("    return model(f'Analyze: {text}').text")
    print()
    
    # Demonstrate different configurations
    configs = {
        'development': {
            'models': {'default': 'gpt-3.5-turbo', 'temperature': 0.9}
        },
        'production': {
            'models': {'default': 'gpt-4', 'temperature': 0.3}
        },
        'testing': {
            'models': {'default': 'mock-model', 'temperature': 0.0}
        }
    }
    
    print("✅ Same code, different environments:")
    for env, config in configs.items():
        ctx = MockContext(config)
        model = ctx.get_model()
        print(f"  {env:>11}: {model}")
    
    print()
    print("💡 Benefits:")
    print("  • One place to change configuration")
    print("  • Easy environment switching")
    print("  • Testable without changing code")
    print("  • Shared configuration across components")


def show_real_scenarios():
    """Show realistic scenarios where context management shines."""
    print("\n\n=== Real-World Scenarios ===\n")
    
    # Mock implementations for demonstration
    class AppService:
        def __init__(self, ctx):
            self.ctx = ctx
            self.model = ctx.get_model()
        
        def process(self, text):
            return f"Processed '{text}' with {self.model}"
    
    class MockContext:
        def __init__(self, config):
            self.config = config
        
        def get_model(self, name=None):
            models_config = self.config.get('models', {})
            default = models_config.get('default', 'gpt-3.5-turbo')
            return f"{name or default}"
    
    print("Scenario 1: Multi-Environment Deployment")
    print("=" * 45)
    
    # Show environment-specific configs
    environments = {
        'development': MockContext({'models': {'default': 'gpt-3.5-turbo'}}),
        'staging': MockContext({'models': {'default': 'gpt-4'}}),
        'production': MockContext({'models': {'default': 'gpt-4-turbo'}})
    }
    
    print("Same application code, different configurations:")
    for env_name, ctx in environments.items():
        service = AppService(ctx)
        result = service.process("user input")
        print(f"  {env_name:>11}: {result}")
    
    print()
    print("Scenario 2: Testing with Mock Models")
    print("=" * 42)
    
    # Production context
    prod_ctx = MockContext({'models': {'default': 'gpt-4'}})
    
    # Test context with mocks
    test_ctx = MockContext({'models': {'default': 'mock-model'}})
    
    print("Production:")
    prod_service = AppService(prod_ctx)
    print(f"  {prod_service.process('real user data')}")
    
    print("Testing:")
    test_service = AppService(test_ctx)
    print(f"  {test_service.process('test data')}")
    
    print()
    print("Scenario 3: Component Sharing")
    print("=" * 33)
    
    # Multiple services sharing the same context
    shared_ctx = MockContext({'models': {'default': 'claude-3'}})
    
    services = {
        'classifier': AppService(shared_ctx),
        'summarizer': AppService(shared_ctx),
        'analyzer': AppService(shared_ctx)
    }
    
    print("All services use the same configuration:")
    for name, service in services.items():
        print(f"  {name}: {service.model}")
    
    print("\n💡 Key insight: Configuration changes in one place!")


def show_practical_usage():
    """Show practical context usage patterns."""
    print("\n\n=== Practical Usage Patterns ===\n")
    
    print("Pattern 1: Explicit Configuration")
    print("=" * 34)
    print("# Create context with specific config")
    print("from ember.context import create_context")
    print()
    print("prod_ctx = create_context(")
    print("    models={'default': 'gpt-4', 'temperature': 0.3}")
    print(")")
    print("model = prod_ctx.get_model()  # Uses gpt-4")
    print()
    
    print("Pattern 2: Configuration Files")
    print("=" * 30)
    print("# Use config files with environment variables")
    print("# ~/.ember/config.yaml:")
    print("models:")
    print("  default: ${MODEL_NAME:-gpt-3.5-turbo}")
    print("  temperature: ${MODEL_TEMP:-0.7}")
    print()
    print("# Then set environment variables")
    print("export MODEL_NAME=gpt-4")
    print("export MODEL_TEMP=0.3")
    print()
    print("# Your code uses the substituted values")
    print("ctx = get_context()  # Loads config with env vars")
    print()
    
    print("Pattern 3: Context Injection")
    print("=" * 30)
    print("# Pass context to components")
    print("class DocumentProcessor:")
    print("    def __init__(self, context):")
    print("        self.ctx = context")
    print("        self.model = context.get_model()")
    print()
    print("# Easy to test and configure")
    print("processor = DocumentProcessor(ctx)")
    print()
    
    print("Pattern 4: Testing with Mock Context")
    print("=" * 37)
    print("# Testing setup")
    print("test_ctx = create_context(")
    print("    models={'default': 'mock-model'},")
    print("    cache={'enabled': False}")
    print(")")
    print("# Use in tests without affecting production config")




def main():
    """Run all context management examples."""
    print_section_header("Context Management")
    
    print("🎯 Why Context Management?")
    print("Learn when and how to manage configuration and state in AI applications.")
    print()
    
    # Start with the problem
    show_the_problem()
    
    # Show the solution
    show_basic_solution()
    
    # Show realistic scenarios
    show_real_scenarios()
    
    # Show practical patterns
    show_practical_usage()
    
    # Simplified best practices
    print("\n" + "="*50)
    print("✅ When to Use Context Management")
    print("="*50)
    print("\n🟢 Use context when you have:")
    print("  • Multiple environments (dev/staging/prod)")
    print("  • Testing that needs different configurations")
    print("  • Multiple components sharing configuration")
    print("  • Complex applications with many settings")
    
    print("\n🔴 Skip context for:")
    print("  • Simple scripts with one model call")
    print("  • Prototypes and quick experiments")
    print("  • Single-environment applications")
    
    print("\n💡 Progressive approach:")
    print("  1. Start simple: Direct model usage")
    print("  2. Add context when you need configuration management")
    print("  3. Use advanced patterns for complex applications")
    
    print("\nNext: Learn about error handling in 'error_handling.py'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())