"""Ember command-line interface implementation.

This package provides the command-line interface for Ember, implementing a
git-style subcommand architecture for configuration, testing, and development.

Architecture Design:
    The CLI follows established Unix principles:
    1. **Subcommand Pattern**: Similar to git, docker, and kubectl for familiarity
    2. **Progressive Disclosure**: Common operations are simple, advanced features discoverable
    3. **Standard Exit Codes**: Follows POSIX conventions for scripting compatibility
    4. **Minimal Dependencies**: CLI loads only required components for fast startup

Package Structure:
    - main: Entry point and command dispatcher
    - commands/: Individual command implementations
        - configure: Configuration management (get/set/list)
        - models: Model discovery and testing
        - setup: Interactive setup wizard
        - registry: Model registry management
        - context: Context inspection and manipulation
    - setup-wizard/: Interactive onboarding experience

Design Rationale:
    Traditional ML frameworks often require complex configuration files or
    environment variables. Ember's CLI provides a conversational interface
    that guides users through setup while teaching best practices.

    The subcommand architecture enables:
    - Lazy loading for fast startup (only load what's needed)
    - Clear separation of concerns (each command in its own module)
    - Easy extensibility (add new commands without touching core)
    - Testability (each command isolated with clear interfaces)

Performance Considerations:
    - Startup time: < 100ms for help, < 200ms for most commands
    - Lazy imports: Commands loaded only when invoked
    - Minimal validation: Fail fast with clear error messages
    - No implicit network calls: Explicit user control

Integration Points:
    - Uses ember.context for configuration management
    - Integrates with model providers for testing
    - Respects XDG Base Directory specification
    - Compatible with shell completion systems
"""
