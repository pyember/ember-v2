# Changelog

All notable changes to Forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Forge
- Intelligent provider routing based on task intent
- OpenAI-compatible API interface
- Support for OpenAI, Anthropic, and ensemble providers
- Automatic intent detection for tool use, planning, code generation
- Configuration system with YAML support
- Cost tracking and usage reporting
- Streaming response support
- Debug mode for transparency
- Comprehensive test suite
- Performance benchmarks

### Security
- Command safety checking for destructive operations
- Blocked command patterns
- Ensemble voting for critical decisions

## [0.1.0] - 2024-01-XX

### Added
- Core EmberBridge integration
- ProviderRouter with intent detection
- ForgeClient with OpenAI compatibility
- Configuration loader with environment variable support
- CLI interface with interactive mode
- Basic documentation and examples

### Fixed
- N/A (initial release)

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Security
- N/A (initial release)

[Unreleased]: https://github.com/ember-ai/forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ember-ai/forge/releases/tag/v0.1.0