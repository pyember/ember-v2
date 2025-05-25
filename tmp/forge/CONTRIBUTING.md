# Contributing to Forge

We welcome contributions to Forge! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/forge.git
   cd forge
   ```
3. Install dependencies:
   ```bash
   npm install
   ```
4. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Node.js >= 18.0.0
- npm >= 8.0.0
- OpenAI API key (for testing)
- Anthropic API key (optional, for full functionality)

### Environment Setup

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key  # Optional
FORGE_DEBUG=true  # Enable debug logging
```

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage

# Run specific test file
npm test -- ember-bridge.test.ts
```

### Type Checking

```bash
npm run typecheck
```

### Linting

```bash
npm run lint

# Auto-fix linting issues
npm run lint -- --fix
```

## Development Guidelines

### Code Style

We follow the Google TypeScript Style Guide with some modifications:
- Use 2 spaces for indentation
- Use single quotes for strings
- Add trailing commas in multi-line objects/arrays
- Prefer `const` over `let` when possible

### Commit Messages

Follow conventional commits format:

```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Build process or auxiliary tool changes

Examples:
```
feat(router): add support for custom intent patterns
fix(bridge): handle streaming errors gracefully
docs(api): update RouterProvider examples
```

### Testing

- Write tests for all new features
- Maintain test coverage above 80%
- Use descriptive test names
- Group related tests using `describe` blocks
- Mock external dependencies

Example test structure:
```typescript
describe('ProviderRouter', () => {
  describe('detectIntent', () => {
    it('should detect tool_use intent when tools are present', () => {
      // Test implementation
    });
  });
});
```

### Documentation

- Update documentation for any API changes
- Add JSDoc comments for public APIs
- Include examples in documentation
- Keep README.md up to date

## Pull Request Process

1. Ensure all tests pass: `npm test`
2. Update documentation if needed
3. Update CHANGELOG.md with your changes
4. Create a pull request with a clear description
5. Link any relevant issues
6. Wait for code review

### PR Title Format

Use the same format as commit messages:
```
feat(router): add support for custom intent patterns
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Updated documentation
- [ ] No new warnings
```

## Architecture Decisions

### Provider Routing

When adding new routing logic:
1. Add intent detection logic to `ProviderRouter.detectIntent`
2. Update default routing configuration
3. Add tests for new patterns
4. Document in routing-flow.md

### Adding New Providers

1. Create provider adapter in `src/providers/`
2. Implement provider-specific logic
3. Add to factory function
4. Update documentation
5. Add integration tests

## Release Process

Releases are automated via GitHub Actions when merging to main:

1. Update version in `package.json`
2. Update CHANGELOG.md
3. Create PR to main
4. After merge, CI will publish to npm

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Join our Discord community (link in README)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Forge! 🔨