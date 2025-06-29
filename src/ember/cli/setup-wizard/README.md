# Ember Setup Wizard

A delightful, interactive setup experience for Ember that gets users from zero to their first API call in under 60 seconds.

## Usage

```bash
npx @ember-ai/setup
```

No installation required - npx runs it directly!

## Features

- 🎨 Beautiful terminal UI with animations
- 🚀 60-second setup experience
- 🔐 Secure API key configuration
- ✅ Automatic connection testing
- 📝 Working example generation
- 🎯 Smart provider recommendations

## Development

```bash
# Install dependencies
npm install

# Run in development
npm run dev

# Build for production
npm run build
```

## Design Philosophy

This setup wizard embodies the principles of simplicity and delight that Steve Jobs would appreciate, with the technical excellence that Jeff Dean and others would expect:

1. **Zero Friction** - Works immediately with `npx`, no installation
2. **Progressive Disclosure** - Shows only what's needed at each step
3. **Instant Feedback** - Visual confirmation at every action
4. **Error Recovery** - Clear guidance when things go wrong
5. **Delightful Details** - Animations, colors, and polish throughout

## Technical Stack

- **Ink** - React for CLIs
- **TypeScript** - Type safety
- **ink-gradient** - Beautiful text effects
- **ink-spinner** - Smooth loading states
- **ink-select-input** - Intuitive selection
- **ink-text-input** - Secure input handling

## Why NPM?

While Ember is a Python library, we chose npm for the setup wizard because:

1. **Superior Terminal UX** - React-based Ink provides unmatched CLI experiences
2. **No Python Dependencies** - Users might not have Python configured yet
3. **Instant Start** - npx requires no installation
4. **Async Excellence** - Better for API calls and progress indication
5. **Delight Factor** - The tools available in the npm ecosystem create joy

This is what happens when you ask "what would create the most delightful user experience?" and refuse to compromise.