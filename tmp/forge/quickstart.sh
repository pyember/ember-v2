#!/bin/bash

# Forge Quickstart Script
# This script sets up and runs Forge for testing

echo "🔨 Forge Quickstart"
echo "=================="
echo ""

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   Tool operations may not work without it"
    echo ""
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not set"
    echo "   Advanced routing features will be limited"
    echo ""
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the project
echo "🔧 Building Forge..."
npm run build

# Create example configuration
echo "📝 Creating example configuration..."
mkdir -p ~/.forge
cat > ~/.forge/config.yaml << EOF
# Forge Configuration
providers:
  default: openai
  
  routing:
    tool_use: openai
    planning: anthropic
    code_gen: anthropic
    synthesis: ensemble
    default: openai

features:
  autoRouting: true
  debug: true
  costTracking: true
EOF

echo "✅ Configuration created at ~/.forge/config.yaml"
echo ""

# Run forge
echo "🚀 Starting Forge..."
echo ""
echo "Try these example commands:"
echo "  - 'List all files in this directory'"
echo "  - 'Explain how this project works'"
echo "  - 'Create a simple web server in Python'"
echo "  - 'help' for more commands"
echo ""

# Run forge with the example prompt
node dist/cli/index.js "$@"