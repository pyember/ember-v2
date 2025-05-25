#!/usr/bin/env node

/**
 * Forge CLI - A powerful coding assistant
 * 
 * Orchestrates multiple AI models for optimal results while maintaining
 * full compatibility with existing Codex workflows.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import { ConfigLoader } from '@forge/config/config-loader';
import { ProviderRouter } from '@forge/core/ember-bridge';
import { createForgeClient } from '@forge/providers/client-factory';
import { runAgentLoop } from './agent-loop';
import { version } from '../../package.json';

const program = new Command();

program
  .name('forge')
  .description('A powerful coding assistant that orchestrates multiple AI models')
  .version(version)
  .argument('[prompt]', 'Initial prompt or command')
  .option('-m, --model <model>', 'Model to use', 'gpt-4')
  .option('-p, --provider <provider>', 'Force specific provider')
  .option('--no-routing', 'Disable automatic provider routing')
  .option('--debug', 'Enable debug output')
  .option('--config <path>', 'Path to configuration file')
  .option('--no-stream', 'Disable streaming responses')
  .option('--ensemble <models...>', 'Use ensemble of models')
  .action(async (prompt, options) => {
    const spinner = ora('Initializing Forge...').start();
    
    try {
      // Load configuration
      const configLoader = new ConfigLoader();
      const config = await configLoader.load();
      
      if (options.debug || config.features?.debug) {
        console.log(chalk.gray('Configuration loaded:'), config);
      }
      
      // Set up provider routing
      let router: ProviderRouter | undefined;
      if (options.routing !== false && config.features?.autoRouting !== false) {
        router = new ProviderRouter(config.providers.routing);
        spinner.text = 'Provider routing enabled';
      }
      
      // Create client
      const client = createForgeClient({
        provider: options.provider || config.providers.default,
        emberBridge: options.routing !== false,
        router,
        debug: options.debug || config.features?.debug
      });
      
      spinner.succeed('Forge initialized');
      
      // Display routing info if enabled
      if (router && (options.debug || config.features?.debug)) {
        console.log(chalk.gray('\nProvider routing:'));
        console.log(chalk.gray('  Tool calls → OpenAI'));
        console.log(chalk.gray('  Planning → Anthropic'));
        console.log(chalk.gray('  Code generation → Anthropic'));
        console.log(chalk.gray('  Critical decisions → Ensemble\n'));
      }
      
      // Run the agent loop
      await runAgentLoop({
        client,
        initialPrompt: prompt,
        model: options.model,
        streaming: options.stream !== false,
        config
      });
      
    } catch (error) {
      spinner.fail('Initialization failed');
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  });

// Additional commands
program
  .command('config')
  .description('Manage Forge configuration')
  .option('--init', 'Initialize configuration file')
  .option('--show', 'Show current configuration')
  .action(async (options) => {
    if (options.init) {
      await initializeConfig();
    } else if (options.show) {
      await showConfig();
    } else {
      program.help();
    }
  });

program
  .command('models')
  .description('List available models and providers')
  .action(async () => {
    console.log(chalk.bold('\nAvailable Providers:'));
    console.log('  • openai     - GPT-4, GPT-4 Turbo, GPT-3.5');
    console.log('  • anthropic  - Claude 3 Opus, Claude 3 Sonnet');
    console.log('  • ensemble   - Multi-model consensus\n');
    
    console.log(chalk.bold('Recommended Usage:'));
    console.log('  • Planning/Architecture: Claude 3 Opus');
    console.log('  • Code Generation: Claude 3 Opus'); 
    console.log('  • Tool Usage: GPT-4');
    console.log('  • Quick Tasks: GPT-3.5 Turbo\n');
  });

async function initializeConfig() {
  const fs = await import('fs');
  const path = await import('path');
  const os = await import('os');
  
  const configDir = path.join(os.homedir(), '.forge');
  const configPath = path.join(configDir, 'config.yaml');
  
  // Create directory if it doesn't exist
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true });
  }
  
  // Write default configuration
  const defaultConfig = `# Forge Configuration
# https://github.com/ember-ai/forge

providers:
  default: openai
  
  # Automatic routing based on task type
  routing:
    tool_use: openai      # Best for function calling
    planning: anthropic   # Superior reasoning
    code_gen: anthropic   # Better code quality
    synthesis: ensemble   # Multiple perspectives
    safety_check: ensemble
    default: openai
  
  # Model configurations
  models:
    openai:
      default: gpt-4
      # apiKey: \${OPENAI_API_KEY}
    
    anthropic:
      default: claude-3-opus
      # apiKey: \${ANTHROPIC_API_KEY}

# Ensemble configurations
ensembles:
  default:
    models: [gpt-4, claude-3-opus]
    strategy: majority
  
  safety:
    models: [gpt-4, claude-3-opus, gpt-4-turbo]
    strategy: unanimous

# Feature flags
features:
  autoRouting: true
  costTracking: true
  debug: false
  streaming: true

# Safety settings
safety:
  confirmCommands: true
  maxCommandLength: 1000
  blockedCommands:
    - rm -rf /
    - format c:
`;
  
  fs.writeFileSync(configPath, defaultConfig);
  console.log(chalk.green(`✓ Configuration initialized at ${configPath}`));
}

async function showConfig() {
  const configLoader = new ConfigLoader();
  const config = await configLoader.load();
  
  console.log(chalk.bold('\nCurrent Configuration:'));
  console.log(JSON.stringify(config, null, 2));
}

// Parse arguments
program.parse(process.argv);

// Show help if no arguments
if (!process.argv.slice(2).length) {
  program.outputHelp();
}