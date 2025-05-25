/**
 * Agent Loop - Core interaction loop for Forge
 * 
 * This is a simplified version that demonstrates the integration.
 * In production, this would integrate with the full Codex agent loop.
 */

import chalk from 'chalk';
import { createInterface } from 'readline';
import { ForgeConfig } from '@forge/config/config-loader';

export interface AgentLoopOptions {
  client: any;
  initialPrompt?: string;
  model: string;
  streaming: boolean;
  config: ForgeConfig;
}

export async function runAgentLoop(options: AgentLoopOptions) {
  const { client, initialPrompt, model, streaming, config } = options;
  
  console.log(chalk.cyan('\n🔨 Forge - Intelligent Coding Assistant\n'));
  
  // Create readline interface
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: chalk.gray('forge> ')
  });
  
  // Conversation history
  const messages: any[] = [
    {
      role: 'system',
      content: 'You are Forge, an advanced coding assistant that orchestrates multiple AI models for optimal results. You have access to shell commands and file operations.'
    }
  ];
  
  // Tool definitions (simplified)
  const tools = [
    {
      type: 'function' as const,
      function: {
        name: 'shell',
        description: 'Execute a shell command',
        parameters: {
          type: 'object',
          properties: {
            command: { 
              type: 'array',
              items: { type: 'string' },
              description: 'Command and arguments to execute'
            }
          },
          required: ['command']
        }
      }
    },
    {
      type: 'function' as const,
      function: {
        name: 'read_file',
        description: 'Read contents of a file',
        parameters: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'File path to read' }
          },
          required: ['path']
        }
      }
    }
  ];
  
  // Process initial prompt if provided
  if (initialPrompt) {
    await processPrompt(initialPrompt);
  } else {
    rl.prompt();
  }
  
  // Handle user input
  rl.on('line', async (input) => {
    const trimmed = input.trim();
    
    if (trimmed === 'exit' || trimmed === 'quit') {
      console.log(chalk.gray('\nGoodbye! 👋\n'));
      rl.close();
      process.exit(0);
    }
    
    if (trimmed === 'clear') {
      console.clear();
      rl.prompt();
      return;
    }
    
    if (trimmed === 'help') {
      showHelp();
      rl.prompt();
      return;
    }
    
    await processPrompt(trimmed);
    rl.prompt();
  });
  
  async function processPrompt(prompt: string) {
    // Add user message
    messages.push({ role: 'user', content: prompt });
    
    try {
      // Determine if we need tools based on the prompt
      const needsTools = prompt.match(/\b(run|execute|read|write|create|delete|list)\b/i);
      
      if (config.features?.debug) {
        console.log(chalk.gray(`\n[Debug] Tools needed: ${needsTools ? 'Yes' : 'No'}`));
      }
      
      // Create completion
      const completion = await client.chat.completions.create({
        model,
        messages,
        tools: needsTools ? tools : undefined,
        stream: streaming
      });
      
      if (streaming) {
        // Handle streaming response
        let response = '';
        process.stdout.write(chalk.green('\nForge: '));
        
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content || '';
          response += content;
          process.stdout.write(content);
        }
        
        process.stdout.write('\n\n');
        messages.push({ role: 'assistant', content: response });
      } else {
        // Handle non-streaming response
        const message = completion.choices[0].message;
        
        if (message.tool_calls) {
          // Handle tool calls
          console.log(chalk.yellow('\n🔧 Executing tools...\n'));
          
          for (const toolCall of message.tool_calls) {
            const args = JSON.parse(toolCall.function.arguments);
            console.log(chalk.gray(`Running: ${toolCall.function.name}(${JSON.stringify(args)})`));
            
            // In production, this would actually execute the tools
            const result = await executeToolMock(toolCall.function.name, args);
            
            messages.push({
              role: 'tool',
              content: result,
              tool_call_id: toolCall.id
            });
          }
          
          // Get final response after tool execution
          const finalCompletion = await client.chat.completions.create({
            model,
            messages,
            stream: false
          });
          
          const finalMessage = finalCompletion.choices[0].message.content;
          console.log(chalk.green('\nForge:'), finalMessage, '\n');
          messages.push({ role: 'assistant', content: finalMessage });
        } else {
          // Regular response
          console.log(chalk.green('\nForge:'), message.content, '\n');
          messages.push(message);
        }
      }
      
      // Show cost if enabled
      if (config.features?.costTracking && !streaming) {
        const usage = completion.usage;
        if (usage) {
          const estimatedCost = (usage.prompt_tokens * 0.01 + usage.completion_tokens * 0.03) / 1000;
          console.log(chalk.gray(`[Tokens: ${usage.total_tokens}, Cost: $${estimatedCost.toFixed(4)}]\n`));
        }
      }
      
    } catch (error) {
      console.error(chalk.red('\nError:'), error);
      console.log();
    }
  }
  
  function showHelp() {
    console.log(chalk.bold('\nAvailable Commands:'));
    console.log('  help   - Show this help message');
    console.log('  clear  - Clear the screen');
    console.log('  exit   - Exit Forge\n');
    
    console.log(chalk.bold('Example Prompts:'));
    console.log('  "List all Python files in this directory"');
    console.log('  "Explain the architecture of this project"');
    console.log('  "Refactor the main function to be more modular"');
    console.log('  "Create a test file for user.py"\n');
  }
  
  async function executeToolMock(name: string, args: any): Promise<string> {
    // Mock tool execution for demonstration
    switch (name) {
      case 'shell':
        return `Executed: ${args.command.join(' ')}\n[Mock output]`;
      case 'read_file':
        return `Contents of ${args.path}:\n[Mock file contents]`;
      default:
        return 'Tool executed successfully';
    }
  }
}