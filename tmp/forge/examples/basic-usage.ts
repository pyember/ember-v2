/**
 * Basic Usage Examples for Forge
 * 
 * Demonstrates how Forge routes different types of requests
 * to optimal providers while maintaining compatibility.
 */

import { createForgeClient, ProviderRouter } from '../src';

async function main() {
  // Create a Forge client with intelligent routing
  const client = createForgeClient({
    emberBridge: true,
    debug: true
  });
  
  console.log('🔨 Forge Examples\n');
  
  // Example 1: Tool usage (routes to OpenAI)
  console.log('1. Tool Usage Request:');
  const toolResponse = await client.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'user', content: 'List all TypeScript files in the src directory' }
    ],
    tools: [{
      type: 'function',
      function: {
        name: 'shell',
        description: 'Execute shell command',
        parameters: {
          type: 'object',
          properties: {
            command: { type: 'array', items: { type: 'string' } }
          }
        }
      }
    }]
  });
  console.log('Provider:', detectProvider(toolResponse));
  console.log('Response:', toolResponse.choices[0].message);
  console.log();
  
  // Example 2: Planning request (routes to Anthropic)
  console.log('2. Planning Request:');
  const planningResponse = await client.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'user', content: 'How should I approach refactoring this monolithic application into microservices?' }
    ]
  });
  console.log('Provider:', detectProvider(planningResponse));
  console.log('Response:', planningResponse.choices[0].message.content);
  console.log();
  
  // Example 3: Code generation (routes to Anthropic)
  console.log('3. Code Generation Request:');
  const codeResponse = await client.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'user', content: 'Implement a binary search tree in Python with insert and search methods' }
    ]
  });
  console.log('Provider:', detectProvider(codeResponse));
  console.log('Response:', codeResponse.choices[0].message.content);
  console.log();
  
  // Example 4: Custom routing
  console.log('4. Custom Routing:');
  const customRouter = new ProviderRouter({
    tool_use: 'openai',
    planning: 'ensemble',  // Use ensemble for planning
    code_gen: 'openai',    // Use OpenAI for code
    default: 'anthropic'
  });
  
  const customClient = createForgeClient({
    emberBridge: true,
    router: customRouter,
    debug: true
  });
  
  const customResponse = await customClient.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'user', content: 'Design a scalable architecture for a real-time chat application' }
    ]
  });
  console.log('Provider:', detectProvider(customResponse));
  console.log('Response:', customResponse.choices[0].message.content);
}

// Helper to detect which provider was used (in real implementation, this would be in metadata)
function detectProvider(response: any): string {
  const content = response.choices[0].message.content || '';
  if (content.includes('[openai]')) return 'OpenAI';
  if (content.includes('[anthropic]')) return 'Anthropic'; 
  if (content.includes('[ensemble]')) return 'Ensemble';
  return 'Unknown';
}

// Run examples
if (require.main === module) {
  main().catch(console.error);
}