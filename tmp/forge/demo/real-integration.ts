/**
 * Real Integration Demo - Shows Forge using actual Ember models
 * 
 * Run this after installing Ember:
 * npm install @ember-ai/ember
 * 
 * Then: npm run demo
 */

import { createForgeClient, ProviderRouter } from '../src';

async function main() {
  console.log('🔨 Forge - Real Ember Integration Demo\n');
  
  // Check if Ember is available
  try {
    await import('@ember-ai/ember');
    console.log('✅ Ember is installed and available\n');
  } catch {
    console.log('⚠️  Ember not installed. Install with: npm install @ember-ai/ember');
    console.log('   Using mock implementation for demo.\n');
  }
  
  // Create client with debug enabled
  const client = createForgeClient({
    emberBridge: true,
    debug: true
  });
  
  console.log('1️⃣  Tool Use Example (routes to OpenAI)\n');
  
  try {
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
      }],
      stream: false
    });
    
    console.log('Response:', toolResponse.choices[0].message);
    if (toolResponse.usage) {
      console.log('Usage:', toolResponse.usage);
    }
  } catch (error) {
    console.error('Tool use failed:', error);
  }
  
  console.log('\n2️⃣  Planning Example (routes to Anthropic)\n');
  
  try {
    const planningResponse = await client.chat.completions.create({
      model: 'gpt-4',
      messages: [
        { role: 'user', content: 'How should I architect a microservices system?' }
      ],
      stream: false
    });
    
    console.log('Response:', planningResponse.choices[0].message.content);
  } catch (error) {
    console.error('Planning failed:', error);
  }
  
  console.log('\n3️⃣  Code Generation Example (routes to Anthropic)\n');
  
  try {
    const codeResponse = await client.chat.completions.create({
      model: 'gpt-4',
      messages: [
        { role: 'user', content: 'Implement a binary search algorithm in Python' }
      ],
      stream: false
    });
    
    console.log('Response:', codeResponse.choices[0].message.content);
  } catch (error) {
    console.error('Code generation failed:', error);
  }
  
  console.log('\n4️⃣  Streaming Example\n');
  
  try {
    const stream = await client.chat.completions.create({
      model: 'gpt-4',
      messages: [
        { role: 'user', content: 'Count to 5 slowly' }
      ],
      stream: true
    });
    
    process.stdout.write('Streaming: ');
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        process.stdout.write(content);
      }
    }
    console.log('\n');
  } catch (error) {
    console.error('Streaming failed:', error);
  }
  
  console.log('\n5️⃣  Custom Routing Example\n');
  
  const customRouter = new ProviderRouter({
    planning: 'gpt-4',      // Use GPT-4 for planning instead of Claude
    code_gen: 'claude-3',   // Explicitly use Claude for code
    default: 'gpt-3.5-turbo' // Use cheaper model by default
  });
  
  const customClient = createForgeClient({
    emberBridge: true,
    router: customRouter,
    debug: true
  });
  
  try {
    const response = await customClient.chat.completions.create({
      model: 'gpt-4',
      messages: [
        { role: 'user', content: 'What is the meaning of life?' }
      ],
      stream: false
    });
    
    console.log('Response (with custom routing):', response.choices[0].message.content);
  } catch (error) {
    console.error('Custom routing failed:', error);
  }
  
  console.log('\n✅ Demo complete!');
  console.log('\nKey insights:');
  console.log('- Tool calls always route to OpenAI (reliable function calling)');
  console.log('- Planning/reasoning routes to Anthropic (superior reasoning)');
  console.log('- Code generation routes to Anthropic (better code quality)');
  console.log('- Streaming works even though Ember doesn\'t support it natively');
  console.log('- Custom routing allows fine-grained control\n');
}

// Run demo
if (require.main === module) {
  main().catch(console.error);
}