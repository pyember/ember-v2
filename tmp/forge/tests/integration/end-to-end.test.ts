/**
 * End-to-end integration tests for Forge
 */

import { createForgeClient, ProviderRouter } from '../../src';
import { ConfigLoader } from '../../src/config/config-loader';

describe('End-to-End Integration', () => {
  describe('Basic Usage', () => {
    it('should handle a complete conversation flow', async () => {
      const client = createForgeClient({ debug: false });
      
      // Initial greeting
      const greeting = await client.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello, Forge!' }],
        stream: false,
      });
      
      expect(greeting.choices[0].message.content).toBeTruthy();
      expect(greeting.usage).toBeDefined();
      
      // Follow-up with tool request
      const messages = [
        { role: 'user', content: 'Hello, Forge!' },
        greeting.choices[0].message,
        { role: 'user', content: 'List the files in the current directory' },
      ];
      
      const toolResponse = await client.chat.completions.create({
        model: 'gpt-4',
        messages,
        tools: [{
          type: 'function',
          function: {
            name: 'ls',
            description: 'List directory contents',
            parameters: {
              type: 'object',
              properties: {
                path: { type: 'string' },
              },
            },
          },
        }],
      });
      
      expect(toolResponse.choices[0].message.tool_calls).toBeDefined();
    });

    it('should handle streaming responses', async () => {
      const client = createForgeClient();
      const chunks: any[] = [];
      
      const stream = await client.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Count to 5' }],
        stream: true,
      });
      
      for await (const chunk of stream) {
        chunks.push(chunk);
      }
      
      expect(chunks.length).toBeGreaterThan(1);
      expect(chunks[0].choices[0].delta.role).toBe('assistant');
      expect(chunks[chunks.length - 1].choices[0].finish_reason).toBe('stop');
    });
  });

  describe('Provider Routing', () => {
    it('should route different intents to appropriate providers', async () => {
      const routingLog: string[] = [];
      
      const router = new ProviderRouter();
      const originalGetProvider = router.getProvider.bind(router);
      router.getProvider = (intent: string) => {
        const provider = originalGetProvider(intent);
        routingLog.push(`${intent} -> ${provider}`);
        return provider;
      };
      
      const client = createForgeClient({ router });
      
      // Tool use -> OpenAI
      await client.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Run ls command' }],
        tools: [{ type: 'function', function: { name: 'shell' } }],
      });
      
      // Planning -> Anthropic
      await client.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'How should I architect this system?' }],
      });
      
      // Code generation -> Anthropic
      await client.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Implement a quicksort algorithm' }],
      });
      
      expect(routingLog).toContain('tool_use -> openai');
      expect(routingLog).toContain('planning -> anthropic');
      expect(routingLog).toContain('code_gen -> anthropic');
    });
  });

  describe('Configuration', () => {
    it('should load and apply configuration', async () => {
      const loader = new ConfigLoader();
      
      // Mock environment variables
      process.env.OPENAI_API_KEY = 'test-key';
      process.env.FORGE_DEBUG = 'true';
      
      const config = await loader.load();
      
      expect(config.providers.models?.openai?.apiKey).toBe('test-key');
      expect(config.features?.debug).toBe(true);
      
      // Create client with config
      const client = createForgeClient({
        router: new ProviderRouter(config.providers.routing),
        debug: config.features?.debug,
      });
      
      expect(client).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      const client = createForgeClient();
      
      try {
        await client.chat.completions.create({
          model: 'invalid-model',
          messages: [{ role: 'user', content: 'Hello' }],
        });
      } catch (error: any) {
        expect(error).toBeDefined();
        expect(error.message).toBeTruthy();
      }
    });

    it('should handle malformed requests', async () => {
      const client = createForgeClient();
      
      try {
        await client.chat.completions.create({
          model: 'gpt-4',
          messages: [], // Empty messages
        });
      } catch (error: any) {
        expect(error).toBeDefined();
      }
    });
  });

  describe('Performance', () => {
    it('should complete routing decisions quickly', async () => {
      const router = new ProviderRouter();
      const iterations = 1000;
      
      const start = Date.now();
      
      for (let i = 0; i < iterations; i++) {
        router.detectIntent([{ role: 'user', content: 'Test message' }]);
        router.getProvider('planning');
      }
      
      const duration = Date.now() - start;
      const avgTime = duration / iterations;
      
      expect(avgTime).toBeLessThan(1); // Less than 1ms per operation
    });
  });
});