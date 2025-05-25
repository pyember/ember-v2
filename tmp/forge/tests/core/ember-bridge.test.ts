/**
 * Tests for EmberBridge core functionality
 */

import { EmberBridge, ProviderRouter } from '@forge/core/ember-bridge';

describe('EmberBridge', () => {
  let bridge: EmberBridge;
  let router: ProviderRouter;

  beforeEach(() => {
    router = new ProviderRouter();
    bridge = new EmberBridge({ router });
  });

  describe('ProviderRouter', () => {
    describe('intent detection', () => {
      it('should detect tool_use intent when tools are present', () => {
        const messages = [{ role: 'user' as const, content: 'List files' }];
        const tools = [{ type: 'function' as const, function: { name: 'ls' } }];
        
        const intent = router.detectIntent(messages, tools);
        expect(intent).toBe('tool_use');
      });

      it('should detect planning intent', () => {
        const messages = [
          { role: 'user' as const, content: 'How should I approach refactoring this codebase?' }
        ];
        
        const intent = router.detectIntent(messages);
        expect(intent).toBe('planning');
      });

      it('should detect code_gen intent', () => {
        const messages = [
          { role: 'user' as const, content: 'Implement a binary search algorithm' }
        ];
        
        const intent = router.detectIntent(messages);
        expect(intent).toBe('code_gen');
      });

      it('should detect safety_check intent', () => {
        const messages = [
          { role: 'user' as const, content: 'Delete all files older than 30 days' }
        ];
        
        const intent = router.detectIntent(messages);
        expect(intent).toBe('safety_check');
      });

      it('should return default for unknown patterns', () => {
        const messages = [
          { role: 'user' as const, content: 'Hello, how are you?' }
        ];
        
        const intent = router.detectIntent(messages);
        expect(intent).toBe('default');
      });
    });

    describe('provider selection', () => {
      it('should route tool_use to openai', () => {
        const provider = router.getProvider('tool_use');
        expect(provider).toBe('openai');
      });

      it('should route planning to anthropic', () => {
        const provider = router.getProvider('planning');
        expect(provider).toBe('anthropic');
      });

      it('should support custom routing', () => {
        const customRouter = new ProviderRouter({
          planning: 'openai',
          code_gen: 'ensemble'
        });
        
        expect(customRouter.getProvider('planning')).toBe('openai');
        expect(customRouter.getProvider('code_gen')).toBe('ensemble');
      });
    });
  });

  describe('EmberBridge', () => {
    describe('createChatCompletion', () => {
      it('should handle non-streaming requests', async () => {
        const params = {
          model: 'gpt-4',
          messages: [{ role: 'user' as const, content: 'Hello' }],
          stream: false as const
        };
        
        const response = await bridge.createChatCompletion(params);
        
        expect(response).toHaveProperty('id');
        expect(response).toHaveProperty('choices');
        expect(response.choices[0].message.role).toBe('assistant');
      });

      it('should handle streaming requests', async () => {
        const params = {
          model: 'gpt-4',
          messages: [{ role: 'user' as const, content: 'Hello' }],
          stream: true as const
        };
        
        const stream = await bridge.createChatCompletion(params);
        
        const chunks = [];
        for await (const chunk of stream) {
          chunks.push(chunk);
        }
        
        expect(chunks.length).toBeGreaterThan(0);
        expect(chunks[0]).toHaveProperty('choices');
      });

      it('should route to correct provider based on intent', async () => {
        const debugBridge = new EmberBridge({ router, debug: true });
        
        // Mock console.log to capture debug output
        const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
        
        await debugBridge.createChatCompletion({
          model: 'gpt-4',
          messages: [{ role: 'user' as const, content: 'Plan a microservices architecture' }],
          stream: false as const
        });
        
        expect(consoleSpy).toHaveBeenCalledWith(
          expect.stringContaining('[Forge] Intent: planning, Provider: anthropic')
        );
        
        consoleSpy.mockRestore();
      });
    });
  });
});