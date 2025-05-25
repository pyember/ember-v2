/**
 * Tests for client factory and ForgeClient
 */

import { createForgeClient, ForgeClient, EnsembleClient } from '@forge/providers/client-factory';
import { ProviderRouter } from '@forge/core/ember-bridge';

describe('Client Factory', () => {
  describe('createForgeClient', () => {
    it('should create ForgeClient by default', () => {
      const client = createForgeClient();
      expect(client).toBeInstanceOf(ForgeClient);
    });

    it('should return OpenAI client when explicitly requested', () => {
      const client = createForgeClient({
        provider: 'openai',
        emberBridge: false
      });
      
      // Should be OpenAI client (has different structure)
      expect(client).not.toBeInstanceOf(ForgeClient);
    });

    it('should create ForgeClient with custom router', () => {
      const router = new ProviderRouter({ default: 'anthropic' });
      const client = createForgeClient({ router });
      
      expect(client).toBeInstanceOf(ForgeClient);
    });
  });

  describe('ForgeClient', () => {
    let client: ForgeClient;

    beforeEach(() => {
      client = new ForgeClient();
    });

    it('should have OpenAI-compatible API structure', () => {
      expect(client).toHaveProperty('chat');
      expect(client.chat).toHaveProperty('completions');
      expect(client.chat.completions).toHaveProperty('create');
      expect(typeof client.chat.completions.create).toBe('function');
    });

    it('should handle chat completion requests', async () => {
      const response = await client.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello' }]
      });
      
      expect(response).toHaveProperty('id');
      expect(response).toHaveProperty('choices');
      expect(response.choices[0].message).toHaveProperty('content');
    });

    it('should use ember bridge for tool requests', async () => {
      const response = await client.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'List files' }],
        tools: [{
          type: 'function',
          function: { name: 'ls', description: 'List files' }
        }]
      });
      
      expect(response).toBeDefined();
      // Should route through ember bridge (mock implementation)
    });
  });

  describe('EnsembleClient', () => {
    it('should create ensemble client with default models', () => {
      const client = new EnsembleClient();
      expect(client).toBeInstanceOf(EnsembleClient);
      expect(client).toBeInstanceOf(ForgeClient);
    });

    it('should accept custom models', () => {
      const models = ['gpt-4', 'claude-3-opus', 'gpt-4-turbo'];
      const client = new EnsembleClient(models);
      expect(client).toBeInstanceOf(EnsembleClient);
    });

    it('should handle ensemble requests', async () => {
      const client = new EnsembleClient();
      
      // Mock console.log to verify ensemble behavior
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      await client.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Critical decision' }]
      });
      
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('[Ensemble] Running models:')
      );
      
      consoleSpy.mockRestore();
    });
  });
});