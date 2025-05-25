/**
 * Tests for actual Ember integration
 */

import { emberIntegration, MessageConverter } from '@forge/core/ember-integration';

describe('EmberIntegration', () => {
  describe('MessageConverter', () => {
    it('should convert simple messages to Ember format', () => {
      const messages = [
        { role: 'system' as const, content: 'You are a helpful assistant' },
        { role: 'user' as const, content: 'Hello, how are you?' }
      ];
      
      const result = MessageConverter.toEmberFormat(messages);
      
      expect(result.prompt).toBe('Hello, how are you?');
      expect(result.context).toBe('System: You are a helpful assistant');
    });
    
    it('should handle multi-turn conversations', () => {
      const messages = [
        { role: 'system' as const, content: 'You are a helpful assistant' },
        { role: 'user' as const, content: 'What is Python?' },
        { role: 'assistant' as const, content: 'Python is a programming language' },
        { role: 'user' as const, content: 'Tell me more' }
      ];
      
      const result = MessageConverter.toEmberFormat(messages);
      
      expect(result.prompt).toBe('Tell me more');
      expect(result.context).toContain('System: You are a helpful assistant');
      expect(result.context).toContain('User: What is Python?');
      expect(result.context).toContain('Assistant: Python is a programming language');
    });
    
    it('should extract tool context', () => {
      const messages = [{ role: 'user' as const, content: 'List files' }];
      const tools = [{
        type: 'function' as const,
        function: {
          name: 'ls',
          description: 'List directory contents'
        }
      }];
      
      const toolContext = MessageConverter.extractToolContext(messages, tools);
      
      expect(toolContext).toContain('You have access to the following tools:');
      expect(toolContext).toContain('- ls: List directory contents');
      expect(toolContext).toContain('{"tool": "tool_name", "arguments": {...}}');
    });
    
    it('should throw error if no user message', () => {
      const messages = [
        { role: 'system' as const, content: 'System prompt' }
      ];
      
      expect(() => MessageConverter.toEmberFormat(messages)).toThrow('No user message found');
    });
  });
  
  describe('EmberIntegration', () => {
    beforeEach(async () => {
      await emberIntegration.initialize();
    });
    
    it('should initialize without errors', () => {
      // If we get here, initialization succeeded (even if using mock)
      expect(emberIntegration).toBeDefined();
    });
    
    it('should invoke model with basic parameters', async () => {
      const response = await emberIntegration.invoke({
        messages: [{ role: 'user', content: 'Hello' }],
        provider: 'openai',
        model: 'gpt-4'
      });
      
      expect(response).toHaveProperty('data');
      expect(response.data).toBeTruthy();
      expect(response).toHaveProperty('usage');
    });
    
    it('should handle different providers', async () => {
      const providers = ['openai', 'anthropic', 'gpt-4', 'claude-3-opus'];
      
      for (const provider of providers) {
        const response = await emberIntegration.invoke({
          messages: [{ role: 'user', content: 'Test' }],
          provider
        });
        
        expect(response.data).toBeTruthy();
      }
    });
    
    it('should convert response to OpenAI format', async () => {
      const emberResponse = {
        data: 'This is a test response',
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30,
          total_cost: 0.001
        }
      };
      
      const openaiFormat = emberIntegration.convertToOpenAIFormat(emberResponse, 'gpt-4');
      
      expect(openaiFormat).toHaveProperty('id');
      expect(openaiFormat.object).toBe('chat.completion');
      expect(openaiFormat.choices[0].message.content).toBe('This is a test response');
      expect(openaiFormat.usage?.total_tokens).toBe(30);
    });
    
    it('should parse tool calls from response', async () => {
      const emberResponse = {
        data: 'I\'ll list the files for you.\n{"tool": "ls", "arguments": {"path": "."}}',
        usage: undefined
      };
      
      const openaiFormat = emberIntegration.convertToOpenAIFormat(emberResponse, 'gpt-4');
      
      expect(openaiFormat.choices[0].message.tool_calls).toBeDefined();
      expect(openaiFormat.choices[0].message.tool_calls?.[0].function.name).toBe('ls');
    });
    
    it('should simulate streaming', async () => {
      const emberResponse = {
        data: 'Hello world test',
        usage: {
          prompt_tokens: 5,
          completion_tokens: 3,
          total_tokens: 8
        }
      };
      
      const chunks: any[] = [];
      for await (const chunk of emberIntegration.simulateStream(emberResponse, 'gpt-4')) {
        chunks.push(chunk);
      }
      
      expect(chunks.length).toBeGreaterThan(3); // At least initial + words + final
      expect(chunks[0].choices[0].delta.role).toBe('assistant');
      expect(chunks[chunks.length - 1].choices[0].finish_reason).toBe('stop');
      
      // Reconstruct message from chunks
      const reconstructed = chunks
        .slice(1, -1)
        .map(c => c.choices[0].delta.content || '')
        .join('');
      expect(reconstructed.trim()).toBe('Hello world test');
    });
  });
});