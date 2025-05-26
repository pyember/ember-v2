/**
 * Ember Bridge for Codex
 * 
 * Provides OpenAI-compatible interface using Ember's model system.
 * Allows seamless provider switching while maintaining compatibility.
 */

import { EventEmitter } from 'events';

// Types matching OpenAI's interface
interface Message {
  role: 'system' | 'user' | 'assistant' | 'function';
  content: string;
  name?: string;
  function_call?: {
    name: string;
    arguments: string;
  };
}

interface Tool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: any;
  };
}

interface CompletionRequest {
  model: string;
  messages: Message[];
  temperature?: number;
  max_tokens?: number;
  tools?: Tool[];
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
  stream?: boolean;
}

interface ChatCompletion {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: string | null;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: {
          name: string;
          arguments: string;
        };
      }>;
    };
    finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface StreamChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: 'assistant';
      content?: string;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: 'function';
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason: 'stop' | 'length' | 'tool_calls' | null;
  }>;
}

/**
 * Provider router for intent-based model selection
 */
export class ProviderRouter {
  private rules: Map<string, string>;
  
  constructor(config?: { [intent: string]: string }) {
    this.rules = new Map(Object.entries(config || {
      tool_use: 'openai',
      planning: 'anthropic', 
      code_gen: 'anthropic',
      synthesis: 'ensemble',
      default: 'openai'
    }));
  }
  
  getProvider(intent: string): string {
    return this.rules.get(intent) || this.rules.get('default') || 'openai';
  }
  
  detectIntent(messages: Message[], tools?: Tool[]): string {
    // If tools are provided, this is a tool use case
    if (tools && tools.length > 0) {
      return 'tool_use';
    }
    
    // Analyze recent messages for intent
    const recentMessages = messages.slice(-3);
    const combinedContent = recentMessages
      .map(m => m.content)
      .join(' ')
      .toLowerCase();
    
    // Simple heuristics - can be enhanced
    if (combinedContent.includes('plan') || combinedContent.includes('think') || 
        combinedContent.includes('approach') || combinedContent.includes('strategy')) {
      return 'planning';
    }
    
    if (combinedContent.includes('code') || combinedContent.includes('implement') ||
        combinedContent.includes('function') || combinedContent.includes('class')) {
      return 'code_gen';
    }
    
    if (combinedContent.includes('summarize') || combinedContent.includes('combine') ||
        combinedContent.includes('synthesize')) {
      return 'synthesis';
    }
    
    return 'default';
  }
}

/**
 * Ember-powered OpenAI-compatible client
 */
export class EmberOpenAIClient {
  private provider: string;
  private model: string;
  private router: ProviderRouter;
  private emberBridge: any; // Will be actual Ember instance
  
  // OpenAI-compatible nested structure
  public chat: {
    completions: {
      create: (params: CompletionRequest) => Promise<ChatCompletion> | AsyncIterable<StreamChunk>;
    };
  };
  
  constructor(options: {
    provider?: string;
    model?: string;
    router?: ProviderRouter;
    apiKey?: string; // For compatibility
  } = {}) {
    this.provider = options.provider || 'openai';
    this.model = options.model || 'gpt-4o';
    this.router = options.router || new ProviderRouter();
    
    // Initialize OpenAI-compatible API structure
    this.chat = {
      completions: {
        create: this.createCompletion.bind(this)
      }
    };
  }
  
  private async createCompletion(params: CompletionRequest): Promise<ChatCompletion> | AsyncIterable<StreamChunk> {
    // Detect intent and potentially override provider
    const intent = this.router.detectIntent(params.messages, params.tools);
    const effectiveProvider = params.tools ? 'openai' : this.router.getProvider(intent);
    
    // Log routing decision for transparency
    console.debug(`[EmberBridge] Intent: ${intent}, Provider: ${effectiveProvider}`);
    
    // For now, return mock response - will integrate with actual Ember
    if (params.stream) {
      return this.streamCompletion(params, effectiveProvider);
    } else {
      return this.completeSync(params, effectiveProvider);
    }
  }
  
  private async completeSync(params: CompletionRequest, provider: string): Promise<ChatCompletion> {
    // TODO: Call actual Ember models API
    // const emberModel = models.instance(`${provider}:${params.model}`);
    // const response = await emberModel.forward(...);
    
    // Mock response for now
    return {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: params.model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: `[${provider}] Response to: ${params.messages[params.messages.length - 1].content}`,
          tool_calls: params.tools ? [{
            id: 'call_' + Date.now(),
            type: 'function',
            function: {
              name: params.tools[0].function.name,
              arguments: '{}'
            }
          }] : undefined
        },
        finish_reason: 'stop'
      }],
      usage: {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150
      }
    };
  }
  
  private async *streamCompletion(params: CompletionRequest, provider: string): AsyncIterable<StreamChunk> {
    // TODO: Integrate with Ember streaming
    // const emberModel = models.instance(`${provider}:${params.model}`);
    // for await (const chunk of emberModel.stream(...)) { ... }
    
    // Mock streaming for now
    const message = params.messages[params.messages.length - 1].content;
    const response = `[${provider}] Streaming response to: ${message}`;
    
    // Initial chunk
    yield {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: params.model,
      choices: [{
        index: 0,
        delta: { role: 'assistant' },
        finish_reason: null
      }]
    };
    
    // Content chunks
    for (const word of response.split(' ')) {
      yield {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: params.model,
        choices: [{
          index: 0,
          delta: { content: word + ' ' },
          finish_reason: null
        }]
      };
    }
    
    // Final chunk
    yield {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: params.model,
      choices: [{
        index: 0,
        delta: {},
        finish_reason: 'stop'
      }]
    };
  }
}

/**
 * Factory function matching Codex's existing pattern
 */
export function createOpenAIClient(config: {
  provider?: string;
  apiKey?: string;
  baseURL?: string;
  timeout?: number;
  defaultHeaders?: Record<string, string>;
}): EmberOpenAIClient {
  // If traditional OpenAI client is requested, return it
  if (config.provider === 'openai' && !process.env.USE_EMBER_BRIDGE) {
    // Return actual OpenAI client
    const OpenAI = require('openai');
    return new OpenAI(config);
  }
  
  // Otherwise return our Ember bridge
  return new EmberOpenAIClient(config);
}

/**
 * Ensemble client for critical operations
 */
export class EmberEnsembleClient extends EmberOpenAIClient {
  private models: string[];
  
  constructor(models: string[] = ['gpt-4o', 'claude-3-opus']) {
    super({ provider: 'ensemble' });
    this.models = models;
  }
  
  // Override to implement ensemble logic
  private async completeSync(params: CompletionRequest, provider: string): Promise<ChatCompletion> {
    // TODO: Run all models in parallel and aggregate
    console.log(`[Ensemble] Running models: ${this.models.join(', ')}`);
    return super.completeSync(params, 'ensemble');
  }
}

// For testing intent detection
if (require.main === module) {
  const router = new ProviderRouter();
  
  const testCases = [
    { messages: [{ role: 'user', content: 'Run npm install' }], tools: [{ type: 'function', function: { name: 'shell' } }] },
    { messages: [{ role: 'user', content: 'How should I approach refactoring this code?' }], tools: [] },
    { messages: [{ role: 'user', content: 'Write a function to parse JSON' }], tools: [] },
    { messages: [{ role: 'user', content: 'Summarize these test results' }], tools: [] },
  ];
  
  testCases.forEach((test, i) => {
    const intent = router.detectIntent(test.messages, test.tools);
    const provider = router.getProvider(intent);
    console.log(`Test ${i + 1}: "${test.messages[0].content}" -> Intent: ${intent}, Provider: ${provider}`);
  });
}