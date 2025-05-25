/**
 * EmberBridge - Core integration between Forge and Ember framework
 * 
 * This module provides the bridge to Ember's model system while maintaining
 * compatibility with OpenAI's interface for seamless integration.
 * 
 * Updated to use actual Ember integration instead of mocks.
 */

import type { 
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletion,
  ChatCompletionChunk
} from 'openai/resources/chat/completions';

import { emberIntegration } from './ember-integration';
import { telemetry } from './telemetry-integration';

export interface ProviderRouterConfig {
  tool_use?: string;
  planning?: string;
  code_gen?: string;
  synthesis?: string;
  safety_check?: string;
  default?: string;
}

export class ProviderRouter {
  private rules: Map<string, string>;
  
  constructor(config: ProviderRouterConfig = {}) {
    this.rules = new Map(Object.entries({
      tool_use: 'openai',
      planning: 'anthropic',
      code_gen: 'anthropic', 
      synthesis: 'ensemble',
      safety_check: 'ensemble',
      default: 'openai',
      ...config
    }));
  }
  
  getProvider(intent: string): string {
    return this.rules.get(intent) || this.rules.get('default') || 'openai';
  }
  
  detectIntent(messages: ChatCompletionMessageParam[], tools?: ChatCompletionTool[]): string {
    // Always use OpenAI for tool calls
    if (tools && tools.length > 0) {
      return 'tool_use';
    }
    
    // Analyze recent messages
    const recentMessages = messages.slice(-3);
    const content = recentMessages
      .map(m => typeof m.content === 'string' ? m.content : '')
      .join(' ')
      .toLowerCase();
    
    // Intent detection heuristics
    if (content.match(/\b(plan|approach|strategy|design|architect)\b/)) {
      return 'planning';
    }
    
    if (content.match(/\b(implement|code|function|class|method|refactor)\b/)) {
      return 'code_gen';
    }
    
    if (content.match(/\b(summarize|combine|merge|synthesize)\b/)) {
      return 'synthesis';
    }
    
    if (content.match(/\b(check|verify|validate|safe|security)\b/)) {
      return 'safety_check';
    }
    
    return 'default';
  }
}

/**
 * EmberBridge - The actual bridge implementation
 * 
 * This class coordinates between OpenAI's interface and Ember's models API,
 * handling format conversion, routing, and streaming simulation.
 */
export class EmberBridge {
  private router: ProviderRouter;
  private debug: boolean;
  private initialized: boolean = false;
  
  constructor(options: {
    router?: ProviderRouter;
    debug?: boolean;
  } = {}) {
    this.router = options.router || new ProviderRouter();
    this.debug = options.debug || false;
  }
  
  /**
   * Ensure Ember integration is initialized
   */
  private async ensureInitialized(): Promise<void> {
    if (!this.initialized) {
      await emberIntegration.initialize();
      await telemetry.initialize();
      this.initialized = true;
    }
  }
  
  async createChatCompletion(
    params: ChatCompletionCreateParamsNonStreaming
  ): Promise<ChatCompletion>;
  async createChatCompletion(
    params: ChatCompletionCreateParamsStreaming
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  async createChatCompletion(
    params: ChatCompletionCreateParamsNonStreaming | ChatCompletionCreateParamsStreaming
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> {
    // Ensure Ember is initialized
    await this.ensureInitialized();
    
    // Detect intent and select provider
    const routingStart = Date.now();
    const intent = this.router.detectIntent(params.messages, params.tools);
    const provider = this.router.getProvider(intent);
    const routingLatency = Date.now() - routingStart;
    
    // Track routing decision
    telemetry.trackRouting(intent, provider, routingLatency);
    
    if (this.debug) {
      console.log(`[Forge] Intent: ${intent}, Provider: ${provider} (${routingLatency}ms)`);
    }
    
    const invocationStart = Date.now();
    let success = false;
    
    try {
      // Invoke through Ember
      const emberResponse = await emberIntegration.invoke({
        messages: params.messages,
        model: params.model,
        temperature: params.temperature,
        maxTokens: params.max_tokens,
        provider,
        tools: params.tools,
      });
      
      const invocationDuration = Date.now() - invocationStart;
      success = true;
      
      // Track successful invocation
      telemetry.trackInvocation(
        provider,
        params.model || 'gpt-4',
        true,
        invocationDuration,
        emberResponse.usage ? {
          promptTokens: emberResponse.usage.prompt_tokens,
          completionTokens: emberResponse.usage.completion_tokens,
          totalCost: emberResponse.usage.total_cost
        } : undefined
      );
      
      if (this.debug) {
        console.log(`[Forge] Ember response:`, {
          length: emberResponse.data.length,
          usage: emberResponse.usage,
          duration: invocationDuration
        });
      }
      
      // Handle streaming vs non-streaming
      if (params.stream) {
        // Track streaming start
        const streamStart = Date.now();
        const stream = emberIntegration.simulateStream(emberResponse, params.model || 'gpt-4');
        
        // Wrap stream to track metrics
        return this.wrapStreamForMetrics(stream, provider, streamStart);
      } else {
        return emberIntegration.convertToOpenAIFormat(emberResponse, params.model || 'gpt-4');
      }
    } catch (error) {
      const invocationDuration = Date.now() - invocationStart;
      
      // Track failed invocation
      telemetry.trackInvocation(
        provider,
        params.model || 'gpt-4',
        false,
        invocationDuration
      );
      
      // Track error
      telemetry.trackError(
        provider,
        error instanceof Error ? error.constructor.name : 'UnknownError',
        false // TODO: Determine if retryable based on error type
      );
      
      // Log error in debug mode
      if (this.debug) {
        console.error('[Forge] Error invoking Ember:', error);
      }
      
      // Re-throw with context
      throw new Error(`Provider error (${provider}): ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  /**
   * Wrap stream to track metrics
   */
  private async *wrapStreamForMetrics(
    stream: AsyncIterable<ChatCompletionChunk>,
    provider: string,
    startTime: number
  ): AsyncIterable<ChatCompletionChunk> {
    let chunkCount = 0;
    
    try {
      for await (const chunk of stream) {
        chunkCount++;
        yield chunk;
      }
    } finally {
      const duration = Date.now() - startTime;
      telemetry.trackStreaming(provider, chunkCount, duration);
    }
  }
  
  /**
   * Get available models from Ember
   */
  async getAvailableModels(): Promise<string[]> {
    await this.ensureInitialized();
    try {
      // This would call emberIntegration.modelsAPI.list()
      // For now, return common models
      return [
        'gpt-4',
        'gpt-4o', 
        'gpt-3.5-turbo',
        'claude-3-opus',
        'claude-3-sonnet',
        'claude-3-haiku',
      ];
    } catch (error) {
      console.warn('Failed to get model list from Ember:', error);
      return [];
    }
  }
  
  /**
   * Get model information
   */
  async getModelInfo(modelId: string): Promise<any> {
    await this.ensureInitialized();
    try {
      // This would call emberIntegration.modelsAPI.info(modelId)
      return {
        id: modelId,
        name: modelId,
        provider: this.getProviderFromModelId(modelId),
      };
    } catch (error) {
      console.warn(`Failed to get info for model ${modelId}:`, error);
      return null;
    }
  }
  
  /**
   * Extract provider from model ID
   */
  private getProviderFromModelId(modelId: string): string {
    if (modelId.includes(':')) {
      return modelId.split(':')[0];
    }
    
    // Common model patterns
    if (modelId.startsWith('gpt-') || modelId === 'o1' || modelId === 'o1-mini') {
      return 'openai';
    }
    if (modelId.startsWith('claude-')) {
      return 'anthropic';
    }
    if (modelId.startsWith('gemini-')) {
      return 'google';
    }
    
    return 'unknown';
  }
}