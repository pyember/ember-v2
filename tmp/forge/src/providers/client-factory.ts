/**
 * Client Factory - Creates appropriate client based on provider selection
 * 
 * This factory maintains compatibility with OpenAI's interface while
 * allowing seamless switching between providers.
 */

import OpenAI from 'openai';
import { EmberBridge, ProviderRouter } from '@forge/core/ember-bridge';

export interface ForgeClientConfig {
  provider?: string;
  apiKey?: string;
  baseURL?: string;
  emberBridge?: boolean;
  router?: ProviderRouter;
  debug?: boolean;
}

/**
 * OpenAI-compatible client that routes to different providers
 */
export class ForgeClient {
  private emberBridge: EmberBridge;
  private openaiClient?: OpenAI;
  
  // Mimic OpenAI client structure
  public chat: {
    completions: {
      create: (...args: any[]) => any;
    };
  };
  
  constructor(config: ForgeClientConfig = {}) {
    // Initialize Ember bridge for multi-provider support
    this.emberBridge = new EmberBridge({
      router: config.router,
      debug: config.debug
    });
    
    // Fallback to direct OpenAI if needed
    if (config.provider === 'openai' && !config.emberBridge) {
      this.openaiClient = new OpenAI({
        apiKey: config.apiKey || process.env.OPENAI_API_KEY,
        baseURL: config.baseURL
      });
    }
    
    // Set up OpenAI-compatible API
    this.chat = {
      completions: {
        create: this.createCompletion.bind(this)
      }
    };
  }
  
  private async createCompletion(...args: any[]): Promise<any> {
    const params = args[0];
    
    // Use direct OpenAI client if available and no tools
    if (this.openaiClient && !params.tools) {
      return this.openaiClient.chat.completions.create(params);
    }
    
    // Otherwise use Ember bridge for intelligent routing
    return this.emberBridge.createChatCompletion(params);
  }
}

/**
 * Factory function that matches Codex's createOpenAIClient pattern
 */
export function createForgeClient(config: ForgeClientConfig = {}): ForgeClient | OpenAI {
  // If explicitly requesting vanilla OpenAI, return it
  if (config.provider === 'openai' && !config.emberBridge && !process.env.FORGE_ENABLE_ROUTING) {
    return new OpenAI({
      apiKey: config.apiKey || process.env.OPENAI_API_KEY,
      baseURL: config.baseURL
    });
  }
  
  // Otherwise return our intelligent routing client
  return new ForgeClient(config);
}

/**
 * Ensemble client for critical operations
 */
export class EnsembleClient extends ForgeClient {
  private models: string[];
  
  constructor(models: string[] = ['gpt-4', 'claude-3-opus'], config: ForgeClientConfig = {}) {
    super({
      ...config,
      emberBridge: true,
      router: new ProviderRouter({
        ...config.router,
        default: 'ensemble'
      })
    });
    this.models = models;
  }
  
  // Override to implement voting logic
  private async createCompletion(params: any): Promise<any> {
    // TODO: Run all models in parallel and aggregate results
    console.log(`[Ensemble] Running models: ${this.models.join(', ')}`);
    return super.createCompletion(params);
  }
}