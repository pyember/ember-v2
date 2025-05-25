/**
 * Ensemble Coordinator - Manages multi-model consensus
 * 
 * Following Jeff Dean's principle: "Use simple, robust algorithms"
 * and Uncle Bob's: "Make it work, make it right, make it fast"
 */

import { emberIntegration } from './ember-integration';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

export type EnsembleStrategy = 'unanimous' | 'majority' | 'best_of' | 'weighted';

export interface EnsembleConfig {
  models: string[];
  strategy: EnsembleStrategy;
  weights?: number[]; // For weighted voting
}

interface ModelResponse {
  model: string;
  content: string;
  confidence?: number;
  usage?: any;
}

/**
 * Simple, robust ensemble implementation
 * 
 * No over-engineering, just practical consensus algorithms
 */
export class EnsembleCoordinator {
  constructor(private config: EnsembleConfig) {
    if (config.models.length < 2) {
      throw new Error('Ensemble requires at least 2 models');
    }
    
    if (config.strategy === 'weighted' && (!config.weights || config.weights.length !== config.models.length)) {
      throw new Error('Weighted strategy requires weights for each model');
    }
  }
  
  /**
   * Run all models in parallel and aggregate results
   */
  async invoke(params: {
    messages: ChatCompletionMessageParam[];
    temperature?: number;
    maxTokens?: number;
  }): Promise<{
    data: string;
    consensus: string;
    responses: ModelResponse[];
    usage?: any;
  }> {
    // Run all models in parallel - Jeff Dean would approve
    const promises = this.config.models.map(model => 
      this.invokeModel(model, params).catch(error => ({
        model,
        content: '',
        error: error.message
      }))
    );
    
    const responses = await Promise.all(promises);
    
    // Filter out failed responses
    const validResponses = responses.filter(r => !('error' in r)) as ModelResponse[];
    
    if (validResponses.length === 0) {
      throw new Error('All models failed to respond');
    }
    
    // Apply consensus strategy
    const consensus = this.applyStrategy(validResponses);
    
    // Aggregate usage stats
    const totalUsage = this.aggregateUsage(validResponses);
    
    return {
      data: consensus,
      consensus: this.config.strategy,
      responses: validResponses,
      usage: totalUsage
    };
  }
  
  /**
   * Invoke a single model
   */
  private async invokeModel(model: string, params: any): Promise<ModelResponse> {
    const response = await emberIntegration.invoke({
      ...params,
      provider: model,
      model: model
    });
    
    return {
      model,
      content: response.data,
      usage: response.usage
    };
  }
  
  /**
   * Apply consensus strategy to responses
   * 
   * Steve Jobs: "Simplicity is the ultimate sophistication"
   */
  private applyStrategy(responses: ModelResponse[]): string {
    switch (this.config.strategy) {
      case 'unanimous':
        return this.unanimousStrategy(responses);
        
      case 'majority':
        return this.majorityStrategy(responses);
        
      case 'best_of':
        return this.bestOfStrategy(responses);
        
      case 'weighted':
        return this.weightedStrategy(responses);
        
      default:
        // Fallback to first response
        return responses[0].content;
    }
  }
  
  /**
   * All models must agree (for safety-critical decisions)
   */
  private unanimousStrategy(responses: ModelResponse[]): string {
    const contents = responses.map(r => r.content.trim());
    
    // Check if all responses are semantically similar
    // For now, use exact match (in production, use semantic similarity)
    const first = contents[0];
    const allAgree = contents.every(c => c === first);
    
    if (!allAgree) {
      // Return a synthesis indicating disagreement
      return `[Ensemble Disagreement]\n\nThe models provided different responses:\n\n${
        responses.map(r => `${r.model}:\n${r.content}`).join('\n\n---\n\n')
      }`;
    }
    
    return first;
  }
  
  /**
   * Majority vote (most common response wins)
   */
  private majorityStrategy(responses: ModelResponse[]): string {
    // Count occurrences of each response
    const voteCounts = new Map<string, number>();
    const responseMap = new Map<string, ModelResponse>();
    
    for (const response of responses) {
      const key = response.content.trim();
      voteCounts.set(key, (voteCounts.get(key) || 0) + 1);
      responseMap.set(key, response);
    }
    
    // Find the response with most votes
    let maxVotes = 0;
    let winner = '';
    
    for (const [content, votes] of voteCounts) {
      if (votes > maxVotes) {
        maxVotes = votes;
        winner = content;
      }
    }
    
    // If no clear majority, return first response with explanation
    if (maxVotes < Math.ceil(responses.length / 2)) {
      return `[No Clear Majority]\n\n${winner}\n\n(${maxVotes}/${responses.length} models agreed)`;
    }
    
    return winner;
  }
  
  /**
   * Best of N (return the highest quality response)
   * For now, use longest response as proxy for quality
   */
  private bestOfStrategy(responses: ModelResponse[]): string {
    // In production, use a judge model to score responses
    // For now, use response length as a simple heuristic
    let best = responses[0];
    let maxLength = best.content.length;
    
    for (const response of responses) {
      if (response.content.length > maxLength) {
        maxLength = response.content.length;
        best = response;
      }
    }
    
    return best.content;
  }
  
  /**
   * Weighted voting based on model confidence/quality
   */
  private weightedStrategy(responses: ModelResponse[]): string {
    if (!this.config.weights) {
      return this.majorityStrategy(responses);
    }
    
    // Create weighted vote counts
    const voteCounts = new Map<string, number>();
    
    for (let i = 0; i < responses.length; i++) {
      const content = responses[i].content.trim();
      const weight = this.config.weights[i];
      voteCounts.set(content, (voteCounts.get(content) || 0) + weight);
    }
    
    // Find highest weighted response
    let maxWeight = 0;
    let winner = '';
    
    for (const [content, weight] of voteCounts) {
      if (weight > maxWeight) {
        maxWeight = weight;
        winner = content;
      }
    }
    
    return winner;
  }
  
  /**
   * Aggregate usage statistics across all models
   */
  private aggregateUsage(responses: ModelResponse[]): any {
    const total = {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      total_cost: 0
    };
    
    for (const response of responses) {
      if (response.usage) {
        total.prompt_tokens += response.usage.prompt_tokens || 0;
        total.completion_tokens += response.usage.completion_tokens || 0;
        total.total_tokens += response.usage.total_tokens || 0;
        total.total_cost += response.usage.total_cost || 0;
      }
    }
    
    return total;
  }
}

/**
 * Factory function for common ensemble configurations
 */
export function createEnsemble(type: 'safety' | 'quality' | 'speed'): EnsembleCoordinator {
  const configs: Record<string, EnsembleConfig> = {
    safety: {
      models: ['gpt-4', 'claude-3-opus'],
      strategy: 'unanimous'
    },
    quality: {
      models: ['gpt-4', 'claude-3-opus', 'gpt-4-turbo'],
      strategy: 'best_of'
    },
    speed: {
      models: ['gpt-3.5-turbo', 'claude-3-haiku'],
      strategy: 'majority'
    }
  };
  
  return new EnsembleCoordinator(configs[type]);
}