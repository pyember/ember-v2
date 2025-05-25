/**
 * Telemetry Integration - Leverages Ember's existing metrics system
 * 
 * Instead of building our own, we use Ember's comprehensive telemetry
 * which provides ~8ns counter increments and Prometheus integration.
 */

import { emberIntegration } from './ember-integration';

// Type definitions matching Ember's metrics API
interface EmberMetrics {
  counter(name: string, increment?: number, tags?: Record<string, string>): void;
  gauge(name: string, value: number, tags?: Record<string, string>): void;
  histogram(name: string, value: number, tags?: Record<string, string>): void;
  timed(name: string, tags?: Record<string, string>): {
    end(): void;
    __enter__(): void;
    __exit__(): void;
  };
}

interface ComponentMetrics {
  counter(name: string, increment?: number, tags?: Record<string, string>): void;
  gauge(name: string, value: number, tags?: Record<string, string>): void;
  histogram(name: string, value: number, tags?: Record<string, string>): void;
  timed(name: string, tags?: Record<string, string>): any;
}

/**
 * Telemetry manager for Forge
 * 
 * This class wraps Ember's metrics system to provide telemetry for:
 * - Routing decisions
 * - Provider usage
 * - Performance tracking
 * - Error rates
 */
export class ForgeTelemetry {
  private metrics?: EmberMetrics;
  private componentMetrics?: ComponentMetrics;
  private initialized: boolean = false;
  
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    try {
      // Import Ember's metrics dynamically
      const ember = await import('@ember-ai/ember');
      
      // Get the metrics singleton
      this.metrics = ember.core?.metrics;
      
      // Create component-specific metrics
      if (ember.core?.metrics?.ComponentMetrics) {
        this.componentMetrics = new ember.core.metrics.ComponentMetrics(
          'forge',
          { service: 'forge', version: '0.1.0' }
        );
      }
      
      this.initialized = true;
      console.log('Forge telemetry initialized with Ember metrics');
    } catch (error) {
      console.warn('Ember metrics not available, telemetry disabled');
    }
  }
  
  /**
   * Track routing decisions
   */
  trackRouting(intent: string, provider: string, latencyMs: number): void {
    if (!this.componentMetrics) return;
    
    // Count routing decisions
    this.componentMetrics.counter('routing_decisions', 1, {
      intent,
      provider
    });
    
    // Track routing latency
    this.componentMetrics.histogram('routing_latency_ms', latencyMs, {
      intent
    });
  }
  
  /**
   * Track provider invocations
   */
  trackInvocation(
    provider: string,
    model: string,
    success: boolean,
    durationMs: number,
    usage?: {
      promptTokens: number;
      completionTokens: number;
      totalCost?: number;
    }
  ): void {
    if (!this.componentMetrics) return;
    
    // Count invocations
    this.componentMetrics.counter('provider_invocations', 1, {
      provider,
      model,
      status: success ? 'success' : 'failure'
    });
    
    // Track duration
    this.componentMetrics.histogram('invocation_duration_ms', durationMs, {
      provider,
      model
    });
    
    // Track usage if available
    if (usage) {
      this.componentMetrics.counter('tokens_used', usage.promptTokens, {
        provider,
        model,
        type: 'prompt'
      });
      
      this.componentMetrics.counter('tokens_used', usage.completionTokens, {
        provider,
        model,
        type: 'completion'
      });
      
      if (usage.totalCost) {
        this.componentMetrics.gauge('invocation_cost_usd', usage.totalCost, {
          provider,
          model
        });
      }
    }
  }
  
  /**
   * Track streaming performance
   */
  trackStreaming(
    provider: string,
    chunks: number,
    totalDurationMs: number
  ): void {
    if (!this.componentMetrics) return;
    
    this.componentMetrics.histogram('stream_chunks', chunks, {
      provider
    });
    
    this.componentMetrics.histogram('stream_duration_ms', totalDurationMs, {
      provider
    });
    
    // Calculate throughput
    const chunksPerSecond = (chunks / totalDurationMs) * 1000;
    this.componentMetrics.gauge('stream_throughput_cps', chunksPerSecond, {
      provider
    });
  }
  
  /**
   * Track errors
   */
  trackError(provider: string, errorType: string, retryable: boolean): void {
    if (!this.componentMetrics) return;
    
    this.componentMetrics.counter('errors', 1, {
      provider,
      error_type: errorType,
      retryable: retryable ? 'true' : 'false'
    });
  }
  
  /**
   * Track ensemble operations
   */
  trackEnsemble(
    strategy: string,
    models: string[],
    consensusReached: boolean,
    durationMs: number
  ): void {
    if (!this.componentMetrics) return;
    
    this.componentMetrics.counter('ensemble_operations', 1, {
      strategy,
      consensus: consensusReached ? 'yes' : 'no',
      model_count: models.length.toString()
    });
    
    this.componentMetrics.histogram('ensemble_duration_ms', durationMs, {
      strategy,
      model_count: models.length.toString()
    });
  }
  
  /**
   * Get Prometheus metrics (if needed)
   * 
   * Following YAGNI principle - only expose this if someone actually needs it
   */
  async getPrometheusMetrics(): Promise<string> {
    // This exists but we don't promote it - YAGNI
    try {
      const ember = await import('@ember-ai/ember');
      if (ember.core?.metrics?.get_prometheus_metrics) {
        return ember.core.metrics.get_prometheus_metrics();
      }
    } catch (error) {
      // Silent fail - metrics are optional
    }
    
    return '# Metrics not available\n';
  }
  
  /**
   * Create a timer for measuring operations
   */
  startTimer(operationName: string, tags?: Record<string, string>): () => void {
    const startTime = Date.now();
    
    return () => {
      const durationMs = Date.now() - startTime;
      if (this.componentMetrics) {
        this.componentMetrics.histogram(`${operationName}_duration_ms`, durationMs, tags);
      }
    };
  }
}

// Singleton instance
export const telemetry = new ForgeTelemetry();

/**
 * Decorator for automatic method timing
 * 
 * Usage:
 * @timed('operation_name')
 * async myMethod() { ... }
 */
export function timed(metricName: string, tags?: Record<string, string>) {
  return function (
    target: any,
    propertyName: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function (...args: any[]) {
      const timer = telemetry.startTimer(metricName, tags);
      try {
        const result = await originalMethod.apply(this, args);
        timer();
        return result;
      } catch (error) {
        timer();
        throw error;
      }
    };
    
    return descriptor;
  };
}