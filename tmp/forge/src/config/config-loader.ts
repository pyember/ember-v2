/**
 * Configuration Loader - Handles Forge configuration
 * 
 * Loads configuration from:
 * 1. Default configuration
 * 2. User's .forge/config.yaml
 * 3. Environment variables
 * 4. Command line arguments
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import yaml from 'js-yaml';
import { ProviderRouterConfig } from '@forge/core/ember-bridge';

export interface ForgeConfig {
  providers: {
    default: string;
    routing?: ProviderRouterConfig;
    models?: {
      [provider: string]: {
        default?: string;
        apiKey?: string;
        baseURL?: string;
      };
    };
  };
  
  ensembles?: {
    [name: string]: {
      models: string[];
      strategy: 'unanimous' | 'majority' | 'best_of';
    };
  };
  
  features?: {
    autoRouting?: boolean;
    costTracking?: boolean;
    debug?: boolean;
    streaming?: boolean;
  };
  
  safety?: {
    confirmCommands?: boolean;
    maxCommandLength?: number;
    blockedCommands?: string[];
  };
}

const DEFAULT_CONFIG: ForgeConfig = {
  providers: {
    default: 'openai',
    routing: {
      tool_use: 'openai',
      planning: 'anthropic',
      code_gen: 'anthropic',
      synthesis: 'ensemble',
      safety_check: 'ensemble',
      default: 'openai'
    },
    models: {
      openai: {
        default: 'gpt-4'
      },
      anthropic: {
        default: 'claude-3-opus'
      }
    }
  },
  
  ensembles: {
    default: {
      models: ['gpt-4', 'claude-3-opus'],
      strategy: 'majority'
    },
    safety: {
      models: ['gpt-4', 'claude-3-opus', 'gpt-4-turbo'],
      strategy: 'unanimous'
    }
  },
  
  features: {
    autoRouting: true,
    costTracking: true,
    debug: false,
    streaming: true
  },
  
  safety: {
    confirmCommands: true,
    maxCommandLength: 1000,
    blockedCommands: ['rm -rf /', 'format c:', 'del /f /s /q']
  }
};

export class ConfigLoader {
  private config: ForgeConfig;
  private configPath?: string;
  
  constructor() {
    this.config = { ...DEFAULT_CONFIG };
  }
  
  /**
   * Load configuration from all sources
   */
  async load(): Promise<ForgeConfig> {
    // 1. Load from user config file
    await this.loadUserConfig();
    
    // 2. Apply environment variables
    this.applyEnvironmentVariables();
    
    // 3. Validate configuration
    this.validateConfig();
    
    return this.config;
  }
  
  /**
   * Load user configuration from .forge/config.yaml
   */
  private async loadUserConfig(): Promise<void> {
    const possiblePaths = [
      path.join(process.cwd(), '.forge', 'config.yaml'),
      path.join(process.cwd(), '.forge', 'config.yml'),
      path.join(os.homedir(), '.forge', 'config.yaml'),
      path.join(os.homedir(), '.forge', 'config.yml')
    ];
    
    for (const configPath of possiblePaths) {
      if (fs.existsSync(configPath)) {
        try {
          const content = fs.readFileSync(configPath, 'utf8');
          const userConfig = yaml.load(content) as Partial<ForgeConfig>;
          this.config = this.mergeConfigs(this.config, userConfig);
          this.configPath = configPath;
          console.log(`Loaded configuration from ${configPath}`);
          break;
        } catch (error) {
          console.warn(`Failed to load config from ${configPath}:`, error);
        }
      }
    }
  }
  
  /**
   * Apply environment variables
   */
  private applyEnvironmentVariables(): void {
    // Provider API keys
    if (process.env.OPENAI_API_KEY) {
      this.config.providers.models = this.config.providers.models || {};
      this.config.providers.models.openai = this.config.providers.models.openai || {};
      this.config.providers.models.openai.apiKey = process.env.OPENAI_API_KEY;
    }
    
    if (process.env.ANTHROPIC_API_KEY) {
      this.config.providers.models = this.config.providers.models || {};
      this.config.providers.models.anthropic = this.config.providers.models.anthropic || {};
      this.config.providers.models.anthropic.apiKey = process.env.ANTHROPIC_API_KEY;
    }
    
    // Feature flags
    if (process.env.FORGE_AUTO_ROUTING !== undefined) {
      this.config.features = this.config.features || {};
      this.config.features.autoRouting = process.env.FORGE_AUTO_ROUTING === 'true';
    }
    
    if (process.env.FORGE_DEBUG !== undefined) {
      this.config.features = this.config.features || {};
      this.config.features.debug = process.env.FORGE_DEBUG === 'true';
    }
    
    // Default provider
    if (process.env.FORGE_DEFAULT_PROVIDER) {
      this.config.providers.default = process.env.FORGE_DEFAULT_PROVIDER;
    }
  }
  
  /**
   * Merge configurations deeply
   */
  private mergeConfigs(base: ForgeConfig, override: Partial<ForgeConfig>): ForgeConfig {
    const merged = { ...base };
    
    if (override.providers) {
      merged.providers = {
        ...base.providers,
        ...override.providers,
        routing: {
          ...base.providers.routing,
          ...override.providers?.routing
        },
        models: {
          ...base.providers.models,
          ...override.providers?.models
        }
      };
    }
    
    if (override.ensembles) {
      merged.ensembles = {
        ...base.ensembles,
        ...override.ensembles
      };
    }
    
    if (override.features) {
      merged.features = {
        ...base.features,
        ...override.features
      };
    }
    
    if (override.safety) {
      merged.safety = {
        ...base.safety,
        ...override.safety
      };
    }
    
    return merged;
  }
  
  /**
   * Validate configuration
   */
  private validateConfig(): void {
    // Check for required API keys
    const providers = this.config.providers.models || {};
    
    if (this.config.providers.default === 'openai' && !providers.openai?.apiKey) {
      throw new Error('OpenAI API key not found. Set OPENAI_API_KEY environment variable.');
    }
    
    if (this.config.providers.routing?.planning === 'anthropic' && !providers.anthropic?.apiKey) {
      console.warn('Anthropic API key not found. Some features may be limited.');
    }
  }
  
  /**
   * Get the loaded configuration
   */
  getConfig(): ForgeConfig {
    return this.config;
  }
  
  /**
   * Get provider routing configuration
   */
  getRoutingConfig(): ProviderRouterConfig | undefined {
    return this.config.providers.routing;
  }
}