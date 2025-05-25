/**
 * Forge - Intelligent coding assistant
 * 
 * Main entry point for the Forge package
 */

// Core exports
export { EmberBridge, ProviderRouter } from './core/ember-bridge';
export type { ProviderRouterConfig } from './core/ember-bridge';

// Provider exports
export { 
  createForgeClient, 
  ForgeClient, 
  EnsembleClient 
} from './providers/client-factory';
export type { ForgeClientConfig } from './providers/client-factory';

// Configuration exports
export { ConfigLoader } from './config/config-loader';
export type { ForgeConfig } from './config/config-loader';

// Version
export { version } from '../package.json';