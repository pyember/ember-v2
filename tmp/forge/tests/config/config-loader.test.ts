/**
 * Tests for configuration loader
 */

import { ConfigLoader } from '@forge/config/config-loader';
import fs from 'fs';
import os from 'os';
import path from 'path';

jest.mock('fs');

describe('ConfigLoader', () => {
  let loader: ConfigLoader;
  const mockFs = fs as jest.Mocked<typeof fs>;

  beforeEach(() => {
    loader = new ConfigLoader();
    jest.clearAllMocks();
    
    // Clear environment variables
    delete process.env.OPENAI_API_KEY;
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.FORGE_AUTO_ROUTING;
    delete process.env.FORGE_DEBUG;
  });

  describe('load', () => {
    it('should load default configuration', async () => {
      mockFs.existsSync.mockReturnValue(false);
      
      const config = await loader.load();
      
      expect(config.providers.default).toBe('openai');
      expect(config.providers.routing?.tool_use).toBe('openai');
      expect(config.providers.routing?.planning).toBe('anthropic');
    });

    it('should load user configuration from home directory', async () => {
      const configPath = path.join(os.homedir(), '.forge', 'config.yaml');
      const userConfig = `
providers:
  default: anthropic
  routing:
    planning: ensemble
`;
      
      mockFs.existsSync.mockImplementation((path) => path === configPath);
      mockFs.readFileSync.mockReturnValue(userConfig);
      
      const config = await loader.load();
      
      expect(config.providers.default).toBe('anthropic');
      expect(config.providers.routing?.planning).toBe('ensemble');
      expect(config.providers.routing?.tool_use).toBe('openai'); // From default
    });

    it('should apply environment variables', async () => {
      mockFs.existsSync.mockReturnValue(false);
      
      process.env.OPENAI_API_KEY = 'test-key';
      process.env.FORGE_DEBUG = 'true';
      process.env.FORGE_DEFAULT_PROVIDER = 'anthropic';
      
      const config = await loader.load();
      
      expect(config.providers.models?.openai?.apiKey).toBe('test-key');
      expect(config.features?.debug).toBe(true);
      expect(config.providers.default).toBe('anthropic');
    });

    it('should validate required configuration', async () => {
      mockFs.existsSync.mockReturnValue(false);
      
      // No OpenAI key when it's the default provider
      await expect(loader.load()).rejects.toThrow('OpenAI API key not found');
    });

    it('should merge configurations deeply', async () => {
      const configPath = path.join(process.cwd(), '.forge', 'config.yaml');
      const userConfig = `
providers:
  models:
    openai:
      default: gpt-4-turbo
features:
  debug: true
`;
      
      mockFs.existsSync.mockImplementation((path) => path === configPath);
      mockFs.readFileSync.mockReturnValue(userConfig);
      process.env.OPENAI_API_KEY = 'test-key';
      
      const config = await loader.load();
      
      expect(config.providers.models?.openai?.default).toBe('gpt-4-turbo');
      expect(config.providers.models?.openai?.apiKey).toBe('test-key');
      expect(config.features?.debug).toBe(true);
      expect(config.features?.autoRouting).toBe(true); // From default
    });
  });

  describe('getRoutingConfig', () => {
    it('should return routing configuration', async () => {
      mockFs.existsSync.mockReturnValue(false);
      process.env.OPENAI_API_KEY = 'test-key';
      
      await loader.load();
      const routing = loader.getRoutingConfig();
      
      expect(routing).toEqual({
        tool_use: 'openai',
        planning: 'anthropic',
        code_gen: 'anthropic',
        synthesis: 'ensemble',
        safety_check: 'ensemble',
        default: 'openai'
      });
    });
  });
});