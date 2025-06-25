export type Provider = 'openai' | 'anthropic' | 'google';

export interface ProviderInfo {
  id: Provider;
  name: string;
  models: string[];
  description: string;
  keyUrl: string;
  envVar: string;
  testModel: string;
  icon: string;
}

export interface SetupState {
  step: 'welcome' | 'provider' | 'apiKey' | 'test' | 'success';
  provider: Provider | null;
  apiKey: string;
  testResult: {success: boolean; message: string} | null;
  exampleFile: string | null;
}

export const PROVIDERS: Record<Provider, ProviderInfo> = {
  openai: {
    id: 'openai',
    name: 'OpenAI',
    models: ['GPT-4', 'GPT-3.5'],
    description: 'Most popular, great for general use',
    keyUrl: 'https://platform.openai.com/api-keys',
    envVar: 'OPENAI_API_KEY',
    testModel: 'gpt-3.5-turbo',
    icon: '🚀',
  },
  anthropic: {
    id: 'anthropic',
    name: 'Anthropic',
    models: ['Claude 3 Opus', 'Claude 3 Sonnet'],
    description: 'Best for complex reasoning',
    keyUrl: 'https://console.anthropic.com/api-keys',
    envVar: 'ANTHROPIC_API_KEY',
    testModel: 'claude-3-haiku',
    icon: '🧠',
  },
  google: {
    id: 'google',
    name: 'Google',
    models: ['Gemini Pro', 'Gemini Vision'],
    description: 'Multimodal capabilities',
    keyUrl: 'https://makersuite.google.com/app/apikey',
    envVar: 'GOOGLE_API_KEY',
    testModel: 'gemini-pro',
    icon: '✨',
  },
};