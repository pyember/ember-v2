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
  color?: string; // Terminal color for the logo
}

export interface SetupState {
  step: 'welcome' | 'setupMode' | 'provider' | 'apiKey' | 'test' | 'success';
  provider: Provider | null;
  apiKey: string;
  testResult: {success: boolean; message: string} | null;
  exampleFile: string | null;
  // Multi-provider setup support
  setupMode?: 'single' | 'all';
  configuredProviders?: Set<Provider>;
  remainingProviders?: Provider[];
}

export const PROVIDERS: Record<Provider, ProviderInfo> = {
  openai: {
    id: 'openai',
    name: 'OpenAI',
    models: ['GPT-4', 'GPT-3.5'],
    description: '',
    keyUrl: 'https://platform.openai.com/settings/organization/api-keys',
    envVar: 'OPENAI_API_KEY',
    testModel: 'gpt-3.5-turbo',
    icon: '◯',
    color: 'green',
  },
  anthropic: {
    id: 'anthropic',
    name: 'Anthropic',
    models: ['Claude 3 Opus', 'Claude 3 Sonnet'],
    description: '',
    keyUrl: 'https://console.anthropic.com/settings/keys',
    envVar: 'ANTHROPIC_API_KEY',
    testModel: 'claude-3-haiku',
    icon: '△',
    color: 'yellow',
  },
  google: {
    id: 'google',
    name: 'Google',
    models: ['Gemini Pro', 'Gemini Vision'],
    description: '',
    keyUrl: 'https://aistudio.google.com/apikey',
    envVar: 'GOOGLE_API_KEY',
    testModel: 'gemini-pro',
    icon: 'G',
    color: 'blue',
  },
};