/**
 * Integration Example: How to minimally modify Codex to use Ember Bridge
 * 
 * This shows the surgical changes needed in the existing Codex codebase
 */

// ============================================
// FILE: codex-cli/src/utils/openai-client.ts
// ============================================
// ORIGINAL CODE:
/*
export function createOpenAIClient(
  config: OpenAIClientConfig | AppConfig,
): OpenAI | AzureOpenAI {
  // ... existing implementation
  return new OpenAI({
    apiKey: getApiKey(config.provider),
    baseURL: getBaseUrl(config.provider),
    timeout: OPENAI_TIMEOUT_MS,
    defaultHeaders: headers,
  });
}
*/

// MODIFIED CODE:
import { EmberOpenAIClient, ProviderRouter } from '@ember/codex-provider';

export function createOpenAIClient(
  config: OpenAIClientConfig | AppConfig,
): OpenAI | AzureOpenAI | EmberOpenAIClient {
  const headers: Record<string, string> = {};
  if (OPENAI_ORGANIZATION) {
    headers["OpenAI-Organization"] = OPENAI_ORGANIZATION;
  }
  if (OPENAI_PROJECT) {
    headers["OpenAI-Project"] = OPENAI_PROJECT;
  }

  // NEW: Check if we should use Ember bridge
  const useEmberBridge = process.env.USE_EMBER_BRIDGE === 'true' || 
                         config.provider?.startsWith('ember:') ||
                         config.useMultiProvider;
  
  if (useEmberBridge) {
    // Use Ember bridge for multi-provider support
    return new EmberOpenAIClient({
      provider: config.provider?.replace('ember:', ''),
      model: config.model,
      router: config.providerRouter,
      apiKey: getApiKey(config.provider),
    });
  }

  // Original logic for standard providers
  if (config.provider?.toLowerCase() === "azure") {
    return new AzureOpenAI({
      apiKey: getApiKey(config.provider),
      baseURL: getBaseUrl(config.provider),
      apiVersion: AZURE_OPENAI_API_VERSION,
      timeout: OPENAI_TIMEOUT_MS,
      defaultHeaders: headers,
    });
  }

  return new OpenAI({
    apiKey: getApiKey(config.provider),
    baseURL: getBaseUrl(config.provider),
    timeout: OPENAI_TIMEOUT_MS,
    defaultHeaders: headers,
  });
}

// ============================================
// FILE: codex-cli/src/utils/agent/agent-loop.ts
// ============================================
// Add these minimal changes:

type AgentLoopParams = {
  model: string;
  provider?: string;
  config?: AppConfig;
  // NEW: Add provider router
  providerRouter?: ProviderRouter;
  // ... rest of existing params
};

export async function agentLoop(params: AgentLoopParams) {
  const {
    model,
    provider,
    config,
    providerRouter,  // NEW
    // ... rest
  } = params;

  // Create client with potential router
  const client = createOpenAIClient({
    ...config,
    provider,
    providerRouter,  // Pass through router
    useMultiProvider: !!providerRouter,  // Enable if router provided
  });

  // Rest of the function remains unchanged!
  // All tool calls automatically go through OpenAI
  // Planning/reasoning can use different providers
}

// ============================================
// FILE: codex-cli/src/cli.tsx (or wherever CLI is initialized)
// ============================================
// Add configuration loading:

import { ProviderRouter } from '@ember/codex-provider';

// In the CLI initialization
function initializeCLI(args: any) {
  // Load routing configuration
  const routingConfig = loadRoutingConfig(); // From .codex/config.yaml
  
  const providerRouter = routingConfig ? 
    new ProviderRouter(routingConfig.routing) : 
    undefined;
  
  // Pass to agent loop
  await agentLoop({
    model: args.model || 'gpt-4',
    provider: args.provider,
    providerRouter,  // NEW: pass router
    // ... other params
  });
}

// ============================================
// Example .codex/config.yaml
// ============================================
/*
# Provider routing configuration
routing:
  # Always use OpenAI for tool execution
  tool_use: openai
  
  # Use Claude for planning and reasoning
  planning: anthropic
  
  # Use Claude for code generation
  code_gen: anthropic
  
  # Use ensemble for critical decisions
  synthesis: ensemble
  
  # Default fallback
  default: openai

# Ensemble configurations
ensembles:
  default:
    models: ["gpt-4o", "claude-3-opus"]
    strategy: majority
*/

// ============================================
// Usage Examples
// ============================================

// 1. Normal usage - no changes
// $ codex "fix the bug in main.py"
// Uses existing OpenAI for everything

// 2. Enable multi-provider via environment
// $ USE_EMBER_BRIDGE=true codex "refactor this codebase"
// Automatically routes: OpenAI for tools, Claude for planning

// 3. Explicit provider selection
// $ codex --provider ember:anthropic "explain this code"
// Forces Anthropic through Ember bridge

// 4. Custom routing config
// $ codex --routing-config ./custom-routing.yaml "build new feature"
// Uses custom routing rules

// ============================================
// Testing the Integration
// ============================================

async function testIntegration() {
  // Test 1: Ensure tool calls still use OpenAI
  const router = new ProviderRouter();
  const client = new EmberOpenAIClient({ router });
  
  const toolResponse = await client.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: 'List files' }],
    tools: [{ type: 'function', function: { name: 'ls' } }]
  });
  
  console.assert(toolResponse.choices[0].message.tool_calls, 'Should have tool calls');
  
  // Test 2: Planning uses different provider
  const planningResponse = await client.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: 'How should I approach this refactor?' }]
  });
  
  console.assert(planningResponse.choices[0].message.content.includes('[anthropic]'), 
    'Should use Anthropic for planning');
  
  console.log('Integration tests passed! ✅');
}

// Run tests if this file is executed directly
if (require.main === module) {
  testIntegration().catch(console.error);
}