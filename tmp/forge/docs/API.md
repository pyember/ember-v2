# Forge API Documentation

## Core Modules

### EmberBridge

The core integration layer between Forge and the Ember framework.

#### `class EmberBridge`

```typescript
constructor(options?: {
  router?: ProviderRouter;
  debug?: boolean;
})
```

##### Methods

###### `createChatCompletion`

Creates a chat completion with intelligent provider routing.

```typescript
async createChatCompletion(
  params: ChatCompletionCreateParamsNonStreaming
): Promise<ChatCompletion>

async createChatCompletion(
  params: ChatCompletionCreateParamsStreaming
): Promise<AsyncIterable<ChatCompletionChunk>>
```

**Example:**
```typescript
const bridge = new EmberBridge();
const response = await bridge.createChatCompletion({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello' }]
});
```

### ProviderRouter

Handles intent detection and provider selection.

#### `class ProviderRouter`

```typescript
constructor(config?: ProviderRouterConfig)
```

##### Methods

###### `detectIntent`

Analyzes messages and tools to determine the intent.

```typescript
detectIntent(
  messages: ChatCompletionMessageParam[], 
  tools?: ChatCompletionTool[]
): string
```

**Returns:** Intent string ('tool_use', 'planning', 'code_gen', 'synthesis', 'safety_check', or 'default')

###### `getProvider`

Returns the provider for a given intent.

```typescript
getProvider(intent: string): string
```

**Returns:** Provider name ('openai', 'anthropic', 'ensemble', etc.)

### ForgeClient

OpenAI-compatible client with intelligent routing.

#### `class ForgeClient`

```typescript
constructor(config?: ForgeClientConfig)
```

##### Properties

###### `chat.completions`

OpenAI-compatible chat completions API.

```typescript
chat: {
  completions: {
    create: (params: CompletionParams) => Promise<ChatCompletion> | AsyncIterable<ChatCompletionChunk>
  }
}
```

**Example:**
```typescript
const client = new ForgeClient();
const response = await client.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Write a function' }],
  stream: true
});

for await (const chunk of response) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

### Configuration

#### `class ConfigLoader`

Loads and manages Forge configuration.

```typescript
constructor()
```

##### Methods

###### `load`

Loads configuration from all sources (files, environment, defaults).

```typescript
async load(): Promise<ForgeConfig>
```

###### `getConfig`

Returns the loaded configuration.

```typescript
getConfig(): ForgeConfig
```

###### `getRoutingConfig`

Returns just the routing configuration.

```typescript
getRoutingConfig(): ProviderRouterConfig | undefined
```

## Types

### Core Types

```typescript
interface ProviderRouterConfig {
  tool_use?: string;
  planning?: string;
  code_gen?: string;
  synthesis?: string;
  safety_check?: string;
  default?: string;
}

interface ForgeClientConfig {
  provider?: string;
  apiKey?: string;
  baseURL?: string;
  emberBridge?: boolean;
  router?: ProviderRouter;
  debug?: boolean;
}

interface ForgeConfig {
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
```

## Factory Functions

### `createForgeClient`

Creates an appropriate client based on configuration.

```typescript
function createForgeClient(
  config?: ForgeClientConfig
): ForgeClient | OpenAI
```

**Example:**
```typescript
// Create intelligent routing client
const client = createForgeClient({
  emberBridge: true,
  debug: true
});

// Create vanilla OpenAI client
const openaiClient = createForgeClient({
  provider: 'openai',
  emberBridge: false
});
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `FORGE_AUTO_ROUTING` | Enable automatic provider routing | `true` |
| `FORGE_DEBUG` | Enable debug logging | `false` |
| `FORGE_DEFAULT_PROVIDER` | Default provider name | `openai` |
| `FORGE_ENABLE_ROUTING` | Force enable routing even for vanilla OpenAI | `false` |

## Error Handling

Forge maintains compatibility with OpenAI error types while adding provider context:

```typescript
try {
  const response = await client.chat.completions.create(params);
} catch (error) {
  if (error.status === 429) {
    // Rate limit error - automatic retry with backoff
  } else if (error.status === 401) {
    // Authentication error - check API keys
  }
}
```

## Integration Examples

### Basic Integration

```typescript
import { createForgeClient } from '@ember-ai/forge';

const client = createForgeClient();

// Works exactly like OpenAI client
const response = await client.chat.completions.create({
  model: 'gpt-4',
  messages: [{ role: 'user', content: 'Hello' }]
});
```

### Custom Routing

```typescript
import { ProviderRouter, createForgeClient } from '@ember-ai/forge';

const router = new ProviderRouter({
  tool_use: 'openai',
  planning: 'ensemble',
  code_gen: 'anthropic',
  default: 'openai'
});

const client = createForgeClient({ router });
```

### Ensemble Operations

```typescript
import { EnsembleClient } from '@ember-ai/forge';

const ensemble = new EnsembleClient([
  'gpt-4',
  'claude-3-opus',
  'gpt-4-turbo'
]);

// All models vote on response
const response = await ensemble.chat.completions.create({
  model: 'ensemble',
  messages: [{ role: 'user', content: 'Should we refactor this code?' }]
});
```