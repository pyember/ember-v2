# Forge Routing Flow

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                              │
│                    "refactor this function"                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Intent Detection                           │
│                                                                 │
│  if (tools present) → 'tool_use'                              │
│  if (planning keywords) → 'planning'                          │
│  if (code keywords) → 'code_gen'                             │
│  if (safety keywords) → 'safety_check'                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Provider Router                             │
│                                                                 │
│  'tool_use' → OpenAI (reliable function calling)              │
│  'planning' → Anthropic (superior reasoning)                  │
│  'code_gen' → Anthropic (better code quality)                │
│  'safety_check' → Ensemble (multiple models vote)             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Ember Bridge                               │
│                                                                 │
│  1. Convert request to Ember format                           │
│  2. Call appropriate Ember model                              │
│  3. Convert response back to OpenAI format                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response to User                             │
│                                                                 │
│  Maintains exact same format as original Codex                │
│  But with optimal model selection behind the scenes           │
└─────────────────────────────────────────────────────────────────┘
```

## Example Routing Decisions

### Example 1: File Operation
```
Input: "List all Python files in src/"
Intent: tool_use (contains 'list')
Provider: OpenAI
Reason: OpenAI has mature, reliable function calling
```

### Example 2: Architecture Planning
```
Input: "How should I structure this microservices architecture?"
Intent: planning (contains 'how should', 'structure')
Provider: Anthropic
Reason: Claude excels at high-level reasoning and planning
```

### Example 3: Code Generation
```
Input: "Implement a binary search tree with balance operations"
Intent: code_gen (contains 'implement')
Provider: Anthropic
Reason: Claude produces higher quality code implementations
```

### Example 4: Dangerous Operation
```
Input: "Delete all test files that haven't been modified in 30 days"
Intent: safety_check (contains 'delete')
Provider: Ensemble (GPT-4 + Claude)
Reason: Multiple models must agree before destructive operations
```

## Configuration Examples

### Default Configuration
```yaml
routing:
  tool_use: openai
  planning: anthropic
  code_gen: anthropic
  synthesis: ensemble
  default: openai
```

### Conservative Configuration
```yaml
routing:
  tool_use: openai
  planning: openai
  code_gen: openai
  synthesis: openai
  default: openai
```

### Aggressive Multi-Model Configuration
```yaml
routing:
  tool_use: openai       # Still use OpenAI for tools
  planning: ensemble     # Multiple models for planning
  code_gen: ensemble     # Multiple models for code
  synthesis: ensemble    # Multiple models for synthesis
  safety_check: ensemble # Multiple models for safety
  default: anthropic     # Default to Claude
```

### Cost-Optimized Configuration
```yaml
routing:
  tool_use: openai
  planning: gpt-3.5-turbo    # Cheaper for simple planning
  code_gen: anthropic        # Quality matters for code
  synthesis: gpt-3.5-turbo   # Cheaper for summaries
  default: gpt-3.5-turbo     # Default to cheapest
```

## Intent Detection Keywords

### Tool Use
- Commands: run, execute, list, create, delete, read, write
- File operations: file, directory, folder
- System: shell, command, terminal

### Planning
- Questions: how should, what approach, best way
- Architecture: design, structure, organize
- Strategy: plan, approach, strategy

### Code Generation
- Implementation: implement, create, write, build
- Code structures: function, class, method, component
- Refactoring: refactor, optimize, improve

### Safety Check
- Destructive: delete, remove, destroy, drop
- Modifications: modify, change, update (in production)
- Security: password, key, secret, credential