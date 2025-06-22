# Getting Started with Ember

Ember is a powerful framework for building LLM-powered applications with a focus on simplicity and performance.

## Installation

```bash
pip install ember-ai
```

## Your First Ember Application

### Hello World

```python
from ember.api import ember

# Simple LLM call
response = await ember.llm("Hello, Ember!")
print(response)
```

### Structured Output

```python
from ember.api import ember
from pydantic import BaseModel

class Greeting(BaseModel):
    message: str
    language: str
    formality: str

# Get structured output
greeting = await ember.llm(
    "Create a greeting for a business meeting",
    output_type=Greeting
)
print(f"{greeting.message} ({greeting.language}, {greeting.formality})")
```

## Creating Operators

Operators are the building blocks of Ember applications. Start simple and add complexity only when needed.

### Simple Function Operator

```python
from ember.api import ember

@ember.op
async def summarize(text: str, max_words: int = 50) -> str:
    """Summarize text to specified length."""
    prompt = f"Summarize this in {max_words} words: {text}"
    return await ember.llm(prompt)

# Use it
summary = await summarize("Long article text here...")
```

### Operator with Structured Output

```python
from ember.api import ember
from pydantic import BaseModel
from typing import List

class Analysis(BaseModel):
    summary: str
    key_points: List[str]
    sentiment: str

@ember.op
async def analyze_document(document: str) -> Analysis:
    """Analyze a document and extract insights."""
    return await ember.llm(
        f"Analyze this document: {document}",
        output_type=Analysis
    )
```

### Composing Operators

```python
from ember.api import ember

@ember.op
async def research_topic(topic: str) -> str:
    """Research a topic by gathering and summarizing information."""
    # Generate search queries
    queries = await ember.llm(
        f"Generate 3 search queries for: {topic}",
        output_type=list[str]
    )
    
    # Search in parallel
    results = await ember.parallel([
        search_web(q) for q in queries
    ])
    
    # Summarize findings
    return await summarize("\n".join(results))
```

## Working with Models

Ember provides a simple interface to various LLM providers:

```python
from ember.api import models

# Use default model
response = await models("What is the capital of France?")

# Use specific model
response = await models("gpt-4", "Explain quantum computing")

# With configuration
response = await models(
    "claude-3",
    "Write a poem",
    temperature=0.8,
    max_tokens=200
)
```

## Data Processing

Load and process data with Ember's data utilities:

```python
from ember.api import data

# Load dataset
dataset = await data.load("path/to/data.json")

# Process in batches
results = await ember.parallel([
    analyze_document(doc) for doc in dataset
])

# Stream results
async for result in ember.stream(dataset, analyze_document):
    print(result.summary)
```

## Performance Optimization

Ember automatically optimizes your code with JIT compilation and parallelization:

```python
from ember.api import ember, jit

# Automatic optimization
@ember.op
@jit  # Enable JIT compilation
async def fast_processor(items: list[str]) -> list[str]:
    return await ember.parallel([
        process_item(item) for item in items
    ])

# Manual performance control
@ember.op
@jit(strategy="aggressive")
async def optimized_pipeline(data):
    # Ember will optimize this entire pipeline
    pass
```

## Error Handling

Ember provides robust error handling:

```python
from ember.api import ember, retry

@ember.op
@retry(max_attempts=3)
async def reliable_operation(input: str) -> str:
    """Operation with automatic retry on failure."""
    return await ember.llm(input)

# With custom error handling
try:
    result = await reliable_operation("Process this")
except Exception as e:
    print(f"Operation failed: {e}")
```

## Next Steps

- Explore [Operators Guide](./quickstart/operators.md) for advanced patterns
- Learn about [Model Configuration](./quickstart/models.md)
- Understand [Data Processing](./quickstart/data.md)
- Dive into [Performance Optimization](./quickstart/performance.md)

## Quick Tips

1. **Start Simple**: Use function operators for most tasks
2. **Type Your Code**: Type hints enable automatic validation
3. **Compose, Don't Inherit**: Build complex behavior by composing simple operators
4. **Let Ember Optimize**: The framework handles parallelization and JIT automatically
5. **Use Structured Output**: Pydantic models make output parsing reliable

## Example: Complete Application

Here's a simple but complete Ember application:

```python
from ember.api import ember, models, data
from pydantic import BaseModel
from typing import List

class EmailAnalysis(BaseModel):
    sender: str
    summary: str
    priority: str
    action_required: bool

@ember.op
async def analyze_email(email: dict) -> EmailAnalysis:
    """Analyze an email and extract key information."""
    return await models(
        "gpt-4",
        f"Analyze this email: {email}",
        output_type=EmailAnalysis
    )

@ember.op
async def process_inbox(emails: List[dict]) -> List[EmailAnalysis]:
    """Process all emails in parallel."""
    return await ember.parallel([
        analyze_email(email) for email in emails
    ])

# Run the application
async def main():
    # Load emails
    emails = await data.load("inbox.json")
    
    # Process them
    analyses = await process_inbox(emails)
    
    # Filter high priority
    urgent = [a for a in analyses if a.priority == "high"]
    print(f"You have {len(urgent)} urgent emails")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Welcome to Ember! Start building powerful LLM applications with simple, composable operators.