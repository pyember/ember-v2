# Ember Prompt Specifications

This guide shows how to create type-safe prompts with structured inputs and outputs using Ember's simplified API.

## Simple Prompting

For most use cases, you don't need explicit specifications - just use type hints:

```python
from ember.api import ember
from pydantic import BaseModel, Field
from typing import List

# Define your output structure
class Analysis(BaseModel):
    summary: str
    key_points: List[str]
    sentiment: str = Field(..., description="positive, negative, or neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)

# Use it directly
@ember.op
async def analyze_text(text: str) -> Analysis:
    """Analyze text and extract structured insights."""
    return await ember.llm(
        f"Analyze this text and provide a summary, key points, sentiment, and confidence: {text}",
        output_type=Analysis
    )
```

## Prompt Templates

For reusable prompt patterns, use template functions:

```python
from ember.api import ember

def qa_prompt(question: str, context: str) -> str:
    """Create a question-answering prompt."""
    return f"""Answer the following question based on the provided context.

Context: {context}

Question: {question}

Provide a clear, concise answer. If the answer cannot be found in the context, say so."""

@ember.op
async def answer_question(question: str, context: str) -> str:
    """Answer a question based on context."""
    return await ember.llm(qa_prompt(question, context))
```

## Structured Prompting

For complex prompts with validation:

```python
from ember.api import ember, validate
from pydantic import BaseModel, Field
from typing import List, Optional

class QAInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    context: str = Field(..., min_length=1)
    max_words: int = Field(default=100, ge=10, le=1000)
    include_confidence: bool = False

class QAOutput(BaseModel):
    answer: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    sources: List[str] = []
    reasoning: Optional[str] = None

@ember.op
@validate
async def advanced_qa(input: QAInput) -> QAOutput:
    """Advanced question answering with validation."""
    prompt = f"""Answer this question in {input.max_words} words or less.
    
Context: {input.context}
Question: {input.question}

{"Include your confidence level (0-1)." if input.include_confidence else ""}
Cite specific parts of the context as sources.
"""
    
    return await ember.llm(prompt, output_type=QAOutput)
```

## Dynamic Prompts

Create prompts that adapt based on inputs:

```python
@ember.op
async def dynamic_analyzer(
    text: str,
    analysis_type: str = "summary",
    detail_level: str = "medium"
) -> dict:
    """Analyze text with dynamic prompt based on parameters."""
    
    # Build prompt based on analysis type
    prompts = {
        "summary": "Summarize the main points",
        "sentiment": "Analyze the emotional tone and sentiment",
        "entities": "Extract all named entities (people, places, organizations)",
        "themes": "Identify the main themes and topics"
    }
    
    # Adjust for detail level
    detail_modifiers = {
        "brief": "very briefly (1-2 sentences)",
        "medium": "in moderate detail",
        "detailed": "comprehensively with examples"
    }
    
    prompt = f"""{prompts.get(analysis_type, prompts['summary'])} {detail_modifiers[detail_level]}:

{text}"""
    
    return await ember.llm(prompt, output_type=dict)
```

## Prompt Composition

Build complex prompts from simpler components:

```python
# Reusable prompt components
def format_examples(examples: List[dict]) -> str:
    """Format examples for few-shot learning."""
    formatted = []
    for ex in examples:
        formatted.append(f"Input: {ex['input']}\nOutput: {ex['output']}")
    return "\n\n".join(formatted)

def format_constraints(constraints: List[str]) -> str:
    """Format constraints as a bulleted list."""
    return "\n".join(f"- {c}" for c in constraints)

@ember.op
async def few_shot_classifier(
    text: str,
    examples: List[dict],
    categories: List[str],
    constraints: Optional[List[str]] = None
) -> str:
    """Classify text using few-shot examples."""
    
    prompt_parts = [
        "Classify the following text into one of these categories:",
        ", ".join(categories),
        "\n\nExamples:",
        format_examples(examples)
    ]
    
    if constraints:
        prompt_parts.extend([
            "\n\nConstraints:",
            format_constraints(constraints)
        ])
    
    prompt_parts.extend([
        "\n\nNow classify this text:",
        text,
        "\n\nCategory:"
    ])
    
    prompt = "\n".join(prompt_parts)
    return await ember.llm(prompt)
```

## Advanced Patterns

### Chain of Thought Prompting

```python
class ReasoningStep(BaseModel):
    step_number: int
    thought: str
    conclusion: str

class ReasonedAnswer(BaseModel):
    reasoning_steps: List[ReasoningStep]
    final_answer: str
    confidence: float

@ember.op
async def reason_step_by_step(question: str) -> ReasonedAnswer:
    """Answer a question with step-by-step reasoning."""
    prompt = f"""Answer this question by thinking through it step by step.

Question: {question}

For each step:
1. State what you're thinking about
2. Draw a conclusion from that thought
3. Use that conclusion in the next step

Provide your reasoning steps and final answer."""
    
    return await ember.llm(prompt, output_type=ReasonedAnswer)
```

### Self-Consistency Prompting

```python
@ember.op
async def self_consistent_answer(question: str, num_attempts: int = 3) -> dict:
    """Get a self-consistent answer by sampling multiple times."""
    
    prompt = f"""Answer this question: {question}
    
Think through different approaches and provide your best answer."""
    
    # Get multiple answers
    answers = await ember.parallel([
        ember.llm(prompt, temperature=0.7) for _ in range(num_attempts)
    ])
    
    # Find most common answer
    from collections import Counter
    answer_counts = Counter(answers)
    best_answer = answer_counts.most_common(1)[0][0]
    confidence = answer_counts[best_answer] / num_attempts
    
    return {
        "answer": best_answer,
        "confidence": confidence,
        "all_answers": answers
    }
```

### Prompt Optimization

```python
@ember.op
async def optimize_prompt(
    task_description: str,
    examples: List[dict],
    current_prompt: Optional[str] = None
) -> str:
    """Use an LLM to optimize a prompt for better performance."""
    
    meta_prompt = f"""You are a prompt engineering expert. 
    
Task: {task_description}

Examples of desired input/output:
{format_examples(examples)}

Current prompt: {current_prompt or "None yet"}

Create an improved prompt that will reliably produce the desired outputs for this task.
Focus on clarity, specificity, and including all necessary context."""
    
    return await ember.llm(meta_prompt)
```

## Best Practices

1. **Use Type Hints**: Enable automatic validation and better IDE support
2. **Structured Output**: Use Pydantic models for reliable parsing
3. **Clear Instructions**: Be explicit about format and constraints
4. **Examples Help**: Include examples for complex tasks
5. **Iterative Refinement**: Test and refine prompts based on outputs
6. **Modular Prompts**: Build complex prompts from reusable components
7. **Error Handling**: Handle cases where LLM output doesn't match expected structure

## Testing Prompts

```python
import pytest

@pytest.mark.asyncio
async def test_qa_prompt():
    """Test question answering prompt."""
    result = await answer_question(
        question="What is the capital of France?",
        context="France is a country in Europe. Its capital is Paris."
    )
    assert "Paris" in result

@pytest.mark.asyncio 
async def test_structured_output():
    """Test structured output parsing."""
    result = await analyze_text("Great product! Highly recommend.")
    assert isinstance(result, Analysis)
    assert result.sentiment == "positive"
    assert 0.0 <= result.confidence <= 1.0
```

## Next Steps

- [Operators Guide](./operators.md) - Building composable operators
- [Models Guide](./models.md) - Working with different LLM providers  
- [Data Processing](./data.md) - Processing data with prompts
- [Advanced Patterns](../advanced/prompting.md) - Advanced prompting techniques