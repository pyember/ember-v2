"""Migration Guide - Before/After examples for the new Ember API.

This guide shows how to migrate from the old complex API to the new simplified API.
Each section demonstrates equivalent functionality in both styles.

Key changes:
1. No more class-based operators (just use functions)
2. Direct models() API instead of LMModule
3. Simple data.stream()/load() instead of DatasetBuilder
4. Zero-config @jit instead of manual optimization
"""

# ============================================================================
# MODELS API - Before and After
# ============================================================================

print("=" * 70)
print("MODELS API MIGRATION")
print("=" * 70)

# OLD WAY - Complex initialization
"""
from ember.model_module import LMModule
from ember._internal.registry import ModelRegistry

# Complex setup
registry = ModelRegistry()
lm = LMModule(model_name="gpt-4", registry=registry)
response = lm("What is AI?")
print(response.data)
"""

# NEW WAY - Direct and simple
from ember.api import models

response = models("gpt-4", "What is AI?")
print(response.text)

# Model binding for reuse
gpt4 = models.instance("gpt-4", temperature=0.7)
response = gpt4("Explain quantum computing")

# ============================================================================
# OPERATORS - Before and After
# ============================================================================

print("\n" + "=" * 70)
print("OPERATORS MIGRATION")
print("=" * 70)

# OLD WAY - Complex class hierarchy
"""
from ember.api.operators import Operator, Specification, EmberModel, Field

class TextInput(EmberModel):
    text: str = Field(..., description="Input text")

class TextOutput(EmberModel):
    processed: str = Field(..., description="Processed text")

class TextProcessorSpec(Specification):
    input_model = TextInput
    structured_output = TextOutput

class TextProcessor(Operator):
    specification = TextProcessorSpec()
    
    def forward(self, *, inputs: TextInput) -> TextOutput:
        processed = inputs.text.upper()
        return TextOutput(processed=processed)

# Usage
processor = TextProcessor()
result = processor(text="hello")
print(result.processed)
"""

# NEW WAY - Just functions!
def text_processor(text: str) -> str:
    """Process text - that's it!"""
    return text.upper()

# Use directly
result = text_processor("hello")
print(result)  # HELLO

# Or with @op decorator for operator features
from ember.api import operators

@operators.op
def advanced_processor(text: str) -> dict:
    """Process with metadata."""
    return {
        "original": text,
        "processed": text.upper(),
        "length": len(text)
    }

# ============================================================================
# DATA LOADING - Before and After
# ============================================================================

print("\n" + "=" * 70)
print("DATA API MIGRATION")
print("=" * 70)

# OLD WAY - Complex builder pattern
"""
from ember.data import DatasetBuilder

# Complex configuration
dataset = (DatasetBuilder()
          .name("mmlu")
          .split("test")
          .streaming(True)
          .batch_size(32)
          .build())

for batch in dataset:
    process_batch(batch)
"""

# NEW WAY - Simple and direct
from ember.api import data

# Stream data (default behavior)
for item in data.stream("mmlu"):
    process(item)

# Or load into memory
dataset = data.load("mmlu", split="test")

# Chain operations fluently
results = (data.stream("mmlu")
          .filter(lambda x: x["score"] > 0.5)
          .transform(lambda x: {**x, "processed": True})
          .first(100))

# ============================================================================
# OPTIMIZATION - Before and After
# ============================================================================

print("\n" + "=" * 70)
print("OPTIMIZATION MIGRATION")
print("=" * 70)

# OLD WAY - Manual optimization
"""
from ember._internal.optimization import Optimizer, OptimizationConfig

config = OptimizationConfig(
    enable_jit=True,
    cache_size=1000,
    parallel_execution=True
)

optimizer = Optimizer(config)
optimized_fn = optimizer.optimize(my_function)
"""

# NEW WAY - Zero configuration!
from ember.api.xcs import jit, vmap

# Just add @jit
@jit
def my_function(x):
    return expensive_computation(x)

# Batch processing
batch_fn = vmap(my_function)
results = batch_fn([item1, item2, item3])

# ============================================================================
# ENSEMBLE PATTERNS - Before and After  
# ============================================================================

print("\n" + "=" * 70)
print("ENSEMBLE MIGRATION")
print("=" * 70)

# OLD WAY - Complex operator composition
"""
class EnsembleOperator(Operator):
    def __init__(self, operators: List[Operator]):
        self.operators = operators
        
    def forward(self, *, inputs):
        results = []
        for op in self.operators:
            results.append(op(inputs=inputs))
        return self.aggregate(results)
"""

# NEW WAY - Simple function composition
def expert1(question): 
    return models("gpt-4", question).text

def expert2(question):
    return models("claude-3", question).text

# Use built-in ensemble
ensemble = operators.ensemble(expert1, expert2)
results = ensemble("What is machine learning?")

# Or compose manually
def my_ensemble(question):
    results = [expert1(question), expert2(question)]
    return max(results, key=len)  # Return longest answer

# ============================================================================
# ERROR HANDLING - Before and After
# ============================================================================

print("\n" + "=" * 70)
print("ERROR HANDLING MIGRATION")
print("=" * 70)

# OLD WAY - Complex exception hierarchy
"""
from ember._internal.exceptions import (
    EmberException,
    ModelException,
    OperatorException,
    ValidationException
)

try:
    result = complex_operation()
except ModelException as e:
    handle_model_error(e)
except OperatorException as e:
    handle_operator_error(e)
"""

# NEW WAY - Simple, focused exceptions
from ember._internal.exceptions import (
    ModelNotFoundError,
    ProviderAPIError
)

try:
    response = models("gpt-4", "Hello")
except ModelNotFoundError:
    # Model doesn't exist
    response = models("gpt-3.5-turbo", "Hello")
except ProviderAPIError as e:
    # API issues (rate limits, auth, etc.)
    print(f"API error: {e}")

# ============================================================================
# COMPLETE EXAMPLE - Before and After
# ============================================================================

print("\n" + "=" * 70)
print("COMPLETE PIPELINE MIGRATION")
print("=" * 70)

# OLD WAY - Complex pipeline
"""
class Pipeline(Operator):
    specification = PipelineSpec()
    
    def __init__(self):
        self.preprocessor = PreprocessOperator()
        self.analyzer = AnalysisOperator()
        self.lm = LMModule("gpt-4")
        
    def forward(self, *, inputs):
        preprocessed = self.preprocessor(inputs=inputs)
        analyzed = self.analyzer(inputs=preprocessed)
        response = self.lm(analyzed.data)
        return PipelineOutput(result=response.data)
"""

# NEW WAY - Simple functions
from ember.api import models
from ember.api.xcs import jit

@jit
def preprocess(text: str) -> str:
    return text.strip().lower()

@jit  
def analyze(text: str) -> dict:
    return {
        "text": text,
        "length": len(text),
        "words": len(text.split())
    }

def pipeline(text: str) -> str:
    # Simple composition
    preprocessed = preprocess(text)
    analysis = analyze(preprocessed)
    
    # Direct model call
    prompt = f"Summarize this {analysis['words']}-word text: {preprocessed}"
    return models("gpt-4", prompt).text

# Use it
result = pipeline("  This is my INPUT text!  ")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("MIGRATION SUMMARY")
print("=" * 70)

print("""
Key Migration Points:

1. MODELS:
   - OLD: LMModule with complex setup
   - NEW: models("gpt-4", "prompt")

2. OPERATORS:
   - OLD: Class inheritance + Specification
   - NEW: Just functions (optional @op decorator)

3. DATA:
   - OLD: DatasetBuilder with configuration
   - NEW: data.stream() or data.load()

4. OPTIMIZATION:
   - OLD: Manual configuration
   - NEW: @jit decorator (zero config)

5. COMPOSITION:
   - OLD: Complex operator hierarchies
   - NEW: Function composition

6. ERROR HANDLING:
   - OLD: Complex exception hierarchy
   - NEW: Focused, specific exceptions

The new API is:
- 10x simpler (less code)
- More Pythonic (just functions)
- Zero configuration
- Better performance (automatic optimization)
- Easier to test and debug
""")