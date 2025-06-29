"""Migration Guide - Before/After examples for the new Ember API.

This guide shows how to migrate from the old complex API to the new simplified API.
Each section demonstrates equivalent functionality in both styles.

NOTE: The "OLD WAY" examples in this guide are hypothetical and represent
patterns from traditional LLM libraries. They show what complex APIs might
look like in other frameworks to contrast with Ember's simplified approach.

Key changes:
1. No more class-based operators (just use functions)
2. Direct models() API instead of LMModule
3. Simple data.stream()/load() instead of DatasetBuilder
4. Zero-config @jit instead of manual optimization
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _shared.conditional_execution import conditional_llm
from ember.api import models, operators, data
from ember.api.xcs import jit


@conditional_llm
def main():
    """Run all migration examples."""

    # ============================================================================
    # MODELS API - Before and After
    # ============================================================================

    print("=" * 70)
    print("MODELS API MIGRATION")
    print("=" * 70)

    # OLD WAY - Complex initialization (hypothetical example)
    print("\nOLD WAY (hypothetical from traditional libraries):")
    print(
        """
    # In traditional LLM libraries, you might see:
    from some_lib.model_module import LMModule
    from some_lib.registry import ModelRegistry
    
    # Complex setup required
    registry = ModelRegistry()
    lm = LMModule(model_name="gpt-4", registry=registry)
    response = lm("What is AI?")
    print(response.data)
    """
    )

    # NEW WAY - Direct and simple
    print("\nNEW WAY - Direct and simple:")
    print("from ember.api import models")
    print("")
    print("response = models('gpt-4', 'What is AI?')")
    print("print(response.text)")

    # Actually run it
    response = models("gpt-4", "What is AI?")
    print(f"\nResult: {response.text}")

    # Model binding for reuse
    print("\n# Model binding for reuse:")
    gpt4 = models.instance("gpt-4", temperature=0.7)
    response = gpt4("Explain quantum computing")
    print(f"Bound model result: {response.text}")

    # ============================================================================
    # OPERATORS - Before and After
    # ============================================================================

    print("\n" + "=" * 70)
    print("OPERATORS MIGRATION")
    print("=" * 70)

    # OLD WAY - Complex operator classes
    print("\nOLD WAY (hypothetical):")
    print(
        """
    class MyOperator(Operator):
        specification = OperatorSpecification(
            input_schema=InputSchema,
            output_schema=OutputSchema
        )
        
        def __init__(self):
            super().__init__()
            self.lm = LMModule("gpt-4")
            
        def execute(self, inputs: InputSchema) -> OutputSchema:
            validated = self.specification.validate_input(inputs)
            result = self.lm(validated.text)
            return self.specification.validate_output({"result": result})
    """
    )

    # NEW WAY - Just functions
    print("\nNEW WAY - Just functions:")
    print("from ember.api import operators")
    print("")
    print("@operators.op  # Optional - adds validation")
    print("def my_function(text: str) -> str:")
    print("    return models('gpt-4', text).text")

    # Simple operator pattern
    @operators.op
    def summarize(text: str) -> str:
        """A simple operator - just a function!"""
        return models("gpt-4", f"Summarize: {text}").text

    result = summarize("Machine learning is a subset of AI...")
    print(f"\nOperator result: {result}")

    # ============================================================================
    # DATA LOADING - Before and After
    # ============================================================================

    print("\n" + "=" * 70)
    print("DATA API MIGRATION")
    print("=" * 70)

    # OLD WAY - Complex builder pattern
    print("\nOLD WAY (hypothetical):")
    print(
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
    )

    # NEW WAY - Simple and direct
    print("\nNEW WAY - Simple and direct:")
    print("from ember.api import data")
    print("")
    print("# Stream data (default behavior)")
    print("for item in data.stream('mmlu'):")
    print("    process(item)")
    print("")
    print("# Or load into memory")
    print("dataset = data.load('mmlu', split='test')")

    # ============================================================================
    # OPTIMIZATION - Before and After
    # ============================================================================

    print("\n" + "=" * 70)
    print("OPTIMIZATION MIGRATION")
    print("=" * 70)

    # OLD WAY - Manual optimization
    print("\nOLD WAY (hypothetical):")
    print(
        """
    # In traditional frameworks:
    from some_lib.optimization import Optimizer, OptimizationConfig
    
    config = OptimizationConfig(
        strategy="dynamic",
        cache_size=1000,
        batch_mode=True
    )
    optimizer = Optimizer(config)
    optimized_fn = optimizer.optimize(my_function)
    """
    )

    # NEW WAY - Zero configuration
    print("\nNEW WAY - Zero configuration:")
    print("from ember.api.xcs import jit")
    print("")
    print("# Just add @jit")
    print("@jit")
    print("def my_function(x):")
    print("    return expensive_computation(x)")

    # ============================================================================
    # OPERATOR COMPOSITION - Before and After
    # ============================================================================

    print("\n" + "=" * 70)
    print("OPERATOR COMPOSITION MIGRATION")
    print("=" * 70)

    # OLD WAY - Complex inheritance
    print("\nOLD WAY (hypothetical):")
    print(
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
    )

    # NEW WAY - Simple function composition
    print("\nNEW WAY - Simple function composition:")

    def expert1(question):
        return models("gpt-4", question).text

    def expert2(question):
        return models("claude-3", question).text

    # Use built-in ensemble
    ensemble = operators.ensemble([expert1, expert2])
    results = ensemble("What is machine learning?")
    print(f"Ensemble results: {results}")

    # Or compose manually
    def my_ensemble(question):
        results = [expert1(question), expert2(question)]
        return max(results, key=len)  # Return longest answer

    result = my_ensemble("What is AI?")
    print(f"Manual ensemble: {result}")

    # ============================================================================
    # ERROR HANDLING - Before and After
    # ============================================================================

    print("\n" + "=" * 70)
    print("ERROR HANDLING MIGRATION")
    print("=" * 70)

    # OLD WAY - Complex exception hierarchy
    print("\nOLD WAY (hypothetical):")
    print(
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
    )

    # NEW WAY - Simple, focused exceptions
    print("\nNEW WAY - Simple, focused exceptions:")
    from ember.api.exceptions import ModelNotFoundError, ProviderAPIError

    try:
        response = models("gpt-4", "Hello")
        print(f"Success: {response.text}")
    except ModelNotFoundError:
        # Model doesn't exist
        response = models("gpt-3.5-turbo", "Hello")
        print(f"Fallback: {response.text}")
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
    print("\nOLD WAY (hypothetical):")
    print(
        """
    class Pipeline(Operator):
        specification = PipelineSpec()
        
        def __init__(self):
            self.preprocessor = PreprocessOperator()
            self.analyzer = AnalyzerOperator()
            self.lm = LMModule("gpt-4")
            
        def execute(self, inputs):
            preprocessed = self.preprocessor(inputs)
            analysis = self.analyzer(preprocessed)
            return self.lm(analysis)
    """
    )

    # NEW WAY - Simple composition
    print("\nNEW WAY - Simple composition:")

    @jit
    def preprocess(text: str) -> str:
        return text.strip().lower()

    @jit
    def analyze(text: str) -> dict:
        return {"text": text, "length": len(text), "words": len(text.split())}

    def pipeline(text: str) -> str:
        # Simple composition
        preprocessed = preprocess(text)
        analysis = analyze(preprocessed)

        # Direct model call
        prompt = f"Summarize this {analysis['words']}-word text: {preprocessed}"
        return models("gpt-4", prompt).text

    # Use it
    result = pipeline("  This is my INPUT text!  ")
    print(f"Pipeline result: {result}")

    # ============================================================================
    # SUMMARY
    # ============================================================================

    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)

    print(
        """
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
"""
    )


if __name__ == "__main__":
    main()
