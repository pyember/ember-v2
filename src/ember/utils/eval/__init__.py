"""Evaluation utilities for Ember core.

This package provides evaluators, pipelines, registries, and stateful evaluators for
assessing system outputs against expected values.

Architecture Design:
    The evaluation system implements a composable, extensible framework for
    assessing AI system outputs. Core principles:

    1. **Protocol-Based Design**: IEvaluator interface enables custom evaluators
    2. **Composability**: Evaluators can be combined via ComposedEvaluator
    3. **Stateful Evaluation**: Support for metrics that aggregate over batches
    4. **Type Safety**: Strong typing for evaluation results and pipelines

Design Rationale:
    Traditional evaluation frameworks often couple evaluation logic with
    specific tasks or models. Ember's design separates concerns:

    - Evaluators are pure functions (stateless) or explicit state machines
    - Results are structured data, not just scores
    - Pipelines handle batching and parallelization
    - Registry enables dynamic evaluator selection

    This enables evaluating any system output (LLM, classical ML, rules-based)
    with the same infrastructure, promoting reuse and consistency.

Performance Considerations:
    - Evaluators are lightweight - minimal overhead per evaluation
    - Batch evaluation uses concurrent execution where possible
    - Stateful evaluators minimize memory via incremental updates
    - Registry uses O(1) lookup for evaluator resolution

Key Components:
    - IEvaluator: Protocol defining evaluator interface
    - EvaluationResult: Structured result with score, passed flag, metadata
    - PipelineEvaluator: Orchestrates evaluation over datasets
    - EvaluatorRegistry: Global registry for evaluator discovery
    - Stateful evaluators: For metrics requiring aggregation (e.g., BLEU)
"""

from .base_evaluator import EvaluationResult, IEvaluator, IStatefulEvaluator
from .evaluators import (
    CodeExecutionEvaluator,
    ComposedEvaluator,
    ExactMatchEvaluator,
    NumericToleranceEvaluator,
    PartialRegexEvaluator,
)
from .pipeline import (
    BatchEvaluationSummary,
    PipelineEvaluator,
    evaluate_batch,
    evaluate_batch_with_summary,
)
from .registry import EvaluatorRegistry
from .stateful_evaluators import AggregatorEvaluator

__all__ = [
    # Base interfaces
    "EvaluationResult",
    "IEvaluator",
    "IStatefulEvaluator",
    # Evaluators
    "CodeExecutionEvaluator",
    "ComposedEvaluator",
    "ExactMatchEvaluator",
    "NumericToleranceEvaluator",
    "PartialRegexEvaluator",
    # Pipeline
    "BatchEvaluationSummary",
    "PipelineEvaluator",
    "evaluate_batch",
    "evaluate_batch_with_summary",
    # Registry
    "EvaluatorRegistry",
    # Stateful
    "AggregatorEvaluator",
]
