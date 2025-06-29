from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

# We define separate type variables for system output and correct answer.
T_out = TypeVar("T_out")
T_ans = TypeVar("T_ans")


@dataclass
class EvaluationResult:
    """Result of evaluating system output.

    Attributes:
        is_correct: Whether output meets criteria.
        score: Numeric quality score.
        metadata: Additional evaluation details.
    """

    is_correct: bool
    score: float
    metadata: Optional[Dict[str, Any]] = None


class IEvaluator(ABC, Generic[T_out, T_ans]):
    """Interface for output evaluation."""

    @abstractmethod
    def evaluate(
        self, system_output: T_out, correct_answer: T_ans, **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate system output against expected answer.

        Args:
            system_output: Raw system output.
            correct_answer: Expected correct answer.
            **kwargs: Additional arguments.

        Returns:
            Evaluation result with score and metadata.
        """
        raise NotImplementedError


class IStatefulEvaluator(ABC, Generic[T_out, T_ans]):
    """Evaluator that accumulates results across samples."""

    @abstractmethod
    def update(self, system_output: T_out, correct_answer: T_ans, **kwargs: Any) -> None:
        """Add sample to internal state.

        Args:
            system_output: System output for sample.
            correct_answer: Expected answer for sample.
            **kwargs: Additional arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> EvaluationResult:
        """Compute aggregated result from all samples.

        Returns:
            EvaluationResult: The aggregated evaluation result.
        """
        raise NotImplementedError

    def evaluate(
        self, system_output: T_out, correct_answer: T_ans, **kwargs: Any
    ) -> EvaluationResult:
        """Convenience method for single-sample evaluation: updates state and computes the result.

        Args:
            system_output (T_out): The system output for the sample.
            correct_answer (T_ans): The expected correct answer.
            **kwargs: Additional keyword arguments.

        Returns:
            EvaluationResult: The evaluation result for the sample.
        """
        self.update(system_output, correct_answer, **kwargs)
        return self.compute()
