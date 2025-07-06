"""Simple cost tracking and reconciliation.

Following Jeff Dean's approach: measure everything, fix what matters.
"""

import logging

from ember.models.schemas import UsageStats

logger = logging.getLogger(__name__)


class CostTracker:
    """Track estimated vs actual costs for accuracy monitoring."""

    def __init__(self):
        self._total_estimated = 0.0
        self._total_actual = 0.0
        self._reconciliation_count = 0
        self._max_deviation = 0.0

    def record_usage(self, usage: UsageStats, model_id: str) -> None:
        """Record usage and check for cost discrepancies.

        This is the key insight from Jeff Dean: measure in production,
        identify discrepancies, fix the important ones.
        """
        if usage.cost_usd is None:
            return

        self._total_estimated += usage.cost_usd

        # If we have actual cost, compare
        if usage.actual_cost_usd is not None:
            self._total_actual += usage.actual_cost_usd
            self._reconciliation_count += 1

            # Calculate deviation
            deviation = abs(usage.cost_usd - usage.actual_cost_usd)
            deviation_pct = (
                (deviation / usage.actual_cost_usd * 100) if usage.actual_cost_usd > 0 else 0
            )

            # Track max deviation
            if deviation > self._max_deviation:
                self._max_deviation = deviation

            # Log significant discrepancies (>5%)
            if deviation_pct > 5:
                logger.warning(
                    f"Cost discrepancy for {model_id}: "
                    f"estimated=${usage.cost_usd:.6f}, "
                    f"actual=${usage.actual_cost_usd:.6f} "
                    f"({deviation_pct:.1f}% difference)"
                )
            elif deviation_pct > 0:
                logger.debug(
                    f"Cost reconciliation for {model_id}: " f"deviation={deviation_pct:.1f}%"
                )

    def get_accuracy_metrics(self) -> dict:
        """Get cost tracking accuracy metrics."""
        if self._reconciliation_count == 0:
            return {"reconciliation_count": 0, "accuracy_pct": 0.0, "max_deviation_usd": 0.0}

        accuracy = 100.0
        if self._total_actual > 0:
            accuracy = (
                100.0 - abs(self._total_estimated - self._total_actual) / self._total_actual * 100
            )

        return {
            "reconciliation_count": self._reconciliation_count,
            "accuracy_pct": round(accuracy, 2),
            "max_deviation_usd": self._max_deviation,
            "total_estimated_usd": self._total_estimated,
            "total_actual_usd": self._total_actual,
        }


# Global tracker instance
_cost_tracker = CostTracker()


def track_usage(usage: UsageStats, model_id: str) -> None:
    """Track usage globally for monitoring."""
    _cost_tracker.record_usage(usage, model_id)


def get_cost_accuracy() -> dict:
    """Get global cost accuracy metrics."""
    return _cost_tracker.get_accuracy_metrics()
