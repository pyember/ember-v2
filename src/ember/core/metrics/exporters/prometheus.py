"""Prometheus metrics exporter.

Converts internal metrics format to Prometheus text format with minimal overhead.
"""

from typing import Any, Dict, List, Tuple

from ..metrics import Metrics


class PrometheusExporter:
    """Efficient Prometheus metrics exporter.

    Converts internal metrics to Prometheus text format with O(n) complexity.
    """

    def __init__(self, metrics: Metrics):
        """Initialize exporter with metrics instance.

        Args:
            metrics: Metrics instance to export from
        """
        self._metrics = metrics

    def export(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        snapshot = self._metrics.get_snapshot()
        lines = []

        for key, value in snapshot.items():
            # Parse metric key
            name, labels = self._parse_key(key)

            if isinstance(value, dict) and "buckets" in value:
                # Histogram
                self._append_histogram(lines, name, value, labels)
            elif isinstance(value, (int, float)):
                # Counter or Gauge
                lines.append(f"{name}{{{labels}}} {value}")

        return "\n".join(lines)

    def _parse_key(self, key: str) -> Tuple[str, str]:
        """Parse metric key into name and Prometheus labels.

        Args:
            key: Metric key in format name[tag1=value1,tag2=value2]

        Returns:
            Tuple of (name, formatted_labels)
        """
        if "[" not in key:
            return key, ""

        name, labels_part = key.split("[", 1)
        labels_part = labels_part.rstrip("]")

        # Convert ember format to Prometheus format
        label_pairs = []
        for pair in labels_part.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                label_pairs.append(f'{k}="{v}"')

        return name, ",".join(label_pairs)

    def _append_histogram(
        self, lines: List[str], name: str, hist: Dict[str, Any], labels: str
    ) -> None:
        """Append histogram metrics in Prometheus format.

        Args:
            lines: Output lines to append to
            name: Metric name
            hist: Histogram snapshot
            labels: Formatted Prometheus labels
        """
        # Summary metrics
        lines.append(f"{name}_sum{{{labels}}} {hist['sum']}")
        lines.append(f"{name}_count{{{labels}}} {hist['count']}")

        # Buckets (cumulative)
        cumulative = 0
        for i, count in enumerate(hist["buckets"]):
            cumulative += count
            boundary = hist["boundaries"][i]

            # Format boundary
            if boundary == float("inf"):
                bucket_label = f'{labels},le="+Inf"' if labels else 'le="+Inf"'
            else:
                bucket_label = (
                    f'{labels},le="{boundary}"' if labels else f'le="{boundary}"'
                )

            lines.append(f"{name}_bucket{{{bucket_label}}} {cumulative}")
