"""Evidence Suite - Prometheus Metrics
Exposes application metrics in Prometheus format.
"""

import os
import time
from collections import defaultdict
from typing import Any

from loguru import logger


class MetricsCollector:
    """Collects and exports application metrics.

    Provides Prometheus-compatible metrics output.
    """

    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._labels: dict[str, dict[str, str]] = {}
        self._start_time = time.time()

    def inc_counter(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        if labels:
            self._labels[key] = labels

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        if labels:
            self._labels[key] = labels

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Observe a value for a histogram metric."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        if labels:
            self._labels[key] = labels

        # Keep only last 1000 observations per metric
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_all(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0,
                }
                for k, v in self._histograms.items()
            },
            "uptime_seconds": time.time() - self._start_time,
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Add help and type comments
        lines.append("# HELP evidence_suite_uptime_seconds Application uptime in seconds")
        lines.append("# TYPE evidence_suite_uptime_seconds gauge")
        lines.append(f"evidence_suite_uptime_seconds {time.time() - self._start_time:.2f}")
        lines.append("")

        # Export counters
        counter_names = set(k.split("{")[0] for k in self._counters)
        for name in sorted(counter_names):
            lines.append(f"# HELP {name} Counter metric")
            lines.append(f"# TYPE {name} counter")
            for key, value in sorted(self._counters.items()):
                if key.startswith(name):
                    lines.append(f"{key} {value}")
            lines.append("")

        # Export gauges
        gauge_names = set(k.split("{")[0] for k in self._gauges)
        for name in sorted(gauge_names):
            lines.append(f"# HELP {name} Gauge metric")
            lines.append(f"# TYPE {name} gauge")
            for key, value in sorted(self._gauges.items()):
                if key.startswith(name):
                    lines.append(f"{key} {value}")
            lines.append("")

        # Export histograms
        histogram_names = set(k.split("{")[0] for k in self._histograms)
        for name in sorted(histogram_names):
            lines.append(f"# HELP {name} Histogram metric")
            lines.append(f"# TYPE {name} histogram")
            for key, values in sorted(self._histograms.items()):
                if key.startswith(name) and values:
                    base_key = key.split("{")[0]
                    labels = key[len(base_key) :] if "{" in key else ""

                    # Calculate histogram buckets
                    buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
                    bucket_counts = [0] * len(buckets)

                    for v in values:
                        for i, b in enumerate(buckets):
                            if v <= b:
                                bucket_counts[i] += 1

                    # Cumulative counts
                    for i in range(1, len(bucket_counts)):
                        bucket_counts[i] += bucket_counts[i - 1]

                    for i, b in enumerate(buckets):
                        if labels:
                            bucket_labels = labels[:-1] + f',le="{b}"}}'
                        else:
                            bucket_labels = f'{{le="{b}"}}'
                        lines.append(f"{base_key}_bucket{bucket_labels} {bucket_counts[i]}")

                    # +Inf bucket
                    if labels:
                        inf_labels = labels[:-1] + ',le="+Inf"}'
                    else:
                        inf_labels = '{le="+Inf"}'
                    lines.append(f"{base_key}_bucket{inf_labels} {len(values)}")

                    # Sum and count
                    if labels:
                        lines.append(f"{base_key}_sum{labels} {sum(values):.6f}")
                        lines.append(f"{base_key}_count{labels} {len(values)}")
                    else:
                        lines.append(f"{base_key}_sum {sum(values):.6f}")
                        lines.append(f"{base_key}_count {len(values)}")
            lines.append("")

        return "\n".join(lines)


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics


# Convenience functions
def inc_request_count(method: str, endpoint: str, status: int) -> None:
    """Increment HTTP request counter."""
    _metrics.inc_counter(
        "evidence_suite_http_requests_total",
        labels={"method": method, "endpoint": endpoint, "status": str(status)},
    )


def observe_request_duration(method: str, endpoint: str, duration: float) -> None:
    """Record HTTP request duration."""
    _metrics.observe_histogram(
        "evidence_suite_http_request_duration_seconds",
        duration,
        labels={"method": method, "endpoint": endpoint},
    )


def inc_analysis_count(evidence_type: str, status: str) -> None:
    """Increment analysis counter."""
    _metrics.inc_counter(
        "evidence_suite_analysis_total",
        labels={"evidence_type": evidence_type, "status": status},
    )


def observe_analysis_duration(evidence_type: str, duration: float) -> None:
    """Record analysis duration."""
    _metrics.observe_histogram(
        "evidence_suite_analysis_duration_seconds",
        duration,
        labels={"evidence_type": evidence_type},
    )


def set_active_connections(count: int) -> None:
    """Set active connection gauge."""
    _metrics.set_gauge("evidence_suite_active_connections", count)


def set_queue_size(queue_name: str, size: int) -> None:
    """Set queue size gauge."""
    _metrics.set_gauge(
        "evidence_suite_queue_size",
        size,
        labels={"queue": queue_name},
    )


async def collect_system_metrics() -> None:
    """Collect system-level metrics."""
    import psutil

    try:
        # CPU usage
        _metrics.set_gauge("evidence_suite_cpu_usage_percent", psutil.cpu_percent())

        # Memory usage
        mem = psutil.virtual_memory()
        _metrics.set_gauge("evidence_suite_memory_usage_bytes", mem.used)
        _metrics.set_gauge("evidence_suite_memory_total_bytes", mem.total)
        _metrics.set_gauge("evidence_suite_memory_usage_percent", mem.percent)

        # Disk usage
        disk = psutil.disk_usage("/")
        _metrics.set_gauge("evidence_suite_disk_usage_bytes", disk.used)
        _metrics.set_gauge("evidence_suite_disk_total_bytes", disk.total)
        _metrics.set_gauge("evidence_suite_disk_usage_percent", disk.percent)

    except Exception as e:
        logger.warning(f"Failed to collect system metrics: {e}")


async def collect_database_metrics() -> None:
    """Collect database-related metrics."""
    try:
        from core.database.monitoring import get_monitor

        monitor = get_monitor()
        pool_stats = monitor.get_pool_stats()

        _metrics.set_gauge(
            "evidence_suite_db_pool_size",
            pool_stats.get("size", 0),
        )
        _metrics.set_gauge(
            "evidence_suite_db_pool_checked_out",
            pool_stats.get("checked_out", 0),
        )
        _metrics.set_gauge(
            "evidence_suite_db_pool_overflow",
            pool_stats.get("overflow", 0),
        )

        query_stats = monitor.get_query_stats()
        _metrics.set_gauge(
            "evidence_suite_db_queries_total",
            query_stats.get("total_queries", 0),
        )
        _metrics.set_gauge(
            "evidence_suite_db_slow_queries_total",
            query_stats.get("slow_queries", 0),
        )

    except Exception as e:
        logger.warning(f"Failed to collect database metrics: {e}")
