"""Evidence Suite - Comprehensive Logging System
Structured logging with monitoring, metrics, and audit trails.
"""

import json
import logging
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any


# Use loguru for enhanced logging if available
try:
    from loguru import logger as loguru_logger

    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False
    loguru_logger = None


class LogLevel(Enum):
    """Log levels with numeric values."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AUDIT = 25  # Custom level for audit logs


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line: int
    extra: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None
    duration_ms: float | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class MetricEntry:
    """Performance metric entry."""

    name: str
    value: float
    unit: str
    timestamp: str
    tags: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1, tags: dict[str, str] | None = None):
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None):
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value

    def histogram(self, name: str, value: float, tags: dict[str, str] | None = None):
        """Record a histogram value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._histograms[key].append(value)
            # Keep last 1000 values
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]

    def _make_key(self, name: str, tags: dict[str, str] | None = None) -> str:
        """Create metric key with tags."""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}{{{tag_str}}}"
        return name

    def get_stats(self) -> dict[str, Any]:
        """Get all metrics statistics."""
        with self._lock:
            stats = {
                "uptime_seconds": time.time() - self._start_time,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }

            for name, values in self._histograms.items():
                if values:
                    sorted_vals = sorted(values)
                    stats["histograms"][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p50": sorted_vals[len(sorted_vals) // 2],
                        "p95": sorted_vals[int(len(sorted_vals) * 0.95)],
                        "p99": sorted_vals[int(len(sorted_vals) * 0.99)],
                    }

            return stats


class EvidenceSuiteLogger:
    """Main logging class for Evidence Suite."""

    def __init__(
        self,
        name: str = "evidence-suite",
        log_dir: str | None = None,
        log_level: str = "INFO",
        json_format: bool = True,
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.json_format = json_format
        self.metrics = MetricsCollector()
        self._trace_id: str | None = None
        self._span_id: str | None = None

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        if HAS_LOGURU:
            self._setup_loguru(enable_console, enable_file)
        else:
            self._setup_stdlib(enable_console, enable_file)

    def _setup_loguru(self, enable_console: bool, enable_file: bool):
        """Setup loguru logger."""
        loguru_logger.remove()

        if enable_console:
            loguru_logger.add(
                sys.stderr,
                level=self.log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                colorize=True,
            )

        if enable_file:
            # Main log file
            loguru_logger.add(
                self.log_dir / "evidence-suite.log",
                rotation="100 MB",
                retention="30 days",
                compression="gz",
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            )

            # Error log file
            loguru_logger.add(
                self.log_dir / "errors.log",
                rotation="50 MB",
                retention="90 days",
                compression="gz",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
            )

            # JSON log file for structured logging
            if self.json_format:
                loguru_logger.add(
                    self.log_dir / "evidence-suite.jsonl",
                    rotation="100 MB",
                    retention="30 days",
                    compression="gz",
                    level=self.log_level,
                    serialize=True,
                )

        self._logger = loguru_logger

    def _setup_stdlib(self, enable_console: bool, enable_file: bool):
        """Setup standard library logger."""
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.log_level)
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )

        if enable_console:
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(self.log_level)
            console.setFormatter(formatter)
            self._logger.addHandler(console)

        if enable_file:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                self.log_dir / "evidence-suite.log",
                maxBytes=100 * 1024 * 1024,  # 100 MB
                backupCount=5,
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def set_trace_context(self, trace_id: str, span_id: str | None = None):
        """Set trace context for distributed tracing."""
        self._trace_id = trace_id
        self._span_id = span_id

    def clear_trace_context(self):
        """Clear trace context."""
        self._trace_id = None
        self._span_id = None

    def _log(
        self,
        level: str,
        message: str,
        extra: dict[str, Any] | None = None,
        exc_info: bool = False,
    ):
        """Internal log method."""
        if HAS_LOGURU:
            log_func = getattr(self._logger, level.lower())
            if extra:
                log_func(f"{message} | {json.dumps(extra)}")
            else:
                log_func(message)
        else:
            log_func = getattr(self._logger, level.lower())
            if extra:
                log_func(f"{message} | {json.dumps(extra)}", exc_info=exc_info)
            else:
                log_func(message, exc_info=exc_info)

        # Update metrics
        self.metrics.increment(f"logs.{level.lower()}")

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, kwargs if kwargs else None)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("INFO", message, kwargs if kwargs else None)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("WARNING", message, kwargs if kwargs else None)

    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log error message."""
        self._log("ERROR", message, kwargs if kwargs else None, exc_info=exc_info)
        self.metrics.increment("errors.total")

    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log critical message."""
        self._log("CRITICAL", message, kwargs if kwargs else None, exc_info=exc_info)
        self.metrics.increment("errors.critical")

    def audit(self, action: str, resource_type: str, resource_id: str, **kwargs):
        """Log audit entry for compliance."""
        entry = {
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }
        self._log("INFO", f"AUDIT: {action} on {resource_type}/{resource_id}", entry)
        self.metrics.increment("audit.total")
        self.metrics.increment(f"audit.{action}")

    @contextmanager
    def timer(self, name: str, tags: dict[str, str] | None = None):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.metrics.histogram(f"{name}.duration_ms", duration_ms, tags)
            self.debug(f"Timer {name}: {duration_ms:.2f}ms", tags=tags)

    def timed(self, name: str | None = None):
        """Decorator for timing functions."""

        def decorator(func: Callable):
            metric_name = name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    self.metrics.increment(f"{metric_name}.success")
                    return result
                except Exception:
                    self.metrics.increment(f"{metric_name}.error")
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    self.metrics.histogram(f"{metric_name}.duration_ms", duration_ms)

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    self.metrics.increment(f"{metric_name}.success")
                    return result
                except Exception:
                    self.metrics.increment(f"{metric_name}.error")
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    self.metrics.histogram(f"{metric_name}.duration_ms", duration_ms)

            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def record_analysis(
        self,
        evidence_id: str,
        agent_type: str,
        duration_ms: float,
        success: bool,
        result_size: int | None = None,
    ):
        """Record analysis metrics."""
        tags = {"agent": agent_type, "status": "success" if success else "error"}
        self.metrics.histogram("analysis.duration_ms", duration_ms, tags)
        self.metrics.increment(f"analysis.{agent_type}.total")
        if success:
            self.metrics.increment(f"analysis.{agent_type}.success")
        else:
            self.metrics.increment(f"analysis.{agent_type}.error")
        if result_size:
            self.metrics.histogram(f"analysis.{agent_type}.result_size", result_size)

    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
    ):
        """Record HTTP request metrics."""
        tags = {"method": method, "status": str(status_code)}
        self.metrics.histogram("http.duration_ms", duration_ms, tags)
        self.metrics.increment(f"http.requests.{status_code}")
        self.metrics.increment("http.requests.total")

    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics."""
        return self.metrics.get_stats()


# Global logger instance
_logger: EvidenceSuiteLogger | None = None


def get_logger() -> EvidenceSuiteLogger:
    """Get or create the global logger."""
    global _logger
    if _logger is None:
        _logger = EvidenceSuiteLogger()
    return _logger


def configure_logging(
    log_dir: str | None = None,
    log_level: str = "INFO",
    json_format: bool = True,
) -> EvidenceSuiteLogger:
    """Configure and return the logger."""
    global _logger
    _logger = EvidenceSuiteLogger(
        log_dir=log_dir,
        log_level=log_level,
        json_format=json_format,
    )
    return _logger


# Convenience functions
def debug(message: str, **kwargs):
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    get_logger().critical(message, **kwargs)


def audit(action: str, resource_type: str, resource_id: str, **kwargs):
    get_logger().audit(action, resource_type, resource_id, **kwargs)
