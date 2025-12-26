"""
Evidence Suite - Database Connection Pool Monitoring
Tracks pool usage, query performance, and connection health.
"""
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.pool import Pool
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    duration_ms: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None


@dataclass
class PoolMetrics:
    """Connection pool metrics snapshot."""
    timestamp: datetime
    pool_size: int
    checked_in: int
    checked_out: int
    overflow: int
    invalid: int

    @property
    def utilization(self) -> float:
        """Calculate pool utilization percentage."""
        total = self.pool_size + self.overflow
        if total == 0:
            return 0.0
        return (self.checked_out / total) * 100


class DatabaseMonitor:
    """
    Monitors database connection pool and query performance.

    Tracks:
    - Connection pool utilization
    - Query latencies
    - Slow queries
    - Connection errors
    """

    def __init__(
        self,
        slow_query_threshold_ms: float = 100.0,
        max_history: int = 1000
    ):
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.max_history = max_history

        # Metrics storage
        self._query_history: deque = deque(maxlen=max_history)
        self._slow_queries: deque = deque(maxlen=100)
        self._pool_snapshots: deque = deque(maxlen=60)  # Last hour at 1/min
        self._errors: deque = deque(maxlen=100)

        # Counters
        self._total_queries = 0
        self._total_errors = 0
        self._total_slow_queries = 0

        # Current state
        self._active_queries: Dict[int, float] = {}

    def record_query(
        self,
        query: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """Record a query execution."""
        metrics = QueryMetrics(
            query=query[:500],  # Truncate long queries
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            success=success,
            error=error
        )

        self._query_history.append(metrics)
        self._total_queries += 1

        if not success:
            self._errors.append(metrics)
            self._total_errors += 1

        if duration_ms > self.slow_query_threshold_ms:
            self._slow_queries.append(metrics)
            self._total_slow_queries += 1

    def record_pool_snapshot(self, pool: Pool):
        """Take a snapshot of pool state."""
        snapshot = PoolMetrics(
            timestamp=datetime.utcnow(),
            pool_size=pool.size(),
            checked_in=pool.checkedin(),
            checked_out=pool.checkedout(),
            overflow=pool.overflow(),
            invalid=pool.invalidatedcount() if hasattr(pool, 'invalidatedcount') else 0
        )
        self._pool_snapshots.append(snapshot)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        if not self._pool_snapshots:
            return {"status": "no_data"}

        latest = self._pool_snapshots[-1]
        return {
            "timestamp": latest.timestamp.isoformat(),
            "pool_size": latest.pool_size,
            "connections_in_use": latest.checked_out,
            "connections_available": latest.checked_in,
            "overflow_connections": latest.overflow,
            "utilization_percent": round(latest.utilization, 2),
            "invalid_connections": latest.invalid
        }

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        if not self._query_history:
            return {
                "total_queries": 0,
                "total_errors": 0,
                "avg_latency_ms": 0,
            }

        recent = list(self._query_history)[-100:]  # Last 100 queries
        latencies = [q.duration_ms for q in recent if q.success]

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else avg_latency

        return {
            "total_queries": self._total_queries,
            "total_errors": self._total_errors,
            "total_slow_queries": self._total_slow_queries,
            "error_rate_percent": round((self._total_errors / max(self._total_queries, 1)) * 100, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "slow_query_threshold_ms": self.slow_query_threshold_ms
        }

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent slow queries."""
        queries = list(self._slow_queries)[-limit:]
        return [
            {
                "query": q.query,
                "duration_ms": round(q.duration_ms, 2),
                "timestamp": q.timestamp.isoformat()
            }
            for q in queries
        ]

    def get_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        errors = list(self._errors)[-limit:]
        return [
            {
                "query": e.query,
                "error": e.error,
                "timestamp": e.timestamp.isoformat()
            }
            for e in errors
        ]

    def get_health(self) -> Dict[str, Any]:
        """Get overall database health status."""
        pool_stats = self.get_pool_stats()
        query_stats = self.get_query_stats()

        # Determine health status
        status = "healthy"
        warnings = []

        if pool_stats.get("utilization_percent", 0) > 80:
            status = "warning"
            warnings.append("High connection pool utilization")

        if query_stats.get("error_rate_percent", 0) > 5:
            status = "degraded"
            warnings.append("High error rate")

        if query_stats.get("p95_latency_ms", 0) > 500:
            if status == "healthy":
                status = "warning"
            warnings.append("High query latency")

        return {
            "status": status,
            "warnings": warnings,
            "pool": pool_stats,
            "queries": query_stats
        }

    def reset_stats(self):
        """Reset all statistics."""
        self._query_history.clear()
        self._slow_queries.clear()
        self._pool_snapshots.clear()
        self._errors.clear()
        self._total_queries = 0
        self._total_errors = 0
        self._total_slow_queries = 0


# Global monitor instance
_monitor: Optional[DatabaseMonitor] = None


def get_monitor() -> DatabaseMonitor:
    """Get or create the global database monitor."""
    global _monitor
    if _monitor is None:
        _monitor = DatabaseMonitor()
    return _monitor


@asynccontextmanager
async def monitored_session(session: AsyncSession):
    """Context manager that monitors session queries."""
    monitor = get_monitor()
    start_time = time.perf_counter()
    query_info = {"last_query": None}

    try:
        yield session

        # Record success
        duration_ms = (time.perf_counter() - start_time) * 1000
        if query_info.get("last_query"):
            monitor.record_query(query_info["last_query"], duration_ms, success=True)

    except Exception as e:
        # Record error
        duration_ms = (time.perf_counter() - start_time) * 1000
        monitor.record_query(
            query_info.get("last_query", "unknown"),
            duration_ms,
            success=False,
            error=str(e)
        )
        raise


def setup_pool_monitoring(engine):
    """Set up event listeners for pool monitoring."""
    monitor = get_monitor()

    @event.listens_for(engine.sync_engine.pool, "checkout")
    def on_checkout(dbapi_conn, connection_record, connection_proxy):
        """Called when a connection is checked out from pool."""
        if hasattr(engine.sync_engine, 'pool'):
            monitor.record_pool_snapshot(engine.sync_engine.pool)

    @event.listens_for(engine.sync_engine.pool, "checkin")
    def on_checkin(dbapi_conn, connection_record):
        """Called when a connection is returned to pool."""
        pass  # Could track connection return times

    @event.listens_for(engine.sync_engine.pool, "invalidate")
    def on_invalidate(dbapi_conn, connection_record, exception):
        """Called when a connection is invalidated."""
        monitor.record_query(
            "CONNECTION_INVALIDATED",
            0,
            success=False,
            error=str(exception) if exception else "Unknown"
        )


async def start_monitoring_task(engine, interval_seconds: int = 60):
    """Background task to periodically capture pool snapshots."""
    monitor = get_monitor()

    while True:
        try:
            if hasattr(engine, 'sync_engine') and hasattr(engine.sync_engine, 'pool'):
                monitor.record_pool_snapshot(engine.sync_engine.pool)
        except Exception:
            pass  # Don't crash on monitoring errors

        await asyncio.sleep(interval_seconds)
