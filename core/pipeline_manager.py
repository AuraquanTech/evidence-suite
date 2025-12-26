"""Evidence Suite - Pipeline Manager
Singleton pipeline manager with caching and circuit breaker.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from core.cache import CacheManager, get_cache
from core.models import EvidencePacket, EvidenceType


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for pipeline failures.
    Prevents cascade failures by stopping calls to failing services.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = 0
    last_failure_time: float = 0
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker entering half-open state")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")

    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened - service still failing")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class PipelineManager:
    """Singleton manager for the evidence pipeline.

    Features:
    - Lazy pipeline initialization
    - Result caching
    - Circuit breaker for failures
    - Metrics aggregation
    """

    _instance: Optional["PipelineManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._pipeline = None
        self._cache: CacheManager | None = None
        self._circuit_breaker = CircuitBreaker()
        self._cache_hits = 0
        self._cache_misses = 0
        self._initialized = True
        logger.info("PipelineManager initialized")

    async def get_pipeline(self):
        """Get or create the singleton pipeline."""
        if self._pipeline is None:
            async with PipelineManager._lock:
                if self._pipeline is None:
                    from pipeline import EvidencePipeline

                    self._pipeline = EvidencePipeline()
                    await self._pipeline.initialize()
                    logger.info("Pipeline singleton created")
        return self._pipeline

    async def get_cache(self) -> CacheManager | None:
        """Get cache manager."""
        if self._cache is None:
            try:
                self._cache = await get_cache()
            except Exception as e:
                logger.warning(f"Cache unavailable: {e}")
        return self._cache

    async def process(
        self,
        content: bytes,
        evidence_type: EvidenceType = EvidenceType.TEXT,
        case_id: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Process evidence with caching and circuit breaker.

        Returns dict with:
        - success: bool
        - cached: bool
        - result: processed data or error
        """
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            return {
                "success": False,
                "cached": False,
                "error": "Circuit breaker open - service temporarily unavailable",
            }

        # Check cache
        if use_cache:
            cache = await self.get_cache()
            if cache and cache.is_connected:
                cache_key = CacheManager.hash_text(content.decode("utf-8", errors="ignore"))
                cached_result = await cache.get_analysis(cache_key)
                if cached_result:
                    self._cache_hits += 1
                    logger.debug(f"Cache hit for {cache_key[:8]}")
                    return {"success": True, "cached": True, "result": cached_result}
                self._cache_misses += 1

        # Process through pipeline
        try:
            pipeline = await self.get_pipeline()

            packet = EvidencePacket(
                raw_content=content, evidence_type=evidence_type, case_id=case_id
            )

            result = await pipeline.process(packet)

            if result.success:
                self._circuit_breaker.record_success()

                # Cache successful result
                if use_cache:
                    cache = await self.get_cache()
                    if cache and cache.is_connected:
                        cache_key = CacheManager.hash_text(content.decode("utf-8", errors="ignore"))
                        result_data = {
                            "behavioral_indicators": result.packet.behavioral_indicators,
                            "fusion_results": result.packet.fusion_results,
                            "extracted_text": result.packet.extracted_text,
                            "stage_times": result.stage_times,
                            "total_time_ms": result.total_time_ms,
                        }
                        await cache.set_analysis(cache_key, result_data)

                return {
                    "success": True,
                    "cached": False,
                    "result": {
                        "packet_id": str(result.packet.id),
                        "behavioral_indicators": result.packet.behavioral_indicators,
                        "fusion_results": result.packet.fusion_results,
                        "extracted_text": result.packet.extracted_text,
                        "stage_times": result.stage_times,
                        "total_time_ms": result.total_time_ms,
                    },
                }
            self._circuit_breaker.record_failure()
            return {
                "success": False,
                "cached": False,
                "error": "; ".join(result.errors) if result.errors else "Unknown error",
            }

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Pipeline error: {e}")
            return {"success": False, "cached": False, "error": str(e)}

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline manager metrics."""
        metrics = {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "circuit_breaker_failures": self._circuit_breaker.failure_count,
        }

        if self._pipeline:
            metrics["pipeline"] = self._pipeline.get_metrics()

        if self._cache_hits + self._cache_misses > 0:
            metrics["cache_hit_rate"] = self._cache_hits / (self._cache_hits + self._cache_misses)

        return metrics

    async def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self._circuit_breaker.state = CircuitState.CLOSED
        self._circuit_breaker.failure_count = 0
        logger.info("Circuit breaker manually reset")

    async def clear_cache(self):
        """Clear pipeline result cache."""
        cache = await self.get_cache()
        if cache and cache.is_connected:
            await cache.invalidate_query_cache("*")
            logger.info("Pipeline cache cleared")

    async def shutdown(self):
        """Shutdown the pipeline."""
        if self._pipeline:
            await self._pipeline.shutdown()
            self._pipeline = None
        logger.info("PipelineManager shutdown complete")


# Convenience function
async def get_pipeline_manager() -> PipelineManager:
    """Get the singleton pipeline manager."""
    return PipelineManager()


async def quick_process(
    content: bytes, evidence_type: EvidenceType = EvidenceType.TEXT, case_id: str | None = None
) -> dict[str, Any]:
    """Quick convenience function for processing evidence.
    Uses singleton pipeline with caching.
    """
    manager = await get_pipeline_manager()
    return await manager.process(content, evidence_type, case_id)
