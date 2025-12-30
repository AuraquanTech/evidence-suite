"""Evidence Suite - Background Worker with ARQ
Redis-backed async job queue for evidence processing.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from arq import create_pool, cron
from arq.connections import ArqRedis, RedisSettings
from loguru import logger

from core.config import redis_settings


# Job functions
async def process_evidence_job(
    ctx: dict,
    evidence_id: str,
    case_id: str,
    evidence_type: str,
    storage_path: str,
) -> dict[str, Any]:
    """Background job to process evidence through the pipeline.

    Args:
        ctx: ARQ context with redis connection
        evidence_id: UUID of the evidence record
        case_id: UUID of the case
        evidence_type: Type of evidence (text, image, audio, etc.)
        storage_path: Path to the evidence file

    Returns:
        Processing result dictionary
    """
    from core.models import EvidencePacket, EvidenceType
    from pipeline import EvidencePipeline

    logger.info(f"Starting background processing for evidence {evidence_id}")

    start_time = datetime.utcnow()

    try:
        # Read evidence file
        with open(storage_path, "rb") as f:
            content = f.read()

        # Create evidence packet
        packet = EvidencePacket(
            raw_content=content,
            evidence_type=EvidenceType(evidence_type),
            case_id=case_id,
        )

        # Get or create pipeline from context
        pipeline = ctx.get("pipeline")
        if pipeline is None:
            pipeline = EvidencePipeline()
            await pipeline.initialize()
            ctx["pipeline"] = pipeline

        # Process through pipeline
        result = await pipeline.process(packet)

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Store result in Redis for retrieval
        result_data = {
            "evidence_id": evidence_id,
            "success": result.success,
            "stages_completed": [s.value for s in result.stages_completed],
            "total_time_ms": result.total_time_ms,
            "agents_used": result.agents_used,
            "errors": result.errors,
            "processing_time_seconds": processing_time,
            "completed_at": datetime.utcnow().isoformat(),
        }

        # Cache result
        redis: ArqRedis = ctx["redis"]
        await redis.setex(
            f"job_result:{evidence_id}",
            3600,  # 1 hour TTL
            str(result_data),
        )

        logger.info(f"Completed processing evidence {evidence_id} in {processing_time:.2f}s")

        return result_data

    except Exception as e:
        logger.error(f"Failed to process evidence {evidence_id}: {e}")
        return {
            "evidence_id": evidence_id,
            "success": False,
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        }


async def batch_process_job(
    ctx: dict,
    evidence_items: list[dict[str, str]],
) -> dict[str, Any]:
    """Background job to process multiple evidence items.

    Args:
        ctx: ARQ context
        evidence_items: List of dicts with evidence_id, case_id, evidence_type, storage_path

    Returns:
        Batch processing results
    """
    logger.info(f"Starting batch processing for {len(evidence_items)} items")

    results = []
    for item in evidence_items:
        result = await process_evidence_job(
            ctx,
            item["evidence_id"],
            item["case_id"],
            item["evidence_type"],
            item["storage_path"],
        )
        results.append(result)

    success_count = sum(1 for r in results if r.get("success"))

    return {
        "total": len(evidence_items),
        "successful": success_count,
        "failed": len(evidence_items) - success_count,
        "results": results,
    }


async def cleanup_old_results(ctx: dict) -> int:
    """Periodic job to clean up old cached results."""
    logger.info("Running cleanup of old job results")

    redis: ArqRedis = ctx["redis"]

    # Find and delete old results
    pattern = "job_result:*"
    deleted = 0

    async for key in redis.scan_iter(pattern):
        ttl = await redis.ttl(key)
        if ttl == -1:  # No TTL set
            await redis.delete(key)
            deleted += 1

    logger.info(f"Cleaned up {deleted} old job results")
    return deleted


async def health_check(ctx: dict) -> dict[str, Any]:
    """Health check job for monitoring."""
    pipeline = ctx.get("pipeline")

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_initialized": pipeline is not None,
        "agent_status": pipeline.get_agent_status() if pipeline else {},
    }


# Startup and shutdown
async def startup(ctx: dict) -> None:
    """Initialize resources on worker startup."""
    logger.info("Evidence Suite worker starting up...")

    # Pre-initialize pipeline
    from pipeline import EvidencePipeline

    pipeline = EvidencePipeline()
    await pipeline.initialize()
    ctx["pipeline"] = pipeline

    logger.info("Worker startup complete")


async def shutdown(ctx: dict) -> None:
    """Clean up resources on worker shutdown."""
    logger.info("Evidence Suite worker shutting down...")

    pipeline = ctx.get("pipeline")
    if pipeline:
        await pipeline.shutdown()

    logger.info("Worker shutdown complete")


# Worker configuration
class WorkerSettings:
    """ARQ worker settings."""

    functions = [
        process_evidence_job,
        batch_process_job,
        cleanup_old_results,
        health_check,
    ]

    cron_jobs = [
        cron(cleanup_old_results, hour=3, minute=0),  # Run at 3 AM daily
    ]

    on_startup = startup
    on_shutdown = shutdown

    redis_settings = RedisSettings(
        host=redis_settings.host,
        port=redis_settings.port,
        password=redis_settings.password,
        database=redis_settings.db,
    )

    max_jobs = 10
    job_timeout = 600  # 10 minutes
    keep_result = 3600  # 1 hour
    health_check_interval = 30


# Client interface
class JobQueue:
    """Client interface for submitting jobs to the worker."""

    def __init__(self):
        self._pool: ArqRedis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._pool is None:
            self._pool = await create_pool(WorkerSettings.redis_settings)
            logger.info("Job queue connected to Redis")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Job queue disconnected")

    async def enqueue_evidence(
        self,
        evidence_id: str,
        case_id: str,
        evidence_type: str,
        storage_path: str,
        defer_by: timedelta | None = None,
    ) -> str:
        """Enqueue evidence for processing.

        Returns:
            Job ID
        """
        await self.connect()

        job = await self._pool.enqueue_job(
            "process_evidence_job",
            evidence_id,
            case_id,
            evidence_type,
            storage_path,
            _defer_by=defer_by,
        )

        logger.info(f"Enqueued evidence {evidence_id} as job {job.job_id}")
        return job.job_id

    async def enqueue_batch(
        self,
        evidence_items: list[dict[str, str]],
    ) -> str:
        """Enqueue batch of evidence for processing.

        Returns:
            Job ID
        """
        await self.connect()

        job = await self._pool.enqueue_job(
            "batch_process_job",
            evidence_items,
        )

        logger.info(f"Enqueued batch of {len(evidence_items)} items as job {job.job_id}")
        return job.job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a job."""
        await self.connect()

        job = await self._pool.job(job_id)
        if job is None:
            return None

        info = await job.info()
        if info is None:
            return {"status": "unknown"}

        return {
            "job_id": job_id,
            "status": info.status,
            "enqueue_time": info.enqueue_time.isoformat() if info.enqueue_time else None,
            "start_time": info.start_time.isoformat() if info.start_time else None,
            "finish_time": info.finish_time.isoformat() if info.finish_time else None,
            "success": info.success,
            "result": info.result,
        }

    async def get_result(self, evidence_id: str) -> dict[str, Any] | None:
        """Get cached result for an evidence item."""
        await self.connect()

        result = await self._pool.get(f"job_result:{evidence_id}")
        if result:
            import ast

            return ast.literal_eval(result.decode())
        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        await self.connect()

        job = await self._pool.job(job_id)
        if job:
            await job.abort()
            return True
        return False

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        await self.connect()

        info = await self._pool.info()

        return {
            "queued_jobs": info.get("queued_jobs", 0),
            "ongoing_jobs": info.get("ongoing_jobs", 0),
            "result_count": info.get("result_count", 0),
        }


# Singleton instance
_job_queue: JobQueue | None = None


async def get_job_queue() -> JobQueue:
    """Get singleton job queue instance."""
    global _job_queue

    if _job_queue is None:
        _job_queue = JobQueue()
        await _job_queue.connect()

    return _job_queue


async def close_job_queue() -> None:
    """Close the job queue connection."""
    global _job_queue

    if _job_queue:
        await _job_queue.disconnect()
        _job_queue = None
