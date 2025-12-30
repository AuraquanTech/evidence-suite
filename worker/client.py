"""Evidence Suite - Worker Client
Client for enqueuing background tasks.
"""

import os
from typing import Any
from uuid import UUID

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from loguru import logger


# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

_pool: ArqRedis | None = None


async def get_worker_pool() -> ArqRedis:
    """Get or create ARQ Redis pool."""
    global _pool

    if _pool is None:
        _pool = await create_pool(
            RedisSettings(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
            )
        )

    return _pool


async def close_worker_pool() -> None:
    """Close the worker pool."""
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None


async def enqueue_analysis(
    evidence_id: str | UUID,
    options: dict[str, Any] | None = None,
    priority: int = 0,
) -> str | None:
    """Enqueue evidence for analysis.

    Args:
        evidence_id: Evidence UUID
        options: Optional processing options
        priority: Job priority (higher = more urgent)

    Returns:
        Job ID if enqueued successfully
    """
    try:
        pool = await get_worker_pool()

        job = await pool.enqueue_job(
            "analyze_evidence_task",
            str(evidence_id),
            options,
            _job_id=f"analyze_{evidence_id}",
        )

        logger.info(f"Enqueued analysis job: {job.job_id}")
        return job.job_id

    except Exception as e:
        logger.error(f"Failed to enqueue analysis: {e}")
        return None


async def enqueue_batch_analysis(
    evidence_ids: list[str | UUID],
    options: dict[str, Any] | None = None,
) -> str | None:
    """Enqueue batch evidence analysis.

    Args:
        evidence_ids: List of evidence UUIDs
        options: Optional processing options

    Returns:
        Job ID if enqueued successfully
    """
    try:
        pool = await get_worker_pool()

        str_ids = [str(eid) for eid in evidence_ids]

        job = await pool.enqueue_job(
            "process_batch_task",
            str_ids,
            options,
        )

        logger.info(f"Enqueued batch job for {len(evidence_ids)} items: {job.job_id}")
        return job.job_id

    except Exception as e:
        logger.error(f"Failed to enqueue batch: {e}")
        return None


async def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Get status of a background job.

    Args:
        job_id: ARQ job ID

    Returns:
        Job status dict or None
    """
    try:
        pool = await get_worker_pool()
        job = await pool.job(job_id)

        if job is None:
            return None

        info = await job.info()

        return {
            "job_id": job_id,
            "status": info.status if info else "unknown",
            "result": info.result if info else None,
            "enqueue_time": info.enqueue_time.isoformat() if info and info.enqueue_time else None,
            "start_time": info.start_time.isoformat() if info and info.start_time else None,
            "finish_time": info.finish_time.isoformat() if info and info.finish_time else None,
        }

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return None


async def cancel_job(job_id: str) -> bool:
    """Cancel a pending job.

    Args:
        job_id: ARQ job ID

    Returns:
        True if cancelled successfully
    """
    try:
        pool = await get_worker_pool()
        job = await pool.job(job_id)

        if job is None:
            return False

        await job.abort()
        logger.info(f"Cancelled job: {job_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        return False


async def get_queue_info() -> dict[str, Any]:
    """Get worker queue statistics."""
    try:
        pool = await get_worker_pool()

        # Get queue length
        queue_key = "arq:queue"
        queue_len = await pool.llen(queue_key)

        # Get active jobs
        in_progress_key = "arq:in-progress"
        in_progress = await pool.zcard(in_progress_key)

        return {
            "queued": queue_len,
            "in_progress": in_progress,
            "connected": True,
        }

    except Exception as e:
        return {
            "queued": 0,
            "in_progress": 0,
            "connected": False,
            "error": str(e),
        }
