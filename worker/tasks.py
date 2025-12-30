"""Evidence Suite - Background Worker Tasks
ARQ task definitions for async evidence processing.
"""

import asyncio
import os
from datetime import datetime
from typing import Any
from uuid import UUID

from arq import cron
from arq.connections import RedisSettings
from loguru import logger


# Worker settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
WORKER_MAX_JOBS = int(os.getenv("WORKER_MAX_JOBS", "10"))
WORKER_JOB_TIMEOUT = int(os.getenv("WORKER_JOB_TIMEOUT", "600"))
WORKER_MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", "3"))


async def startup(ctx: dict[str, Any]) -> None:
    """Worker startup - initialize connections."""
    logger.info("Worker starting up...")

    # Initialize database connection
    from core.database.session import init_db_async

    await init_db_async()

    # Initialize pipeline
    from core.pipeline_manager import get_pipeline

    ctx["pipeline"] = await get_pipeline()

    logger.info("Worker ready to process jobs")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Worker shutdown - cleanup."""
    logger.info("Worker shutting down...")

    # Cleanup pipeline
    if "pipeline" in ctx:
        pipeline = ctx["pipeline"]
        if hasattr(pipeline, "shutdown"):
            await pipeline.shutdown()

    logger.info("Worker shutdown complete")


async def analyze_evidence_task(
    ctx: dict[str, Any],
    evidence_id: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a single piece of evidence.

    Args:
        ctx: ARQ context with pipeline
        evidence_id: UUID of evidence to analyze
        options: Optional processing options

    Returns:
        Analysis result summary
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from core.database import AnalysisJob, EvidenceRecord
    from core.database import EvidenceStatus as DBEvidenceStatus
    from core.database.session import get_async_session

    job_id = ctx.get("job_id", "unknown")
    logger.info(f"[Job {job_id}] Starting analysis for evidence {evidence_id}")

    try:
        evidence_uuid = UUID(evidence_id)
    except ValueError:
        return {"success": False, "error": "Invalid evidence ID"}

    async with get_async_session() as db:
        # Get evidence record
        result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_uuid))
        evidence = result.scalar_one_or_none()

        if not evidence:
            return {"success": False, "error": "Evidence not found"}

        # Update status to processing
        evidence.status = DBEvidenceStatus.PROCESSING
        await db.commit()

        # Create or update analysis job
        job_result = await db.execute(
            select(AnalysisJob).where(AnalysisJob.evidence_id == evidence_uuid)
        )
        job = job_result.scalar_one_or_none()

        if not job:
            job = AnalysisJob(
                evidence_id=evidence_uuid,
                status="running",
                current_stage="initialization",
                started_at=datetime.utcnow(),
            )
            db.add(job)
        else:
            job.status = "running"
            job.started_at = datetime.utcnow()

        await db.commit()

        try:
            # Get pipeline from context
            pipeline = ctx.get("pipeline")
            if not pipeline:
                from core.pipeline_manager import get_pipeline

                pipeline = await get_pipeline()

            # Read evidence file
            if evidence.storage_path and os.path.exists(evidence.storage_path):
                with open(evidence.storage_path, "rb") as f:
                    content = f.read()
            else:
                content = (evidence.extracted_text or "").encode()

            # Update job progress
            job.current_stage = "ocr_processing"
            job.progress_percent = 20
            await db.commit()

            # Process through pipeline
            from core.models import EvidencePacket, EvidenceType

            packet = EvidencePacket(
                evidence_id=str(evidence_uuid),
                case_id=str(evidence.case_id),
                evidence_type=EvidenceType(evidence.evidence_type.value),
                raw_content=content,
                metadata={
                    "filename": evidence.original_filename,
                    "mime_type": evidence.mime_type,
                },
            )

            # Run analysis stages
            job.current_stage = "behavioral_analysis"
            job.progress_percent = 50
            await db.commit()

            result_packet = await pipeline.process(packet)

            job.current_stage = "fusion"
            job.progress_percent = 80
            await db.commit()

            # Update evidence with results
            if result_packet.behavioral_indicators:
                evidence.behavioral_indicators = result_packet.behavioral_indicators.model_dump()

            if result_packet.fusion_result:
                evidence.fusion_results = result_packet.fusion_result.model_dump()
                evidence.fused_score = result_packet.fusion_result.fused_confidence
                evidence.fused_classification = result_packet.fusion_result.primary_classification
                evidence.confidence = result_packet.fusion_result.fused_confidence

            if result_packet.extracted_text:
                evidence.extracted_text = result_packet.extracted_text

            evidence.status = DBEvidenceStatus.ANALYZED
            evidence.analyzed_at = datetime.utcnow()

            # Complete job
            job.status = "completed"
            job.current_stage = "complete"
            job.progress_percent = 100
            job.completed_at = datetime.utcnow()

            await db.commit()

            logger.info(f"[Job {job_id}] Analysis complete for evidence {evidence_id}")

            return {
                "success": True,
                "evidence_id": evidence_id,
                "status": "analyzed",
                "fused_score": evidence.fused_score,
                "classification": evidence.fused_classification,
            }

        except Exception as e:
            logger.error(f"[Job {job_id}] Analysis failed: {e}")

            evidence.status = DBEvidenceStatus.ERROR
            job.status = "failed"
            job.error_message = str(e)
            job.retry_count = (job.retry_count or 0) + 1

            await db.commit()

            return {"success": False, "error": str(e)}


async def process_batch_task(
    ctx: dict[str, Any],
    evidence_ids: list[str],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process multiple evidence items in batch.

    Args:
        ctx: ARQ context
        evidence_ids: List of evidence UUIDs
        options: Optional processing options

    Returns:
        Batch processing results
    """
    logger.info(f"Starting batch processing for {len(evidence_ids)} items")

    results = []
    for evidence_id in evidence_ids:
        result = await analyze_evidence_task(ctx, evidence_id, options)
        results.append(result)

        # Small delay between items
        await asyncio.sleep(0.1)

    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful

    logger.info(f"Batch complete: {successful} successful, {failed} failed")

    return {
        "total": len(evidence_ids),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


async def cleanup_old_jobs(ctx: dict[str, Any]) -> None:
    """Cron task to cleanup old completed jobs."""
    from datetime import timedelta

    from sqlalchemy import delete

    from core.database import AnalysisJob
    from core.database.session import get_async_session

    logger.info("Running job cleanup...")

    cutoff = datetime.utcnow() - timedelta(days=7)

    async with get_async_session() as db:
        result = await db.execute(
            delete(AnalysisJob).where(
                AnalysisJob.status == "completed",
                AnalysisJob.completed_at < cutoff,
            )
        )
        await db.commit()

        logger.info(f"Cleaned up {result.rowcount} old jobs")


# ARQ Worker Settings
class WorkerSettings:
    """ARQ worker configuration."""

    redis_settings = RedisSettings(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
    )

    functions = [
        analyze_evidence_task,
        process_batch_task,
    ]

    cron_jobs = [
        cron(cleanup_old_jobs, hour=3, minute=0),  # Run at 3 AM daily
    ]

    on_startup = startup
    on_shutdown = shutdown

    max_jobs = WORKER_MAX_JOBS
    job_timeout = WORKER_JOB_TIMEOUT
    max_tries = WORKER_MAX_RETRIES

    # Health check
    health_check_interval = 30
    health_check_key = "evidence-suite:worker:health"
