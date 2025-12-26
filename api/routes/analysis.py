"""Evidence Suite - Analysis Routes"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.analysis import (
    AnalysisJobResponse,
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    BehavioralIndicators,
    FusionResults,
)
from core.database import (
    AnalysisJob,
    ChainOfCustodyLog,
    EvidenceRecord,
)
from core.database import (
    EvidenceStatus as DBEvidenceStatus,
)
from core.database.session import get_db


router = APIRouter(prefix="/analysis", tags=["Analysis"])


async def run_analysis_pipeline(evidence_id: UUID, db: AsyncSession):
    """Background task to run the analysis pipeline."""
    # Import pipeline components
    try:
        from core.models import EvidenceType
        from pipeline import EvidencePipeline

        # Get evidence record
        result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
        evidence = result.scalar_one_or_none()
        if not evidence:
            return

        # Update status
        evidence.status = DBEvidenceStatus.PROCESSING
        await db.commit()

        # Get job
        job_result = await db.execute(
            select(AnalysisJob).where(AnalysisJob.evidence_id == evidence_id)
        )
        job = job_result.scalar_one_or_none()
        if job:
            job.status = "running"
            job.started_at = datetime.utcnow()
            await db.commit()

        # Run pipeline
        pipeline = EvidencePipeline()

        # Read evidence content
        if evidence.storage_path:
            with open(evidence.storage_path, "rb") as f:
                content = f.read()

            # Determine evidence type
            etype = EvidenceType.TEXT
            if evidence.mime_type:
                if evidence.mime_type.startswith("image"):
                    etype = EvidenceType.IMAGE
                elif evidence.mime_type.startswith("audio"):
                    etype = EvidenceType.AUDIO

            # Process through pipeline
            packet = pipeline.process(
                content,
                evidence_type=etype,
                case_id=str(evidence.case_id),
            )

            # Update evidence with results
            if packet.behavioral_indicators:
                evidence.behavioral_indicators = packet.behavioral_indicators
            if packet.fusion_results:
                evidence.fusion_results = packet.fusion_results
                evidence.fused_score = packet.fusion_results.get("fused_score")
                evidence.fused_classification = packet.fusion_results.get("classification")
                evidence.confidence = packet.fusion_results.get("confidence")
            if packet.extracted_text:
                evidence.extracted_text = packet.extracted_text

            evidence.status = DBEvidenceStatus.ANALYZED
            evidence.analyzed_at = datetime.utcnow()

            # Update job
            if job:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.progress_percent = 100.0

            # Add custody entry
            custody_entry = ChainOfCustodyLog(
                evidence_id=evidence.id,
                agent_id="pipeline",
                agent_type="analysis",
                action="analysis_completed",
                input_hash=evidence.original_hash,
                output_hash=evidence.original_hash,
                success=True,
            )
            db.add(custody_entry)

            await db.commit()

    except Exception as e:
        # Mark as error
        result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
        evidence = result.scalar_one_or_none()
        if evidence:
            evidence.status = DBEvidenceStatus.ERROR
            await db.commit()

        job_result = await db.execute(
            select(AnalysisJob).where(AnalysisJob.evidence_id == evidence_id)
        )
        job = job_result.scalar_one_or_none()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            await db.commit()


@router.post("/", response_model=AnalysisJobResponse, status_code=202)
async def start_analysis(
    request: AnalysisRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    """Start analysis on evidence."""
    # Verify evidence exists
    result = await db.execute(
        select(EvidenceRecord).where(EvidenceRecord.id == request.evidence_id)
    )
    evidence = result.scalar_one_or_none()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    # Check if already processing
    if evidence.status == DBEvidenceStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Analysis already in progress")

    # Create job
    job = AnalysisJob(
        evidence_id=request.evidence_id,
        status="pending",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Queue background task
    background_tasks.add_task(run_analysis_pipeline, request.evidence_id, db)

    return AnalysisJobResponse(
        id=job.id,
        evidence_id=job.evidence_id,
        status=AnalysisStatus.PENDING,
        created_at=job.created_at,
    )


@router.get("/job/{job_id}", response_model=AnalysisJobResponse)
async def get_job_status(job_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get analysis job status."""
    result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status_map = {
        "pending": AnalysisStatus.PENDING,
        "running": AnalysisStatus.RUNNING,
        "completed": AnalysisStatus.COMPLETED,
        "failed": AnalysisStatus.FAILED,
    }

    return AnalysisJobResponse(
        id=job.id,
        evidence_id=job.evidence_id,
        status=status_map.get(job.status, AnalysisStatus.PENDING),
        current_stage=job.current_stage,
        progress_percent=job.progress_percent or 0.0,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


@router.get("/{evidence_id}", response_model=AnalysisResponse)
async def get_analysis_results(evidence_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get analysis results for evidence."""
    result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
    evidence = result.scalar_one_or_none()

    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    if evidence.status not in [DBEvidenceStatus.ANALYZED, DBEvidenceStatus.VERIFIED]:
        raise HTTPException(status_code=400, detail="Evidence not yet analyzed")

    # Build behavioral indicators
    behavioral = None
    if evidence.behavioral_indicators:
        bi = evidence.behavioral_indicators
        behavioral = BehavioralIndicators(
            sentiment_compound=bi.get("sentiment", {}).get("compound", 0),
            sentiment_positive=bi.get("sentiment", {}).get("pos", 0),
            sentiment_negative=bi.get("sentiment", {}).get("neg", 0),
            sentiment_neutral=bi.get("sentiment", {}).get("neu", 0),
            darvo_score=bi.get("darvo_score", 0),
            gaslighting_score=bi.get("gaslighting_score", 0),
            manipulation_score=bi.get("manipulation_score", 0),
            deception_score=bi.get("deception_score", 0),
            primary_behavior=bi.get("primary_behavior"),
            detected_patterns=bi.get("detected_patterns", []),
        )

    # Build fusion results
    fusion = None
    if evidence.fusion_results:
        fr = evidence.fusion_results
        fusion = FusionResults(
            fused_score=fr.get("fused_score", 0),
            classification=fr.get("classification", "unknown"),
            confidence=fr.get("confidence", 0),
            modality_contributions=fr.get("modality_contributions", {}),
            consensus_achieved=fr.get("consensus_achieved", False),
        )

    return AnalysisResponse(
        evidence_id=evidence.id,
        status=AnalysisStatus.COMPLETED,
        behavioral_indicators=behavioral,
        fusion_results=fusion,
        ocr_text=evidence.extracted_text,
        processing_time_ms=0.0,  # TODO: Calculate from job
        analyzed_at=evidence.analyzed_at or datetime.utcnow(),
    )


@router.post("/batch", response_model=BatchAnalysisResponse, status_code=202)
async def start_batch_analysis(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Start analysis on multiple evidence items."""
    job_ids = []

    for evidence_id in request.evidence_ids:
        # Verify evidence exists
        result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
        evidence = result.scalar_one_or_none()
        if not evidence:
            continue

        # Skip if already processing
        if evidence.status == DBEvidenceStatus.PROCESSING:
            continue

        # Create job
        job = AnalysisJob(
            evidence_id=evidence_id,
            status="pending",
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        job_ids.append(job.id)

        # Queue background task
        background_tasks.add_task(run_analysis_pipeline, evidence_id, db)

    return BatchAnalysisResponse(
        job_ids=job_ids,
        total_items=len(job_ids),
        status="queued",
    )
