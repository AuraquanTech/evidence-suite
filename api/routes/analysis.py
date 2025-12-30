"""Evidence Suite - Analysis Routes

Includes gated output endpoints that enforce the output gate for client-facing data.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
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
    BlockReason,
    ClientAnalysisResponse,
    FusionResults,
    InternalAnalysisResponse,
    RoutingDecision,
    RoutingStatusResponse,
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
from core.output_gate import (
    ClientOutputBlocked,
    compute_routing,
    ensure_client_output_allowed,
    get_client_summary,
    get_internal_summary,
)
from core.output_gate import (
    RoutingDecision as GateRoutingDecision,
)


router = APIRouter(prefix="/analysis", tags=["Analysis"])


async def run_analysis_pipeline(evidence_id: UUID, db: AsyncSession):
    """Background task to run the analysis pipeline with routing decision."""
    # Import pipeline components
    try:
        from core.models import EvidencePacket
        from core.models import EvidenceType as CoreEvidenceType
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
        await pipeline.initialize()

        # Read evidence content
        if evidence.storage_path:
            with open(evidence.storage_path, "rb") as f:
                content = f.read()

            # Determine evidence type
            etype = CoreEvidenceType.TEXT
            if evidence.mime_type:
                if evidence.mime_type.startswith("image"):
                    etype = CoreEvidenceType.IMAGE
                elif evidence.mime_type.startswith("audio"):
                    etype = CoreEvidenceType.AUDIO
                elif "pdf" in evidence.mime_type:
                    etype = CoreEvidenceType.DOCUMENT

            # Create packet and process through pipeline
            packet = EvidencePacket(
                raw_content=content,
                evidence_type=etype,
                case_id=str(evidence.case_id),
            )
            pipeline_result = await pipeline.process(packet)

            # Compute routing decision using the output gate
            routing = compute_routing(pipeline_result)
            logger.info(f"Evidence {evidence_id} routing: {routing.value}")

            # Update evidence with results
            result_packet = pipeline_result.packet
            if result_packet.behavioral_indicators:
                bi = result_packet.behavioral_indicators
                evidence.behavioral_indicators = {
                    "sentiment": {
                        "compound": bi.sentiment_compound,
                        "pos": bi.sentiment_positive,
                        "neg": bi.sentiment_negative,
                        "neu": bi.sentiment_neutral,
                    },
                    "darvo_score": bi.darvo_score,
                    "gaslighting_score": bi.gaslighting_score,
                    "manipulation_score": bi.manipulation_score,
                    "deception_score": bi.deception_indicators,
                    "primary_behavior": bi.primary_behavior_class,
                    "detected_patterns": list(bi.behavior_probabilities.keys())
                    if bi.behavior_probabilities
                    else [],
                }

            # Store fusion results with routing decision
            evidence.fusion_results = {
                "fused_score": result_packet.fused_score,
                "classification": result_packet.fused_classification,
                "confidence": result_packet.fused_score,  # Using fused_score as confidence
                "routing": routing.value,  # Store routing decision
                "fusion_metadata": result_packet.fusion_metadata,
            }
            evidence.fused_score = result_packet.fused_score
            evidence.fused_classification = result_packet.fused_classification
            evidence.confidence = result_packet.fused_score

            if result_packet.extracted_text:
                evidence.extracted_text = result_packet.extracted_text

            evidence.status = DBEvidenceStatus.ANALYZED
            evidence.analyzed_at = datetime.utcnow()

            # Update job with routing info
            if job:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.progress_percent = 100.0
                # Store routing in job metadata if available
                if hasattr(job, "extra_data"):
                    job.extra_data = {"routing": routing.value}

            # Add custody entry
            custody_entry = ChainOfCustodyLog(
                evidence_id=evidence.id,
                agent_id="pipeline",
                agent_type="analysis",
                action="analysis_completed",
                input_hash=evidence.original_hash,
                output_hash=evidence.original_hash,
                success=True,
                extra_data={"routing": routing.value},
            )
            db.add(custody_entry)

            await db.commit()
            await pipeline.shutdown()

    except Exception as e:
        logger.error(f"Pipeline error for evidence {evidence_id}: {e}")
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


# =============================================================================
# GATED OUTPUT ENDPOINTS - Enforce output gate for client-facing data
# =============================================================================


def _get_routing_from_evidence(evidence: EvidenceRecord) -> RoutingDecision:
    """Extract routing decision from evidence fusion results."""
    if evidence.fusion_results:
        routing_str = evidence.fusion_results.get("routing", "review")
        try:
            return RoutingDecision(routing_str)
        except ValueError:
            return RoutingDecision.REVIEW
    return RoutingDecision.REVIEW


def _get_block_reasons_from_evidence(evidence: EvidenceRecord) -> list[BlockReason]:
    """Build block reasons from evidence data."""
    reasons = []

    # Check for low confidence
    if evidence.fused_score is not None and evidence.fused_score < 0.3:
        reasons.append(
            BlockReason(
                code="LOW_FUSION_SCORE",
                detail=f"Fused confidence score only {evidence.fused_score:.2f}",
                severity="high",
            )
        )

    # Check for missing content
    if not evidence.extracted_text or len(evidence.extracted_text.strip()) < 50:
        reasons.append(
            BlockReason(
                code="MISSING_CONTENT",
                detail="Insufficient extracted text content",
                severity="high",
            )
        )

    # Check behavioral indicators
    if evidence.behavioral_indicators:
        bi = evidence.behavioral_indicators
        if bi.get("deception_score", 0) > 0.7:
            reasons.append(
                BlockReason(
                    code="HIGH_DECEPTION_SCORE",
                    detail=f"Deception indicators at {bi.get('deception_score'):.2f}",
                    severity="high",
                )
            )

    # Check fusion metadata for anomalies
    if evidence.fusion_results:
        fusion_meta = evidence.fusion_results.get("fusion_metadata", {})
        anomalies = fusion_meta.get("anomalies", [])
        for anomaly in anomalies:
            if isinstance(anomaly, dict):
                reasons.append(
                    BlockReason(
                        code=anomaly.get("code", "ANOMALY_DETECTED"),
                        detail=anomaly.get("detail", str(anomaly)),
                        severity=anomaly.get("severity", "medium"),
                    )
                )

    return reasons


@router.get("/{evidence_id}/routing", response_model=RoutingStatusResponse)
async def get_routing_status(evidence_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get routing status for evidence - shows if it can go to client."""
    result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
    evidence = result.scalar_one_or_none()

    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    if evidence.status not in [DBEvidenceStatus.ANALYZED, DBEvidenceStatus.VERIFIED]:
        raise HTTPException(status_code=400, detail="Evidence not yet analyzed")

    routing = _get_routing_from_evidence(evidence)
    block_reasons = _get_block_reasons_from_evidence(evidence)

    return RoutingStatusResponse(
        evidence_id=evidence.id,
        routing=routing,
        block_reasons=block_reasons,
        can_send_to_client=routing == RoutingDecision.AUTO_OK,
        requires_review=routing == RoutingDecision.REVIEW,
        missing_documentation=routing == RoutingDecision.NEEDS_DOCS,
    )


@router.get("/{evidence_id}/internal", response_model=InternalAnalysisResponse)
async def get_internal_analysis(evidence_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get FULL analysis results for INTERNAL use.

    This endpoint returns all data including routing decisions and block reasons.
    Use for internal dashboards, review queues, and debugging.
    No output gate enforcement - returns everything.
    """
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

    routing = _get_routing_from_evidence(evidence)
    block_reasons = _get_block_reasons_from_evidence(evidence)

    return InternalAnalysisResponse(
        evidence_id=evidence.id,
        case_id=evidence.case_id,
        status=AnalysisStatus.COMPLETED,
        routing=routing,
        block_reasons=block_reasons,
        behavioral_indicators=behavioral,
        fusion_results=fusion,
        ocr_text=evidence.extracted_text,
        extracted_text_preview=evidence.extracted_text[:500] if evidence.extracted_text else None,
        processing_time_ms=0.0,
        analyzed_at=evidence.analyzed_at or datetime.utcnow(),
        stages_completed=[],  # TODO: Get from job
        errors=[],
    )


@router.get("/{evidence_id}/client", response_model=ClientAnalysisResponse)
async def get_client_analysis(evidence_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get analysis results for CLIENT-FACING use.

    This endpoint ENFORCES the output gate:
    - Returns 403 Forbidden if routing != AUTO_OK
    - Only returns safe, redacted data

    Use this endpoint for client portals, automated reports, external APIs.
    """
    result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
    evidence = result.scalar_one_or_none()

    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    if evidence.status not in [DBEvidenceStatus.ANALYZED, DBEvidenceStatus.VERIFIED]:
        raise HTTPException(status_code=400, detail="Evidence not yet analyzed")

    # Check routing - enforce the gate
    routing = _get_routing_from_evidence(evidence)

    if routing != RoutingDecision.AUTO_OK:
        block_reasons = _get_block_reasons_from_evidence(evidence)
        reason_details = "; ".join(f"{r.code}: {r.detail}" for r in block_reasons[:3])

        raise HTTPException(
            status_code=403,
            detail={
                "message": "Client output blocked by output gate",
                "routing": routing.value,
                "reasons": [{"code": r.code, "detail": r.detail} for r in block_reasons],
                "summary": reason_details or f"Routing is {routing.value}, not auto_ok",
            },
        )

    # Safe to return client data
    summary = "Analysis complete."
    if evidence.fused_classification:
        summary = f"Analysis complete. Classification: {evidence.fused_classification}"

    return ClientAnalysisResponse(
        evidence_id=evidence.id,
        case_id=evidence.case_id,
        status="complete",
        classification=evidence.fused_classification,
        confidence=evidence.fused_score,
        summary=summary,
    )
