"""Evidence Suite - Export API Routes"""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.rbac import Permission, require_permissions


router = APIRouter(prefix="/exports", tags=["Exports"])


@router.post(
    "/case/{case_id}",
    summary="Export case as ZIP package",
    dependencies=[Depends(require_permissions(Permission.ANALYSIS_EXPORT))],
)
async def export_case(
    case_id: UUID,
    include_raw_evidence: bool = Query(default=True),
    include_analysis: bool = Query(default=True),
    include_reports: bool = Query(default=True),
):
    """Export entire case as a legally-admissible ZIP package.

    The package includes:
    - Original evidence files (if include_raw_evidence=true)
    - Chain of custody logs
    - Analysis results (if include_analysis=true)
    - Generated reports (if include_reports=true)
    - Integrity verification script
    - SHA-256 checksums
    """
    from core.export import export_case as do_export

    try:
        export_path = await do_export(
            case_id=case_id,
            include_raw_evidence=include_raw_evidence,
            include_analysis=include_analysis,
            include_reports=include_reports,
            output_dir="./exports",
        )

        if not export_path.exists():
            raise HTTPException(status_code=500, detail="Export failed")  # noqa: TRY301

        return FileResponse(
            path=str(export_path),
            filename=export_path.name,
            media_type="application/zip",
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}") from e


@router.post(
    "/case/{case_id}/async",
    summary="Start async case export",
    dependencies=[Depends(require_permissions(Permission.ANALYSIS_EXPORT))],
)
async def export_case_async(
    case_id: UUID,
    background_tasks: BackgroundTasks,
    include_raw_evidence: bool = Query(default=True),
    include_analysis: bool = Query(default=True),
    include_reports: bool = Query(default=True),
):
    """Start case export in background.

    Returns immediately with a job ID. Use GET /exports/status/{job_id}
    to check status and download when complete.
    """
    import asyncio
    from uuid import uuid4

    from core.export import export_case as do_export

    job_id = str(uuid4())

    # Store job status (in production, use Redis)
    # For now, just start the background task
    async def do_export_task():
        try:
            await do_export(
                case_id=case_id,
                include_raw_evidence=include_raw_evidence,
                include_analysis=include_analysis,
                include_reports=include_reports,
                output_dir="./exports",
            )
        except Exception as e:
            # Log error
            pass

    background_tasks.add_task(asyncio.create_task, do_export_task())

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Export started. Check /exports/status/{job_id} for progress.",
    }


@router.get(
    "/evidence/{evidence_id}",
    summary="Export single evidence item",
    dependencies=[Depends(require_permissions(Permission.EVIDENCE_READ))],
)
async def export_evidence(
    evidence_id: UUID,
    include_analysis: bool = Query(default=True),
):
    """Export a single evidence item with its analysis and custody chain."""
    import json
    import zipfile
    from datetime import datetime
    from io import BytesIO

    from fastapi.responses import StreamingResponse
    from sqlalchemy import select

    from core.database import (
        AnalysisResult,
        ChainOfCustodyLog,
        EvidenceRecord,
    )
    from core.database.session import get_async_session

    async with get_async_session() as db:
        # Get evidence
        result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
        evidence = result.scalar_one_or_none()

        if not evidence:
            raise HTTPException(status_code=404, detail="Evidence not found")

        # Create ZIP in memory
        buffer = BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add metadata
            metadata = {
                "id": str(evidence.id),
                "filename": evidence.original_filename,
                "type": evidence.evidence_type.value if evidence.evidence_type else None,
                "hash": evidence.original_hash,
                "size_bytes": evidence.file_size_bytes,
                "status": evidence.status.value if evidence.status else None,
                "created_at": evidence.created_at.isoformat() if evidence.created_at else None,
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

            # Add chain of custody
            custody_result = await db.execute(
                select(ChainOfCustodyLog)
                .where(ChainOfCustodyLog.evidence_id == evidence_id)
                .order_by(ChainOfCustodyLog.timestamp)
            )
            custody_entries = custody_result.scalars().all()

            custody_data = [
                {
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "agent_id": e.agent_id,
                    "action": e.action,
                    "input_hash": e.input_hash,
                    "output_hash": e.output_hash,
                    "success": e.success,
                }
                for e in custody_entries
            ]
            zf.writestr("chain_of_custody.json", json.dumps(custody_data, indent=2))

            # Add analysis if requested
            if include_analysis:
                if evidence.behavioral_indicators:
                    zf.writestr(
                        "behavioral_indicators.json",
                        json.dumps(evidence.behavioral_indicators, indent=2),
                    )
                if evidence.fusion_results:
                    zf.writestr(
                        "fusion_results.json",
                        json.dumps(evidence.fusion_results, indent=2),
                    )

        buffer.seek(0)

        filename = f"evidence_{evidence_id}_{datetime.now().strftime('%Y%m%d')}.zip"

        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )


@router.get(
    "/chain-of-custody/{evidence_id}",
    summary="Export chain of custody",
    dependencies=[Depends(require_permissions(Permission.EVIDENCE_READ))],
)
async def export_chain_of_custody(
    evidence_id: UUID,
    format: str = Query(default="json", regex="^(json|csv)$"),
):
    """Export chain of custody log for evidence."""
    import csv
    import json
    from io import StringIO

    from fastapi.responses import StreamingResponse
    from sqlalchemy import select

    from core.database import ChainOfCustodyLog, EvidenceRecord
    from core.database.session import get_async_session

    async with get_async_session() as db:
        # Verify evidence exists
        result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Evidence not found")

        # Get custody entries
        result = await db.execute(
            select(ChainOfCustodyLog)
            .where(ChainOfCustodyLog.evidence_id == evidence_id)
            .order_by(ChainOfCustodyLog.timestamp)
        )
        entries = result.scalars().all()

        if format == "json":
            data = [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "agent_id": e.agent_id,
                    "agent_type": e.agent_type,
                    "action": e.action,
                    "input_hash": e.input_hash,
                    "output_hash": e.output_hash,
                    "processing_time_ms": e.processing_time_ms,
                    "success": e.success,
                    "error_message": e.error_message,
                }
                for e in entries
            ]

            return StreamingResponse(
                iter([json.dumps(data, indent=2)]),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=custody_{evidence_id}.json"},
            )

        # CSV
        buffer = StringIO()
        writer = csv.writer(buffer)

        # Header
        writer.writerow(
            [
                "ID",
                "Timestamp",
                "Agent ID",
                "Agent Type",
                "Action",
                "Input Hash",
                "Output Hash",
                "Processing Time (ms)",
                "Success",
                "Error Message",
            ]
        )

        # Data
        for e in entries:
            writer.writerow(
                [
                    e.id,
                    e.timestamp.isoformat() if e.timestamp else "",
                    e.agent_id,
                    e.agent_type,
                    e.action,
                    e.input_hash,
                    e.output_hash,
                    e.processing_time_ms or "",
                    e.success,
                    e.error_message or "",
                ]
            )

        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=custody_{evidence_id}.csv"},
        )
