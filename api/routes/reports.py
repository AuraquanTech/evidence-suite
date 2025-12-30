"""Evidence Suite - Reports API Routes"""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.rbac import Permission, require_permissions


router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get(
    "/case/{case_id}",
    summary="Generate case report",
    dependencies=[Depends(require_permissions(Permission.ANALYSIS_EXPORT))],
)
async def generate_case_report(
    case_id: UUID,
    format: str = Query(default="pdf", regex="^(pdf|html|json|md)$"),
    include_evidence: bool = Query(default=True),
    include_analysis: bool = Query(default=True),
):
    """Generate a forensic analysis report for a case.

    Formats:
    - pdf: PDF document (default)
    - html: HTML document
    - json: JSON data
    - md: Markdown document
    """
    from core.reports import generate_report

    try:
        report_path = await generate_report(
            case_id=case_id,
            format=format,
            output_dir="./reports",
        )

        if not report_path.exists():
            raise HTTPException(status_code=500, detail="Report generation failed")  # noqa: TRY301

        # Determine content type
        content_types = {
            "pdf": "application/pdf",
            "html": "text/html",
            "json": "application/json",
            "md": "text/markdown",
        }

        return FileResponse(
            path=str(report_path),
            filename=report_path.name,
            media_type=content_types.get(format, "application/octet-stream"),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}") from e


@router.get(
    "/case/{case_id}/preview",
    summary="Preview case report data",
    dependencies=[Depends(require_permissions(Permission.ANALYSIS_READ))],
)
async def preview_case_report(
    case_id: UUID,
    include_evidence: bool = Query(default=True),
):
    """Get report data without generating a file.
    Useful for previewing before download.
    """
    from sqlalchemy import select

    from core.database import Case, EvidenceRecord
    from core.database.session import get_async_session

    async with get_async_session() as db:
        # Get case
        result = await db.execute(select(Case).where(Case.id == case_id))
        case = result.scalar_one_or_none()

        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Get evidence summary
        evidence_result = await db.execute(
            select(EvidenceRecord).where(EvidenceRecord.case_id == case_id)
        )
        evidence_records = evidence_result.scalars().all()

        return {
            "case": {
                "id": str(case.id),
                "case_number": case.case_number,
                "title": case.title,
                "status": case.status.value if case.status else None,
            },
            "summary": {
                "total_evidence": len(evidence_records),
                "analyzed": sum(
                    1 for e in evidence_records if e.status and e.status.value == "analyzed"
                ),
                "pending": sum(
                    1 for e in evidence_records if e.status and e.status.value == "pending"
                ),
            },
            "available_formats": ["pdf", "html", "json", "md"],
        }
