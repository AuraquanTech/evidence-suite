"""
Evidence Suite - Case Routes
Optimized with repository pattern and eager loading.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Case, CaseStatus as DBCaseStatus
from core.database.session import get_db
from core.database.repository import CaseRepository, get_case_repository
from api.schemas.cases import (
    CaseCreate,
    CaseUpdate,
    CaseResponse,
    CaseListResponse,
    CaseStatus,
)

router = APIRouter(prefix="/cases", tags=["Cases"])


# Dependency to get case repository
async def get_repo(db: AsyncSession = Depends(get_db)) -> CaseRepository:
    return get_case_repository(db)


@router.post("/", response_model=CaseResponse, status_code=201)
async def create_case(
    case_data: CaseCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new case."""
    # Check for duplicate case number
    existing = await db.execute(
        select(Case).where(Case.case_number == case_data.case_number)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Case number already exists")

    case = Case(
        case_number=case_data.case_number,
        title=case_data.title,
        description=case_data.description,
        client_name=case_data.client_name,
        attorney_name=case_data.attorney_name,
        jurisdiction=case_data.jurisdiction,
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)

    return CaseResponse(
        id=case.id,
        case_number=case.case_number,
        title=case.title,
        description=case.description,
        status=CaseStatus(case.status.value),
        client_name=case.client_name,
        attorney_name=case.attorney_name,
        jurisdiction=case.jurisdiction,
        created_at=case.created_at,
        updated_at=case.updated_at,
        evidence_count=0,
    )


@router.get("/", response_model=CaseListResponse)
async def list_cases(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[CaseStatus] = None,
    search: Optional[str] = None,
    repo: CaseRepository = Depends(get_repo)
):
    """List cases with pagination and filtering.

    Optimized: Uses single query with JOIN for evidence counts,
    avoiding N+1 query problem.
    """
    # Convert status enum
    db_status = DBCaseStatus(status.value) if status else None

    # Use optimized repository method - single query for cases + counts
    cases_with_counts, total = await repo.list_with_evidence_counts(
        page=page,
        page_size=page_size,
        status=db_status,
        search=search
    )

    items = [
        CaseResponse(
            id=case.id,
            case_number=case.case_number,
            title=case.title,
            description=case.description,
            status=CaseStatus(case.status.value),
            client_name=case.client_name,
            attorney_name=case.attorney_name,
            jurisdiction=case.jurisdiction,
            created_at=case.created_at,
            updated_at=case.updated_at,
            closed_at=case.closed_at,
            evidence_count=evidence_count,
        )
        for case, evidence_count in cases_with_counts
    ]

    return CaseListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total > 0 else 0,
    )


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: UUID,
    repo: CaseRepository = Depends(get_repo)
):
    """Get a case by ID.

    Optimized: Uses single query with JOIN for evidence count.
    """
    result = await repo.get_with_evidence_count(case_id)

    if not result:
        raise HTTPException(status_code=404, detail="Case not found")

    case, evidence_count = result

    return CaseResponse(
        id=case.id,
        case_number=case.case_number,
        title=case.title,
        description=case.description,
        status=CaseStatus(case.status.value),
        client_name=case.client_name,
        attorney_name=case.attorney_name,
        jurisdiction=case.jurisdiction,
        created_at=case.created_at,
        updated_at=case.updated_at,
        closed_at=case.closed_at,
        evidence_count=evidence_count,
    )


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: UUID,
    case_data: CaseUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a case."""
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    update_data = case_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if key == "status" and value:
            setattr(case, key, DBCaseStatus(value.value))
            if value == CaseStatus.CLOSED:
                case.closed_at = datetime.utcnow()
        else:
            setattr(case, key, value)

    await db.commit()
    await db.refresh(case)

    return CaseResponse(
        id=case.id,
        case_number=case.case_number,
        title=case.title,
        description=case.description,
        status=CaseStatus(case.status.value),
        client_name=case.client_name,
        attorney_name=case.attorney_name,
        jurisdiction=case.jurisdiction,
        created_at=case.created_at,
        updated_at=case.updated_at,
        closed_at=case.closed_at,
        evidence_count=len(case.evidence_records) if case.evidence_records else 0,
    )


@router.delete("/{case_id}", status_code=204)
async def delete_case(
    case_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a case (soft delete by archiving)."""
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    case.status = DBCaseStatus.ARCHIVED
    await db.commit()
