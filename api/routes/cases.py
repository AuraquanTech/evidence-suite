"""
Evidence Suite - Case Routes
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Case, CaseStatus as DBCaseStatus
from core.database.session import get_db
from api.schemas.cases import (
    CaseCreate,
    CaseUpdate,
    CaseResponse,
    CaseListResponse,
    CaseStatus,
)

router = APIRouter(prefix="/cases", tags=["Cases"])


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
    db: AsyncSession = Depends(get_db)
):
    """List cases with pagination and filtering."""
    query = select(Case)

    if status:
        query = query.where(Case.status == DBCaseStatus(status.value))

    if search:
        query = query.where(
            Case.title.ilike(f"%{search}%") |
            Case.case_number.ilike(f"%{search}%")
        )

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar()

    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    query = query.order_by(Case.created_at.desc())

    result = await db.execute(query)
    cases = result.scalars().all()

    items = [
        CaseResponse(
            id=c.id,
            case_number=c.case_number,
            title=c.title,
            description=c.description,
            status=CaseStatus(c.status.value),
            client_name=c.client_name,
            attorney_name=c.attorney_name,
            jurisdiction=c.jurisdiction,
            created_at=c.created_at,
            updated_at=c.updated_at,
            closed_at=c.closed_at,
            evidence_count=len(c.evidence_records) if c.evidence_records else 0,
        )
        for c in cases
    ]

    return CaseListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get a case by ID."""
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

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
