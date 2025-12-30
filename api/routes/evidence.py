"""Evidence Suite - Evidence Routes
Secured with authentication, file validation, and encryption.
"""

import hashlib
import os
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_active_user, require_role
from api.schemas.evidence import (
    ChainOfCustodyEntry,
    ChainOfCustodyResponse,
    EvidenceListResponse,
    EvidenceResponse,
    EvidenceStatus,
    EvidenceType,
)
from core.database import (
    Case,
    ChainOfCustodyLog,
    EvidenceRecord,
    EvidenceTypeDB,
    User,
)
from core.database import (
    EvidenceStatus as DBEvidenceStatus,
)
from core.database.session import get_db
from core.security import FileValidator, file_validator, generate_safe_storage_path


router = APIRouter(prefix="/evidence", tags=["Evidence"])

# Storage configuration
EVIDENCE_STORAGE_PATH = os.getenv("EVIDENCE_STORAGE_PATH", "./evidence_store")
ENABLE_ENCRYPTION = os.getenv("EVIDENCE_ENCRYPTION_KEY") is not None


def calculate_sha256(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()


@router.post("/upload", response_model=EvidenceResponse, status_code=201)
async def upload_evidence(
    case_id: UUID = Form(...),
    evidence_type: EvidenceType = Form(...),
    file: UploadFile = File(...),
    description: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Upload new evidence to a case.

    Requires authentication. Files are validated for:
    - Allowed MIME types and extensions
    - Maximum file size
    - Path traversal prevention
    - Optional encryption at rest
    """
    # Verify case exists
    case_result = await db.execute(select(Case).where(Case.id == case_id))
    case = case_result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Validate file
    is_valid, error = file_validator.validate(
        filename=file.filename or "unknown",
        content_type=file.content_type,
        file_size=file_size,
        content=content,
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid file: {error}")

    file_hash = calculate_sha256(content)

    # Check for duplicate hash
    existing = await db.execute(
        select(EvidenceRecord).where(
            EvidenceRecord.case_id == case_id, EvidenceRecord.original_hash == file_hash
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409, detail="Evidence with identical hash already exists in this case"
        )

    # Generate safe storage path (prevents path traversal)
    storage_path = generate_safe_storage_path(
        base_dir=EVIDENCE_STORAGE_PATH,
        case_id=str(case_id),
        file_hash=file_hash,
        filename=file.filename or "unknown",
    )

    # Create storage directory
    storage_dir = os.path.dirname(storage_path)
    os.makedirs(storage_dir, exist_ok=True)

    # Encrypt and save file
    if ENABLE_ENCRYPTION:
        from core.encryption import encrypt_evidence

        encrypted_content = encrypt_evidence(content, str(case_id))
        with open(storage_path + ".enc", "wb") as f:
            f.write(encrypted_content)
        storage_path = storage_path + ".enc"
    else:
        with open(storage_path, "wb") as f:
            f.write(content)

    # Create evidence record
    evidence = EvidenceRecord(
        case_id=case_id,
        evidence_type=EvidenceTypeDB(evidence_type.value),
        original_filename=file.filename,
        mime_type=file.content_type,
        file_size_bytes=len(content),
        original_hash=file_hash,
        storage_path=storage_path,
        status=DBEvidenceStatus.PENDING,
        extra_data={"description": description} if description else {},
    )
    db.add(evidence)

    # Create chain of custody entry
    custody_entry = ChainOfCustodyLog(
        evidence_id=evidence.id,
        agent_id="api-upload",
        agent_type="http",
        action="evidence_uploaded",
        input_hash=file_hash,
        output_hash=file_hash,
        success=True,
    )
    db.add(custody_entry)

    await db.commit()
    await db.refresh(evidence)

    return EvidenceResponse(
        id=evidence.id,
        case_id=evidence.case_id,
        evidence_type=EvidenceType(evidence.evidence_type.value),
        original_filename=evidence.original_filename,
        mime_type=evidence.mime_type,
        file_size_bytes=evidence.file_size_bytes,
        original_hash=evidence.original_hash,
        status=EvidenceStatus(evidence.status.value),
        created_at=evidence.created_at,
        metadata=evidence.extra_data,
    )


@router.get("/", response_model=EvidenceListResponse)
async def list_evidence(
    case_id: UUID | None = None,
    status: EvidenceStatus | None = None,
    evidence_type: EvidenceType | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List evidence with pagination and filtering."""
    query = select(EvidenceRecord)

    if case_id:
        query = query.where(EvidenceRecord.case_id == case_id)
    if status:
        query = query.where(EvidenceRecord.status == DBEvidenceStatus(status.value))
    if evidence_type:
        query = query.where(EvidenceRecord.evidence_type == EvidenceTypeDB(evidence_type.value))

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar()

    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    query = query.order_by(EvidenceRecord.created_at.desc())

    result = await db.execute(query)
    records = result.scalars().all()

    items = [
        EvidenceResponse(
            id=e.id,
            case_id=e.case_id,
            evidence_type=EvidenceType(e.evidence_type.value),
            original_filename=e.original_filename,
            mime_type=e.mime_type,
            file_size_bytes=e.file_size_bytes,
            original_hash=e.original_hash,
            status=EvidenceStatus(e.status.value),
            extracted_text=e.extracted_text,
            fused_score=e.fused_score,
            fused_classification=e.fused_classification,
            confidence=e.confidence,
            created_at=e.created_at,
            analyzed_at=e.analyzed_at,
            verified_at=e.verified_at,
            metadata=e.extra_data,
        )
        for e in records
    ]

    return EvidenceListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{evidence_id}", response_model=EvidenceResponse)
async def get_evidence(
    evidence_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get evidence by ID."""
    result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
    evidence = result.scalar_one_or_none()

    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    return EvidenceResponse(
        id=evidence.id,
        case_id=evidence.case_id,
        evidence_type=EvidenceType(evidence.evidence_type.value),
        original_filename=evidence.original_filename,
        mime_type=evidence.mime_type,
        file_size_bytes=evidence.file_size_bytes,
        original_hash=evidence.original_hash,
        status=EvidenceStatus(evidence.status.value),
        extracted_text=evidence.extracted_text,
        fused_score=evidence.fused_score,
        fused_classification=evidence.fused_classification,
        confidence=evidence.confidence,
        created_at=evidence.created_at,
        analyzed_at=evidence.analyzed_at,
        verified_at=evidence.verified_at,
        metadata=evidence.extra_data,
    )


@router.get("/{evidence_id}/chain-of-custody", response_model=ChainOfCustodyResponse)
async def get_chain_of_custody(
    evidence_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get chain of custody for evidence."""
    # Verify evidence exists
    evidence_result = await db.execute(
        select(EvidenceRecord).where(EvidenceRecord.id == evidence_id)
    )
    evidence = evidence_result.scalar_one_or_none()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    # Get custody entries
    result = await db.execute(
        select(ChainOfCustodyLog)
        .where(ChainOfCustodyLog.evidence_id == evidence_id)
        .order_by(ChainOfCustodyLog.timestamp)
    )
    entries = result.scalars().all()

    # Validate chain
    chain_valid = True
    prev_hash = evidence.original_hash
    for entry in entries:
        if entry.input_hash != prev_hash:
            chain_valid = False
            break
        prev_hash = entry.output_hash

    return ChainOfCustodyResponse(
        evidence_id=evidence_id,
        entries=[
            ChainOfCustodyEntry(
                id=e.id,
                timestamp=e.timestamp,
                agent_id=e.agent_id,
                agent_type=e.agent_type,
                action=e.action,
                input_hash=e.input_hash,
                output_hash=e.output_hash,
                processing_time_ms=e.processing_time_ms,
                success=e.success,
                error_message=e.error_message,
            )
            for e in entries
        ],
        chain_valid=chain_valid,
        total_entries=len(entries),
    )


@router.post("/{evidence_id}/verify", response_model=EvidenceResponse)
async def verify_evidence(
    evidence_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin", "analyst")),
):
    """Verify evidence integrity and mark as verified."""
    result = await db.execute(select(EvidenceRecord).where(EvidenceRecord.id == evidence_id))
    evidence = result.scalar_one_or_none()

    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")

    if evidence.status != DBEvidenceStatus.ANALYZED:
        raise HTTPException(status_code=400, detail="Evidence must be analyzed before verification")

    # Verify file hash
    if evidence.storage_path and os.path.exists(evidence.storage_path):
        with open(evidence.storage_path, "rb") as f:
            current_hash = calculate_sha256(f.read())
        if current_hash != evidence.original_hash:
            evidence.status = DBEvidenceStatus.FLAGGED
            await db.commit()
            raise HTTPException(
                status_code=409, detail="Evidence hash mismatch - file may have been tampered"
            )

    evidence.status = DBEvidenceStatus.VERIFIED
    evidence.verified_at = datetime.utcnow()

    # Add custody entry
    custody_entry = ChainOfCustodyLog(
        evidence_id=evidence.id,
        agent_id="api-verify",
        agent_type="http",
        action="evidence_verified",
        input_hash=evidence.original_hash,
        output_hash=evidence.original_hash,
        success=True,
    )
    db.add(custody_entry)

    await db.commit()
    await db.refresh(evidence)

    return EvidenceResponse(
        id=evidence.id,
        case_id=evidence.case_id,
        evidence_type=EvidenceType(evidence.evidence_type.value),
        original_filename=evidence.original_filename,
        mime_type=evidence.mime_type,
        file_size_bytes=evidence.file_size_bytes,
        original_hash=evidence.original_hash,
        status=EvidenceStatus(evidence.status.value),
        extracted_text=evidence.extracted_text,
        fused_score=evidence.fused_score,
        fused_classification=evidence.fused_classification,
        confidence=evidence.confidence,
        created_at=evidence.created_at,
        analyzed_at=evidence.analyzed_at,
        verified_at=evidence.verified_at,
        metadata=evidence.extra_data,
    )
