"""Evidence Suite - Evidence Schemas"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class EvidenceType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    EMAIL = "email"


class EvidenceStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    VERIFIED = "verified"
    FLAGGED = "flagged"
    ERROR = "error"


class EvidenceUpload(BaseModel):
    """Schema for evidence upload metadata."""

    case_id: UUID
    evidence_type: EvidenceType
    description: str | None = None
    source_info: str | None = None
    metadata: dict[str, Any] | None = None


class EvidenceResponse(BaseModel):
    """Schema for evidence responses."""

    id: UUID
    case_id: UUID
    evidence_type: EvidenceType
    original_filename: str | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    original_hash: str
    status: EvidenceStatus
    extracted_text: str | None = None
    fused_score: float | None = None
    fused_classification: str | None = None
    confidence: float | None = None
    created_at: datetime
    analyzed_at: datetime | None = None
    verified_at: datetime | None = None
    metadata: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class EvidenceListResponse(BaseModel):
    """Paginated evidence list response."""

    items: list[EvidenceResponse]
    total: int
    page: int
    page_size: int
    pages: int


class ChainOfCustodyEntry(BaseModel):
    """Single chain of custody entry."""

    id: int
    timestamp: datetime
    agent_id: str
    agent_type: str
    action: str
    input_hash: str
    output_hash: str
    processing_time_ms: float | None = None
    success: bool
    error_message: str | None = None

    class Config:
        from_attributes = True


class ChainOfCustodyResponse(BaseModel):
    """Full chain of custody response."""

    evidence_id: UUID
    entries: list[ChainOfCustodyEntry]
    chain_valid: bool
    total_entries: int
