"""
Evidence Suite - Evidence Schemas
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field


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
    description: Optional[str] = None
    source_info: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EvidenceResponse(BaseModel):
    """Schema for evidence responses."""
    id: UUID
    case_id: UUID
    evidence_type: EvidenceType
    original_filename: Optional[str] = None
    mime_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    original_hash: str
    status: EvidenceStatus
    extracted_text: Optional[str] = None
    fused_score: Optional[float] = None
    fused_classification: Optional[str] = None
    confidence: Optional[float] = None
    created_at: datetime
    analyzed_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class EvidenceListResponse(BaseModel):
    """Paginated evidence list response."""
    items: List[EvidenceResponse]
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
    processing_time_ms: Optional[float] = None
    success: bool
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class ChainOfCustodyResponse(BaseModel):
    """Full chain of custody response."""
    evidence_id: UUID
    entries: List[ChainOfCustodyEntry]
    chain_valid: bool
    total_entries: int
