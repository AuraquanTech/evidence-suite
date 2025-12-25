"""
Evidence Suite - Case Schemas
"""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field


class CaseStatus(str, Enum):
    OPEN = "open"
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    CLOSED = "closed"
    ARCHIVED = "archived"


class CaseBase(BaseModel):
    """Base case schema."""
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    client_name: Optional[str] = Field(None, max_length=200)
    attorney_name: Optional[str] = Field(None, max_length=200)
    jurisdiction: Optional[str] = Field(None, max_length=100)


class CaseCreate(CaseBase):
    """Schema for creating a new case."""
    case_number: str = Field(..., min_length=1, max_length=100)


class CaseUpdate(BaseModel):
    """Schema for updating a case."""
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    status: Optional[CaseStatus] = None
    client_name: Optional[str] = Field(None, max_length=200)
    attorney_name: Optional[str] = Field(None, max_length=200)
    jurisdiction: Optional[str] = Field(None, max_length=100)


class CaseResponse(CaseBase):
    """Schema for case responses."""
    id: UUID
    case_number: str
    status: CaseStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    evidence_count: int = 0

    class Config:
        from_attributes = True


class CaseListResponse(BaseModel):
    """Paginated case list response."""
    items: List[CaseResponse]
    total: int
    page: int
    page_size: int
    pages: int
