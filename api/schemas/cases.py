"""
Evidence Suite - Case Schemas
With enhanced input validation and sanitization.
"""
import re
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class CaseStatus(str, Enum):
    OPEN = "open"
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    CLOSED = "closed"
    ARCHIVED = "archived"


# Regex for input sanitization
SAFE_TEXT_PATTERN = re.compile(r'^[\w\s\-.,;:\'\"!?()@#$%&*+=\[\]{}/<>]+$', re.UNICODE)
CASE_NUMBER_PATTERN = re.compile(r'^[A-Za-z0-9\-_./]+$')


def sanitize_text(text: Optional[str], max_length: int = 10000) -> Optional[str]:
    """Sanitize text input to prevent XSS and injection attacks."""
    if text is None:
        return None

    # Strip and limit length
    text = text.strip()[:max_length]

    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # HTML entity encode dangerous characters
    text = (
        text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )

    return text


class CaseBase(BaseModel):
    """Base case schema with validation."""
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=10000)
    client_name: Optional[str] = Field(None, max_length=200)
    attorney_name: Optional[str] = Field(None, max_length=200)
    jurisdiction: Optional[str] = Field(None, max_length=100)

    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate and sanitize title."""
        v = v.strip()
        if not v:
            raise ValueError('Title cannot be empty')
        return sanitize_text(v, 500)

    @field_validator('description')
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        """Validate and sanitize description."""
        return sanitize_text(v, 10000) if v else None

    @field_validator('client_name', 'attorney_name')
    @classmethod
    def validate_names(cls, v: Optional[str]) -> Optional[str]:
        """Validate person names."""
        if v is None:
            return None
        v = v.strip()
        if len(v) > 200:
            raise ValueError('Name too long')
        return sanitize_text(v, 200)


class CaseCreate(CaseBase):
    """Schema for creating a new case with validated case number."""
    case_number: str = Field(..., min_length=1, max_length=100)

    @field_validator('case_number')
    @classmethod
    def validate_case_number(cls, v: str) -> str:
        """Validate case number format."""
        v = v.strip()
        if not v:
            raise ValueError('Case number cannot be empty')
        if not CASE_NUMBER_PATTERN.match(v):
            raise ValueError('Case number contains invalid characters')
        return v


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
