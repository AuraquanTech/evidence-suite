"""
Evidence Suite - Database Models
SQLAlchemy ORM models for FRE-compliant evidence storage.
Supports PostgreSQL (production) and SQLite (testing).
"""
import os
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any, List
from uuid import uuid4

from sqlalchemy import (
    Column, String, DateTime, Float, Boolean, Text, BigInteger,
    ForeignKey, Enum, JSON, LargeBinary, Index, TypeDecorator
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import CHAR, TypeDecorator as TD

Base = declarative_base()

# Check if we're using SQLite (for tests)
_IS_SQLITE = os.getenv("EVIDENCE_SUITE_ENV") == "test"


class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type when available, otherwise stores as CHAR(36).
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            from uuid import UUID
            return UUID(value)


class JSONType(TypeDecorator):
    """Platform-independent JSON type.
    Uses PostgreSQL's JSONB type when available, otherwise uses JSON.
    """
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())


# Use our custom types
UUID = GUID


class CaseStatus(PyEnum):
    OPEN = "open"
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    CLOSED = "closed"
    ARCHIVED = "archived"


class EvidenceStatus(PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    VERIFIED = "verified"
    FLAGGED = "flagged"
    ERROR = "error"


class EvidenceTypeDB(PyEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    EMAIL = "email"


class Case(Base):
    """Legal case container for evidence."""
    __tablename__ = "cases"

    id = Column(UUID(), primary_key=True, default=uuid4)
    case_number = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    status = Column(Enum(CaseStatus), default=CaseStatus.OPEN)

    # Metadata
    client_name = Column(String(200))
    attorney_name = Column(String(200))
    jurisdiction = Column(String(100))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime)

    # Relationships
    evidence_records = relationship("EvidenceRecord", back_populates="case")

    __table_args__ = (
        Index('ix_cases_status_created', 'status', 'created_at'),
    )


class EvidenceRecord(Base):
    """Primary evidence storage with integrity verification."""
    __tablename__ = "evidence"

    id = Column(UUID(), primary_key=True, default=uuid4)
    case_id = Column(UUID(), ForeignKey("cases.id"), nullable=False)

    # Evidence metadata
    evidence_type = Column(Enum(EvidenceTypeDB), nullable=False)
    original_filename = Column(String(500))
    mime_type = Column(String(100))
    file_size_bytes = Column(BigInteger)

    # Integrity
    original_hash = Column(String(64), nullable=False)  # SHA-256
    storage_path = Column(String(1000))

    # Status
    status = Column(Enum(EvidenceStatus), default=EvidenceStatus.PENDING)

    # Extracted content
    extracted_text = Column(Text)

    # Analysis results (JSON for complex nested data)
    behavioral_indicators = Column(JSONType)
    fusion_results = Column(JSONType)
    ocr_results = Column(JSONType)

    # Scores
    fused_score = Column(Float)
    fused_classification = Column(String(50))
    confidence = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    analyzed_at = Column(DateTime)
    verified_at = Column(DateTime)

    # Extra metadata
    extra_data = Column(JSONType, default={})

    # Relationships
    case = relationship("Case", back_populates="evidence_records")
    custody_entries = relationship("ChainOfCustodyLog", back_populates="evidence")
    analysis_results = relationship("AnalysisResult", back_populates="evidence")

    __table_args__ = (
        Index('ix_evidence_case_status', 'case_id', 'status'),
        Index('ix_evidence_type_created', 'evidence_type', 'created_at'),
        Index('ix_evidence_hash', 'original_hash'),
    )


class ChainOfCustodyLog(Base):
    """Immutable chain of custody audit log (FRE 901 compliant)."""
    __tablename__ = "chain_of_custody"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    evidence_id = Column(UUID(), ForeignKey("evidence.id"), nullable=False)

    # Entry details
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    agent_id = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)
    action = Column(String(200), nullable=False)

    # Integrity hashes
    input_hash = Column(String(64), nullable=False)
    output_hash = Column(String(64), nullable=False)

    # Processing details
    processing_time_ms = Column(Float)
    success = Column(Boolean, default=True)
    error_message = Column(Text)

    # Digital signature (optional, for enhanced verification)
    signature = Column(String(512))
    signer_id = Column(String(100))

    # Extra data
    extra_data = Column(JSONType, default={})

    # Relationship
    evidence = relationship("EvidenceRecord", back_populates="custody_entries")

    __table_args__ = (
        Index('ix_custody_evidence_timestamp', 'evidence_id', 'timestamp'),
    )


class AnalysisResult(Base):
    """Detailed analysis results from each agent."""
    __tablename__ = "analysis_results"

    id = Column(UUID(), primary_key=True, default=uuid4)
    evidence_id = Column(UUID(), ForeignKey("evidence.id"), nullable=False)

    # Agent info
    agent_type = Column(String(50), nullable=False)  # ocr, behavioral, fusion
    agent_id = Column(String(100))

    # Results
    result_data = Column(JSONType, nullable=False)
    confidence = Column(Float)

    # Performance
    processing_time_ms = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    evidence = relationship("EvidenceRecord", back_populates="analysis_results")

    __table_args__ = (
        Index('ix_analysis_evidence_agent', 'evidence_id', 'agent_type'),
    )


class AnalysisJob(Base):
    """Background job tracking for async processing."""
    __tablename__ = "analysis_jobs"

    id = Column(UUID(), primary_key=True, default=uuid4)
    evidence_id = Column(UUID(), ForeignKey("evidence.id"), nullable=False)

    # Job status
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    current_stage = Column(String(50))
    progress_percent = Column(Float, default=0)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Error handling
    error_message = Column(Text)
    retry_count = Column(BigInteger, default=0)

    __table_args__ = (
        Index('ix_jobs_status', 'status'),
    )


class User(Base):
    """System users for audit trail."""
    __tablename__ = "users"

    id = Column(UUID(), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(200))
    role = Column(String(50), default="analyst")  # admin, analyst, reviewer, viewer

    # Authentication (hashed)
    password_hash = Column(String(255))

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)


class AuditLog(Base):
    """System-wide audit log for compliance."""
    __tablename__ = "audit_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    user_id = Column(UUID(), ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(100))

    # Request details
    ip_address = Column(String(45))
    user_agent = Column(String(500))

    # Change details
    old_value = Column(JSONType)
    new_value = Column(JSONType)

    __table_args__ = (
        Index('ix_audit_timestamp', 'timestamp'),
        Index('ix_audit_user_action', 'user_id', 'action'),
    )
