"""Evidence Suite - Analysis Schemas"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BehavioralIndicators(BaseModel):
    """Behavioral analysis results."""

    sentiment_compound: float = Field(..., ge=-1.0, le=1.0)
    sentiment_positive: float = Field(..., ge=0.0, le=1.0)
    sentiment_negative: float = Field(..., ge=0.0, le=1.0)
    sentiment_neutral: float = Field(..., ge=0.0, le=1.0)
    darvo_score: float = Field(..., ge=0.0, le=1.0)
    gaslighting_score: float = Field(..., ge=0.0, le=1.0)
    manipulation_score: float = Field(..., ge=0.0, le=1.0)
    deception_score: float = Field(..., ge=0.0, le=1.0)
    primary_behavior: str | None = None
    detected_patterns: list[str] = []


class FusionResults(BaseModel):
    """Multi-modal fusion results."""

    fused_score: float
    classification: str
    confidence: float
    modality_contributions: dict[str, float] = {}
    consensus_achieved: bool = False


class AnalysisRequest(BaseModel):
    """Request to analyze evidence."""

    evidence_id: UUID
    run_ocr: bool = True
    run_behavioral: bool = True
    run_fusion: bool = True
    priority: int = Field(default=5, ge=1, le=10)


class AnalysisJobResponse(BaseModel):
    """Analysis job status response."""

    id: UUID
    evidence_id: UUID
    status: AnalysisStatus
    current_stage: str | None = None
    progress_percent: float = 0.0
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    class Config:
        from_attributes = True


class AnalysisResponse(BaseModel):
    """Full analysis results response."""

    evidence_id: UUID
    status: AnalysisStatus
    behavioral_indicators: BehavioralIndicators | None = None
    fusion_results: FusionResults | None = None
    ocr_text: str | None = None
    processing_time_ms: float
    analyzed_at: datetime


class BatchAnalysisRequest(BaseModel):
    """Request to analyze multiple evidence items."""

    evidence_ids: list[UUID]
    run_ocr: bool = True
    run_behavioral: bool = True
    run_fusion: bool = True


class BatchAnalysisResponse(BaseModel):
    """Batch analysis status."""

    job_ids: list[UUID]
    total_items: int
    status: str = "queued"
