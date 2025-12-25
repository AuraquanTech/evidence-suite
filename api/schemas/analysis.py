"""
Evidence Suite - Analysis Schemas
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum

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
    primary_behavior: Optional[str] = None
    detected_patterns: List[str] = []


class FusionResults(BaseModel):
    """Multi-modal fusion results."""
    fused_score: float
    classification: str
    confidence: float
    modality_contributions: Dict[str, float] = {}
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
    current_stage: Optional[str] = None
    progress_percent: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class AnalysisResponse(BaseModel):
    """Full analysis results response."""
    evidence_id: UUID
    status: AnalysisStatus
    behavioral_indicators: Optional[BehavioralIndicators] = None
    fusion_results: Optional[FusionResults] = None
    ocr_text: Optional[str] = None
    processing_time_ms: float
    analyzed_at: datetime


class BatchAnalysisRequest(BaseModel):
    """Request to analyze multiple evidence items."""
    evidence_ids: List[UUID]
    run_ocr: bool = True
    run_behavioral: bool = True
    run_fusion: bool = True


class BatchAnalysisResponse(BaseModel):
    """Batch analysis status."""
    job_ids: List[UUID]
    total_items: int
    status: str = "queued"
