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


class RoutingDecision(str, Enum):
    """Routing decisions for analysis output."""

    AUTO_OK = "auto_ok"  # Clean, can go to client
    REVIEW = "review"  # Needs human review
    NEEDS_DOCS = "needs_docs"  # Missing documentation
    BLOCKED = "blocked"  # Hard stop, cannot proceed


class BlockReason(BaseModel):
    """Reason why output was blocked or flagged."""

    code: str
    detail: str
    severity: str = "high"  # critical, high, medium, low


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


# =============================================================================
# GATED OUTPUT SCHEMAS - Separate internal vs client-facing responses
# =============================================================================


class InternalAnalysisResponse(BaseModel):
    """Full analysis response for INTERNAL use only.

    Includes all data, routing decision, and block reasons.
    Use for internal dashboards, review queues, debugging.
    """

    evidence_id: UUID
    case_id: UUID | None = None
    status: AnalysisStatus
    routing: RoutingDecision
    block_reasons: list[BlockReason] = []

    # Full behavioral data
    behavioral_indicators: BehavioralIndicators | None = None
    fusion_results: FusionResults | None = None

    # Raw data
    ocr_text: str | None = None
    extracted_text_preview: str | None = None

    # Metadata
    processing_time_ms: float
    analyzed_at: datetime
    stages_completed: list[str] = []
    errors: list[str] = []


class ClientAnalysisResponse(BaseModel):
    """Safe analysis response for CLIENT-FACING use.

    Only returned when routing == AUTO_OK.
    Sensitive details are redacted.
    """

    evidence_id: UUID
    case_id: UUID | None = None
    status: str = "complete"
    classification: str | None = None
    confidence: float | None = None
    summary: str


class RoutingStatusResponse(BaseModel):
    """Response showing routing status for an evidence item."""

    evidence_id: UUID
    routing: RoutingDecision
    block_reasons: list[BlockReason] = []
    can_send_to_client: bool = False
    requires_review: bool = False
    missing_documentation: bool = False
